import gc

import torch
import pytorch_lightning as pl

from plm_subnetworks.dataset import data_io
from plm_subnetworks.subnetwork.carp_modules import SubnetworkCARP
from plm_subnetworks.utils.metrics import (
    aggregate_over_seq,
    logits_kl,
    MaskedCrossEntropyLoss,
    PerSequenceMaskedCrossEntropyLoss,
)


class CarpMaskLearner(pl.LightningModule):
    def __init__(
        self,
        model,
        mask_learner,
        learning_rate=1e-2,
        lr_hold_epochs=75,
        lr_phaseA=3e-2,
        lr_phaseB=1.5e-3,
        lr_plateau_epochs=25,
        lr_warmup=0.05,
        maintenance_lambda=1,
        maintenance_mlm_lambda=1,
        random_supp_id_path=None,
        sparsity_lambda_final=0.5,
        sparsity_lambda_init=2,
        sparsity_ramp_epochs=150,
        sparsity_warmup_epochs=50,
        suppression_lambda=1.5,
        suppression_level="class",
        suppression_mode="cath",
        suppression_target="1",
    ):
        super().__init__()

        self.mask_learner = mask_learner
        self.suppression_mode = suppression_mode
        self.suppression_level = suppression_level
        self.suppression_target = suppression_target
        self.random_supp_id_path = random_supp_id_path
        self.random_supp_ids = (
            data_io.read_from_txt(random_supp_id_path) if random_supp_id_path else None
        )

        self.suppression_lambda = suppression_lambda
        self.maintenance_lambda = maintenance_lambda
        self.learning_rate = learning_rate
        self.maintenance_mlm_lambda = maintenance_mlm_lambda
        self.maintenance_mlm_lambda_init = maintenance_mlm_lambda

        self.sparsity_lambda_init = sparsity_lambda_init
        self.sparsity_lambda = sparsity_lambda_init
        self.sparsity_lambda_final = sparsity_lambda_final
        self.sparsity_warmup_epochs = sparsity_warmup_epochs
        self.sparsity_ramp_epochs = sparsity_ramp_epochs

        self.perseqmcel = PerSequenceMaskedCrossEntropyLoss()
        self.perresmcel = MaskedCrossEntropyLoss(weight=None, reduction="none")

        self.held_out_ids = []

        self.subnetwork = SubnetworkCARP(model, layers_to_mask=mask_learner.layers_to_mask)

        self.save_hyperparameters(ignore=["model", "mask_learner"])
        self.hparams["mask_config"] = self.mask_learner.config()

        self.lr_phaseA = lr_phaseA
        self.lr_phaseB = lr_phaseB
        self.lr_warmup = lr_warmup
        self.lr_hold_epochs = lr_hold_epochs
        self.lr_plateau_epochs = lr_plateau_epochs

        self.automatic_optimization = True

    def _get_cath_suppression_mask(self, batch):
        if self.suppression_level == "class":
            suppression_mask = torch.tensor(
                [i == int(self.suppression_target) for i in batch["cath_classes"]],
                dtype=torch.bool,
                device=self.device,
            )
        elif self.suppression_level == "architecture":
            suppression_mask = torch.tensor(
                [".".join(i.split(".")[:2]) == self.suppression_target for i in batch["cath_codes"]],
                dtype=torch.bool,
                device=self.device,
            )
        elif self.suppression_level == "topology":
            suppression_mask = torch.tensor(
                [".".join(i.split(".")[:3]) == self.suppression_target for i in batch["cath_codes"]],
                dtype=torch.bool,
                device=self.device,
            )
        elif self.suppression_level == "homologous_superfamily":
            suppression_mask = torch.tensor(
                [".".join(i.split(".")[:4]) == self.suppression_target for i in batch["cath_codes"]],
                dtype=torch.bool,
                device=self.device,
            )
        elif self.suppression_level == "domain_num":
            suppression_mask = torch.tensor(
                [int(i) == int(self.suppression_target) for i in batch["cath_domain_nums"]],
                dtype=torch.bool,
                device=self.device,
            )
        elif self.suppression_level == "random":
            suppression_mask = torch.tensor(
                [cath_id in self.random_supp_ids for cath_id in batch["cath_ids"]],
                dtype=torch.bool,
                device=self.device,
            )
        else:
            raise RuntimeError(f"Invalid suppression level for mode CATH. Got {self.suppression_level}")

        n_suppress = suppression_mask.sum()
        n_maintain = (~suppression_mask).sum()

        return suppression_mask, n_suppress, n_maintain

    def _get_dssp_suppression_mask(self, suppression_target, batch):
        if suppression_target == "helix":
            return batch["helix_mask"].to(self.device)
        if suppression_target == "strand":
            return batch["strand_mask"].to(self.device)
        if suppression_target == "coil":
            return batch["coil_mask"].to(self.device)
        raise RuntimeError(f"Invalid suppression target {suppression_target} for mode DSSP")

    def _uniform_ref(self, logits_like: torch.Tensor) -> torch.Tensor:
        vocab = logits_like.shape[-1]
        return torch.full_like(logits_like, 1.0 / vocab)

    def training_step(self, batch, batch_idx):
        tgt = batch["tgt"].to(self.device)
        src = batch["src"].to(self.device)
        seq_mask = batch["seq_mask"].to(self.device)
        corr_mask = batch["corr_mask"].to(self.device)

        if tgt.size(0) == 0:
            return None

        masks = self.mask_learner()

        with torch.no_grad():
            carp_base = self.subnetwork(
                tgt,
                base=True,
                repr_layers=[-1],
                logits=True,
            )["logits"]

        carp_base = carp_base.detach()
        carp_base_dist = torch.nn.functional.softmax(carp_base, dim=-1)

        subnet_logits = self.subnetwork(
            tgt,
            masks=masks,
            inverse=False,
            repr_layers=[-1],
            logits=True,
        )["logits"]
        subnet_logits_dist = torch.nn.functional.softmax(subnet_logits, dim=-1)
        unif_ref = self._uniform_ref(carp_base_dist)

        per_res_subnet_unif_kl = logits_kl(subnet_logits_dist, unif_ref, seq_mask, epsilon=1e-4)
        per_res_subnet_carp_kl = logits_kl(subnet_logits_dist, carp_base_dist, seq_mask, epsilon=1e-4)

        pred_logits = self.subnetwork(
            src,
            masks=masks,
            repr_layers=[-1],
            logits=True,
        )["logits"]

        if self.suppression_mode == "cath":
            suppression_mask, _, _ = self._get_cath_suppression_mask(batch)

            subnet_unif_kl = aggregate_over_seq(per_res_subnet_unif_kl, seq_mask)
            subnet_carp_kl = aggregate_over_seq(per_res_subnet_carp_kl, seq_mask)

            suppression_kl = (subnet_unif_kl * suppression_mask).sum() / suppression_mask.sum()
            maintenance_kl = (subnet_carp_kl * ~suppression_mask).sum() / (~suppression_mask).sum()

            mlm_loss = self.perseqmcel(pred_logits, tgt, corr_mask)
            maintenance_mlm_loss = (mlm_loss * ~suppression_mask).sum() / (~suppression_mask).sum()

        elif self.suppression_mode == "dssp":
            suppression_mask = self._get_dssp_suppression_mask(self.suppression_target, batch)
            masked_and_not_suppressed = corr_mask.bool() & ~suppression_mask

            suppression_kl = (per_res_subnet_unif_kl * suppression_mask).sum() / suppression_mask.sum()
            maintenance_kl = (per_res_subnet_carp_kl * ~suppression_mask).sum() / (~suppression_mask).sum()

            if masked_and_not_suppressed.sum() == 0 or suppression_mask.sum() == 0 or (~suppression_mask).sum() == 0:
                return None

            maintenance_mlm_loss = self.perresmcel(pred_logits, tgt, masked_and_not_suppressed).mean()
        else:
            raise RuntimeError(f"Invalid suppression mode '{self.suppression_mode}'")

        sparsity_loss = self.mask_learner.compute_sparsity_loss()
        sparsity = self.mask_learner.get_sparsity()
        sparsity_lambda = self.current_sparsity_lambda

        total_loss = (
            self.suppression_lambda * suppression_kl
            + self.maintenance_lambda * maintenance_kl
            + sparsity_lambda * sparsity_loss
            + self.maintenance_mlm_lambda * maintenance_mlm_loss
        )

        self.log("total_loss", total_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("sparsity", sparsity, on_step=True, on_epoch=True, prog_bar=True)
        self.log("sparsity_lambda", sparsity_lambda, on_step=False, on_epoch=True, prog_bar=True)
        self.log("suppression_kl", suppression_kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("maintenance_kl", maintenance_kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("sparsity_loss", sparsity_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("maintenance_mlm_loss", maintenance_mlm_loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": total_loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        src = batch["src"].to(self.device)
        tgt = batch["tgt"].to(self.device)
        corr_mask = batch["corr_mask"].to(self.device)

        if tgt.size(0) == 0:
            return None

        if self.suppression_mode == "cath":
            suppression_mask, n_suppress, n_maintain = self._get_cath_suppression_mask(batch)
            if n_suppress == 0 or n_maintain == 0 or suppression_mask.size(0) == 1:
                self.held_out_ids.extend(batch["cath_ids"])
                return None

        subnet_src_logits = self.subnetwork(
            src,
            masks=self.val_masks,
            repr_layers=[-1],
            logits=True,
        )["logits"]

        if self.suppression_mode == "cath":
            mlm_loss = self.perseqmcel(subnet_src_logits, tgt, corr_mask)
            if n_suppress > 0:
                suppression_mlm_loss = (mlm_loss * suppression_mask).sum() / suppression_mask.sum()
                self.log("val/suppression_mlm_loss", suppression_mlm_loss.item())

            if n_maintain > 0:
                maintenance_mlm_loss = (mlm_loss * ~suppression_mask).sum() / (~suppression_mask).sum()
                self.log("val/maintenance_mlm_loss", maintenance_mlm_loss.item())

            batch_mlm = mlm_loss.mean()

        elif self.suppression_mode == "dssp":
            suppression_mask = self._get_dssp_suppression_mask(self.suppression_target, batch)
            masked_and_suppressed = corr_mask.bool() & suppression_mask
            masked_and_not_suppressed = corr_mask.bool() & ~suppression_mask

            if (
                masked_and_suppressed.sum() == 0
                or masked_and_not_suppressed.sum() == 0
                or suppression_mask.sum() == 0
                or (~suppression_mask).sum() == 0
            ):
                return None

            suppression_mlm_loss = self.perresmcel(subnet_src_logits, tgt, masked_and_suppressed).mean()
            maintenance_mlm_loss = self.perresmcel(subnet_src_logits, tgt, masked_and_not_suppressed).mean()

            batch_mlm = self.perresmcel(subnet_src_logits, tgt, corr_mask).mean()

            self.log("val/suppression_mlm_loss", suppression_mlm_loss.item())
            self.log("val/maintenance_mlm_loss", maintenance_mlm_loss.mean().item())
            mlm_loss = batch_mlm
        else:
            raise RuntimeError(f"Invalid suppression mode '{self.suppression_mode}'")

        self.log_dict(
            {
                "val/batch_mlm_loss": batch_mlm.item(),
                "val/sparsity": self.mask_learner.get_sparsity(),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "val_mlm_loss": mlm_loss.mean(),
        }

    @property
    def current_sparsity_lambda(self) -> float:
        e = self.current_epoch
        sl0 = self.sparsity_lambda_init
        slF = self.sparsity_lambda_final

        if e < self.sparsity_warmup_epochs:
            sparsity_l = sl0
        elif e < self.sparsity_warmup_epochs + self.sparsity_ramp_epochs:
            frac = (e - self.sparsity_warmup_epochs) / self.sparsity_ramp_epochs
            sparsity_l = sl0 + frac * (slF - sl0)
        else:
            sparsity_l = slF

        return sparsity_l

    def on_validation_epoch_start(self):
        if hasattr(self, "val_masks"):
            del self.val_masks
            torch.cuda.empty_cache()

        with torch.no_grad():
            self.val_masks = self.mask_learner()

    def on_validation_batch_start(self, *args, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_batch_start(self, *args, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.mask_learner.parameters(), lr=self.learning_rate)

        def lr_lambda(current_epoch):
            if current_epoch < self.lr_hold_epochs:
                return 1.0
            plateau_start = self.lr_hold_epochs
            plateau_end = plateau_start + self.lr_plateau_epochs
            if plateau_start <= current_epoch < plateau_end:
                return self.lr_phaseA / self.learning_rate
            return self.lr_phaseB / self.learning_rate

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]
