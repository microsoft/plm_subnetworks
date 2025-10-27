import gc
from typing import Dict, Optional

import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR

import plm_subnetworks.dataset.data_io as data_io
from plm_subnetworks.subnetwork.dayhoff_modules import (
    SubnetworkDayhoff,
    WeightedDifferentiableMaskDayhoff,
)
from plm_subnetworks.utils.metrics import (
    aggregate_over_seq,
    logits_kl,
    ce_loss_tokens,
)

# Enable anomaly detection after importing torch
torch.autograd.set_detect_anomaly(True)


def _infer_vocab_size(model: torch.nn.Module) -> Optional[int]:
    vocab_size = getattr(getattr(model, "config", object()), "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return int(model.lm_head.weight.size(0))
    return None


def _zero_scalar_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


class DayhoffMaskLearner(pl.LightningModule):
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
        use_corrupted_inputs=True,
    ):

        super().__init__()
        self.mask_learner = mask_learner
        self.suppression_mode = suppression_mode
        self.suppression_level = suppression_level
        self.suppression_target = suppression_target
        self.random_supp_id_path = random_supp_id_path
        self.random_supp_ids = (
            data_io.read_from_txt(random_supp_id_path)
            if random_supp_id_path
            else None
        )

        self.use_corrupted_inputs = use_corrupted_inputs
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

        self.held_out_ids = []

        self.subnetwork = SubnetworkDayhoff(
            model,
            layers_to_mask=mask_learner.layers_to_mask,
            extra_attention_tokens=mask_learner.extra_attention_tokens,
        )
        self.vocab_size = _infer_vocab_size(model)

        # Save hyperparameters for wandb logging
        self.save_hyperparameters(ignore=["model", "mask_learner"])
        self.hparams["mask_config"] = self.mask_learner.config()

        # Learning-rate schedule
        self.lr_phaseA = lr_phaseA
        self.lr_phaseB = lr_phaseB
        self.lr_warmup = lr_warmup
        self.lr_hold_epochs = lr_hold_epochs
        self.lr_plateau_epochs = lr_plateau_epochs

        self.automatic_optimization = True

    def _seq_mask_from_batch(self, batch):
        """Build a per-token mask excluding BOS/EOS/CLS/SEP/PAD."""
        attn = batch["attention_mask"].to(self.device)
        seq_mask = attn.bool()
        if seq_mask.numel() == 0:
            return seq_mask

        # zero out first token (CLS/BOS)
        seq_mask[:, 0] = False

        lengths = attn.sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        rows = torch.arange(seq_mask.size(0), device=seq_mask.device)
        seq_mask[rows, last_idx] = False
        return seq_mask

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
            raise RuntimeError(
                f"Invalid suppression level '{self.suppression_level}' for mode '{self.suppression_mode}'."
            )
        return suppression_mask, suppression_mask.sum(), (~suppression_mask).sum()

    def _get_dssp_suppression_mask(self, suppression_target, batch):
        if suppression_target == "helix":
            return batch["helix_mask"]
        if suppression_target == "strand":
            return batch["strand_mask"]
        if suppression_target == "coil":
            return batch["coil_mask"]
        raise RuntimeError(f"Unknown DSSP suppression target: {suppression_target}")

    def training_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        attn = batch["attention_mask"]

        if tgt.size(0) == 0:
            return None

        src_inputs = src if self.use_corrupted_inputs else tgt

        seq_mask = self._seq_mask_from_batch(batch)
        masks = self.mask_learner()

        with torch.no_grad():
            base_out = self.subnetwork(input_ids=tgt, attention_mask=attn, base=True, masks=None)
            base_logits = base_out.logits
            base_probs = torch.softmax(base_logits, dim=-1)

        subnet_clean_out = self.subnetwork(input_ids=tgt, attention_mask=attn, masks=masks, inverse=False)
        subnet_logits_clean = subnet_clean_out.logits
        subnet_probs_clean = torch.softmax(subnet_logits_clean, dim=-1)

        vocab_size = self.vocab_size or subnet_probs_clean.size(-1)
        unif_ref = torch.full_like(subnet_probs_clean, 1.0 / vocab_size)

        per_res_subnet_unif_kl = logits_kl(subnet_probs_clean, unif_ref, seq_mask, epsilon=1e-4)
        per_res_subnet_base_kl = logits_kl(subnet_probs_clean, base_probs, seq_mask, epsilon=1e-4)

        subnet_src_out = self.subnetwork(input_ids=src_inputs, attention_mask=attn, masks=masks, inverse=False)
        pred_logits = subnet_src_out.logits

        logits_shifted = pred_logits[:, :-1, :].contiguous()
        tgt_shifted = tgt[:, 1:].contiguous()
        attn_shifted = attn[:, 1:]

        if self.suppression_mode == "cath":
            seq_sup_mask, _, _ = self._get_cath_suppression_mask(batch)

            subnet_unif_kl = aggregate_over_seq(per_res_subnet_unif_kl, seq_mask)
            subnet_base_kl = aggregate_over_seq(per_res_subnet_base_kl, seq_mask)

            suppression_kl = (
                subnet_unif_kl[seq_sup_mask].mean()
                if seq_sup_mask.any()
                else _zero_scalar_like(subnet_unif_kl)
            )
            maintenance_kl = (
                subnet_base_kl[~seq_sup_mask].mean()
                if (~seq_sup_mask).any()
                else _zero_scalar_like(subnet_base_kl)
            )

            per_seq_ce, _, _ = ce_loss_tokens(
                logits_shifted,
                tgt_shifted,
                attention_mask=attn_shifted,
                ignore_index=-100,
            )
            ce_loss_per_seq = torch.as_tensor(per_seq_ce, device=self.device, dtype=pred_logits.dtype)
            valid = torch.isfinite(ce_loss_per_seq)

            valid_supp = seq_sup_mask & valid
            suppression_ce_loss = (
                ce_loss_per_seq[valid_supp].mean()
                if valid_supp.any()
                else _zero_scalar_like(ce_loss_per_seq)
            )

            valid_maint = (~seq_sup_mask) & valid
            maintenance_ce_loss = (
                ce_loss_per_seq[valid_maint].mean()
                if valid_maint.any()
                else _zero_scalar_like(ce_loss_per_seq)
            )

        elif self.suppression_mode == "dssp":
            tok_sup_mask = self._get_dssp_suppression_mask(self.suppression_target, batch).to(self.device)
            tok_sup_mask_shifted = tok_sup_mask[:, 1:]

            suppression_kl = (
                per_res_subnet_unif_kl[tok_sup_mask].mean()
                if tok_sup_mask.any()
                else _zero_scalar_like(per_res_subnet_unif_kl)
            )
            maintenance_kl = (
                per_res_subnet_base_kl[~tok_sup_mask].mean()
                if (~tok_sup_mask).any()
                else _zero_scalar_like(per_res_subnet_base_kl)
            )

            supp_ce, _, _ = ce_loss_tokens(
                logits_shifted,
                tgt_shifted,
                attention_mask=attn_shifted,
                mask=tok_sup_mask_shifted,
                ignore_index=-100,
            )
            supp_tensor = torch.as_tensor(supp_ce, device=self.device, dtype=pred_logits.dtype)
            valid_supp = torch.isfinite(supp_tensor)
            if not valid_supp.any():
                return None
            suppression_ce_loss = supp_tensor[valid_supp].mean()

            maint_mask = (~tok_sup_mask_shifted).bool()
            maint_ce, _, _ = ce_loss_tokens(
                logits_shifted,
                tgt_shifted,
                attention_mask=attn_shifted,
                mask=maint_mask,
                ignore_index=-100,
            )
            maint_tensor = torch.as_tensor(maint_ce, device=self.device, dtype=pred_logits.dtype)
            valid_maint = torch.isfinite(maint_tensor)
            if not valid_maint.any():
                return None
            maintenance_ce_loss = maint_tensor[valid_maint].mean()
        else:
            raise RuntimeError(f"Unknown suppression_mode: {self.suppression_mode}")

        sparsity_loss = self.mask_learner.compute_sparsity_loss()
        sparsity = self.mask_learner.get_sparsity()
        sparsity_lambda = self.current_sparsity_lambda

        total_loss = (
            self.suppression_lambda * suppression_kl
            + self.maintenance_lambda * maintenance_kl
            + sparsity_lambda * sparsity_loss
            + self.maintenance_mlm_lambda * maintenance_ce_loss
        )

        self.log("total_loss", total_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("sparsity", sparsity, on_step=True, on_epoch=True, prog_bar=True)
        self.log("sparsity_lambda", sparsity_lambda, on_step=False, on_epoch=True, prog_bar=True)
        self.log("suppression_kl", suppression_kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("maintenance_kl", maintenance_kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("sparsity_loss", sparsity_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("suppression_ce_loss", suppression_ce_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("maintenance_ce_loss", maintenance_ce_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("suppression_mlm_loss", suppression_ce_loss.item(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("maintenance_mlm_loss", maintenance_ce_loss.item(), on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": total_loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        attn = batch["attention_mask"]

        if tgt.size(0) == 0:
            return None

        src_inputs = src if self.use_corrupted_inputs else tgt

        seq_mask = self._seq_mask_from_batch(batch)

        if not hasattr(self, "val_masks") or self.val_masks is None:
            self.val_masks = self.mask_learner()

        subnet_src_logits = self.subnetwork(
            input_ids=src_inputs,
            attention_mask=attn,
            masks=self.val_masks,
            inverse=False,
        ).logits

        if self.suppression_mode == "cath":
            seq_sup_mask, n_suppress, n_maintain = self._get_cath_suppression_mask(batch)
            logits_shifted = subnet_src_logits[:, :-1, :].contiguous()
            tgt_shifted = tgt[:, 1:].contiguous()
            attn_shifted = attn[:, 1:]
            per_seq_ce, _, _ = ce_loss_tokens(
                logits_shifted,
                tgt_shifted,
                attention_mask=attn_shifted,
                ignore_index=-100,
            )
            ce_loss_per_seq = torch.as_tensor(per_seq_ce, device=self.device, dtype=subnet_src_logits.dtype)
            valid = torch.isfinite(ce_loss_per_seq)

            if (n_suppress == 0) or (n_maintain == 0) or (seq_sup_mask.numel() <= 1):
                self.held_out_ids.extend(batch["cath_ids"])
                return None

            valid_supp = seq_sup_mask & valid
            valid_maint = (~seq_sup_mask) & valid
            if not valid_supp.any() or not valid_maint.any():
                return None

            suppression_ce_loss = ce_loss_per_seq[valid_supp].mean()
            maintenance_ce_loss = ce_loss_per_seq[valid_maint].mean()

        elif self.suppression_mode == "dssp":
            tok_sup_mask = self._get_dssp_suppression_mask(self.suppression_target, batch)
            tok_sup_mask_shifted = tok_sup_mask[:, 1:]

            logits_shifted = subnet_src_logits[:, :-1, :].contiguous()
            tgt_shifted = tgt[:, 1:].contiguous()
            attn_shifted = attn[:, 1:]

            supp_ce, _, _ = ce_loss_tokens(
                logits_shifted,
                tgt_shifted,
                attention_mask=attn_shifted,
                mask=tok_sup_mask_shifted,
                ignore_index=-100,
            )
            supp_tensor = torch.as_tensor(supp_ce, device=self.device, dtype=subnet_src_logits.dtype)
            maintenance_ce, _, _ = ce_loss_tokens(
                logits_shifted,
                tgt_shifted,
                attention_mask=attn_shifted,
                mask=(~tok_sup_mask_shifted).bool(),
                ignore_index=-100,
            )
            maint_tensor = torch.as_tensor(maintenance_ce, device=self.device, dtype=subnet_src_logits.dtype)

            valid_supp = torch.isfinite(supp_tensor)
            valid_maint = torch.isfinite(maint_tensor)
            if not valid_supp.any() or not valid_maint.any():
                return None

            suppression_ce_loss = supp_tensor[valid_supp].mean()
            maintenance_ce_loss = maint_tensor[valid_maint].mean()

        else:
            raise RuntimeError(f"Unknown suppression_mode: {self.suppression_mode}")

        self.log(
            "val/suppression_ce_loss",
            suppression_ce_loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/maintenance_ce_loss",
            maintenance_ce_loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/suppression_mlm_loss",
            suppression_ce_loss.item(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/maintenance_mlm_loss",
            maintenance_ce_loss.item(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/sparsity",
            self.mask_learner.get_sparsity(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return {"val_mlm_loss": maintenance_ce_loss.mean()}

    @property
    def current_sparsity_lambda(self) -> float:
        e = self.current_epoch
        sl0 = self.sparsity_lambda_init
        slF = self.sparsity_lambda_final

        if e < self.sparsity_warmup_epochs:
            return sl0
        if e < self.sparsity_warmup_epochs + self.sparsity_ramp_epochs:
            frac = (e - self.sparsity_warmup_epochs) / max(self.sparsity_ramp_epochs, 1)
            return sl0 + frac * (slF - sl0)
        return slF

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

    def on_train_batch_end(self, *args, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_epoch_end(self, *args, **kwargs):
        self.log(
            "temperature_epoch",
            self.mask_learner.temperature,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_epoch_start(self):
        self.mask_learner.scale_temp(self.current_epoch)

    def on_val_epoch_end(self, *args, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

    def on_save_checkpoint(self, checkpoint: Dict):
        checkpoint["mask_temperature"] = self.mask_learner.temperature
        checkpoint["sparsity_lambda"] = self.sparsity_lambda

    def on_load_checkpoint(self, checkpoint: Dict):
        self.mask_learner.temperature = checkpoint["mask_temperature"]
        self.sparsity_lambda = checkpoint["sparsity_lambda"]

    def configure_optimizers(self):
        mask_params = [
            p
            for n, p in self.named_parameters()
            if "mask_learner" in n and p.requires_grad
        ]
        optimiser = torch.optim.AdamW(
            mask_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.0,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = OneCycleLR(
            optimiser,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=1 / max(self.lr_plateau_epochs, 1),
            div_factor=1.0,
            final_div_factor=30.0,
            anneal_strategy="cos",
            three_phase=False,
        )

        return [optimiser], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "one_cycle_lr",
            }
        ]


__all__ = ["DayhoffMaskLearner"]
