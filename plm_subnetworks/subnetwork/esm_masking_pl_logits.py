import gc

import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR

import plm_subnetworks.dataset.data_io as data_io
from plm_subnetworks.subnetwork.modules import SubnetworkESM
from plm_subnetworks.utils.metrics import (
    aggregate_over_seq,
    logits_kl,
    MaskedCrossEntropyLoss,
    PerSequenceMaskedCrossEntropyLoss,
)

# Enable anomaly detection after importing torch
torch.autograd.set_detect_anomaly(True)

class ESMMaskLearner(pl.LightningModule):
    def __init__(self, 
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
                suppression_target="alpha", 
                ):

        super().__init__()
        self.mask_learner = mask_learner
        self.suppression_mode = suppression_mode
        self.suppression_level = suppression_level
        self.suppression_target = suppression_target
        self.random_supp_id_path = random_supp_id_path
        self.random_supp_ids = data_io.read_from_txt(random_supp_id_path) if random_supp_id_path else None
        
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
        self.perresmcel = MaskedCrossEntropyLoss(weight=None, reduction='none')

        self.held_out_ids = []

        self.subnetwork = SubnetworkESM(model, layers_to_mask=mask_learner.layers_to_mask)

        # Save hyperparameters for wandb logging
        self.save_hyperparameters(ignore=['model', 'mask_learner'])
        self.hparams["mask_config"] = self.mask_learner.config()
    
        # Learning-rate schedule
        self.lr_phaseA = lr_phaseA              # high LR for mask recovery
        self.lr_phaseB = lr_phaseB              # lower LR for pruning / fine-settling
        self.lr_warmup = lr_warmup              # 5% warm-up inside each phase
        self.lr_hold_epochs = lr_hold_epochs    
        self.lr_plateau_epochs = lr_plateau_epochs

        # Enable automatic optimization
        self.automatic_optimization = True


    def _get_cath_suppression_mask(self, batch):
        if self.suppression_level == "class":
             suppression_mask = torch.tensor(
                [i == int(self.suppression_target) for i in batch["cath_classes"]],
                dtype=torch.bool, device=self.device
            )
        elif self.suppression_level == "architecture":

            suppression_mask = torch.tensor(
                ['.'.join(i.split('.')[:2]) == self.suppression_target for i in batch["cath_codes"]],
                dtype=torch.bool, device=self.device
            )

        elif self.suppression_level == "topology":

            suppression_mask = torch.tensor(
                ['.'.join(i.split('.')[:3]) == self.suppression_target for i in batch["cath_codes"]],
                dtype=torch.bool, device=self.device
            )
             
        elif self.suppression_level == "homologous_superfamily":

            suppression_mask = torch.tensor(
                ['.'.join(i.split('.')[:4]) == self.suppression_target for i in batch["cath_codes"]],
                dtype=torch.bool, device=self.device
            )
        elif self.suppression_level == "domain_num":
            suppression_mask = torch.tensor(
                [int(i) == int(self.suppression_target) for i in batch["cath_domain_nums"]],
                dtype=torch.bool, device=self.device
            )
        elif self.suppression_level == "random":
            
            suppression_mask = torch.tensor(
                [cath_id in self.random_supp_ids for cath_id in batch["cath_ids"]],
                dtype=torch.bool, device=self.device
            )
        
        else:
            raise RuntimeError(f"Invalid suppression level for mode CATH. Got {self.level}")
                           
        n_suppress = suppression_mask.sum()
        n_maintain = (~suppression_mask).sum()
        
        return suppression_mask, n_suppress, n_maintain

    def _get_dssp_suppression_mask(self, suppression_target, batch):
        if suppression_target == "helix":
            suppression_mask = batch["helix_mask"]
        elif suppression_target == "strand":
            suppression_mask = batch["strand_mask"]
        elif suppression_target == "coil":
            suppression_mask = batch["coil_mask"]
        return suppression_mask
        
    def training_step(self, batch, batch_idx):

        tgt = batch["tgt"]
        src = batch["src"]

        if tgt.size(0) == 0:
            return None
        
        masks = self.mask_learner()

        with torch.no_grad():
            esm_logits = self.subnetwork(tgt, base=True, masks=None, return_contacts=False)["logits"]

        esm_logits = esm_logits.detach()
        esm_logits_dist = torch.nn.functional.softmax(esm_logits, dim=-1)

        subnet_logits = self.subnetwork(tgt, inverse=False, masks=masks, return_contacts=False)["logits"]
        subnet_logists_dist = torch.nn.functional.softmax(subnet_logits, dim=-1)
        unif_ref = torch.full_like(esm_logits_dist, 1 / len(self.subnetwork.esm.alphabet.all_toks))
        
        per_res_subnet_unif_kl = logits_kl(subnet_logists_dist, unif_ref, batch["seq_mask"], epsilon=1e-4)
        per_res_subnet_esm_kl = logits_kl(subnet_logists_dist, esm_logits_dist, batch["seq_mask"], epsilon=1e-4)

        pred_logits = self.subnetwork(src, masks=masks, return_contacts=False)["logits"]

        if self.suppression_mode == "cath":
            
            suppression_mask, _, _ = self._get_cath_suppression_mask(batch) 
            
            subnet_unif_kl = aggregate_over_seq(per_res_subnet_unif_kl, batch["seq_mask"])
            subnet_esm_kl = aggregate_over_seq(per_res_subnet_esm_kl, batch["seq_mask"])

            suppression_kl = (subnet_unif_kl * suppression_mask).sum() / suppression_mask.sum()
            maintenance_kl = (subnet_esm_kl * ~suppression_mask).sum() / (~suppression_mask).sum()

            mlm_loss = self.perseqmcel(pred_logits, tgt, batch["corr_mask"])
            maintenance_mlm_loss = (mlm_loss * ~suppression_mask).sum() / (~suppression_mask).sum()

        if self.suppression_mode == "dssp":

            suppression_mask = self._get_dssp_suppression_mask(self.suppression_target, batch)
            masked_and_not_suppressed = batch["corr_mask"].bool() & ~suppression_mask   

            suppression_kl = (per_res_subnet_unif_kl * suppression_mask).sum() / suppression_mask.sum()
            maintenance_kl = (per_res_subnet_esm_kl * ~suppression_mask).sum() / (~suppression_mask).sum()
            
            # This happens once every 150k steps
            if masked_and_not_suppressed.sum() == 0 or suppression_mask.sum()==0 or (~suppression_mask).sum() == 0:
                return None

            maintenance_mlm_loss = self.perresmcel(pred_logits, tgt, masked_and_not_suppressed).mean()

        sparsity_loss = self.mask_learner.compute_sparsity_loss()
        sparsity = self.mask_learner.get_sparsity()
        sparsity_lambda = self.current_sparsity_lambda

        total_loss = (
                       self.suppression_lambda * suppression_kl 
                    +  self.maintenance_lambda * maintenance_kl 
                    +  sparsity_lambda * sparsity_loss
                    +  self.maintenance_mlm_lambda * maintenance_mlm_loss
                    )

        self.log("total_loss", total_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("sparsity", sparsity, on_step=True, on_epoch=True, prog_bar=True)
        self.log("sparsity_lambda", sparsity_lambda, on_step=False, on_epoch=True, prog_bar=True)
        self.log("suppression_kl", suppression_kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("maintenance_kl", maintenance_kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("sparsity_loss", sparsity_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('maintenance_mlm_loss', maintenance_mlm_loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": total_loss}
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        
        src = batch["src"]
        tgt = batch["tgt"]
        
        if tgt.size(0) == 0:
            return None

        # per sequence suppression
        if self.suppression_mode == "cath":
            suppression_mask, n_suppress, n_maintain = self._get_cath_suppression_mask(batch)
            if n_suppress == 0 or n_maintain == 0 or suppression_mask.size(0) == 1:
                self.held_out_ids.extend(batch["cath_ids"])
                return None
            
        ## --- Compute MLM Predictions --- ##
        subnet_src_logits = self.subnetwork(src, masks=self.val_masks, return_contacts=False)["logits"]

        if self.suppression_mode == "cath":
            mlm_loss = self.perseqmcel(subnet_src_logits, tgt, batch["corr_mask"])
            if n_suppress > 0:
                suppression_mlm_loss = (mlm_loss * suppression_mask).sum() / suppression_mask.sum()
                self.log('val/suppression_mlm_loss', suppression_mlm_loss.item())

            if n_maintain > 0:
                maintenance_mlm_loss = (mlm_loss * ~suppression_mask).sum() / (~suppression_mask).sum()
                self.log('val/maintenance_mlm_loss', maintenance_mlm_loss.item())

            batch_mlm = mlm_loss.mean()
        
        if self.suppression_mode == "dssp":

            suppression_mask = self._get_dssp_suppression_mask(self.suppression_target, batch)
            masked_and_suppressed = batch["corr_mask"].bool() & suppression_mask        # use for suppression MLM loss
            masked_and_not_suppressed = batch["corr_mask"].bool() & ~suppression_mask   # use for maintenance MLM loss

            if masked_and_suppressed.sum() == 0 or masked_and_not_suppressed.sum() == 0 or suppression_mask.sum()==0 or (~suppression_mask).sum() == 0:
                return None

            suppression_mlm_loss = self.perresmcel(subnet_src_logits, tgt, masked_and_suppressed).mean()
            maintenance_mlm_loss = self.perresmcel(subnet_src_logits, tgt, masked_and_not_suppressed).mean()

            batch_mlm = self.perresmcel(subnet_src_logits, tgt, batch["corr_mask"]).mean()

            self.log('val/suppression_mlm_loss', suppression_mlm_loss.item())
            self.log('val/maintenance_mlm_loss', maintenance_mlm_loss.mean().item())
            mlm_loss = batch_mlm
       

        self.log_dict({
        'val/batch_mlm_loss': batch_mlm.item(),
        'val/sparsity': self.mask_learner.get_sparsity(),
        }, prog_bar=True, on_step=False, on_epoch=True)

      
        return {
            'val_mlm_loss': mlm_loss.mean(),
        }

    # We set the sparsity coefficient to 0 for all subnetworks in our results but 
    # provide for completeness of the framework in https://arxiv.org/abs/2310.03084
    @property
    def current_sparsity_lambda(self) -> float:
        """
        Two‑phase schedule:
        Phase‑1: 0 … warmup_epochs‑1        (λ = λ_init)
        Phase‑2: warmup … warmup+ramp‑1     (linear → λ_final)
        Steady : everything after Phase‑2   (λ = λ_final)
        """
        e   = self.current_epoch
        sl0 = self.sparsity_lambda_init
        slF = self.sparsity_lambda_final

        # Phase‑1: hold 
        if e < self.sparsity_warmup_epochs:
            sparsity_l = sl0

        # Phase‑2: linear ramp 
        elif e < self.sparsity_warmup_epochs + self.sparsity_ramp_epochs:
            frac = (e - self.sparsity_warmup_epochs) / self.sparsity_ramp_epochs  # 0→1
            sparsity_l = sl0 + frac * (slF - sl0)

        # Steady‑state 
        else:
            sparsity_l = slF

        return sparsity_l


    def on_validation_epoch_start(self):
        """Generate and store validation masks."""
        if hasattr(self, 'val_masks'):
            del self.val_masks
            torch.cuda.empty_cache()
        
        # Generate new masks
        with torch.no_grad():  # Ensure no gradients are stored
            self.val_masks = self.mask_learner()

    def on_validation_batch_start(self, *args, **kwargs):
        """Pre-batch memory cleanup for validation."""
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_batch_start(self, *args, **kwargs):
        """Pre-batch memory cleanup for training."""
        torch.cuda.empty_cache()
        gc.collect()
    
    def on_train_batch_end(self, *args, **kwargs):
        """Pre-batch memory cleanup for training."""
        torch.cuda.empty_cache()
        gc.collect()
        
    def on_train_epoch_end(self, *args, **kwargs):
        """Pre-batch memory cleanup for training."""
        self.log("temperature_epoch",
             self.mask_learner.temperature,
             on_step=False, on_epoch=True, prog_bar=True)
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_epoch_start(self):
        self.mask_learner.scale_temp(self.current_epoch)

    def on_val_epoch_end(self, *args, **kwargs):
        """Pre-batch memory cleanup for training."""
        torch.cuda.empty_cache()
        gc.collect()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["mask_temperature"] = self.mask_learner.temperature
        checkpoint["sparsity_lambda"] = self.sparsity_lambda

    def on_load_checkpoint(self, checkpoint):
        self.mask_learner.temperature = checkpoint["mask_temperature"]
        self.sparsity_lambda = checkpoint["sparsity_lambda"]


    def configure_optimizers(self):
        # freeze reference‑model weights 
        for n, p in self.named_parameters():
            if "reference_model" in n:
                p.requires_grad = False

        # mask‑learner parameters only 
        mask_params = [p for n, p in self.named_parameters()
                    if "mask_learner" in n and p.requires_grad]
        print(f"Total mask‑learner parameters to optimise: {len(mask_params):,}")

        # peak / starting LR  
        peak_lr = self.learning_rate          

        optimiser = torch.optim.AdamW(
            mask_params,
            lr=peak_lr,                       
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.0,
        )

        # reverse one‑cycle schedule
        total_steps = int(self.trainer.estimated_stepping_batches)   
        scheduler = OneCycleLR(
            optimiser,
            max_lr=peak_lr,                 
            total_steps=total_steps,
            pct_start= 1 / self.lr_plateau_epochs,                 
            div_factor=1.0,                 
            final_div_factor=30.0,          
            anneal_strategy="cos",          
            three_phase=False,             
        )

        return [optimiser], [{
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "one_cycle_lr",
        }]

