import argparse
import os
import random

import torch
import esm
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from plm_inv_subnetworks.dataset import data_io
from plm_inv_subnetworks.subnetwork.esm_masking_pl_logits import ESMMaskLearner
from plm_inv_subnetworks.subnetwork.modules import WeightedDifferentiableMask
from plm_inv_subnetworks.subnetwork.utils import get_dataloaders

from plm_inv_subnetworks.dataset.data_paths import RUN_DIR_PREFIX

def validate_config(config):
    if config["suppression_mode"] == "cath":
        if config["suppression_level"] is None:
            raise RuntimeError(f"Missing suppression level for suppression_mode 'cath'")
        if config["suppression_target"] is None:
            raise RuntimeError(f"Missing suppression_target for suppression_mode 'cath'")
        if config["suppression_level"] not in ["class", "architecture", "topology", "homologous_superfamily", "domain_num", "random"]:
                    raise RuntimeError(f"Invalid suppression_level '{config['suppression_level']}' for suppression_mode 'cath'")
            
    elif config["suppression_mode"] == "dssp":
        if config["suppression_target"] not in ["helix", "strand", "coil"]:
            raise RuntimeError(f"Invalid suppression_target '{config['suppression_target']}' for suppression_mode 'dssp'")

    else:
        raise RuntimeError(f"Invalid suppression_mode '{config['suppression_mode']}'")


def parse_args():
    parser = argparse.ArgumentParser(description='Training configuration')
    
    # Run configuration
    parser.add_argument('--run_name', type=str, default="test",
                      help='Name of the run')
    parser.add_argument('--resume_last', default=False,
                      help='Resume from last checkpoint')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                      help='WandB run ID')
    parser.add_argument('--run_dir', type=str, default=None,
                      help='Run directory')
    
    # Model parameters
    parser.add_argument('--suppression_mode', type=str, default=None,
                      help='Suppression info - cath or ssp')    
    parser.add_argument('--suppression_level', type=str, default=None,
                      help='Suppression level of CATH, default is None. Ignores for dssp mode.')
    parser.add_argument('--suppression_target', type=str, default=None,
                      help='Suppression target value, type str (will be cast later)')
    parser.add_argument('--suppression_lambda', type=float, default=1,
                      help='Suppression lambda value')
    parser.add_argument('--maintenance_lambda', type=float, default=1,
                      help='Maintenance lambda value')
    parser.add_argument('--maintenance_mlm_lambda', type=float, default=0.5,
                      help='Maintenance MLM lambda value')
    parser.add_argument('--sparsity_lambda_init', type=float, default=1,
                      help='Sparsity lambda init value')
    parser.add_argument('--sparsity_lambda_final', type=float, default=1,
                        help='Sparsity lambda final value')
    parser.add_argument('--sparsity_warmup_epochs', type=int, default=100,
                        help='Sparsity warmup epochs')
    parser.add_argument('--sparsity_ramp_epochs', type=int, default=100,
                        help='Sparsity ramp epochs')
    parser.add_argument('--random_n', type=int, default=None,
                      help='Random n for suppression')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--lr_phaseA', type=float, default=1e-4,
                      help='Learning rate for phase A')
    parser.add_argument('--lr_phaseB', type=float, default=1e-4,
                      help='Learning rate for phase B')
    parser.add_argument('--lr_hold_epochs', type=int, default=0,
                      help='Learning rate hold epoch')
    parser.add_argument('--lr_plateau_epochs', type=int, default=0,
                      help='Learning rate plateau epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size')
    parser.add_argument('--num_examples_per_batch', type=int, default=1,
                      help='Number of examples per batch for evenly sampled batches')
    parser.add_argument('--max_epochs', type=int, default=4,
                      help='Maximum number of epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=16,
                      help='Number of batches to accumulate gradients')
    parser.add_argument('--val_check_interval', type=int, default=1,
                      help='Validation check interval')
    parser.add_argument('--ckpt_freq', type=int, default=50,
                      help='Checkpoint frequency')
    
    # Model architecture
    parser.add_argument('--model_name', type=str, default="esm2_t33_650M_UR50D",
                      help='Model name')
    parser.add_argument('--mask_init_value', type=float, default=-4.595,
                      help='Mask initialization value')
    parser.add_argument('--mask_temp_init', type=float, default=1.0,
                      help='Mask temperature initialization value')
    parser.add_argument('--mask_temp_final', type=float, default=0.1,
                      help='Mask temperature final value')
    parser.add_argument('--mask_temp_decay', type=int, default=50,
                      help='Mask temperature decay epochs')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                      help='Mask threshold for masking')
    
    
    parser.add_argument('--mask_top_layer_frac', type=float, default=0.5,
                      help='Mask top percentage of self-attn layers')
    parser.add_argument('--mask_layer_range', type=lambda x: tuple(map(int, x.strip('()').split(','))), 
                    help='Tuple of (start_layer, end_layer) for masking', default=None)
    
    # Data parameters
    parser.add_argument('--min_n_res', type=int, default=64,
                      help='Minimum number of residues')
    parser.add_argument('--max_n_res', type=int, default=512,
                      help='Maximum number of residues')

    # Hardware/System
    parser.add_argument('--precision', type=str, default="16-mixed",
                      help='Precision for training')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of workers')
    
    # Other parameters
    parser.add_argument('--debug', default=False, type=bool,
                      help='Enable debug to run on 100 examples')
    parser.add_argument('--shuffle', default=True, type=bool,
                      help='Shuffle')
    parser.add_argument('--wandb_dir', type=str, 
                      default="../wandb",
                      help='WandB directory')
    parser.add_argument('--wandb_project', type=str, 
                      default="cath-class-subnetworks-test",
                      help='WandB project')
    
    return parser.parse_args()

if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    os.environ['MASTER_PORT'] = str(random.randint(29500, 29999))
    os.environ['MASTER_ADDR'] = 'localhost'
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    args = parse_args()


    config={
        "batch_size": args.batch_size,
        "num_examples_per_batch": args.num_examples_per_batch,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "val_check_interval": args.val_check_interval,
        "mask_init_value": args.mask_init_value,
        "mask_temperature_init": args.mask_temp_init,
        "mask_temperature_final": args.mask_temp_final,
        "mask_temperature_decay": args.mask_temp_decay,
        "precision": args.precision,
        "num_workers": args.num_workers,
        "ckpt_freq": args.ckpt_freq,
        "resume_last": args.resume_last,
        "wandb_run_id": args.wandb_run_id,

        "debug": args.debug,
        "shuffle": args.shuffle,
        "min_n_res": args.min_n_res,
        "max_n_res": args.max_n_res,
        
        "suppression_mode": args.suppression_mode,
        "suppression_level": args.suppression_level,
        "suppression_target": args.suppression_target,
        "random_n": args.random_n,
        "model": "esm2_t33_650M_UR50D",

        "suppression_lambda": args.suppression_lambda,
        "maintenance_lambda": args.maintenance_lambda,
        "maintenance_mlm_lambda": args.maintenance_mlm_lambda,
        "sparsity_lambda_init": args.sparsity_lambda_init,
        "sparsity_lambda_final": args.sparsity_lambda_final,
        "sparsity_warmup_epochs": args.sparsity_warmup_epochs,
        "mask_top_layer_frac": args.mask_top_layer_frac,
        "mask_layer_range": args.mask_layer_range,
        "lr_phaseA": args.lr_phaseA,
        "lr_phaseB": args.lr_phaseB,
        "lr_hold_epochs": args.lr_hold_epochs,
        "lr_plateau_epochs": args.lr_plateau_epochs,
        "mask_threshold": args.mask_threshold,
        "sparsity_ramp_epochs": args.sparsity_ramp_epochs,

        "wandb_dir": args.wandb_dir,
        "wandb_project": args.wandb_project,
        "run_name": args.run_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown")
    }

    validate_config(config)

    print("################# SUPPRESSION INFO ################# ")
    print(f"Suppression mode: {config['suppression_mode']}")
    print(f"Suppression level: {config['suppression_level']}")
    print(f"Suppression target: {config['suppression_target']}")
    print("#################################################### ")


    # model_dir = f"/users/rvinod/data/rvinod/repos/probing-subnetworks/runs_cath_class/{config['run_name']}_{config['slurm_job_id']}"
    model_dir = f"{RUN_DIR_PREFIX}/{config['run_name']}_{config['slurm_job_id']}"

    config["run_dir"] = model_dir
    ckpt_dir = f"{model_dir}/checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Model dir", model_dir)

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=config,
        group=args.run_name,
    )

    wandb_run_id = wandb.run.id
    config["wandb_run_id"] = wandb_run_id

    wandb_logger = WandbLogger(project=config["wandb_project"], name=config["run_name"], log_model=False)

    use_dssp = config["suppression_mode"] == "dssp"
    print("use_dssp", use_dssp)

    if config["suppression_level"] == "random":
        random_supp_id_path = f"{model_dir}/random_supp_ids.txt"
        config["random_supp_id_path"] = random_supp_id_path
    else:
        config["random_supp_id_path"] = None
        


    if config["suppression_mode"] == "cath" and config["suppression_level"] and config["suppression_target"]:
        train_ids, val_ids, test_ids, train_loader, val_loader = get_dataloaders(batch_size=config["batch_size"], 
                                                                        alphabet=alphabet, 
                                                                        even_sampling=True, 
                                                                        num_workers=config["num_workers"], 
                                                                        use_dssp=use_dssp,
                                                                        level=config["suppression_level"],
                                                                        target=config["suppression_target"],
                                                                        num_samples=config["num_examples_per_batch"],
                                                                        debug=config["debug"], 
                                                                        random_n=config["random_n"],
                                                                        random_supp_id_path=config["random_supp_id_path"],
                                                                        )

    else:
        train_ids, val_ids, test_ids, train_loader, val_loader = get_dataloaders(config["batch_size"], 
                                                                                alphabet, 
                                                                                debug=config["debug"], 
                                                                                shuffle=True,  
                                                                                num_workers=config["num_workers"], 
                                                                                use_dssp=use_dssp,
                                                                            random_n=config["random_n"],
        )



    data_io.write_dict_to_json({"train": train_ids, "val": val_ids, "test": test_ids}, f"{model_dir}/train_val_split.json")
    data_io.write_dict_to_json({"config": config}, f"{model_dir}/config.json")

    print("Created config", config)


    mask_learner = WeightedDifferentiableMask(
        model, 
        temp_init=config["mask_temperature_init"],
        temp_final=config["mask_temperature_final"],
        temp_decay=config["mask_temperature_decay"],
        init_value=config["mask_init_value"],
        mask_top_layer_frac=config["mask_top_layer_frac"],
        mask_layer_range=config["mask_layer_range"],
        mask_threshold=config["mask_threshold"],
    )


    lightning_model = ESMMaskLearner(
        model=model,
        mask_learner=mask_learner,
        suppression_mode=config["suppression_mode"],
        suppression_level=config["suppression_level"],
        suppression_target=config["suppression_target"],
        suppression_lambda=config["suppression_lambda"],
        maintenance_lambda=config["maintenance_lambda"],
        maintenance_mlm_lambda=config["maintenance_mlm_lambda"],
        sparsity_lambda_init=config["sparsity_lambda_init"],
        sparsity_lambda_final=config["sparsity_lambda_final"],
        sparsity_warmup_epochs=config["sparsity_warmup_epochs"], sparsity_ramp_epochs=config["sparsity_ramp_epochs"],
        learning_rate=config["learning_rate"],
        lr_phaseA=config["lr_phaseA"],
        lr_phaseB=config["lr_phaseB"],
        lr_hold_epochs=config["lr_hold_epochs"],
        lr_plateau_epochs=config["lr_plateau_epochs"],
        random_supp_id_path=config["random_supp_id_path"],
    )


    print(f"Number of training sequences: {len(train_ids)}")
    print(f"Number of validation sequences: {len(val_ids)}")


    wandb_logger.watch(
    lightning_model,
    log="gradients",    
    log_freq=4,      
    log_graph=True      
    )

    regular_checkpoint = ModelCheckpoint(
        dirpath=f"{ckpt_dir}/regular_checkpoints",
        filename='{epoch:02d}',
        every_n_epochs=config["ckpt_freq"],
        save_top_k=-1,
        save_last=True,
    )

    best_train_checkpoint = ModelCheckpoint(
        dirpath=f"{ckpt_dir}/best_train_maintenance_kl",
        filename='best-maintenance-kl-{epoch:02d}-{maintenance_kl_epoch:.3f}',
        monitor='maintenance_kl_epoch',
        mode='min',
        save_top_k=2,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')


    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        max_epochs=config["max_epochs"],
        log_every_n_steps=4,
        strategy="auto",
        accumulate_grad_batches=config["accumulate_grad_batches"],
        logger=wandb_logger, # Add WandB logger to trainer
        precision=config["precision"],
        check_val_every_n_epoch=config["val_check_interval"],
        callbacks=[regular_checkpoint, lr_monitor], # best_train_checkpoint

    )

    print("Set up trainer")

    try:
        trainer.fit(lightning_model, train_loader, val_loader) 

    finally:
        data_io.write_dict_to_json({"test": lightning_model.held_out_ids}, f"{model_dir}/test_split.json")
        wandb.finish()