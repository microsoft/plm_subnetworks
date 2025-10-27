import argparse
import os
import random

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from sequence_models.pretrained import load_model_and_alphabet

from plm_subnetworks.dataset import data_io
from plm_subnetworks.subnetwork.carp_masking_pl_logits import CarpMaskLearner
from plm_subnetworks.subnetwork.carp_modules import WeightedDifferentiableMaskCARP
from plm_subnetworks.subnetwork.carp_utils import get_dataloaders
from plm_subnetworks.dataset.data_paths import RUN_DIR_PREFIX


torch.autograd.set_detect_anomaly(True)




def validate_config(config):
    if config["suppression_mode"] == "cath":
        if config["suppression_level"] is None:
            raise RuntimeError("Missing suppression level for suppression_mode 'cath'")
        if config["suppression_target"] is None:
            raise RuntimeError("Missing suppression_target for suppression_mode 'cath'")
        if config["suppression_level"] not in [
            "class",
            "architecture",
            "topology",
            "homologous_superfamily",
            "domain_num",
            "random",
        ]:
            raise RuntimeError(
                f"Invalid suppression_level '{config['suppression_level']}' for suppression_mode 'cath'"
            )

    elif config["suppression_mode"] == "dssp":
        if config["suppression_target"] not in ["helix", "strand", "coil"]:
            raise RuntimeError(
                f"Invalid suppression_target '{config['suppression_target']}' for suppression_mode 'dssp'"
            )

    else:
        raise RuntimeError(f"Invalid suppression_mode '{config['suppression_mode']}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration for CARP subnetworks")

    parser.add_argument("--run_name", type=str, default="test", help="Name of the run")
    parser.add_argument("--resume_last", default=False, help="Resume from last checkpoint")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID")
    parser.add_argument("--run_dir", type=str, default=None, help="Run directory")

    parser.add_argument("--suppression_mode", type=str, default=None, help="Suppression info - cath or dssp")
    parser.add_argument("--suppression_level", type=str, default=None, help="Suppression level of CATH")
    parser.add_argument("--suppression_target", type=str, default=None, help="Suppression target value")
    parser.add_argument("--suppression_lambda", type=float, default=1, help="Suppression lambda value")
    parser.add_argument("--maintenance_lambda", type=float, default=1, help="Maintenance lambda value")
    parser.add_argument(
        "--maintenance_mlm_lambda",
        type=float,
        default=0.5,
        help="Maintenance MLM lambda value",
    )
    parser.add_argument("--sparsity_lambda_init", type=float, default=1, help="Sparsity lambda init value")
    parser.add_argument("--sparsity_lambda_final", type=float, default=1, help="Sparsity lambda final value")
    parser.add_argument("--sparsity_warmup_epochs", type=int, default=100, help="Sparsity warmup epochs")
    parser.add_argument("--sparsity_ramp_epochs", type=int, default=100, help="Sparsity ramp epochs")
    parser.add_argument("--random_n", type=int, default=None, help="Random n for suppression")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_phaseA", type=float, default=1e-4, help="Learning rate for phase A")
    parser.add_argument("--lr_phaseB", type=float, default=1e-4, help="Learning rate for phase B")
    parser.add_argument("--lr_hold_epochs", type=int, default=0, help="Learning rate hold epochs")
    parser.add_argument("--lr_plateau_epochs", type=int, default=0, help="Learning rate plateau epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=4, help="Trainer logging frequency")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--num_examples_per_batch",
        type=int,
        default=1,
        help="Number of examples per batch for evenly sampled batches",
    )
    parser.add_argument("--max_epochs", type=int, default=4, help="Maximum number of epochs")
    parser.add_argument("--accumulate_grad_batches", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--val_check_interval", type=int, default=1, help="Validation check interval")
    parser.add_argument("--ckpt_freq", type=int, default=50, help="Checkpoint frequency")

    parser.add_argument("--mask_init_value", type=float, default=-4.595, help="Mask initialization value")
    parser.add_argument("--mask_temp_init", type=float, default=1.0, help="Mask temperature initial value")
    parser.add_argument("--mask_temp_final", type=float, default=0.1, help="Mask temperature final value")
    parser.add_argument("--mask_temp_decay", type=int, default=50, help="Mask temperature decay epochs")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Mask threshold for masking")

    parser.add_argument(
        "--mask_top_layer_frac",
        type=float,
        default=0.5,
        help="Mask top percentage of ByteNet layers",
    )
    parser.add_argument(
        "--mask_layer_range",
        type=lambda x: tuple(map(int, x.strip("()").split(","))),
        help="Tuple of (start_layer, end_layer) for masking",
        default=None,
    )

    parser.add_argument("--min_n_res", type=int, default=64, help="Minimum number of residues")
    parser.add_argument("--max_n_res", type=int, default=512, help="Maximum number of residues")

    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision for training")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")

    parser.add_argument("--debug", default=False, type=bool, help="Enable debug to run on 100 examples")
    parser.add_argument("--shuffle", default=True, type=bool, help="Shuffle dataset")
    parser.add_argument("--wandb_dir", type=str, default="../wandb", help="WandB directory")
    parser.add_argument("--wandb_project", type=str, default="cath-class-subnetworks-test", help="WandB project")

    return parser.parse_args()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    os.environ["MASTER_PORT"] = str(random.randint(29500, 29999))
    os.environ["MASTER_ADDR"] = "localhost"

    model, _ = load_model_and_alphabet("carp_640M")
    byte_layers = len(model.model.embedder.layers) if hasattr(model, "model") else 0

    args = parse_args()
    config = {
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
        "log_every_n_steps": args.log_every_n_steps,
        "mask_threshold": args.mask_threshold,
        "sparsity_ramp_epochs": args.sparsity_ramp_epochs,
        "wandb_dir": args.wandb_dir,
        "wandb_project": args.wandb_project,
        "run_name": args.run_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
    }

    validate_config(config)

    print("################# SUPPRESSION INFO ################# ")
    print(f"Suppression mode: {config['suppression_mode']}")
    print(f"Suppression level: {config['suppression_level']}")
    print(f"Suppression target: {config['suppression_target']}")
    print("#################################################### ")

    model_dir = f"{RUN_DIR_PREFIX}/{config['run_name']}_{config['slurm_job_id']}"
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "regular_checkpoints"), exist_ok=True)
    config["run_dir"] = model_dir

    if config["suppression_level"] == "random":
        config["random_supp_id_path"] = os.path.join(model_dir, "random_supp_ids.txt")
    else:
        config["random_supp_id_path"] = None

    use_dssp = args.suppression_mode == "dssp"

    if (
        args.suppression_mode == "cath"
        and args.suppression_level is not None
        and args.suppression_target is not None
    ):
        train_ids, val_ids, test_ids, train_loader, val_loader = get_dataloaders(
            batch_size=args.batch_size,
            even_sampling=True,
            num_workers=args.num_workers,
            use_dssp=use_dssp,
            level=args.suppression_level,
            target=args.suppression_target,
            num_samples=args.num_examples_per_batch,
            debug=args.debug,
            random_n=args.random_n,
            random_supp_id_path=config["random_supp_id_path"],
        )
    else:
        train_ids, val_ids, test_ids, train_loader, val_loader = get_dataloaders(
            batch_size=args.batch_size,
            debug=args.debug,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            use_dssp=use_dssp,
            random_n=args.random_n,
            random_supp_id_path=config["random_supp_id_path"],
        )

    wandb.init(
        project=config["wandb_project"],
        name=config["run_name"],
        config=config,
        group=config["run_name"],
    )
    config["wandb_run_id"] = wandb.run.id

    wandb_logger = WandbLogger(
        project=config["wandb_project"],
        name=config["run_name"],
        log_model=False,
    )

    data_io.write_dict_to_json(
        {"train": train_ids, "val": val_ids, "test": test_ids},
        os.path.join(model_dir, "train_val_split.json"),
    )
    data_io.write_dict_to_json({"config": config}, os.path.join(model_dir, "config.json"))

    mask_learner = WeightedDifferentiableMaskCARP(
        model=model,
        temp_init=args.mask_temp_init,
        temp_final=args.mask_temp_final,
        temp_decay=args.mask_temp_decay,
        mask_threshold=args.mask_threshold,
        init_value=args.mask_init_value,
        num_model_layers=byte_layers or 33,
        mask_top_layer_frac=args.mask_top_layer_frac,
        mask_layer_range=args.mask_layer_range,
    )

    random_supp_id_path = config["random_supp_id_path"]

    lightning_model = CarpMaskLearner(
        model=model,
        mask_learner=mask_learner,
        learning_rate=args.learning_rate,
        lr_hold_epochs=args.lr_hold_epochs,
        lr_phaseA=args.lr_phaseA,
        lr_phaseB=args.lr_phaseB,
        lr_plateau_epochs=args.lr_plateau_epochs,
        maintenance_lambda=args.maintenance_lambda,
        maintenance_mlm_lambda=args.maintenance_mlm_lambda,
        random_supp_id_path=random_supp_id_path,
        sparsity_lambda_final=args.sparsity_lambda_final,
        sparsity_lambda_init=args.sparsity_lambda_init,
        sparsity_ramp_epochs=args.sparsity_ramp_epochs,
        sparsity_warmup_epochs=args.sparsity_warmup_epochs,
        suppression_lambda=args.suppression_lambda,
        suppression_level=args.suppression_level,
        suppression_mode=args.suppression_mode,
        suppression_target=args.suppression_target,
    )

    if random_supp_id_path is not None and os.path.exists(random_supp_id_path):
        lightning_model.random_supp_ids = data_io.read_from_txt(random_supp_id_path)

    # wandb_logger.watch(
    #     lightning_model,
    #     log="gradients",
    #     log_freq=4,
    #     log_graph=True,
    # )

    regular_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(ckpt_dir, "regular_checkpoints"),
        filename="{epoch:02d}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=config["ckpt_freq"],
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        strategy="auto",
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=wandb_logger,
        precision=args.precision,
        callbacks=[regular_checkpoint, lr_monitor],
        check_val_every_n_epoch=args.val_check_interval,
    )

    trainer.fit(lightning_model, train_loader, val_loader)
