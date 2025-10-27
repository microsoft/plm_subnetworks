import argparse
import os
import random
from typing import Optional, Sequence, Tuple

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from plm_subnetworks.dataset import data_io, data_paths
from plm_subnetworks.dataset.batch_sampler import CustomBatchSampler
from plm_subnetworks.dataset.cath_dataset import (
    CATHDatabase,
    CATH_ENTRY_FILEPATH,
    CATH_S20_DSSP_FASTA,
)
from plm_subnetworks.dataset.data_paths import CATH_IDS_S20_PDB_CONTACTS, RUN_DIR_PREFIX
from plm_subnetworks.dataset.dayhoff_seq_dataloader import CATHSeqDatasetDayhoff
from plm_subnetworks.subnetwork.dayhoff_masking_pl_logits import DayhoffMaskLearner
from plm_subnetworks.subnetwork.dayhoff_modules import WeightedDifferentiableMaskDayhoff


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
    parser = argparse.ArgumentParser(description="Dayhoff subnetwork training configuration")

    # Run configuration
    parser.add_argument("--run_name", type=str, default="dayhoff-test", help="Name of the run")
    parser.add_argument("--resume_last", default=False, help="Resume from last checkpoint")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID")
    parser.add_argument("--run_dir", type=str, default=None, help="Run directory")

    # Model parameters
    parser.add_argument("--suppression_mode", type=str, default=None, help="Suppression info - cath or dssp")
    parser.add_argument("--suppression_level", type=str, default=None, help="Suppression level of CATH")
    parser.add_argument("--suppression_target", type=str, default=None, help="Suppression target value")
    parser.add_argument("--suppression_lambda", type=float, default=1.0, help="Suppression lambda value")
    parser.add_argument("--maintenance_lambda", type=float, default=1.0, help="Maintenance lambda value")
    parser.add_argument("--maintenance_mlm_lambda", type=float, default=0.5, help="Maintenance MLM lambda value")
    parser.add_argument("--sparsity_lambda_init", type=float, default=1.0, help="Initial sparsity lambda")
    parser.add_argument("--sparsity_lambda_final", type=float, default=1.0, help="Final sparsity lambda")
    parser.add_argument("--sparsity_warmup_epochs", type=int, default=100, help="Sparsity warmup epochs")
    parser.add_argument("--sparsity_ramp_epochs", type=int, default=100, help="Sparsity ramp epochs")
    parser.add_argument("--random_n", type=int, default=None, help="Random n for suppression")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_phaseA", type=float, default=1e-4, help="Learning rate for phase A")
    parser.add_argument("--lr_phaseB", type=float, default=1e-4, help="Learning rate for phase B")
    parser.add_argument("--lr_hold_epochs", type=int, default=0, help="Learning rate hold epochs")
    parser.add_argument("--lr_plateau_epochs", type=int, default=0, help="Learning rate plateau epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_examples_per_batch", type=int, default=1, help="Examples per batch")
    parser.add_argument("--max_epochs", type=int, default=4, help="Maximum epochs")
    parser.add_argument("--accumulate_grad_batches", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--val_check_interval", type=int, default=1, help="Validation interval (epochs)")
    parser.add_argument("--ckpt_freq", type=int, default=50, help="Checkpoint frequency")

    # Model architecture
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Dayhoff-170m-UR90",
        help="HuggingFace identifier for the Dayhoff model",
    )
    parser.add_argument("--mask_init_value", type=float, default=0.5, help="Mask initialization probability or logit")
    parser.add_argument("--mask_temp_init", type=float, default=1.0, help="Mask temperature initial value")
    parser.add_argument("--mask_temp_final", type=float, default=0.1, help="Mask temperature final value")
    parser.add_argument("--mask_temp_decay", type=int, default=50, help="Mask temperature decay epochs")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Mask threshold for binarization")
    parser.add_argument("--mask_top_layer_frac", type=float, default=1.0, help="Fraction of top layers to mask")
    parser.add_argument(
        "--mask_layer_range",
        type=lambda x: tuple(map(int, x.strip("()").split(","))),
        help="Tuple of (start_layer, end_layer) for masking",
        default=None,
    )

    # Data parameters
    parser.add_argument("--min_n_res", type=int, default=64, help="Minimum number of residues")
    parser.add_argument("--max_n_res", type=int, default=512, help="Maximum number of residues")
    parser.add_argument(
        "--disable_input_corruption",
        action="store_true",
        help="Use clean sequences (no random masking) as inputs for the Dayhoff model",
    )
    parser.add_argument(
        "--mask_mamba_projections",
        type=str,
        default="",
        help=(
            "Comma-separated list of additional Mamba projection names to mask "
            "(e.g. 'in_proj,x_proj,dt_proj'); use 'all' to include the default set."
        ),
    )

    # Hardware/System
    parser.add_argument("--precision", type=str, default="16-mixed", help="Precision for training")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader workers")

    # Other parameters
    parser.add_argument("--debug", default=False, type=bool, help="Enable debug mode")
    parser.add_argument("--shuffle", default=True, type=bool, help="Shuffle data")
    parser.add_argument("--wandb_dir", type=str, default="../wandb", help="WandB directory")
    parser.add_argument("--wandb_project", type=str, default="dayhoff-subnetworks", help="WandB project")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow loading custom DayHoff code")

    return parser.parse_args()


def _parse_mask_mamba_projections(arg_value: str) -> Tuple[str, ...]:
    """Parse the command-line projection mask selector."""
    default_tokens = ("in_proj", "x_proj", "dt_proj")
    if not arg_value:
        return ()

    value = arg_value.strip()
    if not value:
        return ()

    if value.lower() == "all":
        return default_tokens

    tokens = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if token not in tokens:
            tokens.append(token)
    return tuple(tokens)




_SUPPORTED_PRECISIONS = {
    "16-mixed",
    "16-true",
    "bf16",
    "bf16-mixed",
    "bf16-true",
    "32-true",
    "64-true",
    "transformer-engine",
    "transformer-engine-float16",
}

_PRECISION_ALIASES = {
    "fp32": "32-true",
    "float32": "32-true",
    "32": "32-true",
    "bf16": "bf16-mixed",
    "bfloat16": "bf16-mixed",
    "fp16": "16-mixed",
    "float16": "16-mixed",
    "half": "16-mixed",
}


def _normalize_precision(requested: Optional[str]) -> str:
    if requested is None:
        return "16-mixed"
    normalized = _PRECISION_ALIASES.get(requested.lower(), requested)
    if normalized not in _SUPPORTED_PRECISIONS:
        allowed = ", ".join(sorted(_SUPPORTED_PRECISIONS))
        raise ValueError(f"Unsupported precision '{requested}'. Allowed values: {allowed}")
    return normalized


def _precision_to_dtype(precision: str) -> Optional[torch.dtype]:
    key = precision.lower()
    if key in {"bf16", "bf16-mixed", "bf16-true"}:
        return torch.bfloat16
    if key == "32-true":
        return torch.float32
    if key == "64-true":
        return torch.float64
    return None


def _infer_num_layers(model: torch.nn.Module) -> int:
    num_layers = getattr(getattr(model, "config", object()), "num_hidden_layers", None)
    if num_layers is not None:
        return int(num_layers)
    for attr in ["encoder", "model", "dayhoff", "transformer"]:
        module = getattr(model, attr, None)
        if module is None:
            continue
        for candidate in ["layer", "layers", "block", "blocks"]:
            seq = getattr(module, candidate, None)
            if isinstance(seq, (torch.nn.ModuleList, list, tuple)):
                return len(seq)
    raise RuntimeError("Unable to infer the number of transformer layers for the Dayhoff model")


def _build_cath_database() -> CATHDatabase:
    cath_db = CATHDatabase()
    cath_db.load_clf(CATH_ENTRY_FILEPATH)
    cath_db.load_sequences(data_paths.get_cath_paths("20")[1])
    try:
        cath_db.load_dssp(CATH_S20_DSSP_FASTA)
    except Exception:
        pass
    return cath_db


def get_dataloaders(
    batch_size: int,
    tokenizer_name: str,
    min_n_res: int,
    max_n_res: int,
    split: float = 0.7,
    even_sampling: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    debug: bool = False,
    use_dssp: bool = False,
    train_ids: Optional[Sequence[str]] = None,
    val_ids: Optional[Sequence[str]] = None,
    test_ids: Optional[Sequence[str]] = None,
    level: Optional[str] = None,
    target: Optional[str] = None,
    num_samples: Optional[int] = None,
    random_n: Optional[int] = None,
    random_supp_id_path: Optional[str] = None,
    mask_inputs: bool = True,
) -> Tuple[Sequence[str], Sequence[str], Sequence[str], DataLoader, DataLoader]:

    cath_ids, seq_filepath = data_paths.get_cath_paths("20")
    cath_ids = data_io.read_from_txt(CATH_IDS_S20_PDB_CONTACTS)
    random.shuffle(cath_ids)

    if debug:
        cath_ids = cath_ids[:100]

    if train_ids is None or val_ids is None or test_ids is None:
        train_ids = cath_ids[: int(len(cath_ids) * split)]
        val_ids = cath_ids[int(len(cath_ids) * split): int(len(cath_ids) * (split + 0.2))]
        test_ids = cath_ids[int(len(cath_ids) * (split + 0.2)) :]

    cath_db = _build_cath_database()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    train_dataset = CATHSeqDatasetDayhoff(
        cath_ids=train_ids,
        cath_database=cath_db,
        tokenizer=tokenizer,
        min_n_res=min_n_res,
        max_n_res=max_n_res,
        masking=mask_inputs,
        use_dssp=use_dssp,
    )

    val_dataset = CATHSeqDatasetDayhoff(
        cath_ids=val_ids,
        cath_database=cath_db,
        tokenizer=tokenizer,
        min_n_res=min_n_res,
        max_n_res=max_n_res,
        masking=mask_inputs,
        use_dssp=use_dssp,
    )

    if even_sampling:
        assert level is not None
        assert target is not None
        assert num_samples is not None

        train_sampler = CustomBatchSampler(
            train_dataset,
            batch_size,
            level=level,
            target=target,
            num_samples=num_samples,
            random_n=random_n,
            random_supp_id_path=random_supp_id_path,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=train_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_sampler = CustomBatchSampler(
            val_dataset,
            batch_size,
            level=level,
            target=target,
            num_samples=num_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=val_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True,
        )

    return train_ids, val_ids, test_ids, train_loader, val_loader


if __name__ == "__main__":

    torch.set_float32_matmul_precision("medium")
    os.environ.setdefault("MASTER_PORT", str(random.randint(29500, 29999)))
    os.environ.setdefault("MASTER_ADDR", "localhost")

    args = parse_args()
    extra_mamba_tokens = _parse_mask_mamba_projections(args.mask_mamba_projections)
    requested_precision = args.precision
    resolved_precision = _normalize_precision(requested_precision)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    dtype = _precision_to_dtype(resolved_precision)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()

    num_layers = _infer_num_layers(model)

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
        "precision": resolved_precision,
        "precision_requested": requested_precision,
        "num_workers": args.num_workers,
        "ckpt_freq": args.ckpt_freq,
        "resume_last": args.resume_last,
        "wandb_run_id": args.wandb_run_id,
        "debug": args.debug,
        "shuffle": args.shuffle,
        "min_n_res": args.min_n_res,
        "max_n_res": args.max_n_res,
        "disable_input_corruption": args.disable_input_corruption,
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
        "sparsity_ramp_epochs": args.sparsity_ramp_epochs,
        "mask_top_layer_frac": args.mask_top_layer_frac,
        "mask_layer_range": args.mask_layer_range,
        "lr_phaseA": args.lr_phaseA,
        "lr_phaseB": args.lr_phaseB,
        "lr_hold_epochs": args.lr_hold_epochs,
        "lr_plateau_epochs": args.lr_plateau_epochs,
        "mask_threshold": args.mask_threshold,
        "mask_mamba_projections": list(extra_mamba_tokens),
        "wandb_dir": args.wandb_dir,
        "wandb_project": args.wandb_project,
        "run_name": args.run_name,
        "model_name": args.model_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "num_layers": num_layers,
    }

    validate_config(config)

    model_dir = f"{RUN_DIR_PREFIX}/{config['run_name']}_{config['slurm_job_id']}"
    config["run_dir"] = model_dir
    ckpt_dir = f"{model_dir}/checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=config,
        group=args.run_name,
        dir=args.wandb_dir,
    )
    config["wandb_run_id"] = wandb.run.id

    wandb_logger = WandbLogger(project=config["wandb_project"], name=config["run_name"], log_model=False)

    use_dssp = config["suppression_mode"] == "dssp"

    if config["suppression_level"] == "random":
        random_supp_id_path = f"{model_dir}/random_supp_ids.txt"
        config["random_supp_id_path"] = random_supp_id_path
    else:
        config["random_supp_id_path"] = None

    if config["suppression_mode"] == "cath" and config["suppression_level"] and config["suppression_target"]:
        train_ids, val_ids, test_ids, train_loader, val_loader = get_dataloaders(
            batch_size=config["batch_size"],
            tokenizer_name=args.model_name,
            min_n_res=config["min_n_res"],
            max_n_res=config["max_n_res"],
            even_sampling=True,
            num_workers=config["num_workers"],
            use_dssp=use_dssp,
            level=config["suppression_level"],
            target=config["suppression_target"],
            num_samples=config["num_examples_per_batch"],
            debug=config["debug"],
            random_n=config["random_n"],
            random_supp_id_path=config["random_supp_id_path"],
            mask_inputs=not config["disable_input_corruption"],
        )
    else:
        train_ids, val_ids, test_ids, train_loader, val_loader = get_dataloaders(
            batch_size=config["batch_size"],
            tokenizer_name=args.model_name,
            min_n_res=config["min_n_res"],
            max_n_res=config["max_n_res"],
            debug=config["debug"],
            shuffle=config["shuffle"],
            num_workers=config["num_workers"],
            use_dssp=use_dssp,
            random_n=config["random_n"],
            mask_inputs=not config["disable_input_corruption"],
        )

    data_io.write_dict_to_json({"train": train_ids, "val": val_ids, "test": test_ids}, f"{model_dir}/train_val_split.json")
    data_io.write_dict_to_json({"config": config}, f"{model_dir}/config.json")

    mask_learner = WeightedDifferentiableMaskDayhoff(
        model,
        temp_init=config["mask_temperature_init"],
        temp_final=config["mask_temperature_final"],
        temp_decay=config["mask_temperature_decay"],
        init_value=config["mask_init_value"],
        num_model_layers=config["num_layers"],
        mask_top_layer_frac=config["mask_top_layer_frac"],
        mask_layer_range=config["mask_layer_range"],
        mask_threshold=config["mask_threshold"],
        extra_attention_tokens=extra_mamba_tokens,
    )

    lightning_model = DayhoffMaskLearner(
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
        sparsity_warmup_epochs=config["sparsity_warmup_epochs"],
        sparsity_ramp_epochs=config["sparsity_ramp_epochs"],
        learning_rate=config["learning_rate"],
        lr_phaseA=config["lr_phaseA"],
        lr_phaseB=config["lr_phaseB"],
        lr_hold_epochs=config["lr_hold_epochs"],
        lr_plateau_epochs=config["lr_plateau_epochs"],
        random_supp_id_path=config["random_supp_id_path"],
        use_corrupted_inputs=not config["disable_input_corruption"],
    )

    wandb_logger.watch(lightning_model, log="gradients", log_freq=4, log_graph=True)

    regular_checkpoint = ModelCheckpoint(
        dirpath=f"{ckpt_dir}/regular_checkpoints",
        filename="{epoch:02d}",
        every_n_epochs=config["ckpt_freq"],
        save_top_k=-1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=config["max_epochs"],
        log_every_n_steps=4,
        strategy="auto",
        accumulate_grad_batches=config["accumulate_grad_batches"],
        logger=wandb_logger,
        precision=resolved_precision,
        check_val_every_n_epoch=config["val_check_interval"],
        callbacks=[regular_checkpoint, lr_monitor],
    )

    try:
        trainer.fit(lightning_model, train_loader, val_loader)
    finally:
        data_io.write_dict_to_json({"test": lightning_model.held_out_ids}, f"{model_dir}/test_split.json")
        wandb.finish()
