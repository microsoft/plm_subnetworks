import argparse
import os
import warnings
from pathlib import Path
import re

import esm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from plm_subnetworks.dataset import data_paths, data_io
from plm_subnetworks.dataset.cath_dataset import CATHDatabase, CATH_ENTRY_FILEPATH
from plm_subnetworks.dataset.data_paths import CATH_S20_DSSP_FASTA, ESM_PPL_METRICS, RUN_DIR_PREFIX
from plm_subnetworks.dataset.esm_seq_dataloader import CATHSeqDatasetESM

from plm_subnetworks.subnetwork.esm_masking_pl_logits import ESMMaskLearner
from plm_subnetworks.subnetwork.modules import WeightedDifferentiableMask

from plm_subnetworks.utils.metrics import PerSequenceMaskedCrossEntropyLoss


# ---------------------- NEW: helpers ----------------------
def parse_epoch_list(epoch_field: str):
    """
    Accepts things like '10', '1,5,10', '01, 07, 12'.
    Returns a list of ints with whitespace trimmed.
    """
    if epoch_field is None or str(epoch_field).strip() == "":
        return []
    parts = re.split(r"[,\s]+", str(epoch_field).strip())
    out = []
    for p in parts:
        if p == "":
            continue
        try:
            out.append(int(p))
        except ValueError:
            # tolerate 'epoch=10' or 'epoch_10_ckpt' in CSV by extracting digits
            m = re.search(r"(\d+)", p)
            if m:
                out.append(int(m.group(1)))
    # keep order but deduplicate
    seen = set()
    uniq = []
    for e in out:
        if e not in seen:
            uniq.append(e)
            seen.add(e)
    return uniq


def find_checkpoint_path(run_dir_from_config: str, epoch: int) -> str | None:
    """
    Search the regular_checkpoints folder for multiple naming conventions:
    - epoch=10.ckpt, epoch=010.ckpt
    - epoch_10.ckpt, epoch_010_ckpt
    Returns the first match (string path) or None if not found.
    """
    ckpt_dir = Path(run_dir_from_config) / "checkpoints" / "regular_checkpoints"
    if not ckpt_dir.exists():
        return None

    # candidate filename patterns
    candidates = [
        f"epoch={epoch}.ckpt",           # epoch=10.ckpt
        f"epoch={epoch:02d}.ckpt",       # epoch=01.ckpt
        f"epoch_{epoch}.ckpt",           # epoch_1.ckpt
        f"epoch_{epoch:02d}.ckpt",       # epoch_01.ckpt
        f"epoch_{epoch}_ckpt",           # epoch_1_ckpt
        f"epoch_{epoch:02d}_ckpt",       # epoch_01_ckpt
    ]
    for name in candidates:
        p = ckpt_dir / name
        if p.exists():
            return str(p)

    # fallback: glob search by integer substring in case other variants exist
    for p in ckpt_dir.glob("*"):
        if re.fullmatch(r".*epoch[^0-9]*0*%d[^0-9]*\.?ckpt.*" % epoch, p.name):
            return str(p)

    return None


def read_runs_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)

    # normalize header case/whitespace
    df.columns = [c.strip().lower() for c in df.columns]

    # flexible rename map
    rename_map = {
        "run name": "run_name",
        "run": "run_name",
        "run_dir": "run_name",   # <— your file
        "rundir": "run_name",
        "epochs": "epoch",
        "category": "category",
        "target": "target",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    # ensure required columns exist
    required = {"run_name", "epoch", "category", "target"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Found columns {list(df.columns)}; expected at least {sorted(required)} "
            f"(tip: include run_dir or run_name, epoch, category, target)."
        )
    return df

def _read_runs_table(path: str) -> pd.DataFrame:
    """
    Reads CSV or XLSX with columns: run_name, epoch, category, target
    """
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    # normalize expected columns
    needed = {"run_name", "epoch", "category", "target"}
    missing = needed - set(df.columns.str.lower())
    if missing:
        # try case-insensitive remap
        lower_map = {c.lower(): c for c in df.columns}
        for col in list(needed):
            if col not in df.columns and col in lower_map:
                df.rename(columns={lower_map[col]: col}, inplace=True)
    assert {"run_name", "epoch", "category", "target"}.issubset(set(df.columns)), \
        "Input file must have columns: run_name, epoch, category, target"
    return df
# ---------------------------------------------------------


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    perseqmcel = PerSequenceMaskedCrossEntropyLoss()

    cath_version = "20"
    _, seq_filepath = data_paths.get_cath_paths(cath_version)
    db = CATHDatabase()
    db.load_clf(CATH_ENTRY_FILEPATH)
    db.load_sequences(seq_filepath)
    db.load_dssp(CATH_S20_DSSP_FASTA)

    # ---------------------- UPDATED: load run specs ----------------------
    if args.csv:
        runs_df = read_runs_table(args.csv)
        # explode rows with multiple epochs into multiple rows
        expanded_rows = []
        for _, row in runs_df.iterrows():
            epochs = parse_epoch_list(row["epoch"])
            if not epochs:
                epochs = [None]
            for e in epochs:
                r = dict(row)
                r["epoch"] = e
                expanded_rows.append(r)
        subnetworks_df_data = [
            (r["run_name"], r["epoch"], r["category"], r["target"]) for r in expanded_rows
        ]
        # prepare metadata output path (next to the input file)
        meta_out_dir = Path(args.csv).parent
        meta_out_path = f"{RUN_DIR_PREFIX}/{meta_out_dir}/{args.csv.split('/')[-1].split('.')[0]}_sparsity.csv"
    else:
        # single run via CLI flags; expand comma/space separated epochs like CSV path
        epochs = parse_epoch_list(args.epoch) if args.epoch is not None else [None]
        if not epochs:
            epochs = [None]
        subnetworks_df_data = [
            (args.run_name, epoch_value, args.category, args.target)
            for epoch_value in epochs
        ]
        meta_out_path = None  # we'll write per-run under its inference dir

    # ---------------------- NEW: list to collect metadata rows ----------------------
    meta_rows = []

    for run_name, epoch, category, target in subnetworks_df_data:

        run_dir = f"{RUN_DIR_PREFIX}/{run_name}"
        # run_dir = f"{run_name}"
        config, split = data_io.get_args_split(run_dir)
        # if epoch is given as string, robustly parse to int (or None)
        if epoch is not None and not isinstance(epoch, (int, np.integer)):
            parsed = parse_epoch_list(str(epoch))
            epoch = parsed[0] if parsed else None

        # find checkpoint with robust patterns
        if epoch is None:
            print(f"[WARN] No epoch provided for run '{run_name}'. Skipping.")
            continue

        ckpt_path = find_checkpoint_path(config["run_dir"], epoch)
        if ckpt_path is None:
            print(f"[WARN] Could not find checkpoint for epoch {epoch} under {config['run_dir']}. Skipping.")
            continue

        print(f"\n>>> Evaluating subnetwork: category = '{category}', target = '{target}', "
              f"epoch = {epoch}, run_dir = '{run_name}'")
        print(f"Checkpoint path: {ckpt_path}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            ckpt = torch.load(ckpt_path, map_location="cpu")

        if args.extend_val:
            val_ids = split["train"] + split["val"] + split["test"]
            print("Eval on full set")
        else:
            val_ids = split["val"] + split["test"]
            print("Eval on val sets")

        if "random" in str(category):
            suppressed_id_path = f"{RUN_DIR_PREFIX}/{run_name}/random_supp_ids.txt"
            # suppressed_id_path = f"{run_name}/random_supp_ids.txt"

            if Path(suppressed_id_path).exists():
                with open(suppressed_id_path, "r") as f:
                    suppressed_ids = f.read().splitlines()
                print("Loaded suppressed ids:", len(suppressed_ids))
                existing_ids = set(val_ids)
                extra_ids = [sid for sid in suppressed_ids if sid not in existing_ids]
                val_ids.extend(extra_ids)
            else:
                print(f"[WARN] random_supp_ids.txt not found for {run_name}")

        if args.verbose:
            print(f"Number of eval seqs: {len(val_ids)}")

        mask_learner = WeightedDifferentiableMask(
            model,
            temp_init=config["mask_temperature_init"],
            temp_final=config["mask_temperature_final"],
            temp_decay=config["mask_temperature_decay"],
            init_value=config["mask_init_value"],
            mask_top_layer_frac=config["mask_top_layer_frac"],
            mask_layer_range=tuple(config["mask_layer_range"]),
            mask_threshold=config["mask_threshold"],
        )

        lightning_model = ESMMaskLearner.load_from_checkpoint(
            ckpt_path,
            model=model,
            mask_learner=mask_learner,
        )

        # ---------------------- NEW: capture sparsity + global steps ----------------------
        mask_temp = ckpt.get("mask_temperature", getattr(lightning_model.mask_learner, "temperature", None))
        sparsity_lambda = ckpt.get("sparsity_lambda", None)
        ckpt_epoch = ckpt.get("epoch", epoch)
        num_training_steps = ckpt.get("global_step", None)

        lightning_model.mask_learner.temperature = mask_temp
        lightning_model.eval()

        model = model.to(device)
        lightning_model = lightning_model.to(device)
        mask_learner = mask_learner.to(device)

        with torch.no_grad():
            masks = lightning_model.mask_learner()

        sparsity = float(lightning_model.mask_learner.get_sparsity().item())

        # queue a metadata row (will be written after inference loop)
        meta_rows.append({
            "run_name": run_name,
            "epoch": ckpt_epoch,
            "category": category,
            "target": target,
            "checkpoint_path": ckpt_path,
            "global_step": num_training_steps,
            "sparsity": sparsity,
        })

        if args.verbose:
            print(f"Mask temperature (runtime): {lightning_model.mask_learner.temperature:.2f}")
            if mask_temp is not None:
                print(f"Mask temperature (from ckpt): {mask_temp:.2f}")
            if sparsity_lambda is not None:
                print(f"Sparsity lambda from checkpoint: {sparsity_lambda:.2f}")
            if num_training_steps is not None:
                print(f"# Steps from checkpoint: {num_training_steps}")
            print(f"Model sparsity: {sparsity:.4f}")
            print(f"Number of passes: {args.n_passes}")

        inference_dir = Path(run_dir) / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)

        if args.extend_val:
            csv_path = inference_dir / f"epoch_{ckpt_epoch}_full_{args.n_passes}_passes.csv"
        else:
            csv_path = inference_dir / f"epoch_{ckpt_epoch}_heldout_{args.n_passes}_passes.csv"

        reuse_existing = csv_path.exists() and not args.overwrite
        if reuse_existing:
            print(f"[INFO] Found existing inference CSV at {csv_path}. Skipping inference run.")
            results_df = pd.read_csv(csv_path)
        else:
            if csv_path.exists():
                print(f"[INFO] {csv_path} already exists. Overwriting as requested.")
            results = []

            for _ in range(args.n_passes):
                val_dataset = CATHSeqDatasetESM(
                    val_ids, db, alphabet,
                    min_n_res=64, max_n_res=512, use_dssp=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=4 if args.override_batch_size else config["batch_size"],
                    shuffle=config["shuffle"],
                    num_workers=config["num_workers"],
                    collate_fn=val_dataset.collate_fn
                )

                for i, batch in enumerate(val_loader):
                    cath_ids = batch["cath_ids"]
                    tgt = batch["tgt"].to(device)
                    src = batch["src"].to(device)
                    corr_mask = batch["corr_mask"].to(device)

                    with torch.no_grad():
                        out = lightning_model.subnetwork(src, masks=masks, return_contacts=False, inverse=False)
                        pred_logits = out["logits"]

                    mlm_loss = perseqmcel(pred_logits, tgt, corr_mask)

                    helix_mask = batch["helix_mask"].to(device)
                    strand_mask = batch["strand_mask"].to(device)
                    coil_mask  = batch["coil_mask"].to(device)

                    helix_corrupted_and_masked  = corr_mask.bool() & helix_mask
                    strand_corrupted_and_masked = corr_mask.bool() & strand_mask
                    coil_corrupted_and_masked   = corr_mask.bool() & coil_mask

                    helix_mlm  = perseqmcel(pred_logits, tgt, helix_corrupted_and_masked)  if helix_corrupted_and_masked.sum()!=0 else np.NaN
                    strand_mlm = perseqmcel(pred_logits, tgt, strand_corrupted_and_masked) if strand_corrupted_and_masked.sum()!=0 else np.NaN
                    coil_mlm   = perseqmcel(pred_logits, tgt, coil_corrupted_and_masked)   if coil_corrupted_and_masked.sum()!=0 else np.NaN

                    batch_data = {
                        "cath_id": cath_ids,
                        "mlm_loss": mlm_loss.cpu().numpy(),
                        "perplexity": np.exp(mlm_loss.cpu().numpy()),
                        "helix_mlm":  helix_mlm.cpu().numpy()  if helix_corrupted_and_masked.sum()!=0  else np.nan,
                        "strand_mlm": strand_mlm.cpu().numpy() if strand_corrupted_and_masked.sum()!=0 else np.nan,
                        "coil_mlm":   coil_mlm.cpu().numpy()   if coil_corrupted_and_masked.sum()!=0   else np.nan,
                    }
                    results.append(pd.DataFrame(batch_data))

            if not results:
                print(f"[WARN] No batches produced results for run '{run_name}'. Skipping.")
                continue

            results_df = pd.concat(results, ignore_index=True)
            results_df.to_csv(csv_path, index=False)
            print("Wrote CSV to", csv_path)

        hydrated_df = data_io.hydrate_df_with_cath_terms(results_df, db)
        gt = pd.read_csv(ESM_PPL_METRICS)

        def print_stats(title, df, prefix=""):
            print(f"{prefix}{title} ({len(df)} seqs): "
                  f"(PPL) {df['perplexity'].mean():.2f}, "
                  f"(MLM) {df['mlm_loss'].mean():.2f}")

        def print_mlm_stats(label, s, prefix=""):
            mean_mlm = s.dropna().mean()
            print(f"{prefix}{label:<8} (PPL): {np.exp(mean_mlm):.2f} / (MLM) {mean_mlm:.2f}")

        hydrated_df_gt = data_io.hydrate_df_with_cath_terms(gt, db)
        hydrated_df_gt = hydrated_df_gt.groupby("cath_id").head(args.n_passes).reset_index(drop=True)

        if "cath" in str(category):
            supp_df = hydrated_df[hydrated_df[category] == target]
            maint_df = hydrated_df[hydrated_df[category] != target]

            print("\n================= Subnetwork =================")
            print_stats("Suppression", supp_df)
            print_stats("Maintenance", maint_df)
            print("=====================================================\n")

            hydrated_df_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(hydrated_df["cath_id"])]
            suppression_gt = hydrated_df_gt[hydrated_df_gt[category] == target]
            maintenance_gt = hydrated_df_gt[hydrated_df_gt[category] != target]

            print("================ ESM-2 (Ground Truth) ================")
            print_stats("Suppression", suppression_gt)
            print_stats("Maintenance", maintenance_gt)
            print("=====================================================\n")

        elif "random" in str(category):
            suppressed_ids = set(suppressed_ids) if 'suppressed_ids' in locals() else set()
            supp_df = hydrated_df[hydrated_df["cath_id"].isin(suppressed_ids)]
            maint_df = hydrated_df[~hydrated_df["cath_id"].isin(suppressed_ids)]

            print("\n================= Random Subnetwork ==================")
            print_stats("Suppression", supp_df)
            print_stats("Maintenance", maint_df)
            print("=====================================================\n")

            hydrated_df_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(hydrated_df["cath_id"])]
            suppression_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(suppressed_ids)]
            maintenance_gt = hydrated_df_gt[~hydrated_df_gt["cath_id"].isin(suppressed_ids)]

            print("================ ESM-2 (Ground Truth) ================")
            print_stats("Suppression", suppression_gt)
            print_stats("Maintenance", maintenance_gt)
            print("=====================================================\n")

        else:
            hydrated_df_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(hydrated_df["cath_id"])]

            print("\n================ Residue Subnetwork ==================")
            print_mlm_stats("Helix",  hydrated_df["helix_mlm"])
            print_mlm_stats("Strand", hydrated_df["strand_mlm"])
            print_mlm_stats("Coil",   hydrated_df["coil_mlm"])
            print("=====================================================\n")

            print("================ ESM-2 (Ground Truth) ================")
            print_mlm_stats("Helix",  hydrated_df_gt["helix_mlm"])
            print_mlm_stats("Strand", hydrated_df_gt["strand_mlm"])
            print_mlm_stats("Coil",   hydrated_df_gt["coil_mlm"])
            print("=====================================================\n")

    # ---------------------- NEW: write metadata_sparsity.csv ----------------------
    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        if meta_out_path is None:
            # single-run CLI: write next to that run's inference folder
            # run_dir = f"{RUN_DIR_PREFIX}/{meta_rows[0]['run_name']}"
            run_dir = f"{meta_rows[0]['run_name']}"
            out_dir = Path(run_dir) / "inference"
            out_dir.mkdir(parents=True, exist_ok=True)
            meta_out_path = out_dir / "metadata_sparsity.csv"
        # append (create if not exists)
        if meta_out_path.exists():
            existing = pd.read_csv(meta_out_path)
            combined = pd.concat([existing, meta_df], ignore_index=True)
            # drop duplicate (run_name, epoch) pairs to keep last occurrence
            combined = combined.drop_duplicates(subset=["run_name", "epoch"], keep="last")
            combined.to_csv(meta_out_path, index=False)
        else:
            meta_df.to_csv(meta_out_path, index=False)
        print(f"Wrote metadata to {meta_out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--override_batch_size', type=bool, default=True,
                        help='Overrides when using smaller GPUs for inference', required=False)
    parser.add_argument('--extend_val', action='store_true',
                        help='Add flag to evaluate on full set instead of heldout set')
    parser.add_argument('--n_passes', type=int, default=1,
                        help='include val seqs in evals')
    parser.add_argument('--pip li', action='store_true',
                        help='Add flag to evaluate random suppression models.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing inference CSVs instead of skipping')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name of run to evaluate')
    parser.add_argument('--epoch', type=str, default=None,
                        help='Epoch to evaluate; can be comma-separated for multiple')
    parser.add_argument('--category', type=str, default=None,
                        help='cath_{level}_code, random_seq, or residue')
    parser.add_argument('--target', type=str, default=None,
                        help='Target to evaluate, e.g., 1.25.40, helix, or random')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV/XLSX with columns: run_name, epoch, category, target')
    parser.add_argument('-V', '--verbose', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()
    main(args)
