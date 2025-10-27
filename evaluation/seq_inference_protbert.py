import argparse
import os
import warnings
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer

from plm_subnetworks.dataset import data_paths, data_io
from plm_subnetworks.dataset.cath_dataset import CATHDatabase, CATH_ENTRY_FILEPATH
from plm_subnetworks.dataset.data_paths import CATH_S20_DSSP_FASTA, PROTBERT_PPL_METRICS, PROTBERT_UR100_PPL_METRICS, RUN_DIR_PREFIX
from plm_subnetworks.dataset.protbert_seq_dataloader import CATHSeqDatasetProtBERT

from plm_subnetworks.subnetwork.protbert_masking_pl_logits import ProtBERTMaskLearner
from plm_subnetworks.subnetwork.protbert_modules import WeightedDifferentiableMaskProtBert

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
            m = re.search(r"(\d+)", p)
            if m:
                out.append(int(m.group(1)))
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
    - epoch_{10}_ckpt variants
    Returns the first match or None if not found.
    """
    ckpt_dir = Path(run_dir_from_config) / "checkpoints" / "regular_checkpoints"
    if not ckpt_dir.exists():
        return None

    candidates = [
        f"epoch={epoch}.ckpt",
        f"epoch={epoch:02d}.ckpt",
        f"epoch_{epoch}.ckpt",
        f"epoch_{epoch:02d}.ckpt",
        f"epoch_{epoch}_ckpt",
        f"epoch_{epoch:02d}_ckpt",
    ]
    for name in candidates:
        candidate_path = ckpt_dir / name
        if candidate_path.exists():
            return str(candidate_path)

    pattern = re.compile(rf".*epoch[^0-9]*0*{epoch}[^0-9]*\.?ckpt.*")
    for path in ckpt_dir.glob("*"):
        if pattern.fullmatch(path.name):
            return str(path)

    return None


def read_runs_table(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        "run name": "run_name",
        "run": "run_name",
        "run_dir": "run_name",
        "rundir": "run_name",
        "epochs": "epoch",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required = {"run_name", "epoch", "category", "target"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Found columns {list(df.columns)}; expected at least {sorted(required)} "
            f"(tip: include run_dir or run_name, epoch, category, target)."
        )
    return df


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    perseqmcel = PerSequenceMaskedCrossEntropyLoss()

    def raw_loss_per_seq(logits, targets, mask):
        mask_bool = mask.to(device=logits.device, dtype=torch.bool)
        token_counts = mask_bool.sum(dim=1)
        if torch.count_nonzero(mask_bool) == 0:
            return torch.full((logits.size(0),), float("nan"), device=logits.device, dtype=logits.dtype)

        B, L, V = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            reduction="none",
        ).view(B, L)
        mask_float = mask_bool.to(dtype=loss.dtype)
        per_seq = (loss * mask_float).sum(dim=1) / token_counts.clamp_min(1.0)
        return per_seq.masked_fill(token_counts == 0, float("nan"))

    cath_version = "20"
    _, seq_filepath = data_paths.get_cath_paths(cath_version)
    db = CATHDatabase()
    db.load_clf(CATH_ENTRY_FILEPATH)
    db.load_sequences(seq_filepath)
    db.load_dssp(CATH_S20_DSSP_FASTA)

    if args.csv:
        runs_df = read_runs_table(args.csv)
        expanded_rows = []
        for _, row in runs_df.iterrows():
            epochs = parse_epoch_list(row["epoch"])
            if not epochs:
                epochs = [None]
            for e in epochs:
                expanded = dict(row)
                expanded["epoch"] = e
                expanded_rows.append(expanded)
        subnetworks_df_data = [
            (r["run_name"], r["epoch"], r["category"], r["target"]) for r in expanded_rows
        ]
        meta_out_dir = Path(args.csv).parent
        meta_out_path = meta_out_dir / f"{args.csv.split('/')[-1].split('.')[0]}_sparsity.csv"
    else:
        epochs = parse_epoch_list(args.epoch) if args.epoch is not None else [None]
        if not epochs:
            epochs = [None]
        subnetworks_df_data = [
            (args.run_name, epoch_value, args.category, args.target) for epoch_value in epochs
        ]
        meta_out_path = None

    meta_rows = []

    for run_name, epoch, category, target in subnetworks_df_data:
        if run_name is None:
            continue

        run_dir = f"{RUN_DIR_PREFIX}/{run_name}"
        config, split = data_io.get_args_split(run_dir)

        if epoch is not None and not isinstance(epoch, (int, np.integer)):
            parsed_epochs = parse_epoch_list(str(epoch))
            epoch = parsed_epochs[0] if parsed_epochs else None

        if epoch is None:
            print(f"[WARN] No epoch provided for run '{run_name}'. Skipping.")
            continue

        ckpt_path = find_checkpoint_path(config["run_dir"], epoch)
        if ckpt_path is None:
            print(f"[WARN] Could not find checkpoint for epoch {epoch} under {config['run_dir']}. Skipping.")
            continue

        print(f"\n>>> Evaluating subnetwork: category = '{category}', target = '{target}', epoch = {epoch}, run_dir = '{run_name}'")
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

        suppressed_ids = None
        if "random" in str(category):
            suppressed_id_path = f"{RUN_DIR_PREFIX}/{run_name}/random_supp_ids.txt"
            # suppressed_id_path = f"{run_name}/random_supp_ids.txt"
            if Path(suppressed_id_path).exists():
                with open(suppressed_id_path, "r") as handle:
                    suppressed_ids = handle.read().splitlines()
                print("Loaded suppressed ids:", len(suppressed_ids))
                existing_ids = set(val_ids)
                extra_ids = [sid for sid in suppressed_ids if sid not in existing_ids]
                val_ids.extend(extra_ids)
            else:
                print(f"[WARN] random_supp_ids.txt not found for {run_name}")

        if args.verbose:
            print(f"Number of eval seqs: {len(val_ids)}")

        print(config)
        model_name = "Rostlab/prot_bert" # config["model_name"]
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = BertForMaskedLM.from_pretrained(model_name)
        model = model.to(device)
        model.eval()

        mask_layer_range = config.get("mask_layer_range")
        mask_learner = WeightedDifferentiableMaskProtBert(
            model,
            temp_init=config["mask_temperature_init"],
            temp_final=config["mask_temperature_final"],
            temp_decay=config["mask_temperature_decay"],
            init_value=config["mask_init_value"],
            mask_top_layer_frac=config["mask_top_layer_frac"],
            mask_layer_range=tuple(mask_layer_range) if mask_layer_range is not None else None,
            mask_threshold=config["mask_threshold"],
        )

        lightning_model = ProtBERTMaskLearner.load_from_checkpoint(
            ckpt_path,
            model=model,
            mask_learner=mask_learner,
        )

        mask_temp = ckpt.get("mask_temperature", getattr(lightning_model.mask_learner, "temperature", None))
        sparsity_lambda = ckpt.get("sparsity_lambda", None)
        ckpt_epoch = ckpt.get("epoch", epoch)
        num_training_steps = ckpt.get("global_step", None)

        if mask_temp is not None:
            lightning_model.mask_learner.temperature = mask_temp

        lightning_model.eval()
        lightning_model = lightning_model.to(device)
        lightning_model.mask_learner = lightning_model.mask_learner.to(device)
        mask_learner = lightning_model.mask_learner

        with torch.no_grad():
            masks = lightning_model.mask_learner()
            masks = {k: v.to(device) for k, v in masks.items()}

        sparsity = float(lightning_model.mask_learner.get_sparsity())

        meta_rows.append({
            "run_name": run_name,
            "epoch": ckpt_epoch,
            "category": category,
            "target": target,
            "checkpoint_path": ckpt_path,
            "global_step": num_training_steps,
            "sparsity": sparsity,
        })

        inference_dir = Path(run_dir) / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)

        if args.extend_val:
            csv_path = inference_dir / f"epoch_{ckpt_epoch}_full_{args.n_passes}_passes.csv"
        else:
            csv_path = inference_dir / f"epoch_{ckpt_epoch}_heldout_{args.n_passes}_passes.csv"

        using_existing = False
        results_df: pd.DataFrame | None = None

        load_attempted = False
        if csv_path.exists():
            if args.overwrite:
                print(f"[INFO] {csv_path} already exists. Overwriting as requested.")
            else:
                load_attempted = True
                print(f"[INFO] {csv_path} already exists. Loading cached results for metrics.")
                try:
                    results_df = pd.read_csv(csv_path)
                except Exception as exc:  # pragma: no cover - defensive branch
                    print(f"[WARN] Failed to load existing results from {csv_path}: {exc}")
                else:
                    using_existing = True

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

        if not using_existing:
            results_frames = []

            for pass_idx in range(args.n_passes):
                val_dataset = CATHSeqDatasetProtBERT(
                    cath_ids=val_ids,
                    cath_database=db,
                    tokenizer=tokenizer,
                    min_n_res=config.get("min_n_res", 64),
                    max_n_res=config.get("max_n_res", 512),
                    masking=True,
                    use_dssp=True,
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=4 if args.override_batch_size else config["batch_size"],
                    shuffle=config.get("shuffle", False),
                    num_workers=config.get("num_workers", 0),
                    collate_fn=val_dataset.collate_fn,
                )

                for batch in val_loader:
                    if batch["src"].numel() == 0:
                        continue

                    cath_ids = batch["cath_ids"]
                    src = batch["src"].to(device)
                    tgt = batch["tgt"].to(device)
                    attn = batch["attention_mask"].to(device)
                    corr_mask = batch["corr_mask"].to(device).bool()
                    helix_mask = batch["helix_mask"].to(device).bool()
                    strand_mask = batch["strand_mask"].to(device).bool()
                    coil_mask = batch["coil_mask"].to(device).bool()

                    with torch.no_grad():
                        sub_out = lightning_model.subnetwork(
                            input_ids=src,
                            attention_mask=attn,
                            masks=masks,
                            inverse=False,
                        )

                    pred_logits = sub_out.logits
                    mlm_loss = perseqmcel(pred_logits, tgt, corr_mask)
                    mlm_loss_raw = raw_loss_per_seq(pred_logits, tgt, corr_mask)

                    helix_corr = corr_mask & helix_mask
                    strand_corr = corr_mask & strand_mask
                    coil_corr = corr_mask & coil_mask

                    helix_mlm = perseqmcel(pred_logits, tgt, helix_corr) if helix_corr.sum() != 0 else torch.full_like(mlm_loss, torch.nan)
                    strand_mlm = perseqmcel(pred_logits, tgt, strand_corr) if strand_corr.sum() != 0 else torch.full_like(mlm_loss, torch.nan)
                    coil_mlm = perseqmcel(pred_logits, tgt, coil_corr) if coil_corr.sum() != 0 else torch.full_like(mlm_loss, torch.nan)

                    helix_mlm_raw = raw_loss_per_seq(pred_logits, tgt, helix_corr) if helix_corr.sum() != 0 else torch.full_like(mlm_loss, torch.nan)
                    strand_mlm_raw = raw_loss_per_seq(pred_logits, tgt, strand_corr) if strand_corr.sum() != 0 else torch.full_like(mlm_loss, torch.nan)
                    coil_mlm_raw = raw_loss_per_seq(pred_logits, tgt, coil_corr) if coil_corr.sum() != 0 else torch.full_like(mlm_loss, torch.nan)

                    batch_data = {
                        "cath_id": cath_ids,
                        "mlm_loss": mlm_loss.detach().cpu().numpy(),
                        "mlm_loss_raw": mlm_loss_raw.detach().cpu().numpy(),
                        "perplexity": np.exp(mlm_loss.detach().cpu().numpy()),
                        "perplexity_raw": np.exp(mlm_loss_raw.detach().cpu().numpy()),
                        "helix_mlm": helix_mlm.detach().cpu().numpy(),
                        "helix_mlm_raw": helix_mlm_raw.detach().cpu().numpy(),
                        "strand_mlm": strand_mlm.detach().cpu().numpy(),
                        "strand_mlm_raw": strand_mlm_raw.detach().cpu().numpy(),
                        "coil_mlm": coil_mlm.detach().cpu().numpy(),
                        "coil_mlm_raw": coil_mlm_raw.detach().cpu().numpy(),
                    }
                    results_frames.append(pd.DataFrame(batch_data))

            if not results_frames:
                print("[WARN] No valid batches were processed. Skipping run.")
                continue

            results_df = pd.concat(results_frames, ignore_index=True)

            results_df.to_csv(csv_path, index=False)
            print("Wrote CSV to", csv_path)
        else:
            if results_df is None or results_df.empty:
                if load_attempted:
                    print(f"[WARN] Existing results at {csv_path} are empty. Recomputing.")
                else:
                    print(f"[WARN] No cached results found at {csv_path}. Recomputing.")
                continue
            print(f"[INFO] Using cached results from {csv_path}")

        hydrated_df = data_io.hydrate_df_with_cath_terms(results_df, db)
        if model_name == "Rostlab/prot_bert":
            print("Using UR100 metrics for ProtBERT base model")
            gt = pd.read_csv(PROTBERT_UR100_PPL_METRICS)
        else:
            print("Using BFD metrics for ProtBERT base model")
            gt = pd.read_csv(PROTBERT_PPL_METRICS)
        hydrated_df_gt = data_io.hydrate_df_with_cath_terms(gt, db)
        hydrated_df_gt = hydrated_df_gt.groupby("cath_id").head(args.n_passes).reset_index(drop=True)
        hydrated_df_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(hydrated_df["cath_id"])]

        def print_stats(title, df, prefix=""):
            if len(df) == 0:
                print(f"{prefix}{title}: no sequences")
                return
            mean_weighted = float(df['mlm_loss'].mean()) if 'mlm_loss' in df else float('nan')
            mean_weighted_ppl = float(df['perplexity'].mean()) if 'perplexity' in df else float('nan')
            if 'mlm_loss_raw' in df and not df['mlm_loss_raw'].isna().all():
                mean_raw = float(df['mlm_loss_raw'].mean())
                mean_raw_ppl = float(df['perplexity_raw'].mean()) if 'perplexity_raw' in df else float('nan')
                print(f"{prefix}{title} ({len(df)} seqs): (PPL-w) {mean_weighted_ppl:.2f}, (MLM-w) {mean_weighted:.2f} | (PPL-raw) {mean_raw_ppl:.2f}, (MLM-raw) {mean_raw:.2f}")
            else:
                print(f"{prefix}{title} ({len(df)} seqs): (PPL) {mean_weighted_ppl:.2f}, (MLM) {mean_weighted:.2f}")

        def print_mlm_stats(label, series, prefix="", raw_series=None):
            s = series.dropna()
            raw_s = raw_series.dropna() if raw_series is not None else None
            if len(s) == 0:
                print(f"{prefix}{label:<8} (PPL): n/a / (MLM) n/a")
                return
            mean_mlm = float(s.mean())
            line = f"{prefix}{label:<8} (PPL): {float(np.exp(mean_mlm)):.2f} / (MLM) {mean_mlm:.2f}"
            if raw_s is not None and len(raw_s) > 0:
                mean_raw = float(raw_s.mean())
                line += f" | raw PPL: {float(np.exp(mean_raw)):.2f} / MLM: {mean_raw:.2f}"
            print(line)

        if "cath" in str(category):
            supp_df = hydrated_df[hydrated_df[category] == target]
            maint_df = hydrated_df[hydrated_df[category] != target]

            print("\n================= Subnetwork =================")
            print_stats("Suppression", supp_df)
            print_stats("Maintenance", maint_df)
            print("=====================================================\n")

            suppression_gt = hydrated_df_gt[hydrated_df_gt[category] == target]
            maintenance_gt = hydrated_df_gt[hydrated_df_gt[category] != target]

            print("================ ProtBERT Base ======================")
            print_stats("Suppression", suppression_gt)
            print_stats("Maintenance", maintenance_gt)
            print("=====================================================\n")

        elif "random" in str(category):
            suppressed_ids = set(suppressed_ids or [])
            supp_df = hydrated_df[hydrated_df["cath_id"].isin(suppressed_ids)]
            maint_df = hydrated_df[~hydrated_df["cath_id"].isin(suppressed_ids)]

            print("\n================= Random Subnetwork ==================")
            print_stats("Suppression", supp_df)
            print_stats("Maintenance", maint_df)
            print("=====================================================\n")

            suppression_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(suppressed_ids)]
            maintenance_gt = hydrated_df_gt[~hydrated_df_gt["cath_id"].isin(suppressed_ids)]

            print("================ ProtBERT Base ======================")
            print_stats("Suppression", suppression_gt)
            print_stats("Maintenance", maintenance_gt)
            print("=====================================================\n")

        else:
            print("\n================ Residue Subnetwork ==================")
            print_mlm_stats("Helix", hydrated_df["helix_mlm"], raw_series=hydrated_df.get("helix_mlm_raw"))
            print_mlm_stats("Strand", hydrated_df["strand_mlm"], raw_series=hydrated_df.get("strand_mlm_raw"))
            print_mlm_stats("Coil", hydrated_df["coil_mlm"], raw_series=hydrated_df.get("coil_mlm_raw"))
            print("=====================================================\n")

            print("================ ProtBERT Base ======================")
            print_mlm_stats("Helix", hydrated_df_gt["helix_mlm"], raw_series=hydrated_df_gt.get("helix_mlm_raw"))
            print_mlm_stats("Strand", hydrated_df_gt["strand_mlm"], raw_series=hydrated_df_gt.get("strand_mlm_raw"))
            print_mlm_stats("Coil", hydrated_df_gt["coil_mlm"], raw_series=hydrated_df_gt.get("coil_mlm_raw"))
            print("=====================================================\n")
    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        if meta_out_path is None:
            run_dir = Path(f"{RUN_DIR_PREFIX}/{meta_rows[0]['run_name']}")
            out_dir = run_dir / "inference"
            out_dir.mkdir(parents=True, exist_ok=True)
            meta_out_path = out_dir / "sparsity.csv"

        if Path(meta_out_path).exists():
            existing = pd.read_csv(meta_out_path)
            combined = pd.concat([existing, meta_df], ignore_index=True)
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
