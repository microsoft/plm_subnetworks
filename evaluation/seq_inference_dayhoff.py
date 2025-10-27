#!/usr/bin/env python3
"""Evaluate Dayhoff subnetworks using cached checkpoints."""

from __future__ import annotations

import argparse
import math
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from plm_subnetworks.dataset import data_io, data_paths
from plm_subnetworks.dataset.cath_dataset import CATHDatabase, CATH_ENTRY_FILEPATH
from plm_subnetworks.dataset.data_paths import CATH_S20_DSSP_FASTA, RUN_DIR_PREFIX
from plm_subnetworks.dataset.dayhoff_seq_dataloader import CATHSeqDatasetDayhoff
from plm_subnetworks.subnetwork.dayhoff_masking_pl_logits import DayhoffMaskLearner
from plm_subnetworks.subnetwork.dayhoff_modules import WeightedDifferentiableMaskDayhoff
from plm_subnetworks.utils.metrics import ce_loss_tokens

DAYHOFF_BASE_METRICS = Path("inference/dayhoff_perplexity.csv")


def parse_epoch_list(epoch_field: str | None) -> List[int]:
    if epoch_field is None or str(epoch_field).strip() == "":
        return []
    parts = re.split(r"[,\s]+", str(epoch_field).strip())
    out: List[int] = []
    for part in parts:
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            match = re.search(r"(\d+)", part)
            if match:
                out.append(int(match.group(1)))
    seen = set()
    deduped: List[int] = []
    for value in out:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def find_checkpoint_path(run_dir_from_config: str, epoch: int) -> Optional[str]:
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
    if not required.issubset(df.columns):
        raise ValueError(
            f"Found columns {list(df.columns)}; expected at least {sorted(required)}"
        )
    return df


def load_dayhoff_base_metrics(n_passes: int, db: CATHDatabase) -> pd.DataFrame:
    if not DAYHOFF_BASE_METRICS.exists():
        print(f"[WARN] Base metrics file not found: {DAYHOFF_BASE_METRICS}")
        return pd.DataFrame()

    df = pd.read_csv(DAYHOFF_BASE_METRICS)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    rename_map = {
        "causal_ce": "mlm_loss",
        "helix_ce": "helix_mlm",
        "strand_ce": "strand_mlm",
        "coil_ce": "coil_mlm",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "perplexity" not in df.columns and "mlm_loss" in df.columns:
        df["perplexity"] = np.exp(df["mlm_loss"])

    for col in ["mlm_loss", "helix_mlm", "strand_mlm", "coil_mlm"]:
        if col not in df.columns:
            df[col] = np.nan
    df["mlm_loss_raw"] = df["mlm_loss"]
    df["perplexity_raw"] = df["perplexity"]
    df["helix_mlm_raw"] = df["helix_mlm"]
    df["strand_mlm_raw"] = df["strand_mlm"]
    df["coil_mlm_raw"] = df["coil_mlm"]

    if "pass" in df.columns:
        df = df.sort_values(["cath_id", "pass"]).groupby("cath_id").head(n_passes)
    hydrated = data_io.hydrate_df_with_cath_terms(df, db)
    return hydrated


def compute_causal_metrics(
    logits: torch.Tensor,
    tgt: torch.Tensor,
    attn: torch.Tensor,
    helix_mask: torch.Tensor,
    strand_mask: torch.Tensor,
    coil_mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    logits_shifted = logits[:, :-1, :].contiguous()
    tgt_shifted = tgt[:, 1:].contiguous()
    attn_shifted = attn[:, 1:].contiguous()

    per_seq_ce, per_seq_ppl, token_counts = ce_loss_tokens(
        logits_shifted,
        tgt_shifted,
        attention_mask=attn_shifted,
        ignore_index=-100,
    )

    metrics: dict[str, np.ndarray] = {
        "mlm_loss": np.asarray(per_seq_ce, dtype=float),
        "perplexity": np.asarray(per_seq_ppl, dtype=float),
        "token_counts": np.asarray(token_counts, dtype=float),
    }

    for name, mask in (
        ("helix", helix_mask),
        ("strand", strand_mask),
        ("coil", coil_mask),
    ):
        mask_shifted = mask[:, 1:].contiguous()
        ce_vals, _, _ = ce_loss_tokens(
            logits_shifted,
            tgt_shifted,
            attention_mask=attn_shifted,
            mask=mask_shifted,
            ignore_index=-100,
        )
        metrics[f"{name}_mlm"] = np.asarray(ce_vals, dtype=float)
    return metrics


def print_stats(title: str, df: pd.DataFrame, prefix: str = "") -> None:
    if df.empty:
        print(f"{prefix}{title}: no sequences")
        return
    mean_loss = float(df["mlm_loss"].mean()) if "mlm_loss" in df else float("nan")
    mean_ppl = float(df["perplexity"].mean()) if "perplexity" in df else float("nan")
    if "mlm_loss_raw" in df and not df["mlm_loss_raw"].isna().all():
        mean_raw = float(df["mlm_loss_raw"].mean())
        mean_raw_ppl = float(df["perplexity_raw"].mean()) if "perplexity_raw" in df else float("nan")
        print(f"{prefix}{title} ({len(df)} seqs): (PPL-w) {mean_ppl:.2f}, (MLM-w) {mean_loss:.2f} | (PPL-raw) {mean_raw_ppl:.2f}, (MLM-raw) {mean_raw:.2f}")
    else:
        print(f"{prefix}{title} ({len(df)} seqs): (PPL) {mean_ppl:.2f}, (MLM) {mean_loss:.2f}")


def print_mlm_stats(label: str, series: pd.Series, prefix: str = "") -> None:
    s = series.dropna()
    if s.empty:
        print(f"{prefix}{label:<8} (PPL): n/a / (MLM) n/a")
        return
    mean_mlm = float(s.mean())
    print(f"{prefix}{label:<8} (PPL): {math.exp(mean_mlm):.2f} / (MLM) {mean_mlm:.2f}")


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cath_version = "20"
    _, seq_filepath = data_paths.get_cath_paths(cath_version)
    cath_db = CATHDatabase()
    cath_db.load_clf(CATH_ENTRY_FILEPATH)
    cath_db.load_sequences(seq_filepath)
    cath_db.load_dssp(CATH_S20_DSSP_FASTA)

    meta_rows: list[dict[str, object]] = []
    meta_out_path: Optional[Path] = None

    if args.csv:
        runs_df = read_runs_table(args.csv)
        expanded: list[tuple[str, Optional[int], str, str]] = []
        for _, row in runs_df.iterrows():
            epochs = parse_epoch_list(row.get("epoch"))
            if not epochs:
                epochs = [None]
            for value in epochs:
                expanded.append((row["run_name"], value, row["category"], row["target"]))
        subnetworks = expanded
        meta_out_path = Path(args.csv).parent / f"{Path(args.csv).stem}_sparsity.csv"
    else:
        epochs = parse_epoch_list(args.epoch) if args.epoch is not None else [None]
        if not epochs:
            epochs = [None]
        subnetworks = [(args.run_name, value, args.category, args.target) for value in epochs]

    for run_name, epoch, category, target in subnetworks:
        if not run_name:
            continue

        run_dir = f"{RUN_DIR_PREFIX}/{run_name}"
        try:
            config, split = data_io.get_args_split(run_dir)
        except FileNotFoundError:
            print(f"[WARN] Could not load config for run '{run_name}'. Skipping.")
            continue

        if epoch is not None and not isinstance(epoch, (int, np.integer)):
            parsed = parse_epoch_list(str(epoch))
            epoch = parsed[0] if parsed else None
        if epoch is None:
            print(f"[WARN] No epoch provided for run '{run_name}'. Skipping.")
            continue

        ckpt_path = find_checkpoint_path(config.get("run_dir", run_dir), int(epoch))
        if ckpt_path is None:
            print(f"[WARN] Could not find checkpoint for epoch {epoch} under {config.get('run_dir', run_dir)}. Skipping.")
            continue

        print(
            f"\n>>> Evaluating Dayhoff subnetwork: category='{category}', target='{target}', epoch={epoch}, run_dir='{run_name}'"
        )
        print(f"Checkpoint path: {ckpt_path}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            ckpt = torch.load(ckpt_path, map_location="cpu")

        if args.extend_val:
            val_ids = split["train"] + split["val"] + split["test"]
            print("Eval on full set")
        else:
            val_ids = split["val"] + split["test"]
            print("Eval on held-out validation/test sequences")

        suppressed_ids: list[str] = []
        if category and "random" in str(category):
            suppressed_id_path = config.get("random_supp_id_path")
            run_dir_path = Path(config.get("run_dir", run_dir))
            if suppressed_id_path:
                candidate = Path(suppressed_id_path)
                if not candidate.is_absolute():
                    suppressed_path = run_dir_path / candidate
                else:
                    suppressed_path = candidate
            else:
                suppressed_path = run_dir_path / "random_supp_ids.txt"

            if suppressed_path.exists():
                suppressed_ids = [
                    line.strip()
                    for line in suppressed_path.read_text().splitlines()
                    if line.strip()
                ]
                if suppressed_ids:
                    existing_ids = set(str(cid) for cid in val_ids)
                    extra_ids = [sid for sid in suppressed_ids if sid not in existing_ids]
                    if extra_ids:
                        val_ids.extend(extra_ids)
                    print(f"Loaded {len(suppressed_ids)} random suppression ids from {suppressed_path}")
                else:
                    print(f"[WARN] random_supp_ids.txt at {suppressed_path} is empty")
            else:
                print(f"[WARN] random_supp_ids.txt not found at {suppressed_path}")

        tokenizer_name = args.tokenizer_name_or_path or config.get("model_name") or args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=config.get("trust_remote_code", False) or args.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.get("model_name", args.model_name_or_path),
            trust_remote_code=config.get("trust_remote_code", False) or args.trust_remote_code,
        )
        model.to(device)
        model.eval()

        mask_layer_range = config.get("mask_layer_range")
        if mask_layer_range is not None and isinstance(mask_layer_range, Sequence):
            mask_layer_range = tuple(mask_layer_range)

        extra_attention_tokens = config.get("mask_mamba_projections") or ()
        if isinstance(extra_attention_tokens, str):
            extra_attention_tokens = tuple(
                token.strip() for token in extra_attention_tokens.split(",") if token.strip()
            )
        else:
            extra_attention_tokens = tuple(extra_attention_tokens)

        mask_learner = WeightedDifferentiableMaskDayhoff(
            model,
            temp_init=config.get("mask_temperature_init", 0.5),
            temp_final=config.get("mask_temperature_final", 0.05),
            temp_decay=config.get("mask_temperature_decay", 50),
            mask_threshold=config.get("mask_threshold", 0.37),
            init_value=config.get("mask_init_value", 0.5),
            num_model_layers=config.get("num_layers", 24),
            mask_top_layer_frac=config.get("mask_top_layer_frac", 1.0),
            mask_layer_range=mask_layer_range,
            temp_hold=config.get("mask_temp_hold", 10),
            extra_attention_tokens=extra_attention_tokens if extra_attention_tokens else None,
        )

        lightning_model = DayhoffMaskLearner.load_from_checkpoint(
            ckpt_path,
            model=model,
            mask_learner=mask_learner,
            suppression_mode=config.get("suppression_mode", "cath"),
            suppression_level=config.get("suppression_level", "class"),
            suppression_target=config.get("suppression_target", "1"),
            suppression_lambda=config.get("suppression_lambda", 1.0),
            maintenance_lambda=config.get("maintenance_lambda", 1.0),
            maintenance_mlm_lambda=config.get("maintenance_mlm_lambda", 1.0),
            sparsity_lambda_init=config.get("sparsity_lambda_init", 0.0),
            sparsity_lambda_final=config.get("sparsity_lambda_final", 0.0),
            sparsity_warmup_epochs=config.get("sparsity_warmup_epochs", 0),
            sparsity_ramp_epochs=config.get("sparsity_ramp_epochs", 0),
            learning_rate=config.get("learning_rate", 1e-3),
            lr_phaseA=config.get("lr_phaseA", 1e-3),
            lr_phaseB=config.get("lr_phaseB", 1e-4),
            lr_hold_epochs=config.get("lr_hold_epochs", 0),
            lr_plateau_epochs=config.get("lr_plateau_epochs", 0),
            random_supp_id_path=config.get("random_supp_id_path"),
            use_corrupted_inputs=not config.get("disable_input_corruption", False),
        )

        lightning_model.eval()
        lightning_model.to(device)
        lightning_model.mask_learner = lightning_model.mask_learner.to(device)

        with torch.no_grad():
            masks = {k: v.to(device) for k, v in lightning_model.mask_learner().items()}
        sparsity = float(lightning_model.mask_learner.get_sparsity())

        meta_rows.append(
            {
                "run_name": run_name,
                "epoch": ckpt.get("epoch", epoch),
                "category": category,
                "target": target,
                "checkpoint_path": ckpt_path,
                "global_step": ckpt.get("global_step", None),
                "sparsity": sparsity,
            }
        )

        inference_dir = Path(config.get("run_dir", run_dir)) / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        if args.extend_val:
            csv_path = inference_dir / f"epoch_{epoch}_full_{args.n_passes}_passes.csv"
        else:
            csv_path = inference_dir / f"epoch_{epoch}_heldout_{args.n_passes}_passes.csv"

        using_existing = False
        results_df: Optional[pd.DataFrame] = None
        load_attempted = False
        if csv_path.exists():
            if args.overwrite:
                print(f"[INFO] {csv_path} already exists. Overwriting as requested.")
            else:
                load_attempted = True
                print(f"[INFO] {csv_path} already exists. Loading cached results for metrics.")
                try:
                    results_df = pd.read_csv(csv_path)
                except Exception as exc:
                    print(f"[WARN] Failed to load existing results from {csv_path}: {exc}")
                else:
                    using_existing = True

        mask_inputs = not config.get("disable_input_corruption", False)
        mask_pct = args.mask_pct if mask_inputs else 0.0
        use_dssp = config.get("suppression_mode", "cath") == "dssp"

        if args.verbose:
            print(f"Masking inputs: {mask_inputs} (mask_pct={mask_pct})")
            print(f"Using DSSP masks: {use_dssp}")
            print(f"# Eval sequences: {len(val_ids)}")
            print(f"Sparsity: {sparsity:.4f}")

        if not using_existing:
            records: list[dict[str, object]] = []
            total_loss = 0.0
            total_tokens = 0

            for pass_idx in range(args.n_passes):
                if args.verbose:
                    print(f"Pass {pass_idx + 1}/{args.n_passes}")

                dataset = CATHSeqDatasetDayhoff(
                    cath_ids=val_ids,
                    cath_database=cath_db,
                    tokenizer=tokenizer,
                    mask_pct=mask_pct,
                    masking=mask_inputs,
                    min_n_res=config.get("min_n_res", 64),
                    max_n_res=config.get("max_n_res", 512),
                    use_dssp=use_dssp,
                )

                loader = DataLoader(
                    dataset,
                    batch_size=4 if args.override_batch_size else config.get("batch_size", 4),
                    shuffle=config.get("shuffle", False),
                    num_workers=config.get("num_workers", 0),
                    collate_fn=dataset.collate_fn,
                )

                for batch in loader:
                    if batch["src"].numel() == 0:
                        continue

                    src = batch["src"].to(device)
                    tgt = batch["tgt"].to(device)
                    attn = batch["attention_mask"].to(device)
                    helix_mask = batch["helix_mask"].to(device)
                    strand_mask = batch["strand_mask"].to(device)
                    coil_mask = batch["coil_mask"].to(device)
                    cath_ids_batch = batch["cath_ids"]

                    inputs = src if lightning_model.use_corrupted_inputs else tgt

                    with torch.no_grad():
                        logits = lightning_model.subnetwork(
                            input_ids=inputs,
                            attention_mask=attn,
                            masks=masks,
                            inverse=False,
                        ).logits

                    metrics = compute_causal_metrics(
                        logits,
                        tgt,
                        attn,
                        helix_mask,
                        strand_mask,
                        coil_mask,
                    )

                    mlm_loss = metrics["mlm_loss"]
                    perplexity = metrics["perplexity"]
                    token_counts = metrics["token_counts"]
                    helix = metrics["helix_mlm"]
                    strand = metrics["strand_mlm"]
                    coil = metrics["coil_mlm"]

                    total_loss += float(np.nan_to_num(mlm_loss) @ token_counts)
                    total_tokens += int(token_counts.sum())

                    for idx, cath_id in enumerate(cath_ids_batch):
                        def finite_or_nan(value: float) -> float:
                            return value if math.isfinite(value) else float("nan")

                        records.append(
                            {
                                "cath_id": cath_id,
                                "mlm_loss": finite_or_nan(mlm_loss[idx]),
                                "mlm_loss_raw": finite_or_nan(mlm_loss[idx]),
                                "perplexity": finite_or_nan(perplexity[idx]),
                                "perplexity_raw": finite_or_nan(perplexity[idx]),
                                "helix_mlm": finite_or_nan(helix[idx]),
                                "helix_mlm_raw": finite_or_nan(helix[idx]),
                                "strand_mlm": finite_or_nan(strand[idx]),
                                "strand_mlm_raw": finite_or_nan(strand[idx]),
                                "coil_mlm": finite_or_nan(coil[idx]),
                                "coil_mlm_raw": finite_or_nan(coil[idx]),
                                "id": cath_id,
                            }
                        )

            if not records:
                print("[WARN] No valid batches processed. Skipping run.")
                continue

            results_df = pd.DataFrame.from_records(records)
            results_df.to_csv(csv_path, index=False)
            dataset_ce = total_loss / max(total_tokens, 1)
            dataset_ppl = math.exp(dataset_ce) if math.isfinite(dataset_ce) else float("nan")
            print(
                f"Processed {len(results_df)} sequence entries.\n"
                f"Dataset mean causal CE: {dataset_ce:.6f} nats (perplexity={dataset_ppl:.2f})"
            )
            print(f"Wrote CSV to {csv_path}")
        else:
            if results_df is None or results_df.empty:
                if load_attempted:
                    print(f"[WARN] Existing results at {csv_path} are empty. Recomputing.")
                else:
                    print(f"[WARN] No cached results found at {csv_path}. Recomputing.")
                continue
            print(f"[INFO] Using cached results from {csv_path}")

        hydrated_df = data_io.hydrate_df_with_cath_terms(results_df, cath_db)
        base_df = load_dayhoff_base_metrics(args.n_passes, cath_db)

        eval_ids: set[str] = set()
        if "cath_id" in hydrated_df.columns:
            eval_ids = set(hydrated_df["cath_id"].astype(str))
        elif "id" in hydrated_df.columns:
            eval_ids = set(hydrated_df["id"].astype(str))

        if eval_ids and not base_df.empty:
            id_col = "cath_id" if "cath_id" in base_df.columns else None
            if id_col is not None:
                base_df = base_df[base_df[id_col].astype(str).isin(eval_ids)]
        if args.verbose:
            print("=== Subnetwork summary ===")
            if "cath_code" in hydrated_df:
                print(hydrated_df.head())

        if category and "cath" in category:
            suppression_df = hydrated_df[hydrated_df[category] == target]
            maintenance_df = hydrated_df[hydrated_df[category] != target]
            print("\n================= Dayhoff Subnetwork =================")
            print_stats("Suppression", suppression_df)
            print_stats("Maintenance", maintenance_df)
            print("=====================================================\n")

            if not base_df.empty and category in base_df.columns:
                base_supp = base_df[base_df[category] == target]
                base_maint = base_df[base_df[category] != target]
                print("================ Dayhoff Base ======================")
                print_stats("Suppression", base_supp)
                print_stats("Maintenance", base_maint)
                print("===================================================\n")
        elif category and "random" in str(category):
            suppressed_set = {str(cid) for cid in suppressed_ids}
            if not suppressed_set:
                print("[WARN] No random suppression ids found; reporting maintenance metrics only")
            cath_col = "cath_id" if "cath_id" in hydrated_df.columns else ("id" if "id" in hydrated_df.columns else None)
            if cath_col is None:
                print("[WARN] Unable to determine identifier column for random suppression evaluation")
                cath_series = hydrated_df.index.to_series().astype(str)
            else:
                cath_series = hydrated_df[cath_col].astype(str)
            suppression_df = hydrated_df[cath_series.isin(suppressed_set)] if suppressed_set else hydrated_df.iloc[0:0]
            maintenance_df = hydrated_df[~cath_series.isin(suppressed_set)] if suppressed_set else hydrated_df
            print("\n================= Random Subnetwork ==================")
            print_stats("Suppression", suppression_df)
            print_stats("Maintenance", maintenance_df)
            print("=====================================================\n")
            if not base_df.empty:
                base_col = None
                if cath_col and cath_col in base_df.columns:
                    base_col = cath_col
                elif "cath_id" in base_df.columns:
                    base_col = "cath_id"
                if base_col:
                    base_series = base_df[base_col].astype(str)
                    base_supp = base_df[base_series.isin(suppressed_set)] if suppressed_set else base_df.iloc[0:0]
                    base_maint = base_df[~base_series.isin(suppressed_set)] if suppressed_set else base_df
                else:
                    base_supp = base_df.iloc[0:0]
                    base_maint = base_df
                print("================ Dayhoff Base ======================")
                print_stats("Suppression", base_supp)
                print_stats("Maintenance", base_maint)
                print("===================================================\n")
        else:
            print("\n================ Residue Subnetwork ==================")
            for name in ["helix_mlm", "strand_mlm", "coil_mlm"]:
                label = name.split("_")[0].capitalize()
                print_mlm_stats(label, hydrated_df[name])
            print("=====================================================\n")
            if not base_df.empty:
                print("================ Dayhoff Base ======================")
                for name in ["helix_mlm", "strand_mlm", "coil_mlm"]:
                    label = name.split("_")[0].capitalize()
                    print_mlm_stats(label, base_df[name])
                print("===================================================\n")

    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        if meta_out_path is None:
            run_dir = Path(f"{RUN_DIR_PREFIX}/{meta_rows[0]['run_name']}")
            out_dir = run_dir / "inference"
            out_dir.mkdir(parents=True, exist_ok=True)
            meta_out_path = out_dir / "metadata_sparsity.csv"
        if meta_out_path.exists():
            existing = pd.read_csv(meta_out_path)
            combined = pd.concat([existing, meta_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["run_name", "epoch"], keep="last")
            combined.to_csv(meta_out_path, index=False)
        else:
            meta_df.to_csv(meta_out_path, index=False)
        print(f"Wrote metadata to {meta_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_batch_size", type=bool, default=True,
                        help="Override batch size when using smaller GPUs", required=False)
    parser.add_argument("--extend_val", action="store_true",
                        help="Evaluate on train+val+test instead of held-out sets")
    parser.add_argument("--n_passes", type=int, default=1,
                        help="Number of evaluation passes for stochastic datasets")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing inference CSVs instead of skipping")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name of run to evaluate")
    parser.add_argument("--epoch", type=str, default=None,
                        help="Epoch(s) to evaluate; comma-separated values allowed")
    parser.add_argument("--category", type=str, default=None,
                        help="Suppression category (e.g., cath_class, random_seq, residue)")
    parser.add_argument("--target", type=str, default=None,
                        help="Suppression target (e.g., 3, helix, random)")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV/XLSX with columns: run_name, epoch, category, target")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Dayhoff-170m-UR90",
                        help="HuggingFace model identifier for Dayhoff")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Separate tokenizer identifier (defaults to model)")
    parser.add_argument("--mask_pct", type=float, default=0.0,
                        help="Masking probability when corruption is enabled")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Enable remote code when loading HuggingFace models", default=True)
    parser.add_argument("-V", "--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()
    evaluate(args)
