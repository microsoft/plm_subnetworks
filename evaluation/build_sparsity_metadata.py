#!/usr/bin/env python3
"""Aggregate sparsity metadata for a set of runs listed in a CSV file.

This utility looks up the ``inference/sparsity.csv`` file that is produced
alongside ProtBERT inference outputs.  For each row in the provided CSV
it extracts the matching epoch entry and writes a consolidated metadata
table next to the CSV that was supplied.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

# Depend on RUN_DIR_PREFIX when available, but fall back to repo-relative path.
try:  # pragma: no cover - defensive import
    from plm_subnetworks.dataset.data_paths import RUN_DIR_PREFIX as _RUN_DIR_PREFIX  # type: ignore
except Exception:  # pylint: disable=broad-except
    _RUN_DIR_PREFIX = None


def default_runs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "runs"


RUNS_DIR = (Path(_RUN_DIR_PREFIX) if _RUN_DIR_PREFIX else default_runs_dir()).resolve()


DEFAULT_FIELD_ORDER = [
    "run_name",
    "epoch",
    "category",
    "target",
    "checkpoint_path",
    "global_step",
    "sparsity",
]


def parse_epoch_list(epoch_field: str | None) -> list[int]:
    """Return a list of integer epochs extracted from a CSV field."""

    if epoch_field is None:
        return []

    cleaned = str(epoch_field).strip()
    if not cleaned:
        return []

    epochs: list[int] = []
    for part in cleaned.replace(";", ",").split(","):
        part_clean = part.strip()
        if not part_clean:
            continue
        try:
            epochs.append(int(part_clean))
        except ValueError:
            digits = ''.join(ch for ch in part_clean if ch.isdigit())
            if digits:
                epochs.append(int(digits))
    return epochs


def normalise_row(row: dict[str, str | None]) -> dict[str, str]:
    """Lower-case and trim keys/values for easier lookup."""

    normalised: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        key_norm = key.strip().lower()
        if not key_norm:
            continue
        if value is None:
            normalised[key_norm] = ""
        else:
            normalised[key_norm] = value.strip()
    return normalised


def read_input_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [normalise_row(row) for row in reader]


def read_sparsity_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tables: list[dict[str, str]] = []
        for row in reader:
            ordered = OrderedDict()
            for key, value in row.items():
                if key is None:
                    continue
                cleaned_key = key.strip()
                if not cleaned_key:
                    continue
                if isinstance(value, str):
                    ordered[cleaned_key] = value.strip()
                elif value is None:
                    ordered[cleaned_key] = ""
                else:
                    ordered[cleaned_key] = str(value)
            tables.append(ordered)
        return tables


def select_epoch_row(rows: Iterable[dict[str, str]], epoch: int) -> dict[str, str] | None:
    for row in rows:
        value = row.get("epoch")
        if value is None:
            continue
        try:
            row_epoch = int(str(value).strip())
        except (TypeError, ValueError):
            continue
        if row_epoch == epoch:
            return dict(row)
    return None


def build_sparsity_records(input_rows: list[dict[str, str]], runs_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for row in input_rows:
        run_name = row.get("run_name") or row.get("run dir") or row.get("run_dir")
        if not run_name:
            print("[WARN] Skipping row with no run_name/run_dir field:", row, file=sys.stderr)
            continue

        epochs = parse_epoch_list(row.get("epoch"))
        if not epochs:
            print(f"[WARN] No epoch specified for run '{run_name}'. Skipping.", file=sys.stderr)
            continue

        meta_path = runs_dir / run_name / "inference" / "metadata_sparsity.csv"
        if not meta_path.exists():
            print(f"[WARN] Missing sparsity metadata at {meta_path}.", file=sys.stderr)
            continue

        sparsity_rows = read_sparsity_table(meta_path)
        if not sparsity_rows:
            print(f"[WARN] No rows found in metadata file {meta_path}.", file=sys.stderr)
            continue

        for epoch in epochs:
            match = select_epoch_row(sparsity_rows, epoch)
            if match is None:
                print(
                    f"[WARN] Epoch {epoch} not found in {meta_path}.",
                    file=sys.stderr,
                )
                continue

            record = dict(match)

            global_step_candidates = (
                record.get("global_step"),
                row.get("global_step"),
            )

            selected_global_step: str | None = None
            for candidate in global_step_candidates:
                if candidate is None:
                    continue
                candidate_str = str(candidate).strip()
                if candidate_str:
                    selected_global_step = candidate_str
                    break

            record["global_step"] = selected_global_step or ""
            records.append(record)

    return records


def compute_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return DEFAULT_FIELD_ORDER

    seen = {key for row in rows for key in row.keys() if key}
    ordered: list[str] = [name for name in DEFAULT_FIELD_ORDER if name in seen]
    remainder = [name for name in seen if name not in ordered]
    ordered.extend(sorted(remainder))
    return ordered


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = compute_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble sparsity metadata for a list of runs.")
    parser.add_argument("--csv", type=Path, help="Path to the CSV containing run entries.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Optional override for the runs directory (defaults to the repository runs/ folder).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path: Path = args.csv

    if not input_path.exists():
        print(f"[ERROR] Input CSV not found: {input_path}", file=sys.stderr)
        return 1

    runs_dir = args.runs_dir.resolve() if args.runs_dir else RUNS_DIR

    if not runs_dir.exists():
        print(f"[ERROR] Runs directory not found: {runs_dir}", file=sys.stderr)
        return 1

    input_rows = read_input_csv(input_path)
    if not input_rows:
        print(f"[ERROR] No rows parsed from {input_path}", file=sys.stderr)
        return 1

    records = build_sparsity_records(input_rows, runs_dir)
    if not records:
        print("[ERROR] No sparsity metadata gathered; nothing to write.", file=sys.stderr)
        return 1

    output_path = input_path.parent / f"{input_path.stem}_sparsity.csv"
    if output_path.exists() and not args.overwrite:
        print(f"[ERROR] Output file already exists: {output_path}. Use --overwrite to replace it.", file=sys.stderr)
        return 1

    write_output(output_path, records)
    print(f"[INFO] Wrote {len(records)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
