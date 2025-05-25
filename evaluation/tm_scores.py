import argparse
import os
import glob
import subprocess
from pathlib import Path

import pandas as pd
import biotite.structure.io as bsio
import torch

from plm_inv_subnetworks.dataset import data_io
from plm_inv_subnetworks.dataset.cath_dataset import get_cath_db
from plm_inv_subnetworks.dataset.data_paths import PDB_DIR, RUN_DIR_PREFIX


def get_rank_001_pdb_path(cath_id: str, directory: str) -> str:
    """
    Return the first .pdb file matching {cath_id}*rank_001*.pdb in directory.
    """
    pattern = os.path.join(directory, f"{cath_id}*rank_001*.pdb")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def calculate_tm_scores(
    input_dir: str,
    reference_dir: str,
    tm_align_path: str,
    cath_ids: list = None,
) -> pd.DataFrame:
    """
    Compute TM-score, RMSD, and pLDDT for each PDB in input_dir against reference_dir.
    """
    records = []

    if cath_ids:
        predictions = [f"{cath_id}.pdb" for cath_id in cath_ids]
    else:
        predictions = os.listdir(input_dir)

    
    for fname in predictions:
        pred_path = os.path.join(input_dir, fname)
        if not fname.endswith('.pdb'):
            continue
        cath_id = Path(fname).stem
        # Load structure
        struct = bsio.load_structure(pred_path, extra_fields=["b_factor"])
        plddt = struct.b_factor.mean()
        # Determine reference PDB
        ref_path = os.path.join(reference_dir, fname)
        # Run TMalign
        result = subprocess.run(
            [tm_align_path, pred_path, ref_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        tm_score, rmsd = None, None
        for line in result.stdout.splitlines():
            if line.startswith("TM-score="):
                try:
                    tm_score = float(line.split()[1])
                except ValueError:
                    pass
            if "RMSD=" in line:
                try:
                    rmsd = float(line.split("RMSD=")[1].split(",")[0])
                except ValueError:
                    pass
        records.append({
            "cath_id": cath_id,
            "TM-score": tm_score,
            "RMSD": rmsd,
            "pLDDT": plddt,
        })
    df = pd.DataFrame(records)
    print(f"Processed {len(records)} structures from {input_dir}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate folding outputs: compute TM-score, RMSD, and pLDDT."
    )
    # CLI args
    parser.add_argument('--run_name', type=str,
                        help='Name of the run (used when --subnetwork_eval is set)')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Epoch number (required for subnetwork_eval)')
    parser.add_argument('--mode', type=str, choices=['pred','gt'], default='pred',
                        help='Prediction mode: pred or gt')
    parser.add_argument('--extend_val', action='store_true',
                        help='Evaluate on full split instead of held-out')
    parser.add_argument('--input_pdb_dir', type=str, default=None,
                        help='Directory of folded PDBs (if not using --subnetwork_eval)')
    parser.add_argument('--reference_pdb_dir', type=str, default=PDB_DIR,
                        help='Directory of reference PDBs (default: PDB_DIR)')
    parser.add_argument('--category', type=str, required='--subnetwork_eval' in os.sys.argv,
                        help='CATH category for grouping (e.g., cath_class_code)')
    parser.add_argument('--target', type=str, required='--subnetwork_eval' in os.sys.argv,
                        help='Target value in category for suppression')
    parser.add_argument('--subnetwork_eval', action='store_true',
                        help='Evaluate run outputs by subnetwork; uses --run_name and --epoch')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to write output CSV (optional)')
    parser.add_argument('--tm_align_path', type=str,default="TMalign",
                        help='Path to TMalign executable')
    parser.add_argument('--cath_ids', type=str, nargs='+', default=None,
                    help='List of CATH IDs to evaluate')

    args = parser.parse_args()

    # Determine input directory and output path
    if args.subnetwork_eval:
        if args.run_name is None or args.epoch is None:
            parser.error("--run_name and --epoch are required with --subnetwork_eval")
        base = f"{RUN_DIR_PREFIX}/{args.run_name}"
        split = "full" if args.extend_val else "val"
        input_dir = f"{base}/folded_epoch_{args.epoch}_{split}/{args.mode}"
        out_label = f"folded_epoch_{args.epoch}_{split}_{args.mode}_tm_scores"
        output_csv = args.output_csv or os.path.join(base, f"{out_label}.csv")
    else:
        input_dir = args.input_pdb_dir
        output_csv = args.output_csv
        if input_dir is None or output_csv is None:
            parser.error("--input_pdb_dir and --output_csv are required when not using --subnetwork_eval")

    print(f"Input dir: {input_dir}")
    print(f"Reference dir: {args.reference_pdb_dir}")
    print(f"Writing results to: {output_csv}")

    # Compute scores
    if args.cath_ids:
        print(f"Evaluating CATH IDs: {args.cath_ids}")
    df = calculate_tm_scores(input_dir, args.reference_pdb_dir, args.tm_align_path, args.cath_ids)
    
    if args.cath_ids:
        print(df)
    else:
        df.to_csv(output_csv, index=False)

        # If subnetwork evaluation, hydrate and summarize
        if args.subnetwork_eval:
            db = get_cath_db()
            hydrated = data_io.hydrate_df_with_cath_terms(df, db)
            supp = hydrated[hydrated[args.category] == args.target]
            maint = hydrated[hydrated[args.category] != args.target]
            for group_name, group_df in zip(["Suppression", "Maintenance"], [supp, maint]):
                print(f"{group_name} Group:")
                print(f"  Mean TM-score: {group_df['TM-score'].mean():.2f}")
                print(f"  Mean RMSD:     {group_df['RMSD'].mean():.2f}")
                print(f"  Mean pLDDT:    {group_df['pLDDT'].mean():.2f}")


if __name__ == "__main__":
    main()
