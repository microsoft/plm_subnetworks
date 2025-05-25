'''Works for the output of ESMFold.'''

from plm_inv_subnetworks.dataset import data_paths 
from plm_inv_subnetworks.dataset.data_paths import PDB_DIR, RUN_DIR_PREFIX

import os
import subprocess
import pandas as pd
import sys
import argparse
from plm_inv_subnetworks.dataset.cath_dataset import CATHDatabase, CATHEntry, CATH_ENTRY_FILEPATH
import biotite.structure.io as bsio
from pathlib import Path
from plm_inv_subnetworks.dataset.cath_dataset import get_cath_db

import torch
from plm_inv_subnetworks.dataset import data_io

import os
import glob

PDB_CHAIN_DIR = PDB_DIR

# Use this function if using ColabFold or AF2 predictions: https://github.com/sokrypton/ColabFold
def get_rank_001_pdb_path(cath_id, directory):
    """
    Returns the path to the .pdb file that starts with the given CATH ID,
    contains 'rank_001', and ends with '.pdb'.
    
    Parameters:
        cath_id (str): The CATH ID prefix of the filename (e.g., "2p84A02").
        directory (str): The directory to search in.
    
    Returns:
        str or None: Full path to the matching .pdb file, or None if not found.
    """
    pattern = os.path.join(directory, f"{cath_id}")
    matches = glob.glob(pattern)
    
    if matches:
        return matches[0]  # Return the first match
    else:
        return None


def calculate_tm_scores(directory1, directory2=PDB_CHAIN_DIR, tm_align_path="TMalign", debug=False):
    """
    Computes TM-scores and RMSD between corresponding PDB files in two directories.

    Args:
    - directory1 (str): Path to the first directory containing PDB files.
    - directory2 (str): Path to the second directory containing PDB files.
    - tm_align_path (str): Path to the TM-align executable.

    Returns:
    - pd.DataFrame: A dataframe containing file names, TM-scores, RMSD, and pLDDT scores.
    """



    pred_pdb_files = [f for f in os.listdir(directory1)] 

    results = []

    for cath_id in pred_pdb_files:

        plddt, tm_score, rmsd = None, None, None
        
        # if using a different model e.g. AF2 or ColabFold, use this:
        # pred_pdb_path = get_rank_001_pdb_path(cath_id, directory1)
        
        pred_pdb_path = os.path.join(directory1, cath_id)

        struct = bsio.load_structure(pred_pdb_path, extra_fields=["b_factor"])
        plddt = struct.b_factor.mean()  # this will be the pLDDT

        # if using a different model e.g. AF2 or ColabFold, use this:
        # gt_pdb_path = f"{os.path.join(directory2, cath_id.split('_')[0])}.pdb"
        
        gt_pdb_path = os.path.join(directory2, cath_id)

        result = subprocess.run(
                    [tm_align_path, pred_pdb_path, gt_pdb_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
        
        for line in result.stdout.split("\n"):
            if "TM-score=" in line and not line.startswith("("):  # TM-score line
                try:
                    tm_score = float(line.split()[1])  # Extract TM-score
                except ValueError:
                    print(f"Could not parse TM-score from: {line}")
            if "Aligned length=" in line and "RMSD=" in line:  # RMSD line
                try:
                    rmsd = float(line.split("RMSD=")[1].split(",")[0].strip())  # Extract RMSD
                except ValueError:
                    print(f"Could not parse RMSD from: {line}")

        results.append({
                        "cath_id": cath_id.split(".")[0],
                        "TM-score": tm_score,
                        "RMSD": rmsd,
                        "pLDDT": plddt,
                    })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    print(f"Processed {directory1}, missing: {len(pred_pdb_files) - len(df)}")
    return df

def main(args):

    print(args.subnetwork_eval)
    if not args.subnetwork_eval:
        output_df_path = args.output_df_path
        print("Writing to...", output_df_path)
        df = calculate_tm_scores(args.input_pdb_dir, args.reference_pdb_dir, tm_align_path="TMalign")
        
        df.to_csv(output_df_path, index=False)

        print("Wrote CSV to", output_df_path)
        print(df.head())

    else:

        if args.epoch is not None:
            input_pdb_dir = f"{RUN_DIR_PREFIX}/{args.run_dir}/folded_epoch_{args.epoch}/{args.mode}"
        
        else:
            print("reading from latest checkpoint...")
            run_dir = f"{RUN_DIR_PREFIX}/{args.run_dir}"
            config, split = data_io.get_args_split(run_dir)
            ckpt_path = f"{config['run_dir']}/checkpoints/regular_checkpoints/last.ckpt"    
            ckpt_epoch = torch.load(ckpt_path, map_location="cpu")["epoch"]
            input_pdb_dir = f"{run_dir}/folded_epoch_{ckpt_epoch}/{args.mode}"

        output_type = input_pdb_dir.split("/")[-1]
        output_dir = str(Path(input_pdb_dir).parent)
        output_df_path = os.path.join(output_dir, f"{output_type}_tm_scores.csv")

        if args.extend_val:
            input_pdb_dir = f"{RUN_DIR_PREFIX}/{args.run_dir}/folded_epoch_{args.epoch}_full/{args.mode}"
            output_dir = str(Path(input_pdb_dir).parent)
            output_df_path = os.path.join(output_dir, f"{output_type}_tm_scores_.csv")
        else:
            input_pdb_dir = f"{RUN_DIR_PREFIX}/{args.run_dir}/folded_epoch_{args.epoch}_val/{args.mode}"
            output_dir = str(Path(input_pdb_dir).parent)
            output_df_path = os.path.join(output_dir, f"{output_type}_tm_scores_val.csv")

        print("Writing to...", output_df_path)
        df = calculate_tm_scores(input_pdb_dir, args.reference_pdb_dir, tm_align_path="TMalign")
        
        df.to_csv(output_df_path, index=False)
        print("Wrote CSV to", output_df_path)

        # hydrate
    
        db = get_cath_db()
        hydrated_df = data_io.hydrate_df_with_cath_terms(df, db)
        suppression = hydrated_df[hydrated_df[args.category] == args.target]
        maintenance = hydrated_df[hydrated_df[args.category] != args.target] 

        # Mean TM-score, RMSD, and pLDDT
        print("Suppression Group:")
        print(f" ###  Mean TM-score: {suppression['TM-score'].mean():.2f}")
        print(f" ### Mean RMSD: {suppression['RMSD'].mean():.2f}")
        print(f" ### Mean pLDDT: {suppression['pLDDT'].mean():.2f}")

        print("Maintenance Group:")
        print(f" ### Mean TM-score: {maintenance['TM-score'].mean():.2f}")
        print(f" ### Mean RMSD: {maintenance['RMSD'].mean():.2f}")
        print(f" ###  Mean pLDDT: {maintenance['pLDDT'].mean():.2f}")

        print(df.head())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_pdb_dir', type=str, default="debug_contact_train",
                      help='ESMFolded structures')
    parser.add_argument('--reference_pdb_dir', type=str, default=PDB_CHAIN_DIR,
                    help='Reference set, defaults to PDB_CHAIN_DIR')
    parser.add_argument('--run_dir', type=str, default="debug_contact_train",
                      help='Name of the run')
    parser.add_argument('--epoch', type=int, default=None,
                    help='Which epoch to load')
    parser.add_argument('--mode', type=str, default="pred",
                    help='pred or gt')
    parser.add_argument('--extend_val', action='store_true',
                    help='Add flag to evaluate on full set instead of heldout set')
    
    parser.add_argument('--category', type=str)
    parser.add_argument('--target', type=str)

    parser.add_argument('--subnetwork_eval', action='store_true',
                    help='Add flag to evaluate on full set instead of heldout set')

    
    parser.add_argument('--output_df_path', type=str, default=None)

    args = parser.parse_args()

    main(args)