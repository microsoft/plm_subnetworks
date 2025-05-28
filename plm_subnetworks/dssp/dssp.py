import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from plm_subnetworks.dataset import data_io, data_paths
from plm_subnetworks.dataset.data_paths import PDB_DIR, PDB_CHAIN_DIR, RUN_DIR_PREFIX, CATH_S20_DSSP_FASTA
from plm_subnetworks.dataset.cath_dataset import CATHDatabase, CATHEntry, CATH_ENTRY_FILEPATH, get_cath_db
from plm_subnetworks.dssp.dssp_utils import get_ssp, compute_3_way_dssp_metrics, compute_8_way_dssp_metrics


def calculate_3_way_ssp_accuracy(directory1, dssp_fasta=CATH_S20_DSSP_FASTA):


    db = get_cath_db()

    pred_pdb_files = [f for f in os.listdir(directory1)] 
    results = []

    missing = 0
    for pdb in pred_pdb_files:
        pdb_path = os.path.join(directory1, pdb)
        pred_ssp = get_ssp(pdb_path)
        cath_id = pdb.split(".")[0]
        gt_ssp = db.query(cath_id).dssp

        # replace unassigned predictions with loop
        pred_ssp = pred_ssp.replace("-", "L")

        # print("gt_ssp", gt_ssp)
        # print("pred_ssp", pred_ssp)

        # print(set(gt_ssp))
        # print(set(pred_ssp))

        # sample_metrics = compute_3_way_dssp_metrics(pred_ssp, gt_ssp)
        sample_metrics = compute_8_way_dssp_metrics(pred_ssp, gt_ssp)

        if sample_metrics is None:
            missing+=1
            continue
        
        sample_metrics["cath_id"] = cath_id
        results.append(sample_metrics)


        print(sample_metrics)

        import sys
        sys.exit()

    print("missing", missing)
    df = pd.DataFrame(results)
    
    return df

def calculate_8_way_ssp_metrics(directory1):
    """
    Calculate secondary structure prediction accuracy for a directory of predicted structures.

    Parameters:
    directory1 (str): Path to the directory containing predicted structure files.

    Returns:
    pd.DataFrame: DataFrame containing accuracy, precision, recall, and F1-score per residue type for each CATH ID.
    """
    db = get_cath_db()

    pred_pdb_files = [f for f in os.listdir(directory1)] 
    results = []

    missing = 0
    for pdb in pred_pdb_files:
        pdb_path = os.path.join(directory1, pdb)
        pred_ssp = get_ssp(pdb_path)
        cath_id = pdb.split(".")[0]
        gt_ssp = db.query(cath_id).dssp

        # Replace unassigned predictions with loop ('L')
        pred_ssp = pred_ssp.replace("-", "L")
        
        try:
            output = compute_8_way_dssp_metrics(pred_ssp, gt_ssp)
        except:
            missing += 1
            continue
       
        output["cath_id"] = cath_id
        
        # Append each row individually to preserve DataFrame structure
        results.extend(output.to_dict(orient="records"))
        
    print("missing", missing)

    # Convert list of dicts into DataFrame
    df = pd.DataFrame(results)

    return df


def main(args):


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
    output_df_path = os.path.join(output_dir, f"{output_type}_ssp_8_way_accuracy.csv")
    # df = calculate_ssp_accuracy(input_pdb_dir, args.reference_pdb_dir)
    
    df = calculate_8_way_ssp_metrics(input_pdb_dir)

    df.to_csv(output_df_path, index=False)

    print("Wrote CSV to", output_df_path)
    print(df.head())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--input_pdb_dir', type=str, default="debug_contact_train",
    #                   help='ESMFolded structures')
    parser.add_argument('--reference_pdb_dir', type=str, default=PDB_CHAIN_DIR,
                    help='Reference set, defaults to PDB_CHAIN_DIR')
    
    parser.add_argument('--run_dir', type=str, default="debug_contact_train",
                      help='Name of the run')
    parser.add_argument('--epoch', type=int, default=None,
                    help='Which epoch to load')
    parser.add_argument('--mode', type=str, default="pred",
                    help='pred or gt')

   
    args = parser.parse_args()

    main(args)