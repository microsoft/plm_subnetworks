import os
import subprocess
import sys

import pandas as pd

from plm_inv_subnetworks.dataset.data_paths import PDB_DIR, PDB_DSSP_DIR
from plm_inv_subnetworks.dataset.cath_dataset import get_cath_db



def extract_chain(pdb_path, chain_id, output_path):
    """
    Extracts a specific chain from a PDB file and saves it to a new file.

    Args:
    - pdb_path (str): Path to the input PDB file.
    - chain_id (str): The chain identifier to extract.
    - output_path (str): Path to save the extracted chain PDB file.
    """
    with open(pdb_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                if line[21] == chain_id:  # Chain ID is at column 22 (index 21 in Python)
                    outfile.write(line)
            elif line.startswith("ENDMDL") or line.startswith("TER"):  
                outfile.write(line)  # Preserve termination lines

def run_dssp(input_file, output_file, dssp_path="mkdssp"):
    result = subprocess.run(
            [dssp_path, "-i", input_file, "-o", output_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    # Check for errors
    if result.returncode != 0:
        print(f"Error running DSSP on {input_file}: {result.stderr}")
       

     
def run_dssp_with_pdb_lookup(input_dir, output_dir, dssp_path="mkdssp"):
    """
    Iterates over files in input_dir, finds corresponding PDB files in pdb_dir, extracts a specified chain,
    runs DSSP on the extracted chain, and saves the output to output_dir.
    Skips files that have already been processed.

    Args:
    - input_dir (str): Directory containing input files (used to determine file names).
    - pdb_dir (str): Directory containing PDB structures.
    - output_dir (str): Directory to store DSSP output files.
    - chain_id (str): The chain identifier to extract.
    - dssp_path (str): Path to the DSSP executable (default is "mkdssp" if it's in PATH).

    Returns:
    - None (saves DSSP output files in output_dir).
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a cache directory for temporary PDB files
    cache_dir = os.path.expanduser("~/.cache/dssp/")
    os.makedirs(cache_dir, exist_ok=True)

    cath_db = get_cath_db()
    pdb_files = [f for f in os.listdir(input_dir) if "rank_001" in f and f.endswith(".pdb")]
    gt_computed_dssp = os.listdir(PDB_DSSP_DIR)


    # Iterate over files in input_dir to determine the file names to search for
    for pdb_file in pdb_files:

        cath_id = pdb_file.split("_")[0]
        chain_id = str(cath_db.query(cath_id).chain_num)  # Get chain ID
        gt_pdb_file_path = os.path.join(PDB_DIR, cath_id)

        print(f"Ground truth PDB: {gt_pdb_file_path}")
        print(f"Prediction PDB: {pdb_file}")

        # compute reference
        if f"{cath_id}.dssp" in gt_computed_dssp:
            print(f"Skipping {cath_id}, DSSP output already exists.")
        else:
            # Define output DSSP file path
            gt_output_dssp_path = os.path.join(PDB_DSSP_DIR, f"{cath_id}.dssp")
             # Extract the specified chain from the PDB structure in PDB_DIR
            temp_pdb_path = os.path.join(cache_dir, f"temp_{pdb_file}")
            extract_chain(gt_pdb_file_path, chain_id, temp_pdb_path)
            run_dssp(temp_pdb_path, gt_output_dssp_path)
            os.remove(temp_pdb_path)
        
        # compute prediction
        pred_input_pdb_path = os.path.join(input_dir, pdb_file)
        pred_output_dssp_path = os.path.join(output_dir, f"{cath_id}.dssp")
        run_dssp(pred_input_pdb_path, pred_output_dssp_path)



def parse_dssp(output_file, debug=False):
    """
    Parses DSSP output and extracts per-residue secondary structure annotations.

    Args:
    - output_file (str): Path to the DSSP output file.

    Returns:
    - pd.DataFrame: DataFrame containing chain ID, residue number, and secondary structure.
    """
    residues = []
    
    with open(output_file, "r") as f:
        dssp_started = False  # Flag to track the start of DSSP data
        for line in f:
            if line.startswith("  #  RESIDUE AA STRUCTURE"):  # DSSP output starts after this
                dssp_started = True
                continue

            if dssp_started:
                if len(line) < 20:  # Ignore empty or invalid lines
                    continue
                
                try:
                    chain_id = line[11]  # Chain ID (Column 12 in DSSP output)
                    res_id = int(line[5:10].strip())  # Residue number
                    ss = line[16]  # Secondary structure
                    
                    # Convert blank spaces (coil) to 'C'
                    if ss == " ":
                        ss = "C"
                    
                    residues.append({"Chain": chain_id, "Residue": res_id, "Secondary Structure": ss})
                
                except ValueError:
                    continue  # Skip invalid lines

    df = pd.DataFrame(residues)

    # Sort by Chain and Residue to ensure consistency
    df = df.sort_values(["Chain", "Residue"]).reset_index(drop=True)
    
    if debug:
        print(f"Parsed {len(df)} residues from {output_file}")
    return df


def compare_dssp(gt_dssp, pred_dssp, debug=False):
    """
    Compares DSSP outputs for ground truth and predicted structures.

    Args:
    - gt_dssp (pd.DataFrame or str): Parsed DSSP DataFrame or path to DSSP file.
    - pred_dssp (pd.DataFrame or str): Parsed DSSP DataFrame or path to DSSP file.

    Returns:
    - dict: Comparison results (accuracy, Q3 score, etc.)
    """
    # If input is a file path, parse it
    if isinstance(gt_dssp, str):
        gt_dssp = parse_dssp(gt_dssp)

    if isinstance(pred_dssp, str):
        pred_dssp = parse_dssp(pred_dssp)

    # Debugging: Print a few residue numbers
    if debug:
        print(f"GT Residues: {gt_dssp['Residue'].tolist()[:10]}")
        print(f"Pred Residues: {pred_dssp['Residue'].tolist()[:10]}")

    # correctly align the residue numbers
    gt_first_residue = gt_dssp["Residue"].min()
    pred_first_residue = pred_dssp["Residue"].min()
    offset = gt_first_residue - pred_first_residue
    pred_dssp["Residue"] += offset  # Apply the shift

    if debug:
        # Debugging: Check alignment
        print(f"Adjusted Pred Residues: {pred_dssp['Residue'].tolist()[:10]}")

    # Merge DataFrames on Residue Number
    merged = pd.merge(gt_dssp, pred_dssp, on=["Residue"], suffixes=("_GT", "_Pred"))

    if len(merged) == 0:
        print(f"No matching residues found after adjusting numbering!")

    # Compute per-residue accuracy
    correct_predictions = (merged["Secondary Structure_GT"] == merged["Secondary Structure_Pred"]).sum()
    total_residues = len(merged)

    # Q3 score calculation
    def map_q3(ss):
        """Maps DSSP structures to Q3 categories."""
        if ss in ["H", "G", "I"]:  # Helices
            return "H"
        elif ss in ["E", "B"]:  # Sheets
            return "E"
        else:  # Coil/Loop
            return "C"

    # Apply Q3 mapping
    merged["Q3_GT"] = merged["Secondary Structure_GT"].apply(map_q3)
    merged["Q3_Pred"] = merged["Secondary Structure_Pred"].apply(map_q3)

    q3_correct_predictions = (merged["Q3_GT"] == merged["Q3_Pred"]).sum()

    return {
        "Total Residues": total_residues,
        "Correct Predictions": correct_predictions,
        "Per-Residue Accuracy": correct_predictions / total_residues if total_residues else 0,
        "Q3 Score": q3_correct_predictions / total_residues if total_residues else 0,
    }


def evaluate_dssp_predictions(pred_dssp_dir, gt_dssp_dir):
    """
    Evaluates secondary structure predictions by comparing DSSP output files.

    Args:
    - pred_dssp_dir (str): Directory containing predicted DSSP files.
    - gt_dssp_dir (str): Directory containing ground truth DSSP files.

    Returns:
    - pd.DataFrame: DataFrame containing evaluation metrics for each structure.
    """
    results = []

    # Get list of predicted DSSP files
    pred_files = {f.replace(".dssp", ""): os.path.join(pred_dssp_dir, f) for f in os.listdir(pred_dssp_dir) if f.endswith(".dssp")}
    gt_files = {f.replace(".dssp", ""): os.path.join(gt_dssp_dir, f) for f in os.listdir(gt_dssp_dir) if f.endswith(".dssp")}

    # Process each file present in both directories
    common_ids = set(pred_files.keys()) & set(gt_files.keys())

    for struct_id in common_ids:
        pred_dssp = parse_dssp(pred_files[struct_id])
        gt_dssp = parse_dssp(gt_files[struct_id])

        # Compute comparison metrics
        metrics = compare_dssp(gt_dssp, pred_dssp)
        metrics["Structure"] = struct_id

        results.append(metrics)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    return df_results


if __name__ == "__main__":

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
   
    # # Example usage
    run_dssp_with_pdb_lookup(
        input_dir,
        output_dir,
        dssp_path="~/.conda/envs/dssp_env/bin/mkdssp"
    )

