from Bio import PDB
import os
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from plm_subnetworks.dataset.constants import DSSP_8, HELIX_SET, STRAND_SET, COIL_SET
from plm_subnetworks.dataset.data_paths import PDB_CHAIN_DIR

def compute_8_way_dssp_metrics(y_pred, y_true):

    residues = {'G', 'H', 'T', 'E', 'S', 'L', 'B', 'I'}  # Updated residue set

    # Ensure input sequences have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input sequences must have the same length.")

    # Convert sequences to lists for computation
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    # Compute metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_list, y_pred_list, labels=list(residues), zero_division=0
    )
    
    # Compute overall accuracy
    accuracy = accuracy_score(y_true_list, y_pred_list)

    # Adjust for missing residues in predictions
    adjusted_metrics = []
    for res in residues:
        if res in y_true_list:
            idx = list(residues).index(res)
            adjusted_metrics.append((res, accuracy, precision[idx], recall[idx], f1[idx]))
        else:
            adjusted_metrics.append((res, None, None, None, None))

    # Create DataFrame
    metrics_df = pd.DataFrame(adjusted_metrics, columns=["Residue", "Accuracy", "Precision", "Recall", "F1-Score"])

    return metrics_df


def compute_3_way_dssp_metrics(pred_dssp, true_dssp):
    """
    Computes per-class accuracy, precision, recall, and F1-score for helices, strands, and coils.

    Args:
        true_dssp (str): Ground truth DSSP sequence.
        pred_dssp (str): Predicted DSSP sequence.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score for helices, strands, and coils.
    """
    if len(true_dssp) != len(pred_dssp):  # DSSP sequences must have the same length
        return None

    # Initialize counters
    metrics = {
        "helix": {"TP": 0, "FP": 0, "FN": 0, "total": 0, "correct": 0},
        "strand": {"TP": 0, "FP": 0, "FN": 0, "total": 0, "correct": 0},
        "coil": {"TP": 0, "FP": 0, "FN": 0, "total": 0, "correct": 0},
    }

    # Iterate through residues
    for true_label, pred_label in zip(true_dssp, pred_dssp):
        if true_label in HELIX_SET:
            metrics["helix"]["total"] += 1
            if pred_label in HELIX_SET:
                metrics["helix"]["TP"] += 1  # True Positive
                metrics["helix"]["correct"] += 1
            else:
                metrics["helix"]["FN"] += 1  # False Negative
        elif true_label in STRAND_SET:
            metrics["strand"]["total"] += 1
            if pred_label in STRAND_SET:
                metrics["strand"]["TP"] += 1
                metrics["strand"]["correct"] += 1
            else:
                metrics["strand"]["FN"] += 1
        elif true_label in COIL_SET:
            metrics["coil"]["total"] += 1
            if pred_label in COIL_SET:
                metrics["coil"]["TP"] += 1
                metrics["coil"]["correct"] += 1
            else:
                metrics["coil"]["FN"] += 1

        # Count False Positives (Predicted but Incorrect)
        if pred_label in HELIX_SET and true_label not in HELIX_SET:
            metrics["helix"]["FP"] += 1
        elif pred_label in STRAND_SET and true_label not in STRAND_SET:
            metrics["strand"]["FP"] += 1
        elif pred_label in COIL_SET and true_label not in COIL_SET:
            metrics["coil"]["FP"] += 1

    # Compute accuracy, precision, recall, and F1-score
    results = {}
    for label in ["helix", "strand", "coil"]:
        TP = metrics[label]["TP"]
        FP = metrics[label]["FP"]
        FN = metrics[label]["FN"]
        total = metrics[label]["total"]
        correct = metrics[label]["correct"]

        # If this structure type is missing in true_dssp, return None for all metrics
        if total == 0:
            results[f"{label}_accuracy"] = None
            results[f"{label}_precision"] = None
            results[f"{label}_recall"] = None
            results[f"{label}_f1"] = None
            continue

        accuracy = correct / total if total > 0 else None
        precision = TP / (TP + FP) if (TP + FP) > 0 else None
        recall = TP / (TP + FN) if (TP + FN) > 0 else None
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision and recall and (precision + recall) > 0) else None

        results[f"{label}_accuracy"] = accuracy
        results[f"{label}_precision"] = precision
        results[f"{label}_recall"] = recall
        results[f"{label}_f1"] = f1_score
    
    return results


def get_ssp(pdb_path, dssp_path='mkdssp'):
    """
    Extracts secondary structure from a PDB file and converts it into UniRep-style labels.

    Args:
        pdb_path (str): Path to the PDB file.
        dssp_path (str): Path to the DSSP executable (default: 'dssp' or 'mkdssp').

    Returns:
        str: String of secondary structure labels using UniRep's exact set {B, E, G, H, I, L, S, T}.
    """
    # Parse PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    # Extract first model (DSSP requires a single model)
    model = structure[0]

    # Run DSSP
    dssp = PDB.DSSP(model, pdb_path, dssp=dssp_path)


    # Convert DSSP output to UniRep labels
    ss_seq = "".join(DSSP_8[dssp[key][2]] for key in dssp.keys())

    ss_seq = "".join(dssp[key][2] for key in dssp.keys())


    return ss_seq



def write_fasta(output_file, sequences):
    """
    Writes DSSP sequences to a FASTA file.

    Args:
        output_file (str): Path to the output FASTA file.
        sequences (dict): Dictionary {PDB_ID: DSSP_sequence}.
    """
    with open(output_file, "w") as f:
        for pdb_id, dssp_seq in sequences.items():
            f.write(f">{pdb_id}\n{dssp_seq}\n")


if __name__ == "__main__":

    FASTA_OUTPUT_FILE = "dssp_annotations.fasta" # TODO: Change to your desired output file path

    missing = 0
    dssp_sequences = {}

    for pdb_file in os.listdir(PDB_CHAIN_DIR):
        if not pdb_file.endswith(".pdb"):
            continue
        
        pdb_id = pdb_file.split(".")[0]  # Extract PDB ID
        pdb_path = os.path.join(PDB_CHAIN_DIR, pdb_file)

        try:
            dssp_seq = get_ssp(pdb_path)
            dssp_sequences[pdb_id] = dssp_seq

        except Exception as e:
            print(f"Skipping {pdb_id} due to error: {e}")
            missing += 1

    # Write to FASTA
    write_fasta(FASTA_OUTPUT_FILE, dssp_sequences)

    print("Done:", len(dssp_sequences))
    print("Missing:", missing)
    print(f"DSSP sequences written to {FASTA_OUTPUT_FILE}")