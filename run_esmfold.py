import pandas as pd
import os

import esm.esmfold.v1
import esm.esmfold.v1.pretrained
import torch
import esm


from plmprobe.dataset.cath_dataset import CATHDatabase, CATHEntry, CATH_ENTRY_FILEPATH, CATH_S40_SEQ_FILEPATH
from plmprobe.dataset.data_paths import CATH_S20_IDS, CATH_S40_IDS, CATH_S20_ATOM_IDS, RUN_DIR_PREFIX
from plmprobe.dataset import data_paths

import esm.esmfold

from plmprobe.esm_modules.esmfold import _load_model

from pathlib import Path

def read_fasta(fasta_path):
    headers = []
    sequences = []
    seq_chunks = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_chunks:
                    sequences.append("".join(seq_chunks))
                    seq_chunks = []
                headers.append(line[1:])  # remove ">"
            else:
                seq_chunks.append(line)
        if seq_chunks:
            sequences.append("".join(seq_chunks))  # last entry

    return headers, sequences

def fold_and_write_pdb(model, id, seq, masks=None, folding_dir=None):

    pred_result_path = f"{folding_dir}/pdbs/{id}.pdb"
    os.makedirs(f"{folding_dir}/pdbs", exist_ok=True)

    # Run inference using modified forward method
    with torch.no_grad():
        output = model.infer_pdb(seq, masks)

    with open(pred_result_path, "w") as f:
        f.write(output)

    import biotite.structure.io as bsio
    struct = bsio.load_structure(pred_result_path, extra_fields=["b_factor"])
    pred_plddt = struct.b_factor.mean()  # this will be the pLDDT


    return pred_plddt


# Example usage
FASTA = Path("/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath_s20_with_pdbs.fasta")
FOLDING_DIR = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data_computed_all_gt_metrics/esmfold_3B"
cath_ids, sequences = read_fasta(FASTA)

cath_version = "20"
_, seq_filepath = data_paths.get_cath_paths(cath_version)
db = CATHDatabase()
db.load_clf(CATH_ENTRY_FILEPATH)
db.load_sequences(seq_filepath)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = _load_model("esmfold_structure_module_only_650M", None)
# model = model.eval().cuda()

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

results = []

for id, seq in zip(cath_ids, sequences):
    seq =  db.query(id).sequence
    class_num = db.query(id).class_num

    pred_plddt = fold_and_write_pdb(model, id, seq, folding_dir=FOLDING_DIR)

    metrics = {
        "id": id,
        "class_num": class_num,
        "pred_plddt": pred_plddt,
    }

    results.append(metrics)

df = pd.DataFrame(results)
df.to_csv(f"{FOLDING_DIR}/plddt.csv")
print(df.head())