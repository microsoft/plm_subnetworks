import plm_inv_subnetworks.dataset.data_io as data_io

# ROOT_DIR = "/users/rvinod/data/rvinod/repos/plm_inv_subnetworks"

# CATH_IDS_S20_PDB_CONTACTS = "../data/cath_ids_with_pdb_contacts.txt" # CATH IDs used for training
# CATH_IDS_S20_PDB_CONTACTS_WITH_ANNOTATIONS = "../data/cath_ids_with_annotations.csv"

# ESM_PPL_METRICS = "../results/esm2_650M_perplexity.csv"
# ESM_TMALIGN_METRICS = "../results/esmfold_650M_tm_align.csv"
# ESMFOLD_650M_PDBS = "../results/esmfold_650M_pdbs"

# PDB_DIR = "../data/dompdb_chain" # prextracted the chains from the PDBs, full PDBs are in dompdb/

# # This needs to be created if you want to run DSSP on the chains; precomputed annotations are provided in CATH_IDS_S20_PDB_CONTACTS_WITH_ANNOTATIONS
# PDB_DSSP_DIR = "../data/cath_pdb/dompdb_dssp_chain" 

# CATH_ENTRY_FILEPATH = "../data/cath-domain-list-v4_3_0.txt"
# CATH_S20_DSSP_FASTA = "../data/dompdb_chain_atom_s20_dssp.fasta"
# CATH_S20_ATOM_IDS = "../data/cath-dataset-nonredundant-S20-atom.txt"
# CATH_S20_ATOM_FILEPATH = "../data/cath-dataset-nonredundant-S20.atom.fa"

# RUN_DIR_PREFIX = "../runs" 

ROOT_DIR = "/users/rvinod/data/rvinod/repos/plm_inv_subnetworks"

CATH_IDS_S20_PDB_CONTACTS = f"{ROOT_DIR}/data/cath_ids_with_pdb_contacts.txt"
CATH_IDS_S20_PDB_CONTACTS_WITH_ANNOTATIONS = f"{ROOT_DIR}/data/cath_ids_with_annotations.csv"

ESM_PPL_METRICS = f"{ROOT_DIR}/results/esm2_650M_perplexity.csv"
ESM_TMALIGN_METRICS = f"{ROOT_DIR}/results/esmfold_650M_tm_align.csv"
ESMFOLD_650M_PDBS = f"{ROOT_DIR}/results/esmfold_650M_pdbs"

PDB_DIR = f"{ROOT_DIR}/data/dompdb_chain"

# This needs to be created if you want to run DSSP on the chains; precomputed annotations are provided in CATH_IDS_S20_PDB_CONTACTS_WITH_ANNOTATIONS
PDB_DSSP_DIR = f"{ROOT_DIR}/data/cath_pdb/dompdb_dssp_chain"

CATH_ENTRY_FILEPATH = f"{ROOT_DIR}/data/cath-domain-list-v4_3_0.txt"
CATH_S20_DSSP_FASTA = f"{ROOT_DIR}/data/dompdb_chain_atom_s20_dssp.fasta"
CATH_S20_ATOM_IDS = f"{ROOT_DIR}/data/cath-dataset-nonredundant-S20-atom.txt"
CATH_S20_ATOM_FILEPATH = f"{ROOT_DIR}/data/cath-dataset-nonredundant-S20.atom.fa"

RUN_DIR_PREFIX = f"{ROOT_DIR}/runs"


# CATH_S20_SEQ_FILEPATH = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath-dataset-nonredundant-S20-v4_3_0.fa"
# CATH_S40_SEQ_FILEPATH = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath-dataset-nonredundant-S40-v4_3_0.fa"
# CATH_S20_ATOM_FILEPATH = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath-dataset-nonredundant-S20.atom.fa"

# CATH_S20_IDS = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath-dataset-ids-nonredundant-S20-v4_3_0.txt"
# CATH_S40_IDS = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath-dataset-ids-nonredundant-S40-v4_3_0.txt"

# CATH_S20_ATOM_IDS = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath-dataset-nonredundant-S20-atom.txt"

# CATH_S20_ATOM_IDS_WITH_PBD_CONTACTS = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data_computed/esm2_650M_contact_metrics_cath_ids_20250125_173917.txt"

# PDB_DIR = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath_pdb/dompdb"
# PDB_DSSP_DIR = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath_pdb/dompdb_dssp_chain"
# PDB_CHAIN_DIR = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath_pdb/dompdb_chain"

# # Containts DSSP annotations for the chains of the PDB that correspond to the cath domains
# CATH_S20_DSSP_FASTA = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/dompdb_chain_atom_s20_dssp.fasta"


# CATH_S20_ATOM_IDS_WITH_PBD_CONTACTS_FASTA = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/cath_s20_with_pdbs.fasta"

# UNIPROT_FASTA = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data/swissprot/uniprot_sprot.fasta"

# ESM_650M_10_PASSES = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data_computed_all_gt_metrics/esm_cath_s20_metric_10_passes.csv"


# # GT_ESM_METRICS = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data_computed_all_gt_metrics/esm_layer_inference_no_diag_bce/esm2_650M_33_20250216_225510.csv"
# GT_ESM_METRICS = "/users/rvinod/data/rvinod/repos/probing-subnetworks/data_computed/esm_cath_s20_metrics.csv"

# RUN_DIR_PREFIX = "/users/rvinod/data/rvinod/repos/probing-subnetworks/runs_cath_class"

def get_cath_paths(cath_version):
    if cath_version == "20":
        cath_ids = data_io.read_from_txt(CATH_S20_ATOM_IDS)
        seq_filepath = CATH_S20_ATOM_FILEPATH
    else:
        pass
        # implement others, e.g below. Make sure to change 
        # the "cath_verion" when calling this function
        # cath_ids = data_io.read_from_txt(CATH_S40_IDS)
        # seq_filepath = CATH_S40_SEQ_FILEPATH

    return cath_ids, seq_filepath

def process_headers(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.startswith('>'):
                # print(line.strip().split('|')[-1].split('/')[0])
                f_out.write(line.strip().split('|')[-1].split('/')[0] + '\n')

# process_headers(CATH_S20_ATOM_FILEPATH, "../data/cath-dataset-nonredundant-S20-atom.txt")