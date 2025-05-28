import plm_subnetworks.dataset.data_io as data_io

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