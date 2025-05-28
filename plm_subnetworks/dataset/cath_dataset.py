'''Load CATH data and access attributes

From https://github.com/wukevin/proteinclip/blob/main/data/cath/README-cath-list-file-format.txt
'''

import re
import sys
from Bio import SeqIO

from plm_subnetworks.dataset import data_paths
from plm_subnetworks.dataset.data_paths import CATH_ENTRY_FILEPATH, CATH_S20_DSSP_FASTA, CATH_S20_ATOM_FILEPATH


def get_cath_db(cath_version="20"):
    _, seq_filepath = data_paths.get_cath_paths(cath_version)
    db = CATHDatabase()
    db.load_clf(CATH_ENTRY_FILEPATH)
    db.load_sequences(seq_filepath)
    db.load_dssp(CATH_S20_DSSP_FASTA)

    return db


class CATHEntry:
    def __init__(self, cath_id, pdb, chain_num, domain_num, class_num, architecture, topology, homologous_superfamily, 
                 s35, s60, s95, s100, s100_count, domain_length, resolution, sequence=None, boundary=None, precut_domain=None):
        self.cath_id = cath_id
        self.boundary = boundary
        self.pdb = pdb
        self.chain_num = chain_num
        self.domain_num = domain_num
        self.class_num = class_num
        self.architecture = architecture
        self.topology = topology
        self.homologous_superfamily = homologous_superfamily
        self.s35 = s35
        self.s60 = s60
        self.s95 = s95
        self.s100 = s100
        self.s100_count = s100_count
        self.domain_length = domain_length
        self.resolution = resolution
        self.sequence = sequence
        self.precut_domain = precut_domain



class CATHDatabase:
    def __init__(self):
        self.entries = {}

    def load_clf(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#') or not line.strip():
                    continue
                parts = re.split(r'\s+', line.strip())
                cath_id = parts[0]
                pdb = cath_id[:4]
                chain_num = cath_id[4]
                domain_num = cath_id[5:]
                self.entries[cath_id] = CATHEntry(
                    cath_id,
                    pdb,
                    chain_num,
                    int(domain_num),
                    int(parts[1]),      # Class
                    int(parts[2]),      # Architecture
                    int(parts[3]),      # Topology
                    int(parts[4]),      # Homologous superfamily
                    int(parts[5]),      # S35
                    int(parts[6]),      # S60
                    int(parts[7]),      # S95
                    int(parts[8]),      # S100
                    int(parts[9]),      # S100 count
                    int(parts[10]),     # Domain length
                    float(parts[11])    # Resolution
                )

    def load_sequences(self, sequence_file_path):
        for record in SeqIO.parse(sequence_file_path, "fasta"):
            # Extract the domain ID from the FASTA header
            header = record.id
            cath_id = header.split('|')[-1].split('/')[0]
            boundary = header.split('|')[-1].split('/')[1]
            if cath_id in self.entries:
                self.entries[cath_id].sequence = str(record.seq)
                self.entries[cath_id].boundary = boundary

    def load_dssp(self, sequence_file_path):
        for record in SeqIO.parse(sequence_file_path, "fasta"):
            # Extract the domain ID from the FASTA header
            cath_id = record.id
            if cath_id in self.entries:
                self.entries[cath_id].dssp = str(record.seq)


    def query(self, cath_id):
        return self.entries.get(cath_id, None)


# def main():
#     # Usage
#     db = CATHDatabase()
#     db.load_clf(CATH_ENTRY_FILEPATH)
#     db.load_sequences(CATH_S20_ATOM_FILEPATH)

#     # Query an entry
#     query_id = "1a1rA02"  # Replace with an actual entry ID
#     entry = db.query(query_id)
    
#     if entry:
#         # Print all attributes of the CATHEntry object
#         for attr, value in vars(entry).items():
#             print(f"{attr}: {value}")
#     else:
#         print("Entry not found")

# if __name__ == "__main__":
#     main()

def main(query_id: str):
    # Usage
    db = CATHDatabase()
    db.load_clf(CATH_ENTRY_FILEPATH)
    db.load_sequences(CATH_S20_ATOM_FILEPATH)

    # Query an entry
    entry = db.query(query_id)
    
    if entry:
        # Print all attributes of the CATHEntry object
        for attr, value in vars(entry).items():
            print(f"{attr}: {value}")
    else:
        print("Entry not found")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cath_dataset.py <query_id>")
        sys.exit(1)

    query_id = sys.argv[1]
    main(query_id)