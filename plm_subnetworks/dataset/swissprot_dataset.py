

import re


class SwissProtEntry:
    def __init__(self, accession, entry_name, protein_name, organism_name, organism_identifier, gene_name, existence, version):
        self.accession = accession
        self.entry_name = entry_name
        self.protein_name = protein_name
        self.organism_name = organism_name
        self.organism_identifier = organism_identifier
        self.gene_name = gene_name
        self.existence = existence
        self.version = version

    def __str__(self):
        return (f"Accession: {self.accession}, Entry Name: {self.entry_name}, Protein Name: {self.protein_name}, "
                f"Organism: {self.organism_name}, Organism ID: {self.organism_identifier}, Gene: {self.gene_name}, "
                f"Existence: {self.existence}, Version: {self.version}")

class SwissProtDatabase:
    def __init__(self):
        self.entries = {}

    def parse_header(self, header):
        """
        Parses a Swiss-Prot FASTA header to extract relevant information.
        """
        # print(f"Parsing header: {header}")  # Debugging print


        FASTA_HEADER_REGEX = re.compile(
                r"^(\w+)\|([\w\d]+)\|([\w\d_]+)\s(.+?)\sOS=(.+?)\sOX=(\d+)(?:\sGN=([\S]+))?\sPE=(\d+)\sSV=(\d+)"
    )

        match = FASTA_HEADER_REGEX.match(header)

        if match:
            groups = match.groups()
            return SwissProtEntry(
                accession=groups[1],  # Accession number
                entry_name=groups[2],  # Entry name
                protein_name=groups[3],  # Protein name
                organism_name=groups[4],  # Organism name
                organism_identifier=groups[5],  # Organism ID
                gene_name=groups[6] if groups[6] else None,  # Gene Name (optional)
                existence=groups[7],  # Protein Existence
                version=groups[8]  # Sequence Version
            )
        else:
            return None  # If the header doesn't match, return None

    def load_clf(self, sequence_file_path):
        """
        Reads a FASTA file and populates the database with Swiss-Prot entries.
        """
        from Bio import SeqIO

        for record in SeqIO.parse(sequence_file_path, "fasta"):
            header = record.description  # Get the full FASTA header
            entry = self.parse_header(header)

            if entry:
                self.entries[entry.accession] = entry  # Store by accession number
            
        print(f"Total entries loaded: {len(self.entries)}")  # Debugging print

    def display_entries(self, n=2):
        """
        Displays the first 'n' stored Swiss-Prot entries.
        """
        n_accessions = 0
        for accession, entry in self.entries.items():
            print(entry)
            n_accessions += 1
            if n_accessions >= n:
                break


    def query(self, accession):
        return self.entries[accession] if accession in self.entries else None


# Usage Example
if __name__ == "__main__":
   
    db = SwissProtDatabase()
    db.load_clf("swissprot.fasta")  # Replace with your file path
    db.display_entries()
    print(" ")

    print("Test query:")
    test = db.query("Q6GZX2")
    print(test)


