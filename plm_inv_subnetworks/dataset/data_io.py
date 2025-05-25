import json
import os
from pathlib import Path
import pandas as pd
from plm_inv_subnetworks.dataset.constants import DSSP_3

def read_from_txt(file_path):
    with open(file_path, 'r') as file:
        # Read the lines and remove any surrounding whitespace
        data_list = [line.strip() for line in file]
    return data_list


def write_dict_to_json(dict, filename):
    with open(filename, 'w') as f:
        json.dump(dict, f, indent=4)

    return dict

def read_json_to_dict(filename):
    with open(filename, 'r') as f:
        dict = json.load(f)
    return dict

def get_args_split(dir_path):
    config_path = os.path.join(dir_path, "config.json")
    split_path = os.path.join(dir_path, "train_val_split.json")
    config = read_json_to_dict(config_path)["config"]
    split = read_json_to_dict(split_path)
    return config, split

def hydrate_df_with_dssp(df, db, id_field="cath_id"):
    def get_cath_data(row):
        cath_entry = db.query(row[id_field])
        dssp_8 = cath_entry.dssp
        dssp_3 = ''.join(DSSP_3.get(char, char) for char in dssp_8)
        helix_pct = (dssp_3.count('H') / len(dssp_3) * 100) if dssp_3 else 0
        strand_pct = (dssp_3.count('E') / len(dssp_3) * 100) if dssp_3 else 0
        coil_pct = (dssp_3.count('L') / len(dssp_3) * 100) if dssp_3 else 0
                
        return pd.Series({
            'dssp': cath_entry.dssp,
            'helix_pct': helix_pct,
            'strand_pct': strand_pct,
            'coil_pct': coil_pct,
        })

    df = df.merge(df.apply(get_cath_data, axis=1), left_index=True, right_index=True)
    return df

def hydrate_df_with_cath_terms(df, db, id_field="cath_id"):
    def get_cath_data(row):
        cath_entry = db.query(row[id_field])

        return pd.Series({
            'chain_id': cath_entry.chain_num,
            'length': len(cath_entry.sequence),
            'cath_class': cath_entry.class_num,
            'cath_architecture': cath_entry.architecture,
            'cath_topology': cath_entry.topology,
            'cath_homologous_superfamily': cath_entry.homologous_superfamily,
            'seq_len': len(cath_entry.sequence),
            'cath_domain_num': cath_entry.domain_num,
            'cath_class_code': f"{cath_entry.class_num}",
            'cath_architecture_code': f"{cath_entry.class_num}.{cath_entry.architecture}",
            'cath_topology_code': f"{cath_entry.class_num}.{cath_entry.architecture}.{cath_entry.topology}",
            'cath_homologous_superfamily_code': f"{cath_entry.class_num}.{cath_entry.architecture}.{cath_entry.topology}.{cath_entry.homologous_superfamily}",
            'cath_domain_code': f"{cath_entry.class_num}.{cath_entry.architecture}.{cath_entry.topology}.{cath_entry.homologous_superfamily}.{cath_entry.domain_num}",
            'cath_code': f"{cath_entry.class_num}.{cath_entry.architecture}.{cath_entry.topology}.{cath_entry.homologous_superfamily}.{cath_entry.domain_num}"
        })

    df = df.merge(df.apply(get_cath_data, axis=1), left_index=True, right_index=True)
    return df

def dataframe_to_fasta(df, output_file):
    with open(output_file, "w") as fasta_file:
        for _, row in df.iterrows():
            header = f">{row['cath_id']}"
            sequence = row['reconstructed_seq']
            fasta_file.write(f"{header}\n{sequence}\n")

    print(f"FASTA file '{output_file}' has been created successfully.")