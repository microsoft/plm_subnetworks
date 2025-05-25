import random
import torch

from torch.utils.data import Dataset, DataLoader

import esm

import plm_inv_subnetworks.dataset.data_paths as data_paths
from plm_inv_subnetworks.dataset import constants
from plm_inv_subnetworks.dataset.cath_dataset import CATHDatabase
from plm_inv_subnetworks.dataset.constants import HELIX_SET, STRAND_SET, COIL_SET


# from plm_inv_subnetworks.dataset.data_paths import CATH_S20_IDS, CATH_S40_IDS, CATH_S20_SEQ_FILEPATH, CATH_S40_SEQ_FILEPATH, CATH_ENTRY_FILEPATH


def get_corr_mask(n_corr, seq_lens, max_len):
    masks = []
    for i in range(int(n_corr.size()[0])):
        n = n_corr[i].item()
        length = seq_lens[i].item()
        mask = torch.zeros(max_len + 2) #, dtype=torch.int8)  # Use int8 instead of default int64
        mask.index_fill_(0, torch.tensor(random.sample(range(1, length + 1), n)), 1)
        masks.append(mask)
    return torch.stack(masks)

class CATHSeqDatasetESM(Dataset):
    def __init__(self, cath_ids, cath_database, alphabet, mask_pct=0.2, min_n_res=8, max_n_res=512, masking=True, use_dssp=False):
        self.cath_database = cath_database
        self.cath_ids = cath_ids
        self.masking = masking
        self.min_n_res = min_n_res
        self.max_n_res = max_n_res
        self.alphabet = alphabet
        self.tokenizer = alphabet.get_batch_converter()
        self.mask_pct = mask_pct
        self.use_dssp = use_dssp
        self.standard_toks = torch.tensor([alphabet.encode(tok)[0] for tok in alphabet.standard_toks]) 
        self.use_dssp = use_dssp


    def __len__(self):
        return len(self.cath_ids)

    def __getitem__(self, idx):

        try:
            cath_entry = self.cath_database.query(self.cath_ids[idx])
            sequence = cath_entry.sequence
            dssp = cath_entry.dssp
            # gt_contacts = torch.tensor(get_gt_contacts(cath_entry.cath_id, cath_entry.chain_num)) # ,dtype=torch.float16)
            cath_class = cath_entry.class_num
            cath_architecture = cath_entry.architecture
            cath_topology = cath_entry.topology
            cath_homologous_superfamily = cath_entry.homologous_superfamily
            cath_domain_num = int(cath_entry.domain_num)

        except Exception as e:
            print(f"Error fetching {self.cath_ids[idx]}: {e}")
            return None
    
        if (
            not sequence 
            or len(sequence) < self.min_n_res 
            or len(sequence) > self.max_n_res 
            ):
                return None

        return {
            "cath_id": cath_entry.cath_id,
            "sequence": sequence,
            "cath_class": cath_class,
            "cath_architecture": cath_architecture,
            "cath_topology": cath_topology,
            "cath_homologous_superfamily": cath_homologous_superfamily,
            "cath_domain_num": cath_domain_num,
            "dssp": dssp,
            "cath_code": f"{cath_class}.{cath_architecture}.{cath_topology}.{cath_homologous_superfamily}.{cath_domain_num}",
        }

    def collate_fn(self, batch):
        alphabet = self.alphabet

        batch = [item for item in batch if item is not None]

        if len(batch) == 0:
        # Return empty batch with same structure
            return {
                "cath_ids": [],
                # "cath_entries": [],
                "sequences": [],
                "src": torch.empty(0),
                "tgt": torch.empty(0),
                "corr_mask": torch.empty(0),
                "seq_lens": torch.empty(0),
                "batch_lens": torch.empty(0)
            }

        cath_ids = [item["cath_id"] for item in batch]
        sequences = [item["sequence"] for item in batch]
        cath_classes = [item["cath_class"] for item in batch]
        cath_architectures = [item["cath_architecture"] for item in batch]
        cath_topologies = [item["cath_topology"] for item in batch]
        cath_homologous_superfamilies = [item["cath_homologous_superfamily"] for item in batch]
        cath_domain_nums = [item["cath_domain_num"] for item in batch]
        dssp = [item["dssp"] for item in batch]
        cath_codes = [item["cath_code"] for item in batch]
        
        data = list(zip(cath_ids, sequences))  # [(cath_id, sequence)]
        cath_ids, sequences, tgt = alphabet.get_batch_converter()(data)
        batch_lens = (tgt != alphabet.padding_idx).sum(1)

        standard_toks = torch.tensor([alphabet.encode(tok)[0] for tok in alphabet.standard_toks])

        seq_lens = torch.sum((tgt[:, :, None] == standard_toks).any(dim=2), dim=1).unsqueeze(0)[0]
        
        # Randomly choose 15% of residues to mask; 80% of those will be replaced with [MASK], 10% with random tokens, and 10% will remain unchanged 
        n_corr = (seq_lens * 0.15).to(torch.long)
        max_len = seq_lens.max().item()

        # Use boolean type for binary masks
        positions = torch.arange(batch_lens.max())
        seq_mask = torch.zeros((len(batch_lens), positions.size(0))) #, dtype=torch.bool)
        seq_mask.masked_fill_((positions[None, :] >= 1) & (positions[None, :] < (batch_lens - 1)[:, None]), True)

    
        assert (seq_lens == torch.sum(seq_mask, dim=1)).all()

        helix_mask = torch.zeros_like(seq_mask, dtype=torch.bool)
        strand_mask = torch.zeros_like(seq_mask, dtype=torch.bool)
        coil_mask = torch.zeros_like(seq_mask, dtype=torch.bool)

        # Loop through batch and align DSSP annotations
        for i, dssp_seq in enumerate(dssp):
            dssp_idx = 0  # Track DSSP character index
            for j in range(seq_mask.shape[1]):  # Loop over max sequence length
                if seq_mask[i, j]:  # If this position is a valid residue
                    if dssp_idx < len(dssp_seq):  # Ensure we don't go out of bounds
                        dssp_char = dssp_seq[dssp_idx]
                        
                        if dssp_char in HELIX_SET:
                            helix_mask[i, j] = True
                        elif dssp_char in STRAND_SET:
                            strand_mask[i, j] = True
                        elif dssp_char in COIL_SET:
                            coil_mask[i, j] = True
                        
                        dssp_idx += 1  # Move to the next DSSP character

        # Ensure padding positions remain False
        helix_mask[seq_mask == 0] = False
        strand_mask[seq_mask == 0] = False
        coil_mask[seq_mask == 0] = False


        src = tgt.clone()

        # SEQ SUPPRESSION
        masks = get_corr_mask(n_corr, seq_lens, max_len)
        # src[masks == 1] = alphabet.mask_idx

        # get masked positions
        masked_pos = (masks == 1).nonzero(as_tuple=False)
        # Shuffle the indices
        perm = torch.randperm(masked_pos.size(0), device=src.device)
        n_total = masked_pos.size(0)
        n_mask = int(0.8 * n_total)
        n_rand = int(0.1 * n_total)

        # 80% → replace with [MASK]
        idx_mask = masked_pos[perm[:n_mask]]
        src[idx_mask[:, 0], idx_mask[:, 1]] = alphabet.mask_idx

        # 10% → replace with random tokens (excluding mask token)
        idx_rand = masked_pos[perm[n_mask:n_mask + n_rand]]
        vocab = list(range(len(alphabet.all_toks)))
        vocab.remove(alphabet.mask_idx)  # don't sample the mask token
        rand_indices = torch.randint(low=0, high=len(vocab), size=(idx_rand.size(0),), device=src.device)
        rand_tokens = torch.tensor([vocab[i] for i in rand_indices.tolist()], device=src.device)
        rand_tokens = rand_tokens.to(dtype=src.dtype)
        src[idx_rand[:, 0], idx_rand[:, 1]] = rand_tokens

       
        return {
            "cath_ids": cath_ids,
            "cath_classes": cath_classes,
            "cath_architectures": cath_architectures,
            "cath_topologies": cath_topologies,
            "cath_homologous_superfamilies": cath_homologous_superfamilies,
            "cath_domain_nums": cath_domain_nums,
            "sequences": sequences,
            "src": src,
            "tgt": tgt,
            "corr_mask": masks,
            "seq_lens": seq_lens,
            "batch_lens": batch_lens,
            "seq_mask": seq_mask,
            "helix_mask": helix_mask,
            "strand_mask": strand_mask,
            "coil_mask": coil_mask,
            "dssp": dssp,
            "cath_codes" : cath_codes,

        }



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    model = model.to(device)
    

    cath_version = "20" # {20, 40}
    cath_ids, seq_filepath = data_paths.get_cath_paths(cath_version)
    cath_ids = cath_ids[:100]
    # Usage
    db = CATHDatabase()
    db.load_clf(CATH_ENTRY_FILEPATH)
    db.load_sequences(seq_filepath)
    cath_dataset = CATHSeqDatasetESM(model, cath_ids, db, alphabet)

    min_n_res = 8
    max_n_res = 512
    batch_size = 4


    # Create DataLoader with custom collate_fn
    cath_dataloader = DataLoader(
        cath_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=cath_dataset.collate_fn  # Pass the function directly
    )

    for i, batch in enumerate(cath_dataloader):

        batch["src"] = batch["src"].to(device)
        batch["tgt"] = batch["tgt"].to(device)
        batch["corr_mask"] = batch["corr_mask"].to(device)
     
        with torch.no_grad():
            results = model(batch["src"], repr_layers=[], return_contacts=False)
        token_representations = results["representations"]
        logits = results["logits"]
        mcel = MaskedCrossEntropyLoss(weight=None, reduction='mean')
        loss = mcel(logits, batch["tgt"], batch["corr_mask"])
        print(f"Batch {i+1}: {loss}")
