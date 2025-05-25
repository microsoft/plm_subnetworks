
from torch.utils.data import BatchSampler, DataLoader, Sampler
import random
import esm
import datetime
import torch

from plm_inv_subnetworks.dataset.cath_dataset import CATHDatabase, CATH_ENTRY_FILEPATH, CATH_S20_DSSP_FASTA
from plm_inv_subnetworks.dataset.esm_seq_dataloader import CATHSeqDatasetESM
from plm_inv_subnetworks.dataset import data_paths
from torch.utils.data import Subset


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, level, target, num_samples=2, random_n=2500, random_supp_id_path=None):

        self.dataset = dataset
        self.batch_size = batch_size
        self.level = level
        self.target = target
        self.num_samples = num_samples
        self.random_n = random_n
        self.random_supp_id_path = random_supp_id_path

        if level == "class":
            L = 1
        elif level == "architecture":
            L = 2
        elif level == "topology":
            L = 3
        elif level == "homologous_superfamily":
            L = 4
        else:
            L = 5

        if self.level == "random":

            print(f"Suppressing {random_n} RANDOM sequences!")


            self.valid_indices = [i for i, data in enumerate(self.dataset) if data and i < random_n]
            self.invalid_indices = [i for i, data in enumerate(self.dataset) if data and i > random_n]

            suppressed_ids = [self.dataset[i]["cath_id"] for i in self.valid_indices]
            print(f"Suppressed {len(suppressed_ids)} IDs: {suppressed_ids}")
            # Write them to a file
            if random_supp_id_path:
                with open(random_supp_id_path, "w") as f:
                    for item in suppressed_ids:
                        f.write(f"{item}\n")

        else:
            self.valid_indices = [i for i, data in enumerate(self.dataset) if data and '.'.join(data["cath_code"].split('.')[:L])  == target]
            self.invalid_indices = [i for i, data in enumerate(self.dataset) if data and '.'.join(data["cath_code"].split('.')[:L])  != target]

        
        assert len(self.valid_indices) >= num_samples, "Not enough L-labeled samples to satisfy constraint!"
        
        # Shuffle indices
        random.shuffle(self.valid_indices)
        random.shuffle(self.invalid_indices)
        
        # Limit valid indices to only those that can be used in complete batches
        self.num_batches = len(self.valid_indices) // num_samples
        self.remaining_indices = self.valid_indices[:self.num_batches * num_samples]  # Trim excess valid samples
        
    def __iter__(self):
        L_pool = self.remaining_indices[:]
        other_pool = self.invalid_indices[:]
        
        for i in range(self.num_batches):
            # Select `num_samples` L samples
            batch_L = L_pool[i * self.num_samples:(i + 1) * self.num_samples]
            
            # Select remaining samples from other classes (without repetition)
            batch_other = other_pool[:self.batch_size - self.num_samples]  # Take from the front
            other_pool = other_pool[self.batch_size - self.num_samples:]  # Remove from the pool
            
            batch = batch_L + batch_other
            random.shuffle(batch)
            yield batch
        
    def __len__(self):
        return self.num_batches  # Stop when all target samples are used

if __name__ == "__main__":

    cath_version = "20"  
    min_n_res = 64
    max_n_res = 512
    batch_size = 4
    num_train = 32
    random_supp_id_path = "test_rand_supp.txt"

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    cath_ids, seq_filepath = data_paths.get_cath_paths(cath_version)
    db = CATHDatabase()
    db.load_clf(CATH_ENTRY_FILEPATH)
    db.load_sequences(seq_filepath)
    db.load_dssp(CATH_S20_DSSP_FASTA)

    train_ids = cath_ids[:num_train]
    train_dataset = CATHSeqDatasetESM(
        train_ids, db, alphabet,
        min_n_res=min_n_res, max_n_res=max_n_res
    )

    train_batch_sampler = CustomBatchSampler(
        train_dataset, batch_size,
        level="random", target=0, num_samples=2,
        random_n=10, random_supp_id_path=random_supp_id_path
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=train_dataset.collate_fn
    )

    starttime = datetime.datetime.now()
    print(f"Total train batches: {len(train_dataloader)}")

    for batch in train_dataloader:
        print(batch["cath_ids"])
        break

    print(f"Total time: {(datetime.datetime.now() - starttime) / 60}")

