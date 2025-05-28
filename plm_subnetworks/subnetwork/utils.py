import random
from torch.utils.data import DataLoader

from plm_subnetworks.dataset import data_io, data_paths
from plm_subnetworks.dataset.batch_sampler import CustomBatchSampler
from plm_subnetworks.dataset.cath_dataset import CATHDatabase, CATH_ENTRY_FILEPATH, CATH_S20_DSSP_FASTA
from plm_subnetworks.dataset.esm_seq_dataloader import CATHSeqDatasetESM
from plm_subnetworks.dataset.data_paths import CATH_IDS_S20_PDB_CONTACTS

def sanitize_name(name):
    return name.replace('.', '_')

def get_dataloaders(batch_size, alphabet, split=0.7, even_sampling=False,shuffle=True, num_workers=0, 
                    debug=False, train_ids=None, val_ids=None, test_ids=None, use_dssp=False,
                    level=None, target=None, num_samples=None, random_n=None, random_supp_id_path=None):
    
    cath_version = "20"
    cath_ids, seq_filepath = data_paths.get_cath_paths(cath_version)
    cath_ids = data_io.read_from_txt(CATH_IDS_S20_PDB_CONTACTS) # these are the CATH IDs used in trainign models in the paper
    random.shuffle(cath_ids)

    if debug:
        cath_ids = cath_ids[:100]

    if train_ids and val_ids and test_ids:
        train_ids = train_ids
        val_ids = val_ids
        test_ids = test_ids
    else:
        train_ids = cath_ids[: int(len(cath_ids) * split)]
        val_ids = cath_ids[int(len(cath_ids) * split): int(len(cath_ids) * (split + 0.2))]
        test_ids = cath_ids[int(len(cath_ids) * (split + 0.2)):]


    db = CATHDatabase()
    db.load_clf(CATH_ENTRY_FILEPATH)
    db.load_sequences(seq_filepath)
    db.load_dssp(CATH_S20_DSSP_FASTA)


    print(f"Using {num_workers} workers")

    # Create datasets and dataloaders
    train_dataset = CATHSeqDatasetESM(train_ids, db, alphabet,
                                   min_n_res=64, 
                                   max_n_res=512,
                                   use_dssp=use_dssp)
    
    val_dataset = CATHSeqDatasetESM(val_ids, db, alphabet, 
                                   min_n_res=64, 
                                   max_n_res=512,
                                   use_dssp=use_dssp)
    
    if even_sampling:
        
        assert level is not None 
        assert target is not None
        assert num_samples is not None

        print(" ########### Using even sampling: ###########")
        print("level: ", level)
        print("target: ", target)
        print("batch_size: ", batch_size)
        print("num_examples per batch: ", num_samples)
        print("num train examples: ", len(train_dataset))
        print("num val examples: ", len(val_dataset))
        print("debug mode: ", debug)
        print("#############################################")
        

        print("Getting TRAIN loader...")
        train_batch_sampler = CustomBatchSampler(train_dataset, batch_size, level=level, target=target, num_samples=num_samples, random_n=random_n, random_supp_id_path=random_supp_id_path)
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=train_dataset.collate_fn, num_workers=num_workers, pin_memory=True)

        print("Getting VAL loader...")
        val_batch_sampler = CustomBatchSampler(val_dataset, batch_size, level=level, target=target, num_samples=num_samples, random_n=random_n)
        val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, collate_fn=train_dataset.collate_fn, num_workers=num_workers, pin_memory=True)


    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,  # Set to 0 to avoid multiprocessing issues
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,

        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True,
        )
    
    for batch in train_loader:
        print(batch["cath_ids"])
        break

    return train_ids, val_ids, test_ids, train_loader, val_loader

