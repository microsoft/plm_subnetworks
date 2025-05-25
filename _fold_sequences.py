import argparse
import pandas as pd

import esm.esmfold.v1
import esm.esmfold.v1.pretrained
import torch
import esm
import warnings
import esm.esmfold
from esm_modules.esmfold import ESMRepFold

from plmprobe.subnetwork.modules import SubnetworkESM, WeightedDifferentiableMask
from plmprobe.subnetwork.esm_masking_pl_logits import ESMMaskLearner

from plmprobe.dataset.cath_dataset import CATHDatabase, CATHEntry, CATH_ENTRY_FILEPATH, CATH_S40_SEQ_FILEPATH
from plmprobe.dataset import data_io
from plmprobe.dataset.data_paths import CATH_S20_IDS, CATH_S40_IDS, CATH_S20_ATOM_IDS, RUN_DIR_PREFIX
from plmprobe.dataset import data_paths

from plmprobe.esm_modules.esmfold import _load_model
import biotite.structure.io as bsio

import os

def fold_and_write_pdb(model, id, seq, masks=None, folding_dir=None):

    pred_result_path = f"{folding_dir}/pred/{id}.pdb"
    gt_result_path = f"{folding_dir}/gt/{id}.pdb"
    os.makedirs(f"{folding_dir}/pred", exist_ok=True)
    # os.makedirs(f"{folding_dir}/gt", exist_ok=True)

    # Run inference using modified forward method
    with torch.no_grad():
        output = model.infer_pdb(seq, masks)

    with open(pred_result_path, "w") as f:
        f.write(output)

    struct = bsio.load_structure(pred_result_path, extra_fields=["b_factor"])
    pred_plddt = struct.b_factor.mean()  # this will be the pLDDT

    # with torch.no_grad():
    #     output = model.infer_pdb(seq, None)

    # with open(gt_result_path, "w") as f:
    #     f.write(output)

    # import biotite.structure.io as bsio
    # struct = bsio.load_structure(gt_result_path, extra_fields=["b_factor"])
    # gt_plddt = struct.b_factor.mean()  # this will be the pLDDT

    return pred_plddt #, gt_plddt



def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    dir = f"{RUN_DIR_PREFIX}/{args.run_dir}"
    config, split = data_io.get_args_split(dir)

    if args.epoch is not None:
        ckpt_path = f"{config['run_dir']}/checkpoints/regular_checkpoints/epoch={args.epoch}.ckpt"
    else:
        ckpt_path = f"{config['run_dir']}/checkpoints/regular_checkpoints/last.ckpt"
    
    ckpt_epoch = torch.load(ckpt_path, map_location="cpu")["epoch"]

    if args.extend_val:
        val_ids = split["train"] + split["val"]
        folding_dir = f"{config['run_dir']}/folded_epoch_{ckpt_epoch}_full"
    else:
        val_ids = split["val"]
        folding_dir = f"{config['run_dir']}/folded_epoch_{ckpt_epoch}_val"

    if args.random_model:
        suppressed_id_path = f"{config['run_dir']}/random_supp_ids.txt"
        with open(suppressed_id_path, "r") as f:
            suppressed_ids = f.read().splitlines()
        print("Loaded suppressed ids:", len(suppressed_ids))
        val_ids.extend(suppressed_ids)

    if args.cath_ids:
        val_ids = args.cath_ids
        print("Only folding ----", val_ids)
        folding_dir = "/users/rvinod/data/rvinod/repos/probing-subnetworks/folded_pdbs"

    print("Number of eval seqs:", len(val_ids))

    print("Loading from checkpoint", ckpt_epoch)

    
    os.makedirs(folding_dir, exist_ok=True)


    cath_version = "20"
    _, seq_filepath = data_paths.get_cath_paths(cath_version)
    db = CATHDatabase()
    db.load_clf(CATH_ENTRY_FILEPATH)
    db.load_sequences(seq_filepath)

    # Initialize mask learner with the same hyperparameters
    mask_learner = WeightedDifferentiableMask(
        esm_model, 
        temp_init=config["mask_temperature_init"],
        temp_final=config["mask_temperature_final"],
        temp_decay=config["mask_temperature_decay"],
        init_value=config["mask_init_value"],
        mask_top_layer_frac=config["mask_top_layer_frac"],
        mask_layer_range=config["mask_layer_range"],
        mask_threshold=config["mask_threshold"],
    )



     # Load checkpoint weights
    lightning_model = ESMMaskLearner.load_from_checkpoint(ckpt_path, 
                                                        model=esm_model, 
                                                        mask_learner=mask_learner)

    print("Checkpoint path:", ckpt_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        ckpt = torch.load(ckpt_path, map_location="cpu")

    print(f"Mask temperature: {lightning_model.mask_learner.temperature:.2f}")
    mask_temp = ckpt["mask_temperature"]
    sparsity_lambda = ckpt["sparsity_lambda"]
    ckpt_epoch = ckpt["epoch"]
    num_training_steps = ckpt["global_step"]
    print(f"Mask temperature from checkpoint: {mask_temp:.2f}")
    print(f"Sparsity lambda from checkpoint: {sparsity_lambda:.2f}")
    print(f"# Steps from checkpoint: {num_training_steps}")
    # Set the model to evaluation mode
    lightning_model.mask_learner.temperature = mask_temp
    lightning_model.eval()
    esm_model = esm_model.to(device)
    lightning_model = lightning_model.to(device)
    mask_learner = mask_learner.to(device)
   
    sparsity = lightning_model.mask_learner.get_sparsity().item()
    print(f"Model sparsity: {sparsity}")


    with torch.no_grad():
        masks = lightning_model.mask_learner()

    esm_subnetwork = SubnetworkESM(esm_model, mask_learner.layers_to_mask)


    model = _load_model("esmfold_structure_module_only_650M", esm_subnetwork)
    model = model.eval().cuda()

    results = []

    for id in val_ids:
        seq =  db.query(id).sequence
        class_num = db.query(id).class_num

        # pred_plddt, gt_plddt = fold_and_write_pdb(model, id, seq, masks, folding_dir)
        pred_plddt = fold_and_write_pdb(model, id, seq, masks, folding_dir)

        metrics = {
            "id": id,
            "class_num": class_num,
            "pred_plddt": pred_plddt,
            # "gt_plddt": gt_plddt,
        }

        results.append(metrics)
    
    df = pd.DataFrame(results)
    print(df.head())
        

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default="debug_contact_train",
                      help='Name of the run')
    parser.add_argument('--epoch', type=str, default=None,
                    help='Which epoch to load')
    parser.add_argument('--extend_val', action='store_true',
                    help='Add flag to evaluate on full set instead of heldout set')
    parser.add_argument('--random_model', action='store_true',
                    help='Add flag to evaluate on full set instead of heldout set')
    
    parser.add_argument('--cath_ids', 
                    nargs='+',  # Accepts one or more arguments
                    help='List of CATH IDs')
    
    args = parser.parse_args()

    main(args)

    
