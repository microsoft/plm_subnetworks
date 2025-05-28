import argparse
import os
import warnings

import esm
import torch
import pandas as pd
import biotite.structure.io as bsio

from plm_subnetworks.dataset import data_io
from plm_subnetworks.dataset import data_paths
from plm_subnetworks.dataset.data_paths import RUN_DIR_PREFIX, CATH_S20_ATOM_IDS
from plm_subnetworks.dataset.cath_dataset import CATHDatabase, CATH_ENTRY_FILEPATH
from plm_subnetworks.subnetwork.modules import SubnetworkESM, WeightedDifferentiableMask
from plm_subnetworks.subnetwork.esm_masking_pl_logits import ESMMaskLearner
from plm_subnetworks.esm_modules.esmfold import _load_model


def fold_and_write_pdb(model, cid, seq, masks, folding_dir):
    pred_dir = os.path.join(folding_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)
    pred_pdb = os.path.join(pred_dir, f"{cid}.pdb")

    with torch.no_grad():
        pdb_str = model.infer_pdb(seq, masks)

    with open(pred_pdb, "w") as f:
        f.write(pdb_str)

    struct = bsio.load_structure(pred_pdb, extra_fields=["b_factor"])
    return struct.b_factor.mean()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model, _ = esm.pretrained.esm2_t33_650M_UR50D()

    # Build list of runs to process
    if args.csv:
        run_df = pd.read_csv(args.csv)
        runs = list(zip(run_df["run_name"], run_df["epoch"],
                        run_df["category"]))
    else:
        runs = [(args.run_name, args.epoch,
                 args.category)]

    for run_name, epoch, category in runs:
        run_dir = f"{RUN_DIR_PREFIX}/{run_name}"
        config, split = data_io.get_args_split(run_dir)

        # Default held-out vs full set
        if args.extend_val:
            val_ids = split["train"] + split["val"] + split["test"]
            suffix = "full"
        else:
            val_ids = split["val"] + split["test"]
            suffix = "val"

        # Random suppression IDs
        if "random" in category:
            supp_path = os.path.join(run_dir, "random_supp_ids.txt")
            with open(supp_path) as f:
                extra = f.read().splitlines()
            val_ids.extend(extra)
            if args.verbose:
                print(f"[{run_name}] loaded {len(extra)} random IDs")

        # Override with user-provided CATH IDs?
        if args.cath_ids:
            val_ids = args.cath_ids
            print(f"[{run_name}] overriding val_ids with: {val_ids}")
            # (optional) you could also redirect folding output to a different folder here

        if args.verbose:
            print(f"[{run_name}] will fold {len(val_ids)} sequences (epoch={epoch})")

        # Load checkpoint and prepare output dir
        ckpt_file = f"{config['run_dir']}/checkpoints/regular_checkpoints/epoch={epoch or 'last'}.ckpt"
        ckpt_meta = torch.load(ckpt_file, map_location="cpu")
        actual_epoch = ckpt_meta["epoch"]
        fold_dir = f"{config['run_dir']}/folded_epoch_{actual_epoch}_{suffix}"
        os.makedirs(fold_dir, exist_ok=True)

        cath_version = "20"
        _, seq_filepath = data_paths.get_cath_paths(cath_version)
        db = CATHDatabase()
        db.load_clf(CATH_ENTRY_FILEPATH)
        db.load_sequences(seq_filepath)

        # Build mask learner + Lightning model
        mask_learner = WeightedDifferentiableMask(
            esm_model,
            temp_init = config["mask_temperature_init"],
            temp_final= config["mask_temperature_final"],
            temp_decay= config["mask_temperature_decay"],
            init_value= config["mask_init_value"],
            mask_top_layer_frac = config["mask_top_layer_frac"],
            mask_layer_range      = config["mask_layer_range"],
            mask_threshold        = config["mask_threshold"],
        )
        lightning = ESMMaskLearner.load_from_checkpoint(
            ckpt_file, model=esm_model, mask_learner=mask_learner
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            ckpt_meta = torch.load(ckpt_file, map_location="cpu")

        lightning.mask_learner.temperature = ckpt_meta["mask_temperature"]
        lightning.eval()
        esm_model.to(device)
        lightning.to(device)
        mask_learner.to(device)

        if args.verbose:
            print(f" mask_temp={lightning.mask_learner.temperature:.2f}, "
                  f"sparsity={lightning.mask_learner.get_sparsity().item():.4f}")

        with torch.no_grad():
            masks = lightning.mask_learner()

        subnet = SubnetworkESM(esm_model, mask_learner.layers_to_mask)
        fold_model = _load_model("esmfold_structure_module_only_650M", subnet)
        fold_model = fold_model.eval().to(device)

        # Fold each sequence
        records = []
        for cid in val_ids:
            seq = db.query(cid).sequence
            # setting masks to None will use the default ESM-650M representation => ESMFold (650M)
            plddt = fold_and_write_pdb(fold_model, cid, seq, masks, fold_dir)
            records.append({"cath_id": cid, "pred_plddt": plddt})

        out_df = pd.DataFrame(records)
        if args.cath_ids:
            print(out_df)
        else:
           
            out_csv = os.path.join(fold_dir, "pred_plddt.csv")
            out_df.to_csv(out_csv, index=False)
            print(f"[{run_name}] wrote results to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv',      type=str,
                        help='CSV with [run_name,epoch,category,target]')
    parser.add_argument('--run_name', type=str,
                        help='Name of a single run to fold')
    parser.add_argument('--epoch',    type=str,
                        help='Epoch to load (omit for last)')
    parser.add_argument('--category', type=str,
                        help='cath_{level}_code, random_seq, or residue')
    parser.add_argument('--extend_val', action='store_true',
                        help='Include train+val+test instead of val+test')
    parser.add_argument('--cath_ids', nargs='+',
                        help='List of CATH IDs to fold (overrides splits)')
    parser.add_argument('--verbose',   action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()
    main(args)
