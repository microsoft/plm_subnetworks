import argparse
import os
import warnings

import esm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from plm_inv_subnetworks.dataset import data_paths, data_io
from plm_inv_subnetworks.dataset.cath_dataset import CATHDatabase, CATH_ENTRY_FILEPATH
from plm_inv_subnetworks.dataset.data_paths import CATH_S20_DSSP_FASTA, ESM_PPL_METRICS, RUN_DIR_PREFIX
from plm_inv_subnetworks.dataset.esm_seq_dataloader import CATHSeqDatasetESM

from plm_inv_subnetworks.subnetwork.esm_masking_pl_logits import ESMMaskLearner
from plm_inv_subnetworks.subnetwork.modules import WeightedDifferentiableMask

from plm_inv_subnetworks.utils.metrics import PerSequenceMaskedCrossEntropyLoss



def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    perseqmcel = PerSequenceMaskedCrossEntropyLoss()


    cath_version = "20"
    _, seq_filepath = data_paths.get_cath_paths(cath_version)
    db = CATHDatabase()
    db.load_clf(CATH_ENTRY_FILEPATH)
    db.load_sequences(seq_filepath)
    db.load_dssp(CATH_S20_DSSP_FASTA)

    if args.csv:
        subnetworks_df = pd.read_csv(args.csv)
        subnetworks_df_data = list(zip(subnetworks_df["run_name"], subnetworks_df["epoch"], subnetworks_df["category"], subnetworks_df["target"]))
    else:

        subnetworks_df_data = [(args.run_name, args.epoch, args.category, args.target)]
    
    for run_name, epoch, category, target in subnetworks_df_data:
        
        run_dir = f"{RUN_DIR_PREFIX}/{run_name}"
        config, split = data_io.get_args_split(run_dir)
        ckpt_path = f"{config['run_dir']}/checkpoints/regular_checkpoints/epoch={epoch}.ckpt"

        print(f"\n>>> Evaluating inv. subnetwork: category = '{category}', target = '{target}', "
            f"epoch = {epoch}, run_dir = '{run_name}'")
        print(f"Checkpoint path: {ckpt_path}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            ckpt = torch.load(ckpt_path, map_location="cpu")

        if args.extend_val:
            val_ids = split["train"] + split["val"] + split["test"]
            print("Eval on full set")
        else:
            val_ids = split["val"] + split["test"]
            print("Eval on val sets")

        if "random" in category:
            suppressed_id_path = f"{RUN_DIR_PREFIX}/{run_name}/random_supp_ids.txt"
            with open(suppressed_id_path, "r") as f:
                suppressed_ids = f.read().splitlines()
            print("Loaded suppressed ids:", len(suppressed_ids))

            val_ids.extend(suppressed_ids)
        if args.verbose:
            print(f"Number of eval seqs: {len(val_ids)}")

        mask_learner = WeightedDifferentiableMask(
            model,
            temp_init=config["mask_temperature_init"],
            temp_final=config["mask_temperature_final"],
            temp_decay=config["mask_temperature_decay"],
            init_value=config["mask_init_value"],
            mask_top_layer_frac=config["mask_top_layer_frac"],
            mask_layer_range=config["mask_layer_range"],
            mask_threshold=config["mask_threshold"],
        )

        lightning_model = ESMMaskLearner.load_from_checkpoint(
            ckpt_path,
            model=model,
            mask_learner=mask_learner,
        )

        mask_temp = ckpt["mask_temperature"]
        sparsity_lambda = ckpt["sparsity_lambda"]
        ckpt_epoch = ckpt["epoch"]
        num_training_steps = ckpt["global_step"]

        lightning_model.mask_learner.temperature = mask_temp
        lightning_model.eval()

        model = model.to(device)
        lightning_model = lightning_model.to(device)
        mask_learner = mask_learner.to(device)

        with torch.no_grad():
            masks = lightning_model.mask_learner()

        sparsity = lightning_model.mask_learner.get_sparsity().item()

        if args.verbose:
            print(f"Mask temperature (runtime): {lightning_model.mask_learner.temperature:.2f}")
            print(f"Mask temperature (from ckpt): {mask_temp:.2f}")
            print(f"Sparsity lambda from checkpoint: {sparsity_lambda:.2f}")
            print(f"# Steps from checkpoint: {num_training_steps}")
            print(f"Model sparsity: {sparsity:.4f}")
            print(f"Number of passes: {args.n_passes}")

        results = []

        for _ in range(args.n_passes):

            val_dataset = CATHSeqDatasetESM(val_ids, db, alphabet, 
                                        min_n_res=64, 
                                        max_n_res=512,
                                        use_dssp=True)
            val_loader = DataLoader(
                val_dataset,
                batch_size= 4 if args.override_batch_size else config["batch_size"],
                shuffle=config["shuffle"],
                num_workers=config["num_workers"],  # Set to 0 to avoid multiprocessing issues
                collate_fn=val_dataset.collate_fn
            )

            for i, batch in enumerate(val_loader):

                cath_ids = batch["cath_ids"]
                tgt = batch["tgt"].to(device)
                src = batch["src"].to(device)
                corr_mask = batch["corr_mask"].to(device)
                seq_lens = batch["seq_lens"].to(device)

                with torch.no_grad():
                    pred_logits = lightning_model.subnetwork(src, masks=masks, return_contacts=False, inverse=False)["logits"]
                

                mlm_loss = perseqmcel(pred_logits, tgt, corr_mask)

                helix_mask = batch["helix_mask"].to(device)
                strand_mask = batch["strand_mask"].to(device)
                coil_mask = batch["coil_mask"].to(device)
                
                # Note: corr_mask.sum() << helix_corrupted_and_masked.sum()
                helix_corrupted_and_masked = corr_mask.bool() & helix_mask
                strand_corrupted_and_masked = corr_mask.bool() & strand_mask
                coil_corrupted_and_masked = corr_mask.bool() & coil_mask

                helix_mlm = perseqmcel(pred_logits, tgt, helix_corrupted_and_masked) if helix_corrupted_and_masked.sum()!=0 else np.NaN
                strand_mlm = perseqmcel(pred_logits, tgt, strand_corrupted_and_masked) if strand_corrupted_and_masked.sum()!=0 else np.NaN
                coil_mlm = perseqmcel(pred_logits, tgt, coil_corrupted_and_masked) if coil_corrupted_and_masked.sum()!=0 else np.NaN



                # Create a dictionary for this batch
                batch_data = {
                    "cath_id": cath_ids,
                    "mlm_loss": mlm_loss.cpu().numpy(),
                    "perplexity": np.exp(mlm_loss.cpu().numpy()), 
                    "helix_mlm": helix_mlm.cpu().numpy() if helix_corrupted_and_masked.sum()!=0 else np.nan,
                    "strand_mlm": strand_mlm.cpu().numpy() if strand_corrupted_and_masked.sum()!=0 else np.nan,
                    "coil_mlm": coil_mlm.cpu().numpy() if coil_corrupted_and_masked.sum()!=0 else np.nan,
                }
            
                # Convert to DataFrame and append to results list
                batch_df = pd.DataFrame(batch_data)
                results.append(batch_df)

                

        # Concatenate all batch DataFrames
        results_df = pd.concat(results, ignore_index=True)

        inference_dir = f"{run_dir}/inference"
        
        if args.extend_val:
            csv_path = f"{inference_dir}/epoch_{ckpt_epoch}_full_{args.n_passes}_passes.csv"
        else:
            csv_path = f"{inference_dir}/epoch_{ckpt_epoch}_heldout_{args.n_passes}_passes.csv"
            # fasta_path = f"{inference_dir}/epoch_{ckpt_epoch}_avg.fasta"

       
        os.makedirs(inference_dir, exist_ok=True)

        results_df.to_csv(csv_path, index=False)
        print("Wrote CSV to", csv_path)
        # data_io.dataframe_to_fasta(results_df, fasta_path)

        hydrated_df = data_io.hydrate_df_with_cath_terms(results_df, db)
       
        gt = pd.read_csv(ESM_PPL_METRICS)



    def print_stats(title, df, prefix=""):
        print(f"{prefix}{title} ({len(df)} seqs): "
        f"(PPL) {df['perplexity'].mean():.2f}, "
        f"(MLM) {df['mlm_loss'].mean():.2f}")

    def print_mlm_stats(label, df, prefix=""):
        mean_mlm = df.dropna().mean()
        print(f"{prefix}{label:<8} (PPL): {np.exp(mean_mlm):.2f} / (MLM) {mean_mlm:.2f}")

    hydrated_df_gt = data_io.hydrate_df_with_cath_terms(gt, db)
    hydrated_df_gt = hydrated_df_gt.groupby("cath_id").head(args.n_passes).reset_index(drop=True)


    if "cath" in category:
        supp_df = hydrated_df[hydrated_df[category] == target]
        maint_df = hydrated_df[hydrated_df[category] != target]

        print("\n================= Inverse Subnetwork =================")
        print_stats("Suppression", supp_df)
        print_stats("Maintenance", maint_df)
        print("=====================================================\n")
        
        hydrated_df_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(hydrated_df["cath_id"])]
        suppression_gt = hydrated_df_gt[hydrated_df_gt[category] == target]
        maintenance_gt = hydrated_df_gt[hydrated_df_gt[category] != target]

        print("================ ESM-2 (Ground Truth) ================")
        print_stats("Suppression", suppression_gt)
        print_stats("Maintenance", maintenance_gt)
        print("=====================================================\n")

    elif "random" in category:

        supp_df = hydrated_df[hydrated_df["cath_id"].isin(suppressed_ids)]
        maint_df = hydrated_df[~hydrated_df["cath_id"].isin(suppressed_ids)]

        print("\n================= Random Subnetwork ==================")
        print_stats("Suppression", supp_df)
        print_stats("Maintenance", maint_df)
        print("=====================================================\n")

        hydrated_df_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(hydrated_df["cath_id"])]
        suppression_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(suppressed_ids)]
        maintenance_gt = hydrated_df_gt[~hydrated_df_gt["cath_id"].isin(suppressed_ids)]

        print("================ ESM-2 (Ground Truth) ================")
        print_stats("Suppression", suppression_gt)
        print_stats("Maintenance", maintenance_gt)
        print("=====================================================\n")

    else:
        hydrated_df_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(hydrated_df["cath_id"])]

        print("\n================ Residue Subnetwork ==================")
        print_mlm_stats("Helix", hydrated_df["helix_mlm"])
        print_mlm_stats("Strand", hydrated_df["strand_mlm"])
        print_mlm_stats("Coil", hydrated_df["coil_mlm"])
        print("=====================================================\n")

        print("================ ESM-2 (Ground Truth) ================")
        print_mlm_stats("Helix", hydrated_df_gt["helix_mlm"])
        print_mlm_stats("Strand", hydrated_df_gt["strand_mlm"])
        print_mlm_stats("Coil", hydrated_df_gt["coil_mlm"])
        print("=====================================================\n")



if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--override_batch_size', type=bool, default=True,
                        help='Overrides when using smaller GPUs for inference', required=False)
    parser.add_argument('--extend_val', action='store_true',
                    help='Add flag to evaluate on full set instead of heldout set')
    parser.add_argument('--n_passes', type=int, default=1,
                        help='include val seqs in evals')
    parser.add_argument('--random', action='store_true',
                    help='Add flag to evaluate random suppression models.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name of run to evaluate')
    parser.add_argument('--epoch', type=str, default=None,
                        help='Epoch to evaluate')
    parser.add_argument('--category', type=str, default=None,
                        help='cath_{level}_code, random_seq, or residue')
    parser.add_argument('--target', type=str, default=None,
                        help='Target to evaluate, e..g., 1.25.40, helix, or random')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to csv file to evaluate')
    parser.add_argument('-V', '--verbose', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    main(args)

    

    
