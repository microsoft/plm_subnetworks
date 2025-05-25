from plm_inv_subnetworks.dataset.cath_dataset import CATHDatabase, CATHEntry, CATH_ENTRY_FILEPATH
from plm_inv_subnetworks.dataset import data_io
from plm_inv_subnetworks.dataset import data_paths
from plm_inv_subnetworks.dataset.esm_seq_dataloader import CATHSeqDatasetESM
from torch.utils.data import DataLoader
import numpy as np
from plm_inv_subnetworks.subnetwork.modules import WeightedDifferentiableMask
from plm_inv_subnetworks.subnetwork.esm_masking_pl_logits import ESMMaskLearner
import torch
import esm
import warnings
import pandas as pd

import argparse
import os
import torch.nn as nn

from plm_inv_subnetworks.dataset.data_paths import ESM_PPL_METRICS, RUN_DIR_PREFIX, CATH_S20_DSSP_FASTA

from plm_inv_subnetworks.utils.metrics import MaskedCrossEntropyLoss, PerSequenceMaskedCrossEntropyLoss




def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # mcel_seq = MaskedCrossEntropyLoss(weight=None, reduction='mean')
    # mcel_res = MaskedCrossEntropyLoss(weight=None, reduction='none')
    perseqmcel = PerSequenceMaskedCrossEntropyLoss()


    cath_version = "20"
    _, seq_filepath = data_paths.get_cath_paths(cath_version)
    db = CATHDatabase()
    db.load_clf(CATH_ENTRY_FILEPATH)
    db.load_sequences(seq_filepath)
    db.load_dssp(CATH_S20_DSSP_FASTA)

    if args.csv:
        # subnetworks_df = pd.read_csv("/users/rvinod/data/rvinod/repos/probing-subnetworks/data_computed_all_gt_metrics/all_subnetworks_.csv")
        subnetworks_df = pd.read_csv(args.csv)
        subnetworks_df_data = list(zip(subnetworks_df["run_name"], subnetworks_df["epoch"], subnetworks_df["category"], subnetworks_df["target"]))
    else:
        # subnetworks_df = pd.read_csv("/users/rvinod/data/rvinod/repos/probing-subnetworks/data_computed_all_gt_metrics/all_subnetworks.csv")
        # subnetworks_df_data = list(zip(subnetworks_df["run_dir"], subnetworks_df["epoch"], subnetworks_df["level"], subnetworks_df["target"]))
        # print(subnetworks_df_data)
        # exit()

        subnetworks_df_data = [(args.run_name, args.epoch, args.category, args.target)]
    
        # subnetworks_df_data = [
        #                     # ("v24-supp1-p0.96-smooth-temp-3.2_11392022", 5, "cath_class_code", "1"),
        #                     ("v27-supp-strand-TEST-maintkl-mimic_11502861", "02", "residue", "strand"), 
        #                     # ("v25-random100-p0.96-smooth-temp-3_11445994", 23, "random_seq", "random"),
        #                     ]
    
    
    # for subnetwork_run_dir, epoch, in subnetworks_df_data:
    
    for subnetwork_run_dir, epoch, category, target, in subnetworks_df_data:
    

        print(subnetwork_run_dir, epoch)
        run_dir = f"{RUN_DIR_PREFIX}/{subnetwork_run_dir}"
        # subset = subnetworks_df[subnetworks_df["run_dir"] == subnetwork_run_dir]
        # category = subset["level"].iloc[0]
        # target = subset["target"].iloc[0]
        print(category, target)
        plotting_dir = f"{run_dir}/figures"
        os.makedirs(plotting_dir, exist_ok=True)

        config, split = data_io.get_args_split(run_dir)
        
        ckpt_path = f"{config['run_dir']}/checkpoints/regular_checkpoints/epoch={epoch}.ckpt"
        print("Checkpoint path:", ckpt_path)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            ckpt = torch.load(ckpt_path, map_location="cpu")

        if args.extend_val:
            print("Eval on full set")
            val_ids = split["val"] + split["train"]
        else:
            print("Eval on val set")
            val_ids = split["val"]

        if args.random:
            suppressed_id_path = f"{RUN_DIR_PREFIX}/{subnetwork_run_dir}/random_supp_ids.txt"
            with open(suppressed_id_path, "r") as f:
                suppressed_ids = f.read().splitlines()
            print("Loaded suppressed ids:", len(suppressed_ids))

            val_ids.extend(suppressed_ids)


        print("Number of eval seqs:", len(val_ids))

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
        


        # Load checkpoint weights
        lightning_model = ESMMaskLearner.load_from_checkpoint(ckpt_path, 
                                                            model=model, 
                                                            mask_learner=mask_learner)

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

        model = model.to(device)
        lightning_model = lightning_model.to(device)
        mask_learner = mask_learner.to(device)

        with torch.no_grad():
            masks = lightning_model.mask_learner()

        sparsity = lightning_model.mask_learner.get_sparsity().item()
        print(f"Model sparsity: {sparsity}")

    
        results = []

        print("Number of passes:", args.n_passes)

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

                break
                

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

        if "cath" in category:

            supp_df = hydrated_df.loc[hydrated_df[category] == target]
            maint_df = hydrated_df.loc[hydrated_df[category] != target]

            print(f"### Suppression ({len(supp_df)} seqs): "
                f"(PPL) {supp_df['perplexity'].mean():.2f}, "
                f"(MLM) {supp_df['mlm_loss'].mean():.2f}, ")
                # f"(exp(MLM)) {np.exp(supp_df['mlm_loss'].mean()):.2f}")

            print(f"### Maintenance ({len(maint_df)} seqs): "
                f"(PPL) {maint_df['perplexity'].mean():.2f}, "
                f"(MLM) {maint_df['mlm_loss'].mean():.2f}, ")
                # f"(exp(MLM)) {np.exp(maint_df['mlm_loss'].mean()):.2f}")

                                
            print(f"Helix (PPL):  {np.exp(hydrated_df['helix_mlm'].dropna().mean()):.2f}")
            print(f"Strand (PPL): {np.exp(hydrated_df['strand_mlm'].dropna().mean()):.2f}")
            print(f"Coil (PPL):   {np.exp(hydrated_df['coil_mlm'].dropna().mean()):.2f}")

            # REPORT GT METRICS TODO: change path when 10 passes are done.
            hydrated_df_gt = data_io.hydrate_df_with_cath_terms(gt, db)
            suppression_gt = hydrated_df_gt[hydrated_df_gt[category] == target]
            maintenance_gt = hydrated_df_gt[hydrated_df_gt[category] != target]
            print(f"ESM-2 Suppression --- PPL: {suppression_gt['perplexity'].mean():.2f}, MLM: {suppression_gt['mlm_loss'].mean():.2f}")
            print(f"ESM-2 Maintenance --- PPL: {maintenance_gt['perplexity'].mean():.2f}, MLM: {maintenance_gt['mlm_loss'].mean():.2f}")

        if "random" in category:

            
            supp_df = hydrated_df[hydrated_df["cath_id"].isin(suppressed_ids)]
            maint_df = hydrated_df[~hydrated_df["cath_id"].isin(suppressed_ids)]
            
            print(f"### Suppression ({len(supp_df)} seqs): "
                f"(PPL) {supp_df['perplexity'].mean():.2f}, "
                f"(MLM) {supp_df['mlm_loss'].mean():.2f}, ")
                # f"(exp(MLM)) {np.exp(supp_df['mlm_loss'].mean()):.2f}")

            print(f"### Maintenance ({len(maint_df)} seqs): "
                f"(PPL) {maint_df['perplexity'].mean():.2f}, "
                f"(MLM) {maint_df['mlm_loss'].mean():.2f}, ")
                # f"(exp(MLM)) {np.exp(maint_df['mlm_loss'].mean()):.2f}")

            hydrated_df_gt = data_io.hydrate_df_with_cath_terms(gt, db)
            suppression_gt = hydrated_df_gt[hydrated_df_gt["cath_id"].isin(suppressed_ids)]
            maintenance_gt = hydrated_df_gt[~hydrated_df_gt["cath_id"].isin(suppressed_ids)]

            print(f"ESM-2 Suppression --- PPL: {suppression_gt['perplexity'].mean():.2f}, MLM: {suppression_gt['mlm_loss'].mean():.2f}")
            print(f"ESM-2 Maintenance --- PPL: {maintenance_gt['perplexity'].mean():.2f}, MLM: {maintenance_gt['mlm_loss'].mean():.2f}")
        else:
                                
            print(f"Helix (PPL):  {np.exp(hydrated_df['helix_mlm'].dropna().mean()):.2f} / (MLM) {hydrated_df['helix_mlm'].dropna().mean():.2f}")
            print(f"Strand (PPL): {np.exp(hydrated_df['strand_mlm'].dropna().mean()):.2f} / (MLM) {hydrated_df['strand_mlm'].dropna().mean():.2f}")
            print(f"Coil (PPL):   {np.exp(hydrated_df['coil_mlm'].dropna().mean()):.2f} / (MLM) {hydrated_df['coil_mlm'].dropna().mean():.2f}")

            print(f"ESM-2 Suppression --- ")

            print(f"Helix (PPL):  {np.exp(gt['helix_mlm'].mean()):.2f} / (MLM) {gt['helix_mlm'].dropna().mean():.2f}")
            print(f"Strand (PPL): {np.exp(gt['strand_mlm'].mean()):.2f} / (MLM) {gt['strand_mlm'].dropna().mean():.2f}")
            print(f"Coil (PPL):   {np.exp(gt['coil_mlm'].mean()):.2f} / (MLM) {gt['coil_mlm'].dropna().mean():.2f}")

        print("****************************************************************")






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
    
    args = parser.parse_args()

    main(args)

    

    
