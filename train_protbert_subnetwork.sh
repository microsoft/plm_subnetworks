#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%j.out
#SBATCH --error=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%j.err

#SBATCH --gres-flags=enforce-binding 

##SBATCH --constraint=h100
#SBATCH -p gpu-he --gres=gpu:1

##SBATCH -p gpu-b200 --gres=gpu:1


##SBATCH --ntasks-per-node=4

 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4 # this is for num workers
#SBATCH --mem=25G

# Request an hour of runtime:
#SBATCH --time=24:00:00

#SBATCH -J  TRAIN-protbert-UR100-cath-class-3

source /users/rvinod/data/rvinod/repos/plm_subnetworks/.venv_protbert/bin/activate

export CUDA_LAUNCH_BLOCKING=1

echo "Running from: $(pwd)"
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

cd /users/rvinod/data/rvinod/repos/plm_subnetworks/plm_subnetworks

ml python/3.11.0s-ixrhc3q 

srun /oscar/data/lcrawfo1/rvinod/repos/plm_subnetworks/.venv_protbert/bin/python subnetwork/protbert_train_logits.py \
    --run_name protbert-cath-class-3 \
    --wandb_project protbert-full-masked-UR100 \
    --model_name Rostlab/prot_bert \
    --batch_size 16 \
    --max_epochs 100 \
    --mask_init_value 0.998 \
    --suppression_mode cath \
    --suppression_level class \
    --suppression_target 3 \
    --num_examples_per_batch 4 \
    --learning_rate 1e-1 \
    --precision bf16 \
    --maintenance_lambda 9 \
    --suppression_lambda 10 \
    --maintenance_mlm_lambda 1 \
    --num_workers 4 \
    --accumulate_grad_batches 2 \
    --mask_top_layer_frac 0.8 \
    --sparsity_lambda_init 0 \
    --sparsity_lambda_final 0.0 \
    --sparsity_warmup_epochs 200 \
    --mask_temp_init 1 \
    --mask_temp_final 0.1 \
    --mask_temp_decay 100 \
    --lr_phaseA 1e-1 \
    --lr_phaseB 5e-4 \
    --lr_plateau_epochs 125 \
    --lr_hold_epochs 50 \
    --mask_threshold 0.43 \
    --ckpt_freq 1 \
    --sparsity_ramp_epochs 150 \
    --mask_layer_range 0,30

# srun /oscar/data/lcrawfo1/rvinod/repos/plm_subnetworks/.venv_protbert/bin/python subnetwork/protbert_train_logits.py \
#     --run_name protbert-cath-class-2-maxepochs100-init0.97-initmask0.45-lambflip \
#     --wandb_project protbert-full-masked-UR100 \
#     --model_name Rostlab/prot_bert \
#     --batch_size 16 \
#     --max_epochs 100 \
#     --mask_init_value 0.998 \
#     --suppression_mode cath \
#     --suppression_level class \
#     --suppression_target 2 \
#     --num_examples_per_batch 4 \
#     --learning_rate 1e-1 \
#     --precision bf16 \
#     --maintenance_lambda 10 \
#     --suppression_lambda 5 \
#     --maintenance_mlm_lambda 1 \
#     --num_workers 4 \
#     --accumulate_grad_batches 2 \
#     --mask_top_layer_frac 0.8 \
#     --sparsity_lambda_init 0 \
#     --sparsity_lambda_final 0.0 \
#     --sparsity_warmup_epochs 200 \
#     --mask_temp_init 1 \
#     --mask_temp_final 0.1 \
#     --mask_temp_decay 100 \
#     --lr_phaseA 1e-1 \
#     --lr_phaseB 5e-4 \
#     --lr_plateau_epochs 125 \
#     --lr_hold_epochs 50 \
#     --mask_threshold 0.43 \
#     --ckpt_freq 1 \
#     --sparsity_ramp_epochs 150 \
#     --mask_layer_range 0,30 \
   
    # for random baseline
    # --random_n 100 
    # --suppression_level random 
