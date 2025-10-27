#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%j.out
#SBATCH --error=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%j.err

#SBATCH --gres-flags=enforce-binding 


#SBATCH --constraint=h100
#SBATCH -p gpu-he --gres=gpu:1


 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2 # this is for num workers
#SBATCH --mem=25G

# Request an hour of runtime:
#SBATCH --time=72:00:00

#SBATCH -J  TRAIN-esm-class-2

source /users/rvinod/data/rvinod/repos/plm_subnetworks/.venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1

echo "Running from: $(pwd)"
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

cd /users/rvinod/data/rvinod/repos/plm_subnetworks/plm_subnetworks

ml python/3.11.0s-ixrhc3q 

srun /oscar/data/lcrawfo1/rvinod/repos/plm_subnetworks/.venv/bin/python subnetwork/train_logits.py \
    --run_name esm-cath-class-2 \
    --wandb_project esm-seeds \
    --batch_size 16 \
    --max_epochs 1000 \
    --mask_init_value 0.96 \
    --suppression_mode cath \
    --suppression_level class \
    --suppression_target 2 \
    --num_examples_per_batch 4 \
    --learning_rate 1e-1 \
    --precision bf16 \
    --maintenance_lambda 7 \
    --suppression_lambda 10 \
    --maintenance_mlm_lambda 1 \
    --num_workers 4 \
    --accumulate_grad_batches 2 \
    --mask_top_layer_frac 0.8 \
    --sparsity_lambda_init 0 \
    --sparsity_lambda_final 0.0 \
    --sparsity_warmup_epochs 200 \
    --mask_temp_init 3 \
    --mask_temp_final 0.01 \
    --mask_temp_decay 100 \
    --lr_phaseA 1e-1 \
    --lr_phaseB 5e-4 \
    --lr_plateau_epochs 125 \
    --lr_hold_epochs 50 \
    --mask_threshold 0.40 \
    --ckpt_freq 5 \
    --sparsity_ramp_epochs 150 \
    --mask_layer_range 0,33
   
    # for random baseline
    # --random_n 100 
    # --suppression_level random 
