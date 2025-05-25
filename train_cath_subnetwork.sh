#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/data/rvinod/repos/probing-subnetworks/slurm_jobs/output-%j.out
#SBATCH --error=/users/rvinod/data/rvinod/repos/probing-subnetworks/slurm_jobs/output-%j.err

#SBATCH --gres-flags=enforce-binding 


#SBATCH --constraint=h100
#SBATCH -p gpu-he --gres=gpu:1


##SBATCH --ntasks-per-node=4

 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4 # this is for num workers
#SBATCH --mem=25G

# Request an hour of runtime:
#SBATCH --time=48:00:00

#SBATCH -J  TRAIN-v28-supp-1

source /users/rvinod/data/rvinod/repos/plm_inv_subnetworks/h100env/bin/activate

export CUDA_LAUNCH_BLOCKING=1

echo "Running from: $(pwd)"
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

cd /users/rvinod/data/rvinod/repos/plm_inv_subnetworks/plm_inv_subnetworks


srun python subnetwork/train_logits.py \
    --run_name v28-supp-1 \
    --wandb_project cath-class-subnetworks-v6 \
    --batch_size 16 \
    --max_epochs 1000 \
    --mask_init_value 0.96 \
    --suppression_mode cath \
    --suppression_level class \
    --suppression_target 1 \
    --num_examples_per_batch 4 \
    --learning_rate 1e-1 \
    --precision bf16 \
    --maintenance_lambda 7 \
    --suppression_lambda 10 \
    --maintenance_mlm_lambda 1 \
    --num_workers 4 \
    --accumulate_grad_batches 4 \
    --mask_top_layer_frac 0.8 \
    --sparsity_lambda_init 0 \
    --sparsity_lambda_final 0 \
    --sparsity_warmup_epochs 200 \
    --mask_temp_init 3.2 \
    --mask_temp_final 0.01 \
    --mask_temp_decay 100 \
    --lr_phaseA 1e-1 \
    --lr_phaseB 5e-4 \
    --lr_plateau_epochs 125 \
    --lr_hold_epochs 50 \
    --mask_threshold 0.40 \
    --ckpt_freq 5 \
    --sparsity_ramp_epochs 150 
   
    # for random baseline
    # --random_n 100 
    # --suppression_level random 
