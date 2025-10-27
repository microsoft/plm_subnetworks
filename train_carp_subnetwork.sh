#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%j.out
#SBATCH --error=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%j.err

#SBATCH --gres-flags=enforce-binding 

#SBATCH --constraint=h100
#SBATCH -p gpu-he --gres=gpu:1

##SBATCH -p gpu-b200 --gres=gpu:1


##SBATCH --ntasks-per-node=4

 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4 # this is for num workers
#SBATCH --mem=50G

# Request an hour of runtime:
#SBATCH --time=48:00:00

#SBATCH -J  TRAIN-carp-cath-arch-1.10


export PYTHONPATH=/users/rvinod/data/rvinod/repos/plm_subnetworks:$PYTHONPATH



source /users/rvinod/data/rvinod/repos/plm_subnetworks/.venv_carp/bin/activate

export CUDA_LAUNCH_BLOCKING=1

echo "Running from: $(pwd)"
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

cd /users/rvinod/data/rvinod/repos/plm_subnetworks/plm_subnetworks

ml python/3.11.0s-ixrhc3q 

srun /oscar/data/lcrawfo1/rvinod/repos/plm_subnetworks/.venv_carp/bin/python subnetwork/carp_train_logits.py \
    --run_name carp-cath-arch-1.10 \
    --wandb_project carp-full-masked \
    --batch_size 16 \
    --max_epochs 100 \
    --mask_init_value 0.995 \
    --suppression_mode cath \
    --suppression_level architecture \
    --suppression_target 1.10 \
    --num_examples_per_batch 4 \
    --learning_rate 1e-1 \
    --precision bf16-mixed \
    --maintenance_lambda 9 \
    --suppression_lambda 10 \
    --maintenance_mlm_lambda 1 \
    --num_workers 4 \
    --accumulate_grad_batches 4 \
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
    --mask_threshold 0.40 \
    --ckpt_freq 1 \
    --sparsity_ramp_epochs 150 \
    --mask_layer_range 0,56
