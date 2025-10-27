#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%j.out
#SBATCH --error=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%j.err

#SBATCH --gres-flags=enforce-binding 

#SBATCH --constraint=h100
#SBATCH -p gpu-he --gres=gpu:1


##SBATCH --ntasks-per-node=4

 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4 # this is for num workers
#SBATCH --mem=50G

# Request an hour of runtime:
#SBATCH --time=24:00:00

#SBATCH -J  TRAIN-dayhoff-UR90-cath-class-3-mamba


export CUDA_LAUNCH_BLOCKING=1


module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate dayhoff
export PYTHONPATH=/users/rvinod/data/rvinod/repos/plm_subnetworks:$PYTHONPATH


echo "Running from: $(pwd)"
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

cd /users/rvinod/data/rvinod/repos/plm_subnetworks/plm_subnetworks

srun /users/rvinod/data/rvinod/.conda/envs/dayhoff/bin/python subnetwork/dayhoff_train_logits.py \
  --run_name dayhoff-cath-class-3-mamba \
  --wandb_project dayhoff-full-masked \
  --model_name microsoft/Dayhoff-170m-UR90 \
  --trust_remote_code \
  --batch_size 64 \
  --num_examples_per_batch 16 \
  --max_epochs 200 \
  --learning_rate 1e-1 \
  --accumulate_grad_batches 4 \
  --num_workers 4 \
  --precision fp32 \
  --mask_init_value 0.99 \
  --mask_top_layer_frac 0.80 \
  --mask_layer_range 0,24 \
  --mask_threshold 0.40 \
  --mask_temp_init 1 \
  --mask_temp_final 0.1 \
  --mask_temp_decay 100 \
  --sparsity_lambda_init 0 \
  --sparsity_lambda_final 1 \
  --sparsity_warmup_epochs 65 \
  --sparsity_ramp_epochs 15 \
  --suppression_mode cath \
  --suppression_level class \
  --suppression_target 3 \
  --maintenance_lambda 4 \
  --suppression_lambda 4 \
  --maintenance_mlm_lambda 4 \
  --lr_phaseA 1e-1 \
  --lr_phaseB 5e-4 \
  --lr_plateau_epochs 125 \
  --lr_hold_epochs 5 \
  --ckpt_freq 10 \
  --disable_input_corruption \
  --mask_mamba_projections=all