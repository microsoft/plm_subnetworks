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
#SBATCH --time=36:00:00

#SBATCH -J  TRAIN-v27-test-helix 

# cd /users/rvinod/data/rvinod/repos/probing-subnetworks/plmprobe
# source /users/rvinod/data/rvinod/repos/probing-subnetworks/h100env/bin/activate

# ## export PYTHONPATH=/users/rvinod/data/rvinod/repos/probing-subnetworks
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

# echo "Running from: $(pwd)"
# echo "Python: $(which python)"
# echo "PYTHONPATH: $PYTHONPATH"

source /users/rvinod/data/rvinod/repos/plm_subnetworks/h100env/bin/activate

export CUDA_LAUNCH_BLOCKING=1

echo "Running from: $(pwd)"
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

cd /users/rvinod/data/rvinod/repos/plm_subnetworks/plm_subnetworks

srun python subnetwork/train_logits.py \
   --run_name v27-test-helix \
   --wandb_project dssp-suppression \
   --batch_size 16 \
   --max_epochs 800 \
   --mask_init_value 0.96 \
   --suppression_mode dssp \
   --suppression_target helix \
   --learning_rate 1e-1 \
   --precision bf16 \
   --maintenance_lambda 100 \
   --suppression_lambda 50 \
   --maintenance_mlm_lambda 1.2 \
   --num_workers 4 \
   --accumulate_grad_batches 4 \
   --mask_top_layer_frac 0.8 \
   --sparsity_lambda_init 0 \
   --sparsity_lambda_final 0 \
   --sparsity_warmup_epochs 200 \
   --mask_temp_init 3.3 \
   --mask_temp_final 0.01 \
   --mask_temp_decay 100 \
   --lr_phaseA 1e-1 \
   --lr_phaseB 5e-4 \
   --lr_plateau_epochs 125 \
   --lr_hold_epochs 50 \
   --mask_threshold 0.38 \
   --ckpt_freq 1 \
   --sparsity_ramp_epochs 150

