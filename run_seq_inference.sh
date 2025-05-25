#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/data/rvinod/repos/probing-subnetworks/slurm_jobs/output-%j.out
#SBATCH --error=/users/rvinod/data/rvinod/repos/probing-subnetworks/slurm_jobs/output-%j.err

#SBATCH --nodes=1

#SBATCH -A cbc-a5000-gcondo
##SBATCH --constraint=h100
#SBATCH -p gpu --gres=gpu:1
#SBATCH --gres-flags=enforce-binding 

#SBATCH --cpus-per-task=1
 
# Request an hour of runtime
#SBATCH --time=24:00:00

#SBATCH --mem=20G

#SBATCH -J EVAL-v27-test 

cd /users/rvinod/data/rvinod/repos/plm_inv_subnetworks/evaluation
source /users/rvinod/data/rvinod/repos/plm_inv_subnetworks/h100env/bin/activate


python seq_inference.py \
    --run_name v27-test_11552001 \
    --epoch 02 \
    --category cath_homologous_superfamily_code \
    --target 3.40.30.10 \
    --n_passes 10 \
    --extend_val
