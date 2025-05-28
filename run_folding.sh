#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/data/rvinod/repos/probing-subnetworks/slurm_jobs/output-%j.out
#SBATCH --error=/users/rvinod/data/rvinod/repos/probing-subnetworks/slurm_jobs/output-%j.err

#SBATCH -A cbc-a5000-gcondo

#SBATCH --nodes=1

#SBATCH -p gpu --gres=gpu:1
#SBATCH --gres-flags=enforce-binding 

#SBATCH --cpus-per-task=1
 
# Request an hour of runtime
#SBATCH --time=8:00:00

#SBATCH --mem=50G

#SBATCH -J FOLD-v27-test


cd /users/rvinod/data/rvinod/repos/plm_subnetworks/evaluation

ml miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate esmfold_new3


echo "Running from: $(pwd)"
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"


#— user-configurable variables —#
RUN_NAME="v27-test_11552001"
EPOCH="04"
CATEGORY="cath_homologous_superfamily_code"
TARGET="3.40.30.10"
CATH_IDS="4aw9A00"   # if you have multiple, you can do: "4aw9A00 4aqzA00"


#— call your Python script using those variables —#
~/.conda/envs/esmfold_new3/bin/python fold_sequences.py \
  --run_name "${RUN_NAME}" \
  --epoch "${EPOCH}" \
  --category "${CATEGORY}" \
  --cath_ids "${CATH_IDS}" 

~/.conda/envs/esmfold_new3/bin/python tm_scores.py \
  --run_name "${RUN_NAME}" \
  --mode pred \
  --epoch "${EPOCH}" \
  --category "${CATEGORY}" \
  --target "${TARGET}" \
  --subnetwork_eval \
  --cath_ids "${CATH_IDS}" 
