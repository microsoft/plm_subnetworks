#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%j.out
#SBATCH --error=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%j.err

 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4 # this is for num workers
#SBATCH --mem=50G

# Request an hour of runtime:
#SBATCH --time=72:00:00

#SBATCH -J  PLOT-layer-activations


ml python/3.11.0s-ixrhc3q 

source /users/rvinod/data/rvinod/repos/plm_subnetworks/.venv/bin/activate

python -m pip install --quiet seaborn

export CUDA_LAUNCH_BLOCKING=1

echo "Running from: $(pwd)"
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

cd /users/rvinod/data/rvinod/repos/plm_subnetworks




python notebooks/layer_activation_mean_all_models.py --model dayhoff
python notebooks/layer_activation_mean_all_models.py --model esm2
python notebooks/layer_activation_mean_all_models.py --model carp
python notebooks/layer_activation_mean_all_models.py --model protbert_ur100
