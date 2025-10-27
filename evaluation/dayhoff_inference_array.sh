#!/bin/bash
#
# EVAL (ARRAY): Dayhoff runs via seq_inference_dayhoff.py driven from a CSV file
#

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm_array/output-%A_%a.out
#SBATCH --error=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm_array/output-%A_%a.err

#SBATCH --nodes=1
##SBATCH -A cbc-a5000-gcondo
#SBATCH --constraint=h100
#SBATCH -p gpu-he --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --mem=60G
#SBATCH -J EVAL-dayhoff-array-mamba
#SBATCH --array=0-0%8

set -euo pipefail

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
CONDA_ENV="dayhoff"
CONDA_RUN=(conda run --no-capture-output -n "$CONDA_ENV")

echo "[DEBUG] verifying conda environment '$CONDA_ENV'"
if ! "${CONDA_RUN[@]}" python - <<'PYENV'
import sys
try:
    import numpy as np
except Exception as exc:
    print(f"[DEBUG] numpy import failed: {exc}")
    sys.exit(1)
else:
    print(f"[DEBUG] numpy version = {np.__version__}")
print(f"[DEBUG] sys.executable = {sys.executable}")
PYENV
then
    echo "[ERROR] Failed to initialize conda environment '$CONDA_ENV'" >&2
    exit 1
fi

DEFAULT_CSV="/users/rvinod/data/rvinod/repos/plm_subnetworks/results_figures/dayhoff-v4.csv"
# DEFAULT_CSV="/users/rvinod/data/rvinod/repos/plm_subnetworks/results_figures/dayhoff-mamba.csv"
COMMON_ARGS=(
    --n_passes 10
    --extend_val
    --overwrite
    --trust_remote_code
)

MAX_PARALLEL=${MAX_PARALLEL:-6}
if [[ -z "$MAX_PARALLEL" ]]; then
    MAX_PARALLEL=6
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]]; then
    echo "MAX_PARALLEL must be an integer; got '$MAX_PARALLEL'" >&2
    exit 1
fi

if (( MAX_PARALLEL < 1 )); then
    MAX_PARALLEL=1
fi

CSV_PATH="${1:-$DEFAULT_CSV}"
if [[ ! -f "$CSV_PATH" ]]; then
    echo "Error: CSV file not found at $CSV_PATH" >&2
    exit 1
fi

CSV_ABS=$(realpath "$CSV_PATH")

# Build task list from the CSV (skip rows marked done/status)
mapfile -t TASKS < <(
python3 - <<'PY' "$CSV_ABS"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
with csv_path.open(newline='') as fh:
    reader = csv.DictReader(fh)
    if reader.fieldnames is None:
        sys.exit(0)
    for row in reader:
        lowered = {}
        for key, value in row.items():
            if key is None:
                continue
            lowered[key.strip().lower()] = (value.strip() if isinstance(value, str) else value)

        status = lowered.get('done') or lowered.get('status')
        if status is not None:
            normalized = str(status).strip().lower()
            if normalized in {'done', 'true', '1', 'yes', 'y', 'complete', 'finished'}:
                continue

        run_name = lowered.get('run_name') or lowered.get('run dir') or lowered.get('run_dir')
        epoch = lowered.get('epoch')
        category = lowered.get('category')
        target = lowered.get('target')

        if not run_name or epoch in (None, '') or not category or not target:
            continue

        print(f"{run_name}|{epoch}|{category}|{target}")
PY
)

TASK_COUNT=${#TASKS[@]}
if (( TASK_COUNT == 0 )); then
    echo "No runnable tasks discovered in $CSV_ABS" >&2
    exit 1
fi

echo "Discovered $TASK_COUNT runnable tasks"

if (( TASK_COUNT < MAX_PARALLEL )); then
    EFFECTIVE_PARALLEL=$TASK_COUNT
else
    EFFECTIVE_PARALLEL=$MAX_PARALLEL
fi

if (( EFFECTIVE_PARALLEL < 1 )); then
    EFFECTIVE_PARALLEL=1
fi

ARRAY_SPEC="0-$((TASK_COUNT-1))"
if (( TASK_COUNT > 1 )); then
    ARRAY_SPEC+="%$EFFECTIVE_PARALLEL"
fi

need_resubmit=false
if (( TASK_COUNT > 1 )) && [[ -z "${FORCE_SERIAL:-}" ]]; then
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        need_resubmit=true
    elif [[ "${SLURM_ARRAY_TASK_COUNT:-1}" -le 1 ]]; then
        need_resubmit=true
    fi
fi

if [[ $need_resubmit == true ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Current submission only has a single array task; re-submitting with --array=$ARRAY_SPEC" >&2
    else
        echo "Submitting array job with --array=$ARRAY_SPEC (max parallel=$EFFECTIVE_PARALLEL)" >&2
    fi
    sbatch --export=ALL,MAX_PARALLEL=$MAX_PARALLEL --array="$ARRAY_SPEC" "$0" "$CSV_ABS"
    status=$?
    if (( status != 0 )); then
        echo "Failed to submit array job" >&2
    fi
    exit $status
fi

run_task() {
    local idx=$1
    IFS='|' read -r RUN_NAME EPOCH CATEGORY TARGET <<< "${TASKS[$idx]}"

    echo "[$(date)] Task $idx starting"
    echo "  CSV: $CSV_ABS"
    echo "  Run: $RUN_NAME"
    echo "  Epoch: $EPOCH"
    echo "  Category: $CATEGORY"
    echo "  Target: $TARGET"

    cd /users/rvinod/data/rvinod/repos/plm_subnetworks/
    PYTHONPATH=/users/rvinod/data/rvinod/repos/plm_subnetworks:$PYTHONPATH
    export PYTHONPATH

    "${CONDA_RUN[@]}" python evaluation/seq_inference_dayhoff.py \
        --run_name "$RUN_NAME" \
        --epoch "$EPOCH" \
        --category "$CATEGORY" \
        --target "$TARGET" \
        "${COMMON_ARGS[@]}"

    echo "[$(date)] Task $idx done"
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "No SLURM_ARRAY_TASK_ID detected; running all $TASK_COUNT tasks sequentially." >&2
    for idx in "${!TASKS[@]}"; do
        run_task "$idx"
    done
    exit 0
fi

if (( SLURM_ARRAY_TASK_ID >= TASK_COUNT )); then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range for $TASK_COUNT tasks; nothing to do." >&2
    exit 0
fi

run_task "$SLURM_ARRAY_TASK_ID"

export PYTHONPATH=/users/rvinod/data/rvinod/repos/plm_subnetworks:$PYTHONPATH
cd /users/rvinod/data/rvinod/repos/plm_subnetworks/evaluation

"${CONDA_RUN[@]}" python evaluation/build_csv.py --csv "$DEFAULT_CSV" 

echo "[$(date)] All done"