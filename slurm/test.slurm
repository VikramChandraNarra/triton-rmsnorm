#!/bin/bash
# =============================================================================
# Correctness gate on Trillium: run the pytest suite on ONE H100.
#
# Submit from your $SCRATCH copy of the repo (output must land on scratch):
#     cd $SCRATCH/triton-rmsnorm
#     sbatch slurm/test.slurm
#
# Single-GPU job == 1/4 node == 24 cores / 188 GiB / 1 H100. The suite is light;
# 30 min is plenty even with first-run Triton autotune/compile.
# =============================================================================
#SBATCH --job-name=rmsnorm-test
#SBATCH --output=%x_%j.out          # %x=job-name %j=jobid; written to $SCRATCH
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1            # single H100 (only 1 or 4 are allowed)
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL
## #SBATCH --account=def-YOURPI      # uncomment+set if you have multiple accounts

set -euo pipefail

# --- modules: pin the same versions used at setup time ----------------------
module purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

# --- activate the venv built on the login node ------------------------------
source "$HOME/venvs/triton-rmsnorm/bin/activate"

# --- redirect every writable cache to $SCRATCH ------------------------------
# $HOME is READ-ONLY on compute nodes. Without these, Triton/torch/matplotlib
# try to write under ~/.cache or ~/.triton and the job dies. Triton's autotune
# cache in particular MUST be writable for the kernel to compile.
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export MPLCONFIGDIR="$SCRATCH/.cache/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.cache/inductor"   # for the torch.compile baseline later
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR"

echo ">> Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# --- run from the repo root so `import kernels` resolves --------------------
cd "$SLURM_SUBMIT_DIR"
echo ">> Running pytest"
pytest -q
echo ">> pytest exit code: $?"
