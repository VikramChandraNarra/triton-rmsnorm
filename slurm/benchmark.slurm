#!/bin/bash
# =============================================================================
# Benchmark sweep on Trillium: latency + effective bandwidth on ONE H100.
#
#     cd $SCRATCH/triton-rmsnorm
#     sbatch slurm/benchmark.slurm
#
# NOTE: bench/benchmark.py is added AFTER the correctness gate passes. This
# script is staged and ready so there is zero scramble once tests are green.
# Single GPU is the right choice -- the kernel is per-row, so one H100 gives a
# clean, contention-free measurement.
# =============================================================================
#SBATCH --job-name=rmsnorm-bench
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
## #SBATCH --account=def-YOURPI

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

source "$HOME/venvs/triton-rmsnorm/bin/activate"

export TRITON_CACHE_DIR="$SCRATCH/.cache/triton"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export MPLCONFIGDIR="$SCRATCH/.cache/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.cache/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR"

# H100 SXM HBM3 theoretical peak = 3.35 TB/s. The benchmark reports effective
# bandwidth as a % of this. (A100 80GB would be 2039 GB/s.) benchmark.py reads
# this env var so the device peak isn't hard-coded.
export RMSNORM_PEAK_GBPS=3350

echo ">> Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "$SLURM_SUBMIT_DIR"
echo ">> Running benchmark sweep -> bench/results/"
python bench/benchmark.py --out bench/results
echo ">> Done. CSV + plots are under bench/results/ (copy them back to inspect)."
