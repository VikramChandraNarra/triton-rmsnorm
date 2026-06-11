#!/bin/bash
# =============================================================================
# One-time environment setup for Trillium (SciNet H100 cluster).
#
# RUN THIS ON THE GPU LOGIN NODE, NOT INSIDE A JOB:
#     ssh MYUSER@trillium-gpu.scinet.utoronto.ca
#     bash slurm/setup_trillium.sh
#
# Why on the login node: Trillium compute nodes have NO internet access, so
# `pip install` cannot run there. We build the virtualenv once here (login
# nodes have internet) and the job scripts just `source` it. The venv lives in
# $HOME so it survives $SCRATCH purges; $HOME is read-only on compute nodes but
# importing from it is fine -- we redirect all *writable* caches to $SCRATCH in
# the job scripts.
#
# Packages come from the Alliance wheelhouse via `--no-index` (their prebuilt,
# cluster-tuned torch/triton), not PyPI. requirements.txt is only for generic
# cloud GPU boxes.
# =============================================================================
set -euo pipefail

VENV="${VENV:-$HOME/venvs/triton-rmsnorm}"

echo ">> Loading modules (pin versions for reproducibility)"
module purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

echo ">> Creating virtualenv at $VENV"
# --no-download builds the venv from the cluster's local copies, no internet.
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"

echo ">> Upgrading pip from the wheelhouse"
pip install --no-index --upgrade pip

echo ">> Checking which wheels the cluster actually has"
# avail_wheels is an Alliance helper; informational, never fails the script.
avail_wheels torch triton numpy pandas matplotlib pytest 2>/dev/null || true

echo ">> Installing dependencies from the Alliance wheelhouse"
# torch on the Alliance linux wheels bundles a matching Triton; we still ask for
# triton explicitly so the import is guaranteed. If `triton` is not a separate
# wheel on this cluster, the torch-bundled one is already present and this line
# is a harmless no-op-ish resolve.
pip install --no-index torch numpy pandas matplotlib pytest
pip install --no-index triton || echo "   (triton not a standalone wheel; using the torch-bundled Triton)"

echo ">> Sanity check: versions + GPU visibility from the login node"
python - <<'PY'
import torch
print("torch   :", torch.__version__)
try:
    import triton
    print("triton  :", triton.__version__)
except Exception as e:
    print("triton  : IMPORT FAILED ->", e)
print("cuda ok :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device  :", torch.cuda.get_device_name(0))
PY

echo
echo ">> Done. Venv: $VENV"
echo ">> Next: clone the repo into \$SCRATCH and submit slurm/test.slurm from there:"
echo "      cd \$SCRATCH && git clone <repo-url> triton-rmsnorm && cd triton-rmsnorm"
echo "      sbatch slurm/test.slurm"
