# Running on Trillium (SciNet H100 cluster)

Trillium-specific instructions for the Digital Research Alliance of Canada
cluster. Each GPU node has **4× NVIDIA H100 SXM 80GB** (HBM3, **3.35 TB/s** peak
bandwidth). For the generic cloud-GPU path (Lambda/RunPod) see `RUN.md` instead.

Three Trillium facts drive everything below:

1. **Compute nodes have no internet.** All package installs happen on the login
   node, once, into a venv the jobs reuse.
2. **`$HOME` and `$PROJECT` are read-only on compute nodes.** Every writable
   cache (Triton autotune, torch.compile inductor, matplotlib) is redirected to
   `$SCRATCH` inside the job scripts — otherwise the kernel can't even compile.
3. **Whole-GPU scheduling.** You request exactly 1 GPU (¼ node: 24 cores,
   188 GiB) or a multiple of 4. We use 1 — the kernel is per-row, so a single
   H100 gives a clean, contention-free measurement.

---

## 1. Log in to the GPU login node
SSH keys must be uploaded to CCDB first (Trillium allows key auth only).
```bash
ssh -i /PATH/TO/SSH_PRIVATE_KEY MYUSER@trillium-gpu.scinet.utoronto.ca
```

## 2. One-time environment setup (on the login node)
```bash
cd $SCRATCH
git clone https://github.com/VikramChandraNarra/triton-rmsnorm.git
cd triton-rmsnorm
bash slurm/setup_trillium.sh
```
This builds `~/venvs/triton-rmsnorm` from the Alliance wheelhouse
(`pip install --no-index`) and prints a torch/triton/GPU sanity check. The venv
lives in `$HOME` so it survives `$SCRATCH` purges; the repo lives in `$SCRATCH`
because **jobs must be submitted from, and write output to, scratch.**

## 3. Run the correctness gate
Submit the batch job (from the `$SCRATCH` repo copy):
```bash
sbatch slurm/test.slurm
squeue -u $USER            # watch it
# output appears as rmsnorm-test_<jobid>.out in the current dir
```

**Faster, interactive alternative** — grab an H100 for 2 hours and run pytest
live (great for a first sanity pass):
```bash
debugjob -g 1                                  # interactive 1-GPU shell, 2h
module load StdEnv/2023 python/3.11.5 cuda/12.6
source ~/venvs/triton-rmsnorm/bin/activate
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton  # writable cache (see fact #2)
export XDG_CACHE_HOME=$SCRATCH/.cache
mkdir -p "$TRITON_CACHE_DIR"
cd $SCRATCH/triton-rmsnorm && pytest -q
```
> `debugjob` has no internet and read-only `$HOME` too — but the venv is already
> built and caches point at scratch, so it just works.

Expect every parametrization to pass: forward across {2048,4096,8192} hidden
dims × {512,1024,2048,4096} rows × {fp16,bf16}, the non-power-of-two and 3D
cases, and the gradient checks. **Send me the output before we benchmark — no
perf numbers until this is green.**

## 4. Benchmark (only after step 3 passes)
```bash
sbatch slurm/benchmark.slurm     # reports bandwidth as % of the H100's 3.35 TB/s
```
(`bench/benchmark.py` is added once correctness is confirmed; the SLURM script
is already staged.)

---

## Cluster quick reference
| Need | Command |
|------|---------|
| GPU login node | `ssh MYUSER@trillium-gpu.scinet.utoronto.ca` |
| Interactive 1× H100 (2h) | `debugjob -g 1` |
| Submit a job | `sbatch slurm/test.slurm` |
| Watch your queue | `squeue -u $USER` |
| Live job CPU/mem | `jobperf <JOBID>` |
| Cancel | `scancel <JOBID>` |
| Storage quotas | `diskusage_report` |
| GPU status (on node) | `nvidia-smi` |

**Account:** for most users SLURM picks the account automatically. If you have
multiple allocations, uncomment and set `#SBATCH --account=def-YOURPI` in the
job scripts.

**Don't** set `--mem` (ignored — you always get the node/GPU's full memory) and
**don't** request 2 or 3 GPUs (only 1 or 4 are valid).
