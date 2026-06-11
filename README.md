# fusedkernel — Fused Triton RMSNorm

A single-file, autotuned [Triton](https://github.com/openai/triton) kernel for
**RMSNorm** (Zhang & Sennrich, 2019) — the normalization used by Llama, Qwen,
Mistral, and Gemma — with both forward and backward passes wired into PyTorch
autograd.

RMSNorm is a textbook *memory-bound* op: the arithmetic (a few multiplies and one
`rsqrt` per element) is trivial next to the cost of streaming the activation
through HBM. The eager PyTorch spelling
(`square → mean → add → rsqrt → mul → mul`) launches a separate kernel for each
step, round-tripping the activation through memory every time. This kernel fuses
the whole sequence so each row is **read once and written once**.

## Design

- **One program per row.** The entire row lives in SRAM/registers, so the
  forward pass reads `x` exactly once and writes `y` exactly once.
- **fp32 reductions.** All sums accumulate in fp32 even when `x` is fp16/bf16 —
  the sum of `N` squares is where low precision actually bites. We upcast on load
  and only downcast the final stored result.
- **`rstd` is cached** by the forward pass and handed to the backward, so the
  reduction is never recomputed.
- **Backward `dw` uses no atomics** — lock-free per-program partial buffers plus a
  grid-stride loop, finished with a small `torch.sum`.
- **Autotuned per-`N`** over `num_warps` and `num_stages`; `BLOCK_SIZE` is fixed to
  `next_power_of_2(N)` so the row fits in one tile.

## Usage

```python
import torch
from kernels import rmsnorm

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.ones(4096, device="cuda", dtype=torch.bfloat16)

y = rmsnorm(x, w, eps=1e-6)   # differentiable: y.sum().backward() works
```

`kernels.rmsnorm_reference` provides the fp32-oracle reference used by the tests.

## Layout

```
kernels/rmsnorm.py     the fused forward + backward Triton kernel
tests/test_rmsnorm.py  correctness + gradient tests vs an fp32 oracle
bench/                 benchmark outputs (results/)
slurm/                 SLURM scripts for the Trillium H100 cluster
```

## Requirements

A single NVIDIA GPU (A100 80GB or H100), CUDA 12.x, Python 3.10+. Install with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> This kernel cannot run without an NVIDIA GPU; it has been authored but not yet
> executed on hardware. `pytest -q` is the correctness gate.

## Testing

```bash
pytest -q
```

Tests compare against an fp32 oracle (not a bit-match against one eager spelling)
across `{2048, 4096, 8192} × {512, 1024, 2048, 4096} × {fp16, bf16}`, plus
non-power-of-two and 3D shapes, and analytic-vs-autograd gradient checks. The
first run is slower while Triton autotunes and compiles each `(N, kernel)`;
results cache under `.triton/`.

## Running on a GPU

- **Generic cloud GPU** (Lambda / RunPod / etc.): see [`RUN.md`](RUN.md).
- **Trillium (SciNet H100 cluster):** see [`TRILLIUM.md`](TRILLIUM.md) — it handles
  the no-internet compute nodes, read-only `$HOME`, SLURM scripts, and the H100's
  3.35 TB/s bandwidth denominator.

## License

See [`LICENSE`](LICENSE).
