# fusedkernel — Fused Triton RMSNorm (forward + backward)

A single-file, autotuned [Triton](https://github.com/openai/triton) kernel for
**RMSNorm** (Zhang & Sennrich, 2019) — the normalization used by Llama, Qwen,
Mistral, and Gemma — with both the forward and backward passes wired into
PyTorch autograd. Validated for correctness and benchmarked end-to-end on an
**NVIDIA H100 80GB SXM (HBM3)**.

**Headline (H100, forward, M=4096, N=8192):**

| dtype | fused latency | speedup vs eager PyTorch | effective bandwidth | % of H100 peak (3.35 TB/s) |
|-------|--------------:|-------------------------:|--------------------:|---------------------------:|
| bf16  | 0.0893 ms     | **5.96×**                | 1503.4 GB/s         | 44.9%                      |
| fp16  | 0.0970 ms     | **5.48×**                | 1384.1 GB/s         | 41.3%                      |

---

## 1. What we are trying to do

RMSNorm normalizes a row vector `x ∈ ℝ^N` by its root-mean-square and rescales it
by a learned gain `w ∈ ℝ^N`:

```
rms(x) = sqrt( mean_j(x_j²) + eps )
y_i    = (x_i / rms(x)) · w_i
```

Unlike LayerNorm there is **no mean-subtraction and no bias** — it is purely a
rescaling by the root-mean-square. The arithmetic is trivial: a couple of
multiplies and a single `rsqrt` per element. That makes RMSNorm a textbook
**memory-bound** operation. The cost is dominated entirely by moving `x` and `y`
through HBM, not by the math.

The eager PyTorch spelling makes this worse than it needs to be:

```python
variance = x.pow(2).mean(dim=-1, keepdim=True)   # square → mean
x = x * torch.rsqrt(variance + eps)              # add → rsqrt → mul
return weight * x                                # mul
```

Each step launches a **separate CUDA kernel**, and each kernel round-trips the
full activation through memory. For a memory-bound op, the number of HBM
round-trips *is* the runtime.

**The goal:** fuse the entire sequence into one kernel so each row is **read once
and written once**, and verify — on real hardware — both that it is numerically
correct and that it approaches the device's memory-bandwidth ceiling.

## 2. How we did it

The kernel lives in [`kernels/rmsnorm.py`](kernels/rmsnorm.py). The design, point
by point:

### One Triton program per row
The entire row fits in SRAM/registers, so the forward pass reads `x` exactly once
and writes `y` exactly once — there is no second read. `BLOCK_SIZE` is fixed per
call to `next_power_of_2(N)` so a single tile spans a full row; the tail is masked
when `N` is not a power of two. We guard `N ≤ 65536` so a too-wide row fails loud
rather than silently corrupting (a row that big would need a multi-pass variant).

### fp32 reductions regardless of storage dtype
Every reduction accumulates in fp32 even when `x` is fp16/bf16. The sum of `N`
squares is exactly where low precision bites, so we **upcast on load** and only
**downcast at the store boundary**. This is what lets the kernel hold a tight
tolerance against an fp32 oracle (see §3).

### `rstd` is cached for the backward pass
The forward pass saves `rstd = 1/rms` (one fp32 per row — negligible traffic) and
hands it to the backward pass, so the reduction is **never recomputed** during the
gradient.

### Lock-free backward `dw` — no atomics
The gradients (all reductions over the `N` columns of a row):

```
xhat_i = x_i · rstd                 (normalized activation)
dyw_i  = dy_i · w_i                 (upstream grad folded with the gain)

dx_i   = rstd · ( dyw_i − xhat_i · mean_j(dyw_j · xhat_j) )
dw_i   = Σ_over_rows( dy_i · xhat_i )         (a reduction across the M axis)
```

`dx` is a per-row reduction — embarrassingly parallel, one program per row. `dw`
is the awkward one: it sums one contribution per row into a single `[N]` vector.
Instead of atomics, **each program gets its own private `[N]` accumulator** and
uses a **grid-stride loop** to fold many rows into it with zero cross-program
contention. A final small `torch.sum` reduces the `[n_programs, N]` partial buffer
to `[N]`. The program count is capped at the device SM count, so the partial
buffer stays small while still saturating the machine.

### Autotuned per-`N`
`@triton.autotune(key=["N"])` sweeps `num_warps ∈ {1,2,4,8,16,32}` and
`num_stages ∈ {1,2,4}`. For a memory-bound kernel, `num_warps` (how many threads
cooperate to stream one row) and `num_stages` (software pipelining of the global
loads) are the knobs that actually move the needle. `BLOCK_SIZE` is deliberately
*not* autotuned — correctness requires `BLOCK_SIZE ≥ N`, so it is pinned.

### PyTorch integration
A `torch.autograd.Function` flattens any `[..., N]` input to `[M, N]` (RMSNorm
always acts on the last dim, so `[B, S, H]` and `[B·S, H]` are the same problem),
runs the kernels, and reshapes back. `rmsnorm(x, w, eps)` is a drop-in,
fully differentiable op.

## 3. How we verified correctness

Tests live in [`tests/test_rmsnorm.py`](tests/test_rmsnorm.py). The methodology is
deliberate:

- **We do *not* bit-match a same-dtype eager implementation.** Two correct fp16
  kernels legitimately differ in their last bit depending on rounding order.
  Instead we compare against an **fp32 oracle** — the mathematically true value,
  computed end-to-end in fp32 and rounded to the test dtype only at the very end —
  and assert closeness with dtype-appropriate tolerances (tighter for fp16, looser
  for bf16). This is the honest way to make a precision claim: *"within X of the
  true value,"* not *"identical to one particular spelling."*
- **Backward is checked the same way:** analytic Triton gradients vs autograd
  gradients of the fp32 oracle.
- **Coverage:** forward across `{2048, 4096, 8192}` hidden dims ×
  `{512, 1024, 2048, 4096}` rows × `{fp16, bf16}`, plus non-power-of-two and 3D
  shapes, and analytic-vs-autograd gradient checks.

**Result on H100:** `40 passed` (Slurm job `604545`, node `trig0006`, 00:01:57,
exit `0:0`).

## 4. How we benchmarked

The sweep lives in [`bench/benchmark.py`](bench/benchmark.py). Because RMSNorm is
memory-bound, the headline metric is **not** raw FLOP/s — it is **effective HBM
bandwidth as a fraction of the device peak**.

- **Timing** uses `triton.testing.do_bench`, which warms up, runs many reps, and
  returns the median — it handles CUDA stream sync and discards cold launches, so
  autotuning and one-time compilation never pollute the measurement.
- **Bytes moved** counts the dominant activation traffic: forward = read `x` +
  write `y` = `2·M·N·elem_size`; backward = read `x` + read `dy` + write `dx` =
  `3·M·N·elem_size`. The `O(M+N)` terms (`w`, `rstd`, `dw` partials) are
  negligible and excluded — that is the honest denominator for "% of peak."
- **`% of peak`** divides effective bandwidth by the device peak, supplied via
  `RMSNORM_PEAK_GBPS` (3350 for H100 SXM HBM3; 2039 for A100 80GB). Nothing is
  hard-coded.
- **Same shapes as the correctness suite**, so a green test run and the perf
  numbers describe the exact same kernels.

Output: `bench/results/rmsnorm_benchmark.csv` (columns: `direction, dtype, M, N,
triton_ms, torch_ms, speedup, gbps, pct_peak`) plus per-dtype latency and
%-of-peak plots.

## 5. Results (NVIDIA H100 80GB SXM, HBM3)

Benchmark Slurm job `604586`, node `trig0024`, runtime 00:01:12, exit `0:0`.
Peak bandwidth denominator: **3.35 TB/s** (H100 SXM HBM3).

**Forward, the largest sweep point (M=4096, N=8192):**

| dtype | fused (`triton_ms`) | eager (`torch_ms`) | speedup | GB/s   | % of peak |
|-------|--------------------:|-------------------:|--------:|-------:|----------:|
| bf16  | 0.0893 ms           | 0.5322 ms          | 5.96×   | 1503.4 | 44.9%     |
| fp16  | 0.0970 ms           | 0.5319 ms          | 5.48×   | 1384.1 | 41.3%     |

(`torch_ms` shown for the headline point is implied by `speedup × triton_ms`; the
full sweep across all `M × N × dtype × {forward, backward}` is in the generated
CSV.)

### What the numbers mean

- **The ~6× speedup is the result the design predicts.** Eager RMSNorm is six
  kernels, each re-streaming the activation through HBM; the fused kernel reads
  once and writes once. The speedup ≈ the number of HBM round-trips eliminated.
  The benchmark confirms the fusion does exactly what it was built to do.
- **Bandwidth is the honest efficiency score.** At ~45% of the H100's 3.35 TB/s
  ceiling, the kernel is genuinely **streaming-limited** — not launch-bound or
  compute-bound — which is the right regime for a memory-bound op. It is *not yet
  saturating the bus*: well-tuned pure-streaming kernels on H100 reach ~70–80% of
  peak, so there is real, identifiable headroom (tile/vectorization width, the
  per-row program scheme, `num_warps`/`num_stages` at large `N`).
- **bf16 edges out fp16** (1503 vs 1384 GB/s) at identical byte counts — a
  microarchitectural/conversion-path difference, not an algorithmic one.

**Honest bottom line:** correct and shippable, a clean ~6× over eager, ~45% of
peak bandwidth — a solid, defensible result with a clearly understood path to
push utilization higher.

## 6. Usage

```python
import torch
from kernels import rmsnorm

x = torch.randn(4096, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.ones(8192, device="cuda", dtype=torch.bfloat16)

y = rmsnorm(x, w, eps=1e-6)   # differentiable: y.sum().backward() works
```

`kernels.rmsnorm_reference` provides the fp32-oracle reference used by the tests
(matched to Hugging Face's `LlamaRMSNorm`).

## 7. Repository layout

```
kernels/rmsnorm.py      the fused forward + backward Triton kernel + autograd glue
tests/test_rmsnorm.py   correctness + gradient tests vs an fp32 oracle
bench/benchmark.py      latency + effective-bandwidth sweep
bench/results/          generated CSV + plots
slurm/                  SLURM scripts for the Trillium H100 cluster
RUN.md                  generic cloud-GPU (Lambda/RunPod) run guide
TRILLIUM.md             SciNet Trillium H100 cluster run guide
```

## 8. Requirements

A single NVIDIA GPU (A100 80GB or H100), CUDA 12.x, Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 9. Testing

```bash
pytest -q
```

The suite skips cleanly on a CPU-only machine (so collection is green
everywhere) and only *runs* on a CUDA box. The first run is slower while Triton
autotunes and compiles each `(N, kernel)`; results cache under `.triton/`.

## 10. Running on a GPU

- **Generic cloud GPU** (Lambda / RunPod / etc.): see [`RUN.md`](RUN.md).
- **Trillium (SciNet H100 cluster):** see [`TRILLIUM.md`](TRILLIUM.md) — it handles
  the no-internet compute nodes, read-only `$HOME`, SLURM scripts, and the H100's
  3.35 TB/s bandwidth denominator.

## License

See [`LICENSE`](LICENSE).
