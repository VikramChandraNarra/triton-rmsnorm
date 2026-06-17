# A fused RMSNorm kernel in Triton, and what its bandwidth number actually means

This is a writeup of a from-scratch Triton kernel for RMSNorm, the normalization
used in Llama, Qwen, Mistral, and Gemma. The forward and backward passes are each
fused into a single kernel launch and wired into PyTorch autograd. I ran it on an
H100 and it lands at about 6x over the eager PyTorch spelling and about 45% of the
card's peak HBM bandwidth. The interesting part of that sentence is the 45%, so
most of this post is about where it comes from and what is left on the table.

Everything here is reproducible. The kernel is one file
(`kernels/rmsnorm.py`), the benchmark is another (`bench/benchmark.py`), and the
numbers below come from two SLURM jobs on a single H100 80GB SXM. Job IDs and node
names are in the results section so you can check the timestamps.

## What RMSNorm is and why fusing both passes matters

RMSNorm takes a row $x \in \mathbb{R}^N$, divides it by its root mean square, and
scales by a learned gain $w$:

$$
\mathrm{rms}(x) = \sqrt{\frac{1}{N}\sum_j x_j^2 + \epsilon}, \qquad
y_i = \frac{x_i}{\mathrm{rms}(x)}\, w_i
$$

There is no mean subtraction and no bias, unlike LayerNorm. So the arithmetic per
element is a couple of multiplies and one reciprocal square root. That is nothing.
The cost is moving $x$ and $y$ through HBM. RMSNorm is memory bound, and for a
memory bound op the only thing that moves the wall clock is how many times you
touch memory.

The eager PyTorch version touches it many times. This line of math

```python
variance = x.pow(2).mean(dim=-1, keepdim=True)
x = x * torch.rsqrt(variance + eps)
return weight * x
```

compiles to a chain of separate CUDA kernels: square, mean, add epsilon, rsqrt,
two multiplies. Each one reads the activation back from HBM and writes a result
out again. On a memory bound op those round trips are the runtime. There is also
fixed launch overhead per kernel, which matters more at small batch sizes where
each kernel is short enough that the launch is a real fraction of its time.

A fused kernel reads each row once, does all the math in registers, and writes the
result once. The speedup you should expect is roughly the number of HBM round
trips you removed. That is a useful prediction to carry into the results, because
if the measured speedup matches it, the win is structural rather than luck.

Fusing the backward matters for the same reason and adds one more. The backward
needs the normalization scale, and if you do not save it from the forward you have
to recompute the reduction over the row. Saving one float per row during the
forward and reading it back in the backward removes that recompute entirely. The
cost is a tiny extra write, $O(M)$, which is rounding error next to the $O(MN)$
activation traffic.

## Design decisions

### One program per row, single launch

Each Triton program owns one row and keeps the whole row resident in
registers and SRAM. The forward reads `x` once and writes `y` once, with no second
pass over the data. The tile width is fixed per call to `next_power_of_2(N)` so a
single tile spans the row, and the tail is masked when `N` is not a power of two.

```python
row = tl.program_id(0)
cols = tl.arange(0, BLOCK_SIZE)          # BLOCK_SIZE = next_power_of_2(N)
mask = cols < N

x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
mean_sq = tl.sum(x * x, axis=0) / N
rstd = tl.math.rsqrt(mean_sq + eps)
tl.store(Rstd + row, rstd)                # saved for the backward

w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
y = x * rstd * w
tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
```

The tradeoff is the upper bound on `N`. The whole row has to fit in one tile, so
this design caps the hidden dimension. I guard it at 65536 and raise rather than
silently produce garbage. Real hidden dims (2048 to 8192) are comfortably inside
that, but it is a real limit, and a model with a very wide last dimension would
need a multi pass variant. I would rather state the limit than pretend the kernel
is general.

### fp32 reductions regardless of storage dtype

Every reduction accumulates in fp32 even when `x` is fp16 or bf16. The kernel
upcasts on load and only rounds back down at the final store. The sum of `N`
squares is exactly where low precision bites, because you are adding `N` positive
numbers and the running total grows while the increments stay small. bf16 has 8
mantissa bits, so a large running sum stops being able to represent small
additions, and the error is systematic rather than averaging out.

Doing the reduction in fp32 costs almost nothing here because the op is memory
bound, not compute bound. The extra fp32 registers are not the bottleneck, the
memory bus is. So this is a free correctness win in this regime. It would be a real
cost in a compute bound kernel, where the extra register pressure could lower
occupancy, but that is not the situation.

### Atomic free weight gradient

The two gradients, with $\hat{x}_i = x_i\,\mathrm{rstd}$ and
$\mathrm{dyw}_i = dy_i\, w_i$:

$$
dx_i = \mathrm{rstd}\left(\mathrm{dyw}_i - \hat{x}_i \cdot \tfrac{1}{N}\textstyle\sum_j \mathrm{dyw}_j\,\hat{x}_j\right), \qquad
dw_i = \sum_{\text{rows}} dy_i\,\hat{x}_i
$$

`dx` is a per row reduction, so it parallelizes one program per row with no
coordination. `dw` is the awkward one. It sums one contribution from every row into
a single vector of length `N`, which is a reduction across the row axis. The
obvious way to write that is an atomic add into a shared `dw` buffer, and it works,
but atomics on the same addresses from many programs serialize and the contention
gets worse as you add parallelism.

Instead, each program gets its own private accumulator of length `N` and walks a
contiguous chunk of rows with a grid stride loop, folding each row into its private
buffer. No two programs ever write the same address. A small `torch.sum` at the end
reduces the `[n_programs, N]` partial buffer down to `[N]`.

```python
dw_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
for row in range(row_start, row_end):     # this program's chunk of rows
    x  = tl.load(X  + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + row)
    xhat = x * rstd
    # ... dx computed and stored here, reusing the same x and dy loads ...
    dw_acc += dy * xhat                    # private, no atomics
tl.store(DW_partial + pid * N + cols, dw_acc, mask=mask)
```

The number of programs is capped at the SM count, so the partial buffer stays
small, a few hundred rows of `N` floats, and each program strides over
`ceil(M / n_programs)` rows. The tradeoff is the extra `[n_programs, N]` buffer and
a second small kernel for the final sum. That memory is negligible and the final
sum is cheap, so trading it for zero atomic contention is the right call. The other
quiet win in that loop is that `dx` and `dw` share the same `x` and `dy` loads, so
the backward reads each of them once even though it produces two gradients.

### Autotuning

The kernel is autotuned per `N` over `num_warps` in {1, 2, 4, 8, 16, 32} and
`num_stages` in {1, 2, 4}. For a streaming kernel those are the two knobs that
matter: how many warps cooperate on one row, which sets how much memory level
parallelism you get, and how deep the load pipeline is. `BLOCK_SIZE` is
deliberately not in the autotune space, because correctness requires
`BLOCK_SIZE >= N`, so it is pinned rather than searched. This is also the honest
caveat on the autotuning: the search space is small. It covers the knobs I was
confident mattered and not much else, which is part of why there is bandwidth left
on the table (more on that at the end).

The whole thing sits behind a `torch.autograd.Function` that flattens any
`[..., N]` input to `[M, N]`, since RMSNorm only ever acts on the last dimension,
and reshapes back on the way out. So `rmsnorm(x, w)` is a drop in differentiable
op.

## How I measured

The headline metric is not FLOP/s. For a memory bound op FLOP/s is misleading, the
number that means something is effective bandwidth as a fraction of what the card
can deliver.

Timing goes through `triton.testing.do_bench`, which warms up, runs many
repetitions, returns the median, and discards the cold launches. That keeps
autotuning and one time compilation out of the measurement, which otherwise would
dominate the first few calls.

For bandwidth I count only the activation traffic the kernel is obliged to move.
Forward is read `x` plus write `y`, so $2MN$ elements. Backward is read `x`, read
`dy`, write `dx`, so $3MN$. The weight, the saved `rstd`, and the partial buffer
are all $O(M + N)$, negligible next to $2MN$, so I leave them out. That is the
honest denominator for a percent of peak claim, since padding the byte count with
traffic the op does not really need would inflate the number.

The arithmetic is plain enough to check by hand. For bf16 at M=4096, N=8192,
forward:

```
bytes  = 2 * 4096 * 8192 * 2          = 134,217,728 bytes  = 0.1342 GB
time   = 0.0893 ms                    = 8.93e-5 s
gbps   = 0.1342 GB / 8.93e-5 s        = 1503 GB/s
% peak = 1503 / 3350                  = 44.9%
```

The peak is supplied by an environment variable, 3350 GB/s for the H100 SXM HBM3
part, 2039 for an A100, so nothing about the device is hardcoded.

### Correctness

I do not bit match against an eager fp16 implementation, because two correct fp16
kernels can differ in the last bit just from rounding order, and matching one
particular spelling would be testing the wrong thing. Instead I compare against an
fp32 oracle: the value computed end to end in fp32 and only rounded to fp16 or
bf16 at the very end. That gives an honest claim, "within X of the true value,"
rather than "identical to some other code I wrote."

Tolerances are set per dtype against that oracle: fp16 at atol 1e-3, rtol 1e-2, and
bf16 looser at atol 1e-2, rtol 2e-2, since bf16 has fewer mantissa bits. The
backward is checked the same way, analytic Triton gradients against autograd
gradients of the oracle. gradcheck wants float64, which Triton does not support, so
the fp32 oracle comparison is the right tool.

The sweep covers hidden dims {2048, 4096, 8192} crossed with row counts {512, 1024,
2048, 4096} crossed with {fp16, bf16}, plus non power of two and 3D shapes, plus
the gradient checks. On the H100 the full suite is 40 passed (SLURM job 604545,
node trig0006, 1m57s). The shapes in the benchmark are the same shapes, so the perf
numbers describe the exact kernels that passed correctness.

## Results

Run on one NVIDIA H100 80GB SXM, HBM3, 3.35 TB/s peak. Benchmark SLURM job 604586,
node trig0024, 1m12s. Forward pass at the largest shape in the sweep, M=4096,
N=8192:

| dtype | fused     | eager     | speedup | GB/s   | % of peak |
| ----- | --------- | --------- | ------- | ------ | --------- |
| bf16  | 0.0893 ms | 0.5322 ms | 5.96x   | 1503.4 | 44.9%     |
| fp16  | 0.0970 ms | 0.5319 ms | 5.48x   | 1384.1 | 41.3%     |

The full grid across every M, N, dtype, and both directions is in the generated
CSV. These are the headline rows.

The 6x is the part that matches the prediction. Eager is roughly six kernels each
re-reading the activation, the fused kernel reads once and writes once, and the
measured speedup lining up with the number of round trips removed is the
reassuring sign that the win is structural and not a measurement artifact.

The 45% is the number worth being precise about. Being bandwidth bound is the good
news in it: the kernel is not stalling on launches and not spending its time on
arithmetic, it is limited by the memory bus, which is the correct place to be
stuck for an op like this. But 45% is not the ceiling. A well tuned pure streaming
kernel on H100 reaches roughly 70 to 80% of peak, so there is real room between
where this is and where a memory bound kernel can get.

Two honesty notes on the number itself. First, 1503 GB/s is an effective
bandwidth from a model, bytes I counted divided by time, not a hardware counter. It
is a fair model because the byte count is the true minimum traffic, but it is not
the same thing as the DRAM throughput counter, and the first thing I would do to
push further is reconcile the two. Second, bf16 edges fp16, 1503 against 1384 GB/s
at the same byte count, which is a conversion path difference and not anything
algorithmic.

So the summary I would actually stand behind: correct across the sweep, a clean 6x
over eager, bandwidth bound at about 45% of peak, with roughly half the bandwidth
budget still unclaimed.

## What I would do to push toward peak

I have a set of hypotheses for the gap, not a fixed answer, because I have not put
this under Nsight Compute yet. The honest first step is to profile, then chase the
biggest item. In rough order of what I would try:

Profile first. Pull `dram__bytes` and the achieved DRAM throughput from ncu and
compare against the 1503 GB/s model number. If they disagree the kernel is moving
more traffic than the minimum, for example reloading something or not coalescing,
and that is a different fix than if they agree and the kernel is simply latency
bound.

Reconsider one program per row at large N. At N=8192 a bf16 row is 16KB. A single
program streaming 16KB may not have enough memory requests in flight to hide HBM
latency, even with several warps. I would try a layout that puts more rows or more
of the machine on the same memory stream, for example multiple rows per program or
a 2D tiling, so more warps are issuing loads concurrently and the latency is
hidden by parallelism rather than by pipeline depth alone.

Check vectorization. I want to confirm Triton is emitting 128 bit loads, the
`ld.global.v4` form, on the activation reads. The masked load for non power of two
`N` can block vectorization, but for the power of two shapes it should vectorize,
and if it is not that is throughput left on the floor.

Widen the autotune space. Right now it only sweeps `num_warps` and `num_stages`.
Adding the rows per program or tile shape as a tuned dimension, once the layout
above supports it, would let the search find the configuration that actually
saturates the bus per shape rather than per the two knobs I picked.

Fold the backward's final reduction. The `dw` path ends with a separate
`torch.sum` over the `[n_programs, N]` partial buffer, which is a second small
launch. A tree reduction inside the kernel, or a second tiny fused pass, would
remove that launch. It is small, so this is a tidy up rather than the main lever,
but it is on the list.

None of these are guaranteed wins, which is the point of profiling first. The gap
from 45% to the 70 to 80% a memory bound kernel can reach is real and I can see the
candidate causes, and the next stretch of work is measuring which one is actually
costing the bandwidth.
