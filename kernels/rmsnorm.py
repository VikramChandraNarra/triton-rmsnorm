"""
Fused RMSNorm (forward + backward) in Triton.

RMSNorm (Zhang & Sennrich, 2019) is the normalization used by Llama, Qwen,
Mistral, Gemma, etc. For a row vector x in R^N and a learned gain w in R^N:

    rms(x) = sqrt( mean_j(x_j^2) + eps )
    y_i    = (x_i / rms(x)) * w_i

Unlike LayerNorm there is no mean-subtraction and no bias: it is purely a
re-scaling by the root-mean-square. That makes it a textbook *memory-bound*
op -- the arithmetic (a couple of multiplies and one rsqrt per element) is
trivial next to the cost of moving x and y through HBM. The whole game is
therefore to touch HBM the minimum number of times, which is exactly what a
fused kernel buys us over the eager PyTorch sequence
(square -> mean -> add -> rsqrt -> mul -> mul), each step of which is its own
kernel launch round-tripping the activation through memory.

Design summary (defensible line by line):
  * One Triton program == one row. The entire row lives in SRAM/registers,
    so we read x exactly once and write y exactly once. Forward is a single
    pass; there is no second read of x.
  * All reductions accumulate in fp32 even when x is fp16/bf16. The sum of
    N squares is where low precision actually bites, so we upcast on load and
    only downcast the final stored result. This is what lets us hold a tight
    tolerance against the reference.
  * rstd = 1/rms is saved by the forward pass and handed to the backward pass,
    so we never recompute the reduction during the backward.
  * Backward dx is one-program-per-row (embarrassingly parallel). Backward dw
    is a cross-row reduction, which we do with lock-free per-program partial
    buffers + a grid-stride loop, then a final small torch.sum. No atomics.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configuration
# ---------------------------------------------------------------------------
# RMSNorm is memory-bound, so the knob that actually moves the needle is the
# number of warps per program: it sets how many threads cooperate to stream
# one row and run the in-SRAM reduction. Too few warps under-utilizes memory
# parallelism on wide rows; too many wastes occupancy on narrow rows. We let
# the autotuner pick per-N (key=['N']) and also sweep num_stages, which
# controls software pipelining of the global loads.
#
# BLOCK_SIZE is deliberately *not* in here -- it is fixed per call to
# next_power_of_2(N) so the whole row fits in one tile. Autotuning it would be
# meaningless because correctness requires BLOCK_SIZE >= N.
def _autotune_configs():
    configs = []
    for num_warps in (1, 2, 4, 8, 16, 32):
        for num_stages in (1, 2, 4):
            configs.append(triton.Config({}, num_warps=num_warps, num_stages=num_stages))
    return configs


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@triton.autotune(configs=_autotune_configs(), key=["N"])
@triton.jit
def _rmsnorm_fwd_kernel(
    X,            # *Pointer* to input   [M, N]
    W,            # *Pointer* to weight  [N]
    Y,            # *Pointer* to output  [M, N]
    Rstd,         # *Pointer* to saved 1/rms [M]
    stride_xm,    # row stride of X (elements, not bytes)
    stride_ym,    # row stride of Y
    N,            # number of columns (hidden dim)
    eps,          # epsilon inside the sqrt
    BLOCK_SIZE: tl.constexpr,  # = next_power_of_2(N); one tile spans a full row
):
    # Each program owns exactly one row. program_id(0) in [0, M).
    row = tl.program_id(0)

    # Advance the row pointers. We index columns with a single arange tile and
    # mask off the tail when N is not a power of two.
    X += row * stride_xm
    Y += row * stride_ym
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Single read of x from HBM. `other=0.0` makes masked lanes contribute 0 to
    # the sum of squares. Upcast to fp32 immediately: every reduction below is
    # done in fp32 regardless of the storage dtype.
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    # mean of squares -> rms -> reciprocal. Dividing the sum by N (a python int
    # promoted to fp32) keeps this in fp32. tl.math.rsqrt is the single
    # transcendental in the whole forward pass.
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = tl.math.rsqrt(mean_sq + eps)

    # Stash rstd for the backward pass (one fp32 per row -- negligible traffic).
    tl.store(Rstd + row, rstd)

    # Load the gain, scale, and write y exactly once. We cast the result back to
    # the output's element type only at the store boundary.
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------
# Gradient derivation (all reductions are over the N columns of a single row):
#
#   Let xhat_i = x_i * rstd                      (the normalized activation)
#       dyw_i  = dy_i * w_i                       (upstream grad folded with gain)
#
#   dx_i = rstd * ( dyw_i - xhat_i * mean_j(dyw_j * xhat_j) )
#   dw_i = sum_over_rows( dy_i * xhat_i )         (reduction across the M axis)
#
# The dx term is a per-row reduction (mean of dyw*xhat) and is trivially
# parallel across rows. dw is the awkward one: it sums one contribution per row
# into a single [N] vector. We avoid atomics by giving each *program* its own
# private [N] partial accumulator and using a grid-stride loop so a program
# folds many rows into its private buffer with zero cross-program contention.
# A final torch.sum reduces the small [n_programs, N] partial buffer to [N].
@triton.autotune(configs=_autotune_configs(), key=["N"])
@triton.jit
def _rmsnorm_bwd_kernel(
    DX,            # *Pointer* to grad input  [M, N]  (output)
    DY,            # *Pointer* to grad output [M, N]  (input)
    X,             # *Pointer* to input       [M, N]  (input, saved)
    W,             # *Pointer* to weight      [N]
    Rstd,          # *Pointer* to saved 1/rms [M]
    DW_partial,    # *Pointer* to partial dw  [n_programs, N]  (output)
    stride_m,      # row stride shared by DX, DY, X (we keep them contiguous)
    N,             # number of columns
    M,             # number of rows (needed for the grid-stride bound)
    rows_per_prog, # how many rows each program strides over
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Weight is row-invariant: load it once and reuse across all rows this
    # program handles.
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Private dw accumulator for this program, kept in fp32 registers.
    dw_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Grid-stride loop: program `pid` owns rows [pid*rows_per_prog,
    # (pid+1)*rows_per_prog). Contiguous row chunks per program give clean,
    # coalesced access and let one warm `w` load amortize across the chunk.
    row_start = pid * rows_per_prog
    row_end = tl.minimum(row_start + rows_per_prog, M)
    for row in range(row_start, row_end):
        off = row * stride_m + cols
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)

        xhat = x * rstd
        dyw = dy * w

        # mean_j(dyw_j * xhat_j) -- the only reduction in the dx path.
        mean_dyw_xhat = tl.sum(dyw * xhat, axis=0) / N
        dx = rstd * (dyw - xhat * mean_dyw_xhat)
        tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=mask)

        # Fold this row's dw contribution into the private accumulator.
        dw_acc += dy * xhat

    # Flush the private partial. The host reduces [n_programs, N] -> [N].
    tl.store(DW_partial + pid * N + cols, dw_acc, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers + autograd glue
# ---------------------------------------------------------------------------
def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _check_last_dim(n: int):
    # The whole row must fit in one SRAM tile. 2^16 fp32 = 256KB, already past
    # what a single program should hold; our target hidden dims (<=8192) are
    # comfortably inside this. We guard so a bad shape fails loud, not silent.
    if n > 65536:
        raise ValueError(
            f"hidden dim {n} exceeds the single-tile design limit (65536). "
            "This kernel keeps a whole row resident; very wide rows need a "
            "multi-pass variant."
        )


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # Flatten all leading dims into M rows of N columns. RMSNorm always acts
        # on the last dim, so [B, S, H] and [B*S, H] are the same problem.
        orig_shape = x.shape
        N = orig_shape[-1]
        _check_last_dim(N)
        x2d = x.reshape(-1, N)
        M = x2d.shape[0]

        # We require contiguous rows so stride math is a single row stride. The
        # .contiguous() is a no-op for the standard activation layout.
        x2d = x2d.contiguous()
        weight = weight.contiguous()

        y = torch.empty_like(x2d)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        BLOCK_SIZE = _next_power_of_2(N)
        grid = (M,)
        _rmsnorm_fwd_kernel[grid](
            x2d, weight, y, rstd,
            x2d.stride(0), y.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x2d, weight, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, weight, rstd = ctx.saved_tensors
        N = x2d.shape[-1]
        M = x2d.shape[0]

        dy2d = dy.reshape(-1, N).contiguous()
        dx = torch.empty_like(x2d)

        # Number of programs for the dw reduction. We cap parallelism at the
        # device's SM count so the partial buffer stays small (few hundred rows
        # of N) while still saturating the machine; each program then strides
        # over ceil(M / n_programs) rows. Falls back to 64 if we cannot query.
        try:
            sm_count = torch.cuda.get_device_properties(x2d.device).multi_processor_count
        except Exception:
            sm_count = 64
        n_programs = min(M, sm_count)
        rows_per_prog = triton.cdiv(M, n_programs)
        # Recompute the *actual* number of programs we need to cover M given the
        # rounded-up rows_per_prog, so no program runs past the end.
        n_programs = triton.cdiv(M, rows_per_prog)

        dw_partial = torch.empty((n_programs, N), dtype=torch.float32, device=x2d.device)

        BLOCK_SIZE = _next_power_of_2(N)
        grid = (n_programs,)
        _rmsnorm_bwd_kernel[grid](
            dx, dy2d, x2d, weight, rstd, dw_partial,
            x2d.stride(0),
            N, M, rows_per_prog,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Reduce the partials to the final [N] weight grad, then match weight's
        # dtype. Accumulation happened in fp32 inside the kernel.
        dw = dw_partial.sum(dim=0).to(weight.dtype)

        # Grad w.r.t. x, weight, eps. eps is not differentiable -> None.
        return dx.reshape(ctx.orig_shape), dw, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm. `x` is normalized over its last dim and scaled by `weight`.

    Args:
        x:      input tensor, any shape [..., N]
        weight: gain vector [N]
        eps:    stabilizer inside the sqrt (default 1e-6, matching HF Llama)
    Returns:
        tensor with the same shape and dtype as `x`.
    """
    return _RMSNormFunction.apply(x, weight, eps)


# ---------------------------------------------------------------------------
# Pure-PyTorch reference (the correctness oracle; also a benchmark baseline)
# ---------------------------------------------------------------------------
def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eager reference, written to match Hugging Face's LlamaRMSNorm exactly:
    the normalization math is done in fp32 and the result is cast back to the
    input dtype before the gain multiply."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x.to(input_dtype))
