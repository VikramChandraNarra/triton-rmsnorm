"""
Correctness tests for the fused Triton RMSNorm.

Philosophy: we do NOT bit-match a same-dtype eager implementation, because two
correct fp16 implementations legitimately differ in their last bit depending on
rounding order. Instead we compare against an fp32 *oracle* (the mathematically
"true" value, computed entirely in fp32 and only then rounded to the test
dtype) and assert closeness with tolerances appropriate to the storage dtype.
This is the honest way to make a precision claim: "within X of the true value,"
not "identical to one particular eager spelling."

Backward is checked the same way -- analytic Triton grads vs autograd grads of
the fp32 oracle. (torch.autograd.gradcheck needs float64, which Triton kernels
do not support, so the fp32-oracle comparison is the right tool here.)

Run on a CUDA box:  pytest -q tests/test_rmsnorm.py
"""

import pytest
import torch

# Skip the whole module cleanly on a machine without CUDA + Triton (e.g. a
# laptop), so `pytest` is green to collect everywhere and only *runs* on GPU.
triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required for the Triton RMSNorm tests", allow_module_level=True)

from kernels.rmsnorm import rmsnorm, rmsnorm_reference  # noqa: E402


# Target shapes: hidden dims from Llama-3-8B / Qwen2-7B and a batch*seq sweep.
HIDDEN_DIMS = [2048, 4096, 8192]
ROW_COUNTS = [512, 1024, 2048, 4096]
DTYPES = [torch.float16, torch.bfloat16]

# Per-dtype tolerances against the fp32 oracle. fp16 has ~3-4 decimal digits,
# bf16 only ~2-3, so bf16 gets the looser bound. These are the conventional
# tolerances used for normalization kernels (e.g. Triton's own LayerNorm tests).
TOL = {
    torch.float16: dict(atol=1e-3, rtol=1e-2),
    torch.bfloat16: dict(atol=1e-2, rtol=2e-2),
}

EPS = 1e-6


def _oracle_forward(x, weight, eps=EPS):
    """The mathematically true RMSNorm, computed end-to-end in fp32 and rounded
    to x's dtype only at the very end."""
    xf = x.float()
    wf = weight.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps) * wf
    return y.to(x.dtype)


def _make_inputs(M, N, dtype, requires_grad=False, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(M, N, device="cuda", dtype=dtype, requires_grad=requires_grad)
    # Weight initialized near 1 (as RMSNorm gains are in practice), with spread.
    weight = torch.randn(N, device="cuda", dtype=dtype, requires_grad=requires_grad)
    weight = (weight * 0.1 + 1.0).detach().requires_grad_(requires_grad)
    return x, weight


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", HIDDEN_DIMS)
@pytest.mark.parametrize("M", ROW_COUNTS)
def test_forward_matches_oracle(M, N, dtype):
    x, weight = _make_inputs(M, N, dtype)
    y = rmsnorm(x, weight, EPS)
    y_oracle = _oracle_forward(x, weight, EPS)
    torch.testing.assert_close(y, y_oracle, **TOL[dtype])


def test_forward_non_power_of_two_hidden():
    # Tail-masking path: hidden dim that is not a power of two.
    x, weight = _make_inputs(1024, 4000, torch.float16)
    y = rmsnorm(x, weight, EPS)
    y_oracle = _oracle_forward(x, weight, EPS)
    torch.testing.assert_close(y, y_oracle, **TOL[torch.float16])


def test_forward_3d_shape_is_flattened():
    # [B, S, H] must behave identically to [B*S, H].
    B, S, H = 4, 512, 4096
    torch.manual_seed(1)
    x = torch.randn(B, S, H, device="cuda", dtype=torch.float16)
    weight = torch.randn(H, device="cuda", dtype=torch.float16) * 0.1 + 1.0
    y = rmsnorm(x, weight, EPS)
    assert y.shape == (B, S, H)
    y_oracle = _oracle_forward(x, weight, EPS)
    torch.testing.assert_close(y, y_oracle, **TOL[torch.float16])


def test_reference_helper_agrees_with_oracle():
    # Sanity-check our HF-style eager reference against the fp32 oracle so the
    # benchmark baseline is itself trustworthy.
    x, weight = _make_inputs(1024, 4096, torch.float16)
    y_ref = rmsnorm_reference(x, weight, EPS)
    y_oracle = _oracle_forward(x, weight, EPS)
    torch.testing.assert_close(y_ref, y_oracle, **TOL[torch.float16])


# ---------------------------------------------------------------------------
# Backward correctness (analytic Triton grads vs autograd of the fp32 oracle)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", HIDDEN_DIMS)
@pytest.mark.parametrize("M", [512, 2048])
def test_backward_matches_oracle(M, N, dtype):
    # Triton path.
    x, weight = _make_inputs(M, N, dtype, requires_grad=True, seed=2)
    y = rmsnorm(x, weight, EPS)
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    dx_triton, dw_triton = x.grad, weight.grad

    # fp32-oracle path: same inputs upcast to fp32, autograd through the oracle,
    # grads rounded back to the test dtype for comparison.
    xf, wf = _make_inputs(M, N, dtype, requires_grad=False, seed=2)
    xf = xf.float().detach().requires_grad_(True)
    wf = wf.float().detach().requires_grad_(True)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y_ref = xf * torch.rsqrt(var + EPS) * wf
    y_ref.backward(grad_out.float())
    dx_ref = xf.grad.to(dtype)
    dw_ref = wf.grad.to(dtype)

    # dw is a sum over M rows, so its error grows with M -> looser bound there.
    tol = TOL[dtype]
    torch.testing.assert_close(dx_triton, dx_ref, **tol)
    torch.testing.assert_close(
        dw_triton, dw_ref,
        atol=tol["atol"] * 10, rtol=tol["rtol"] * 5,
    )


def test_backward_dw_accumulates_across_all_rows():
    # Regression guard for the grid-stride dw reduction: if a program skipped or
    # double-counted rows, dw would be wrong by a row-dependent factor. Compare
    # against a plain fp32 sum reduction.
    M, N, dtype = 4096, 2048, torch.float16
    x, weight = _make_inputs(M, N, dtype, requires_grad=True, seed=3)
    y = rmsnorm(x, weight, EPS)
    grad_out = torch.randn_like(y)
    y.backward(grad_out)

    # Reference dw via the oracle.
    xf = x.detach().float().requires_grad_(True)
    wf = weight.detach().float().requires_grad_(True)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y_ref = xf * torch.rsqrt(var + EPS) * wf
    y_ref.backward(grad_out.float())

    torch.testing.assert_close(
        weight.grad, wf.grad.to(dtype), atol=1e-2, rtol=5e-2,
    )
