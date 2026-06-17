"""
Benchmark sweep for the fused Triton RMSNorm.

RMSNorm is memory-bound (see kernels/rmsnorm.py): the arithmetic is trivial next
to the cost of streaming the activation through HBM. So the headline number is
*not* raw FLOP/s -- it is **effective bandwidth as a fraction of the device's HBM
peak**. A fused kernel that reads x once and writes y once should approach that
peak; the eager PyTorch sequence (square -> mean -> add -> rsqrt -> mul -> mul)
cannot, because each step round-trips the activation through memory in its own
launch. The speedup column makes that gap concrete.

What we measure, per (direction, dtype, M, N):
  * triton_ms : median latency of the fused kernel
  * torch_ms  : median latency of the eager fp32-accumulating reference
  * speedup   : torch_ms / triton_ms
  * gbps      : effective HBM bandwidth of the fused kernel (moved bytes / time)
  * pct_peak  : gbps as a % of the device peak (env RMSNORM_PEAK_GBPS, e.g.
                3350 for an H100 SXM HBM3, 2039 for an A100 80GB)

Timing uses triton.testing.do_bench, which warms up, runs many reps, and returns
the median -- it handles CUDA stream sync and discards the first (cold) launches,
so autotuning and one-time compilation do not pollute the measurement.

Usage:
    python bench/benchmark.py --out bench/results
    RMSNORM_PEAK_GBPS=2039 python bench/benchmark.py --out bench/results   # A100
"""

import argparse
import os
import sys

import torch

# Allow `python bench/benchmark.py` from the repo root without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Shapes mirror the correctness suite (tests/test_rmsnorm.py): hidden dims drawn
# from Llama-3-8B / Qwen2-7B and a batch*seq sweep. Same shapes everywhere means
# a green correctness run and the perf numbers describe the exact same kernels.
HIDDEN_DIMS = [2048, 4096, 8192]
ROW_COUNTS = [512, 1024, 2048, 4096]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _bytes_moved(M: int, N: int, elem_size: int, direction: str) -> int:
    """HBM traffic the fused kernel is *obliged* to move, for the bandwidth calc.

    Forward: read x [M,N] + write y [M,N]. The w [N] read, rstd [M] write, and
    per-row rstd are O(M+N) -- negligible next to the 2*M*N activation traffic,
    so we count the dominant term, which is the honest denominator for "% of
    peak" on a memory-bound op.

    Backward: read x [M,N] + read dy [M,N] + write dx [M,N] = 3*M*N. The dw
    partials are O(n_programs*N) and again negligible.
    """
    activation = M * N * elem_size
    return (2 if direction == "forward" else 3) * activation


def _bench_one(M, N, dtype, direction, peak_gbps):
    # Local imports so `--help` works on a machine without a GPU / triton: the
    # kernel module hard-imports triton at top level, so we defer it to call time.
    import triton
    from kernels.rmsnorm import rmsnorm, rmsnorm_reference

    elem_size = torch.tensor([], dtype=dtype).element_size()
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)

    if direction == "forward":
        triton_fn = lambda: rmsnorm(x, weight)              # noqa: E731
        torch_fn = lambda: rmsnorm_reference(x, weight)     # noqa: E731
    else:
        # Backward: build a graph once, then time only the backward pass. We
        # retain_graph so do_bench can call backward repeatedly on the same graph.
        xr = x.clone().requires_grad_(True)
        wr = weight.clone().requires_grad_(True)
        y = rmsnorm(xr, wr)
        dy = torch.randn_like(y)
        triton_fn = lambda: y.backward(dy, retain_graph=True)   # noqa: E731

        xo = x.clone().requires_grad_(True)
        wo = weight.clone().requires_grad_(True)
        yo = rmsnorm_reference(xo, wo)
        torch_fn = lambda: yo.backward(dy, retain_graph=True)   # noqa: E731

    triton_ms = triton.testing.do_bench(triton_fn)
    torch_ms = triton.testing.do_bench(torch_fn)

    moved = _bytes_moved(M, N, elem_size, direction)
    gbps = moved / (triton_ms * 1e-3) / 1e9
    pct_peak = 100.0 * gbps / peak_gbps if peak_gbps else float("nan")

    return {
        "direction": direction,
        "dtype": {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype],
        "M": M,
        "N": N,
        "triton_ms": round(triton_ms, 4),
        "torch_ms": round(torch_ms, 4),
        "speedup": round(torch_ms / triton_ms, 3),
        "gbps": round(gbps, 1),
        "pct_peak": round(pct_peak, 2),
    }


def _maybe_plot(df, out_dir, peak_gbps):
    """Per-dtype forward-pass plots: latency and % of peak bandwidth vs N. Plots
    are a convenience; the CSV is the source of truth, so a missing matplotlib
    must not fail the run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - plotting is best-effort
        print(f">> matplotlib unavailable ({e}); skipping plots, CSV is written.")
        return

    fwd = df[df["direction"] == "forward"]
    for dtype in sorted(fwd["dtype"].unique()):
        sub = fwd[fwd["dtype"] == dtype]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for M in sorted(sub["M"].unique()):
            s = sub[sub["M"] == M].sort_values("N")
            ax1.plot(s["N"], s["triton_ms"], marker="o", label=f"M={M}")
            ax2.plot(s["N"], s["pct_peak"], marker="o", label=f"M={M}")
        ax1.set(xlabel="hidden dim N", ylabel="latency (ms)",
                title=f"Fused RMSNorm forward latency ({dtype})")
        ax2.set(xlabel="hidden dim N", ylabel="% of HBM peak",
                title=f"Effective bandwidth ({dtype}, peak={peak_gbps} GB/s)")
        ax2.axhline(100, color="grey", ls="--", lw=0.8)
        for ax in (ax1, ax2):
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"rmsnorm_forward_{dtype}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f">> wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="bench/results",
                        help="directory for the CSV and plots (created if absent)")
    parser.add_argument("--directions", nargs="+", default=["forward", "backward"],
                        choices=["forward", "backward"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA device required to benchmark the Triton RMSNorm kernel.")

    # Device HBM peak in GB/s. The SLURM script exports this (H100 SXM = 3350);
    # default keeps a single-machine run honest if the env var is unset.
    peak_gbps = float(os.environ.get("RMSNORM_PEAK_GBPS", "3350"))

    os.makedirs(args.out, exist_ok=True)
    dev = torch.cuda.get_device_name(0)
    print(f">> device: {dev}   HBM peak: {peak_gbps} GB/s")

    rows = []
    for direction in args.directions:
        for name, dtype in DTYPES.items():
            for M in ROW_COUNTS:
                for N in HIDDEN_DIMS:
                    rec = _bench_one(M, N, dtype, direction, peak_gbps)
                    rows.append(rec)
                    print(f"   {direction:8s} {name} M={M:5d} N={N:5d}  "
                          f"triton={rec['triton_ms']:.4f}ms  "
                          f"speedup={rec['speedup']:.2f}x  "
                          f"{rec['gbps']:.1f}GB/s  {rec['pct_peak']:.1f}% peak")

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "rmsnorm_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f">> wrote {csv_path}  ({len(df)} rows)")

    _maybe_plot(df, args.out, peak_gbps)


if __name__ == "__main__":
    main()
