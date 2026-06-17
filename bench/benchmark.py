#!/usr/bin/env python3
"""Benchmark fused RMSNorm against the eager PyTorch reference.

This is intentionally small and boring: it avoids optional plotting
dependencies so the Trillium batch job has one job, which is to produce a CSV
without falling over on compute nodes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import triton  # noqa: F401
except Exception as exc:  # pragma: no cover - exercised on cluster only
    raise SystemExit(f"Triton import failed: {exc}") from exc

from kernels import rmsnorm, rmsnorm_reference


EPS = 1e-6
DTYPES = (torch.float16, torch.bfloat16)
SHAPES = (
    (512, 2048),
    (1024, 4096),
    (2048, 4096),
    (4096, 8192),
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this benchmark on a GPU node.")


def _make_inputs(m: int, n: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(m, n, device="cuda", dtype=dtype)
    weight = torch.randn(n, device="cuda", dtype=dtype) * 0.1 + 1.0
    grad = torch.randn_like(x)
    return x, weight, grad


def _time_cuda(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return median(times_ms)


def _forward_bytes(m: int, n: int, dtype: torch.dtype) -> int:
    elem = torch.tensor([], dtype=dtype).element_size()
    # Read x and weight, write y, write one fp32 rstd per row.
    return (2 * m * n + n) * elem + m * 4


def _backward_bytes(m: int, n: int, dtype: torch.dtype) -> int:
    elem = torch.tensor([], dtype=dtype).element_size()
    # Approximate: read x/dy, read weight, write dx/dw, plus rstd traffic.
    return (3 * m * n + 2 * n) * elem + m * 4


def _gbps(num_bytes: int, ms: float) -> float:
    return num_bytes / (ms / 1_000.0) / 1e9


def run(out: Path, warmup: int, repeat: int) -> Path:
    _require_cuda()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "rmsnorm_benchmark.csv"
    peak_gbps = float(os.environ.get("RMSNORM_PEAK_GBPS", "3350"))

    rows = []
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch : {torch.__version__}")

    for dtype in DTYPES:
        for m, n in SHAPES:
            x, weight, grad = _make_inputs(m, n, dtype)

            def fused_forward():
                with torch.no_grad():
                    rmsnorm(x, weight, EPS)

            def eager_forward():
                with torch.no_grad():
                    rmsnorm_reference(x, weight, EPS)

            x_bw = x.detach().requires_grad_(True)
            w_bw = weight.detach().requires_grad_(True)

            def fused_backward():
                x_bw.grad = None
                w_bw.grad = None
                y = rmsnorm(x_bw, w_bw, EPS)
                y.backward(grad)

            fused_fwd_ms = _time_cuda(fused_forward, warmup, repeat)
            eager_fwd_ms = _time_cuda(eager_forward, warmup, repeat)
            fused_bwd_ms = _time_cuda(fused_backward, warmup, repeat)

            fwd_gbps = _gbps(_forward_bytes(m, n, dtype), fused_fwd_ms)
            bwd_gbps = _gbps(_backward_bytes(m, n, dtype), fused_bwd_ms)

            row = {
                "dtype": str(dtype).replace("torch.", ""),
                "M": m,
                "N": n,
                "fused_forward_ms": f"{fused_fwd_ms:.4f}",
                "eager_forward_ms": f"{eager_fwd_ms:.4f}",
                "forward_speedup": f"{eager_fwd_ms / fused_fwd_ms:.3f}",
                "fused_forward_gbps": f"{fwd_gbps:.1f}",
                "fused_forward_pct_peak": f"{100 * fwd_gbps / peak_gbps:.2f}",
                "fused_backward_ms": f"{fused_bwd_ms:.4f}",
                "fused_backward_gbps": f"{bwd_gbps:.1f}",
                "fused_backward_pct_peak": f"{100 * bwd_gbps / peak_gbps:.2f}",
            }
            rows.append(row)
            print(
                f"{row['dtype']:>8} M={m:<5} N={n:<5} "
                f"fwd={row['fused_forward_ms']} ms "
                f"speedup={row['forward_speedup']}x "
                f"bwd={row['fused_backward_ms']} ms"
            )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("bench/results"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    csv_path = run(args.out, args.warmup, args.repeat)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
