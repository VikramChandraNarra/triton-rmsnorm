# Running on a rented GPU (Lambda / RunPod / etc.)

This kernel cannot be exercised without an NVIDIA GPU. These are the exact
commands to validate **correctness** on a fresh single-A100 or single-H100 box.
(The benchmark + plots are a separate step, added only after these tests pass.)

## 0. Box requirements
- 1x A100 80GB or H100 (any single-GPU instance works; the kernel is per-row).
- CUDA 12.x driver, Python 3.10+.

## 1. Clone + install
```bash
git clone <your-repo-url> fusedkernel && cd fusedkernel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Confirm the GPU is visible
```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
```

## 3. Run the correctness + gradient tests
```bash
pytest -q
```
Expect every parametrization to pass (forward across {2048,4096,8192} x
{512,1024,2048,4096} x {fp16,bf16}, the non-power-of-two and 3D shape cases,
and the backward/gradient checks). The first run is slower because Triton
autotunes and compiles each (N, kernel) the first time it is seen; results are
cached under `.triton/` for subsequent runs.

If anything fails, capture the full output and send it back before we proceed
to benchmarking — we do not write a single perf number until correctness is
green.

> Status: the kernel + tests are written but have NOT been executed on a GPU
> yet (this repo was authored on a CPU-only machine). Step 3 is the
> sanity-check gate.
