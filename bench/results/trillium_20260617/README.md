# Trillium H100 RMSNorm Run - 2026-06-17

Staged checkout:

```text
/scratch/aneela/triton-rmsnorm_20260617T200445Z
```

Environment:

- Cluster: SciNet Trillium
- Partition: `compute`
- Account: `def-mponce-ac`
- QOS: `normal`
- Node type: 1x NVIDIA H100 80GB HBM3
- Modules: `StdEnv/2023`, `python/3.11.5`, `cuda/12.6`
- Python environment: `/home/aneela/venvs/triton-rmsnorm`
- Torch: `2.12.0`
- Triton: `3.6.0`

Correctness gate:

- Slurm job: `604545`
- Node: `trig0006`
- Runtime: `00:01:57`
- Result: `40 passed`, exit code `0:0`
- Log: `rmsnorm-test_604545.out`

Benchmark:

- Slurm job: `604586`
- Node: `trig0024`
- Runtime: `00:01:12`
- Result: exit code `0:0`
- Log: `rmsnorm-bench_604586.out`
- CSV: `../rmsnorm_benchmark.csv`

Headline forward results:

| dtype | M | N | fused ms | eager ms | speedup | fused GB/s | H100 peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 4096 | 8192 | 0.0970 | 0.5319 | 5.484x | 1384.1 | 41.32% |
| bf16 | 4096 | 8192 | 0.0893 | 0.5322 | 5.960x | 1503.4 | 44.88% |
