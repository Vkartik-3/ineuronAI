# CPU vs GPU — Model-Compute Benchmark

- Timestamp: 2026-07-24T06:42:08.688029+00:00
- Git commit: 5977b30
- Host: macOS-26.5.2-arm64-arm-64bit / arm
- torch: 2.5.1  |  CUDA available: False  |  GPU: None
- Measures: model compute only (forward pass); h2d/d2h transfer reported for GPU

## Results (model compute)

| device | batch | p50 ms | p95 ms | p99 ms | throughput (samples/s) |
|---|---|---|---|---|---|
| cpu | 1 | 0.0563 | 0.0594 | 0.0662 | 17473.8 |
| cpu | 8 | 0.075 | 0.0824 | 0.1027 | 103444.9 |
| cpu | 16 | 0.0807 | 0.0817 | 0.0851 | 198288.5 |
| cpu | 32 | 0.089 | 0.0912 | 0.1009 | 357442.7 |
| cuda | — | — | — | — | SKIPPED: No CUDA device available on this host — GPU path NOT measure… |

## Conclusion

GPU was UNAVAILABLE on this host, so only CPU was measured. The single-request CPU p99 is well under the 300 ms SLO (batch_1 p99 = 0.0662 ms), so the low-latency path is served on CPU. The 'GPU acceleration' claim remains a code-supported capability that is UNVERIFIED on hardware here — it must be benchmarked on a CUDA host to assert a throughput benefit.
