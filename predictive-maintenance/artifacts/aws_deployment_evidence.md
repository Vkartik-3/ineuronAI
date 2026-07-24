# AWS Deployment Evidence

**Status: DEPLOYED, INVOKED, BENCHMARKED, then TORN DOWN.** The inference
service was really deployed to AWS ECS Fargate behind an ALB with ElastiCache
Redis, served live predictions, was latency-benchmarked, and then all billable
resources were destroyed to stop cost.

| Evidence item | Value |
|---|---|
| Deployment date | 2026-07-24 (us-east-2) |
| AWS account | 845242190324 (IAM user `pm-demo`) |
| Region | us-east-2 (Ohio) |
| Service | ECS Fargate `inference-service` on `predictive-maintenance-cluster`, 2 vCPU / 4 GB, 1 task |
| Image | `…/predictive-maintenance-inference:v1` |
| Image digest | `sha256:65dd7039f463b37e44a8af43c110701b31fdb7b4134491f41d918a8f11c255d1` |
| Model version | v1.0.0 (PyTorch MLP temporal checkpoint, baked into image) |
| Endpoint | `http://pm-alb-1719162413.us-east-2.elb.amazonaws.com` (ALB, now deleted) |
| ALB target health | `healthy` |
| `/health` (deployed) | HTTP 200 |
| Deployed prediction | `/predict/rul` → RUL 79.38 cycles, health "warning", confidence 0.74 (engine_042) |
| Redis (ElastiCache) | `pm-redis` cache.t3.micro, status `available`, connected from task |
| CloudWatch logs | group `/ecs/predictive-maintenance-inference`, stream `ecs/inference-service/0b536c4553…` |
| Rate limiter (prod) | verified ACTIVE — burst >100/min returned HTTP 429 |
| Deployed latency | `artifacts/deployed_api_latency.json` (see below) |
| Teardown | all resources deleted — see `aws_teardown_evidence` in this file |

## Deployed latency (real HTTP over public internet → ALB → Fargate)

Paced under the 100/min rate limit, 0 errors. Source:
`artifacts/deployed_api_latency.json`.

| Path | p50 | p95 | p99 | SLO<300ms |
|---|---|---|---|---|
| cache-miss (full inference) | 107 ms | 174 ms | 261 ms | ✅ |
| cache-hit | 108 ms | 188 ms | 199 ms | ✅ |

**Honest reading:** latency is dominated by the client's home-internet RTT to
us-east-2 (~100 ms baseline). Server compute is sub-millisecond (see
`model_compute_latency.json`), so cache-hit vs miss barely differ over this link
— the network, not the model, is the floor here. A same-region client (e.g. a
load generator in us-east-2) would show much lower numbers; that was not run.

## What was fixed to make the deploy actually work

The pre-existing Docker/AWS scaffolding did **not** run. Fixes made this session:
- Wrong entrypoint/context and package-relative imports → new
  `inference_service/Dockerfile.aws` (context = `predictive-maintenance/`,
  entrypoint `inference_service.api.main:app`, `PYTHONPATH=/app`).
- `requirements.txt` pulled **tensorflow** and omitted **torch + redis** → new
  `requirements-aws.txt` with the correct runtime set (CPU torch).
- `inference_engine.py` evaluated a `tf.keras.Model` annotation at import →
  added `from __future__ import annotations`.
- `kafka-python` made the optional consumer block startup on DNS retries →
  omitted from the image so it is skipped cleanly.
- Model checkpoint (gitignored `*.pt`) baked into the image and symlinked to the
  config's expected path.

## Teardown evidence
All of: ECS service + cluster, task definitions, ALB + target group + listener,
ElastiCache `pm-redis` + subnet group, ECR images, security groups, CloudWatch
log group — deleted. Verified `describe` calls return empty/absent. The IAM
roles (`ecsTaskExecutionRole`, `predictiveMaintTaskRole`) are free and left in
place.
