# AWS Architecture

**Status: prepared and validated, NOT deployed.** No AWS resources have been
created and no endpoint exists (no credentials in this environment). Definitions
live in [`infra/aws/`](../../infra/aws/). See
[`artifacts/aws_deployment_evidence.md`](../artifacts/aws_deployment_evidence.md)
for the executed-vs-unverified breakdown.

## Target topology (smallest defensible path)

```
Internet
   │
   ▼
 ALB  ──►  ECS Fargate task: inference-service (:8000)
                 │   cpu 2048 / mem 4096  (ecs_task_definition.json)
                 ├── ECR              image: predictive-maintenance-inference
                 ├── ElastiCache Redis  prediction cache (300s TTL, db 0)
                 ├── MLflow           model registry (poll 60s)  [external/self-hosted]
                 ├── CloudWatch Logs  structured JSON logs
                 └── Secrets Manager / SSM  API keys, DB creds
```

Only what the architecture needs is included: **ECR** (image), **ECS Fargate**
(stateless API), **ElastiCache** (the Redis cache the code already uses),
**ALB** (ingress + health checks on `/health`), **CloudWatch** (logs),
**Secrets Manager/SSM** (no secrets in source). TimescaleDB (RDS) and Kafka
(MSK) are referenced by env in the task def but are not required for the core
`/predict/rul` path.

## Config surface (all via env, 12-factor)

`MLFLOW_TRACKING_URI`, `REDIS_HOST/PORT`, `KAFKA_BOOTSTRAP_SERVERS`, `API_KEYS`,
`ADMIN_API_KEYS`, `DB_*`. The service reads env over config defaults (the
`MLFLOW_TRACKING_URI` override is implemented and tested locally).

## IAM

Least-privilege deploy policy in `infra/aws/iam_deploy_policy.json` (ECR
push/pull, ECS register/update, pass-role scoped). Task pull policy in
`ecr_policy.json`. Task role vs execution role separated in the task def.

## Health & readiness

ALB target-group health check → `GET /health` (returns `degraded` if no model
loaded or a dependency is down). `cuda_available` + `model_devices` in the
health body give a live device-placement evidence trail.

## Cost (rough order-of-magnitude, us-east-1, NOT a quote)

1× Fargate task (2 vCPU/4 GB) ≈ $0.10/hr; ALB ≈ $0.025/hr + LCU; ElastiCache
`cache.t3.micro` ≈ $0.017/hr. A short demo deployment is a few dollars; tear
down after (see teardown runbook) to avoid standing cost.

## Rollback

`deployment.enable_rollback` + ECS keeping the previous task-definition
revision; `infra/aws/verify_deployment.sh` smoke-checks the new revision before
it is trusted. Model-level rollback is handled by the retrain guard (last
known-good preserved).
