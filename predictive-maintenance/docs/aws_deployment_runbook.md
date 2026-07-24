# AWS Deployment Runbook

**Not yet executed.** These are the exact steps to deploy; running them requires
AWS credentials with the permissions in `infra/aws/iam_deploy_policy.json`.

## Prerequisites
- AWS CLI v2 configured (`aws configure`), Docker running, `jq` installed.
- Env: `AWS_ACCOUNT_ID`, `AWS_REGION`, `ECS_CLUSTER`, `ECS_SERVICE`.

## Steps

```bash
# 0. One-time infra (ECR repo, cluster, ALB, ElastiCache, roles) — via console
#    or IaC; ARNs feed into ecs_task_definition.json placeholders.

# 1. Build, push, register, roll out (idempotent)
export AWS_ACCOUNT_ID=... AWS_REGION=us-east-1
export ECS_CLUSTER=predictive-maintenance-cluster ECS_SERVICE=inference-service
bash infra/aws/deploy.sh          # authenticates ECR, builds, pushes, updates ECS, waits to stabilise

# 2. Smoke-verify the deployed revision
bash infra/aws/verify_deployment.sh   # hits /health + a sample /predict/rul

# 3. Capture evidence into artifacts/aws_deployment_evidence.md:
#    - deployment timestamp, region, service name, image digest, model version
#    - /health output, one prediction result, deployed p50/p99 latency
#    - Redis connectivity, a CloudWatch log group reference
```

## Post-deploy checks
- `GET /health` → `healthy`, `models_loaded` shows the MLP true.
- `GET /metrics` scrapeable; `model_version_info` reflects the deployed version.
- Run `inference_service/benchmark_http.py --url <ALB-URL>` to fill
  `artifacts/deployed_api_latency.json` — **this is the only legitimate source
  of the deployed-latency number.** Do not substitute local numbers.

## If a step fails
Deployment does not flip traffic until the new task is healthy; ECS keeps the
prior revision for rollback (see teardown/rollback). Record the failure in the
evidence file rather than claiming success.
