# AWS ECS Fargate Deployment

Deploys the **predictive maintenance inference service** (PyTorch MLP backend)
to AWS ECS Fargate with Redis caching, CloudWatch logging, and Secrets Manager
for credentials.

## Architecture

```
Internet → ALB → ECS Fargate (inference-service)
                      │
                      ├── ECR  (Docker image)
                      ├── Redis ElastiCache (prediction cache, 300 s TTL)
                      ├── MLflow (model registry, polls every 60 s)
                      ├── TimescaleDB RDS (sensor data)
                      └── Kafka MSK (real-time prediction streaming)
```

## Prerequisites

| Tool | Install |
|---|---|
| AWS CLI v2 | `brew install awscli` |
| Docker | https://docs.docker.com/get-docker/ |
| jq | `brew install jq` |
| envsubst | `brew install gettext` |

Configure AWS credentials:
```bash
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region:        us-east-1
```

The deploying IAM user/role must have the permissions in [iam_deploy_policy.json](iam_deploy_policy.json).

## Quick Start

```bash
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1
export ECS_CLUSTER=predictive-maintenance-cluster
export ECS_SERVICE=inference-service

./infra/aws/deploy.sh
```

## What `deploy.sh` does

1. Authenticates Docker with ECR
2. Creates the ECR repository if it doesn't exist (with `scanOnPush=true`)
3. Builds the inference-service Docker image from `inference_service/Dockerfile`
4. Tags and pushes to ECR (image tagged with git short-SHA + `latest`)
5. Registers a new ECS task definition revision from `ecs_task_definition.json`
6. Updates the Fargate service to the new revision (`--force-new-deployment`)
7. Waits for the service to stabilise

## Task Definition

[ecs_task_definition.json](ecs_task_definition.json) configures:

| Parameter | Value |
|---|---|
| Launch type | Fargate |
| CPU / Memory | 2 vCPU / 4 GB |
| Container port | 8000 |
| Health check | `GET /health` every 30 s |
| Log driver | `awslogs` → CloudWatch `/ecs/predictive-maintenance-inference` |
| Secrets | API key, DB password, Redis password via Secrets Manager |

## Secrets Manager setup

Create secrets before first deploy:

```bash
aws secretsmanager create-secret \
  --name pm/api-key \
  --secret-string "your-api-key" \
  --region us-east-1

aws secretsmanager create-secret \
  --name pm/db-password \
  --secret-string "your-db-password" \
  --region us-east-1

aws secretsmanager create-secret \
  --name pm/redis-password \
  --secret-string "your-redis-password" \
  --region us-east-1
```

## Retraining pipeline on AWS

The automated retraining pipeline (`ml_pipeline/retrain/retrain_pipeline.py`)
runs as a scheduled ECS task (cron via EventBridge):

```bash
# Create EventBridge rule — run retraining check every Sunday at 02:00 UTC
aws events put-rule \
  --name "pm-retrain-weekly" \
  --schedule-expression "cron(0 2 ? * SUN *)" \
  --state ENABLED

# Point it at the retrain ECS task
aws events put-targets \
  --rule pm-retrain-weekly \
  --targets '[{
    "Id": "retrain-task",
    "Arn": "arn:aws:ecs:us-east-1:ACCOUNT:cluster/predictive-maintenance-cluster",
    "RoleArn": "arn:aws:iam::ACCOUNT:role/ecsEventsRole",
    "EcsParameters": {
      "TaskDefinitionArn": "arn:aws:ecs:us-east-1:ACCOUNT:task-definition/retrain-pipeline",
      "TaskCount": 1,
      "LaunchType": "FARGATE",
      "NetworkConfiguration": {
        "awsvpcConfiguration": {
          "Subnets": ["subnet-xxxxx"],
          "SecurityGroups": ["sg-xxxxx"],
          "AssignPublicIp": "DISABLED"
        }
      }
    }
  }]'
```

## MLflow artifact storage (S3)

MLflow is configured to store artifacts in S3:
```
s3://mlflow-artifacts/retraining
```

Create the bucket:
```bash
aws s3 mb s3://mlflow-artifacts --region us-east-1
aws s3api put-bucket-versioning \
  --bucket mlflow-artifacts \
  --versioning-configuration Status=Enabled
```
