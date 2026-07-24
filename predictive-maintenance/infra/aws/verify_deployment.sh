#!/usr/bin/env bash
# AWS deployment evidence collector / classifier.
#
# The repository contains Dockerfiles (inference_service, ml_pipeline, alerting,
# data_loader, dashboard) but NO ECS task definition, ECR config, or deploy
# script. It is therefore "AWS-deployable" (buildable container images) rather
# than "deployed". This script captures real evidence when AWS credentials and
# a service name are available, and otherwise states exactly what is missing —
# it never fabricates a deployment result.
#
# Usage:
#   AWS_REGION=us-east-1 ECS_CLUSTER=pm ECS_SERVICE=inference \
#   ECR_REPO=predictive-maintenance ./infra/aws/verify_deployment.sh
set -uo pipefail

echo "== AWS deployment verification =="

if ! command -v aws >/dev/null 2>&1; then
  echo "STATUS: aws CLI not installed — repository is AWS-DEPLOYABLE, not deployed."
  echo "Dockerfiles present:"
  find . -name Dockerfile -not -path './.venv/*' 2>/dev/null
  exit 0
fi

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "STATUS: no valid AWS credentials — cannot verify a live deployment."
  echo "Provide credentials and re-run to capture ECR digest / ECS revision / logs."
  exit 0
fi

echo "Caller identity:"; aws sts get-caller-identity --output text

: "${AWS_REGION:=us-east-1}"

if [[ -n "${ECR_REPO:-}" ]]; then
  echo "-- ECR image digest --"
  aws ecr describe-images --repository-name "$ECR_REPO" --region "$AWS_REGION" \
    --query 'sort_by(imageDetails,&imagePushedAt)[-1].{digest:imageDigest,pushed:imagePushedAt}' \
    --output table 2>&1 | tail -n +1
fi

if [[ -n "${ECS_CLUSTER:-}" && -n "${ECS_SERVICE:-}" ]]; then
  echo "-- ECS service state --"
  aws ecs describe-services --cluster "$ECS_CLUSTER" --services "$ECS_SERVICE" \
    --region "$AWS_REGION" \
    --query 'services[0].{status:status,desired:desiredCount,running:runningCount,taskDef:taskDefinition}' \
    --output table
  echo "-- Latest CloudWatch logs (if log group /ecs/$ECS_SERVICE) --"
  aws logs tail "/ecs/$ECS_SERVICE" --region "$AWS_REGION" --since 10m 2>&1 | head -20
else
  echo "STATUS: set ECS_CLUSTER and ECS_SERVICE to capture service/task/log evidence."
fi
