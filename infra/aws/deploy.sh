#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Build, push, and deploy the inference service to AWS ECS Fargate
#
# What this script does:
#   1. Authenticates Docker with Amazon ECR
#   2. Builds the inference-service Docker image (PyTorch MLP backend)
#   3. Tags and pushes the image to ECR
#   4. Registers a new ECS task definition revision
#   5. Updates the ECS Fargate service to use the new revision
#   6. Waits for the deployment to stabilise
#
# Prerequisites:
#   - AWS CLI v2 configured (aws configure) with a user/role that has the
#     permissions in iam_deploy_policy.json
#   - Docker running locally
#   - jq installed (brew install jq / apt install jq)
#
# Required environment variables (set before running):
#   AWS_ACCOUNT_ID   — 12-digit AWS account number
#   AWS_REGION       — e.g. us-east-1
#   ECS_CLUSTER      — ECS cluster name (e.g. predictive-maintenance-cluster)
#   ECS_SERVICE      — ECS service name (e.g. inference-service)
#
# Optional:
#   IMAGE_TAG        — Docker image tag (default: git short-SHA or 'latest')
#   DOCKERFILE_PATH  — Path to Dockerfile (default: auto-detected)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }
require() { command -v "$1" &>/dev/null || die "'$1' not found — please install it"; }

require aws
require docker
require jq

# ---------------------------------------------------------------------------
# Config — override via environment variables
# ---------------------------------------------------------------------------
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
AWS_REGION="${AWS_REGION:?Set AWS_REGION}"
ECS_CLUSTER="${ECS_CLUSTER:?Set ECS_CLUSTER}"
ECS_SERVICE="${ECS_SERVICE:?Set ECS_SERVICE}"

REPO_NAME="predictive-maintenance-inference"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO="${ECR_REGISTRY}/${REPO_NAME}"

# Image tag: use git SHA if inside a repo, otherwise 'latest'
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"
FULL_IMAGE="${ECR_REPO}:${IMAGE_TAG}"

# Locate Dockerfile
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKERFILE="${DOCKERFILE_PATH:-${REPO_ROOT}/predictive-maintenance/inference_service/Dockerfile}"
CONTEXT_DIR="$(dirname "${DOCKERFILE}")"

TASK_DEF_JSON="${SCRIPT_DIR}/ecs_task_definition.json"

log "=== Predictive Maintenance — ECS Fargate Deployment ==="
log "Region      : ${AWS_REGION}"
log "Account     : ${AWS_ACCOUNT_ID}"
log "Cluster     : ${ECS_CLUSTER}"
log "Service     : ${ECS_SERVICE}"
log "Image       : ${FULL_IMAGE}"
log "Dockerfile  : ${DOCKERFILE}"

# ---------------------------------------------------------------------------
# Step 1 — ECR authentication
# ---------------------------------------------------------------------------
log "Step 1/6: Authenticating with ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# ---------------------------------------------------------------------------
# Step 2 — Ensure ECR repository exists
# ---------------------------------------------------------------------------
log "Step 2/6: Ensuring ECR repository exists..."
aws ecr describe-repositories \
  --repository-names "${REPO_NAME}" \
  --region "${AWS_REGION}" > /dev/null 2>&1 \
|| aws ecr create-repository \
     --repository-name "${REPO_NAME}" \
     --region "${AWS_REGION}" \
     --image-scanning-configuration scanOnPush=true \
     --encryption-configuration encryptionType=AES256

# Apply lifecycle policy — keep only last 10 images
aws ecr put-lifecycle-policy \
  --repository-name "${REPO_NAME}" \
  --region "${AWS_REGION}" \
  --lifecycle-policy-text '{
    "rules": [{
      "rulePriority": 1,
      "description": "Keep last 10 images",
      "selection": {"tagStatus": "any", "countType": "imageCountMoreThan", "countNumber": 10},
      "action": {"type": "expire"}
    }]
  }' > /dev/null

# ---------------------------------------------------------------------------
# Step 3 — Build Docker image
# ---------------------------------------------------------------------------
log "Step 3/6: Building Docker image..."
docker build \
  --file "${DOCKERFILE}" \
  --tag "${FULL_IMAGE}" \
  --tag "${ECR_REPO}:latest" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  "${CONTEXT_DIR}"

# ---------------------------------------------------------------------------
# Step 4 — Push to ECR
# ---------------------------------------------------------------------------
log "Step 4/6: Pushing image to ECR..."
docker push "${FULL_IMAGE}"
docker push "${ECR_REPO}:latest"
log "Pushed: ${FULL_IMAGE}"

# ---------------------------------------------------------------------------
# Step 5 — Register ECS task definition
# ---------------------------------------------------------------------------
log "Step 5/6: Registering ECS task definition..."

# Substitute shell variables in the JSON template
TASK_DEF_RENDERED=$(
  AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID}" \
  AWS_REGION="${AWS_REGION}" \
  envsubst < "${TASK_DEF_JSON}"
)

# Override the container image with the freshly pushed tag
TASK_DEF_RENDERED=$(
  echo "${TASK_DEF_RENDERED}" \
  | jq --arg img "${FULL_IMAGE}" \
       '.containerDefinitions[0].image = $img'
)

TASK_DEF_ARN=$(
  aws ecs register-task-definition \
    --region "${AWS_REGION}" \
    --cli-input-json "${TASK_DEF_RENDERED}" \
  | jq -r '.taskDefinition.taskDefinitionArn'
)
log "Registered task definition: ${TASK_DEF_ARN}"

# ---------------------------------------------------------------------------
# Step 6 — Update ECS service
# ---------------------------------------------------------------------------
log "Step 6/6: Updating ECS service..."
aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --task-definition "${TASK_DEF_ARN}" \
  --force-new-deployment \
  --region "${AWS_REGION}" > /dev/null

log "Waiting for service to stabilise (this can take 2–5 min)..."
aws ecs wait services-stable \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" \
  --region "${AWS_REGION}"

log "=== Deployment complete ==="
log "Service ${ECS_SERVICE} is running task definition:"
log "  ${TASK_DEF_ARN}"
