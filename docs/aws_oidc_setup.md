# AWS OIDC Deploy Setup (GitHub Actions, no static keys)

This wires GitHub Actions → AWS via **OpenID Connect**: Actions assumes a scoped
IAM role and AWS issues short-lived credentials. **No AWS access key is ever
stored.** This is the production-correct deploy path.

Do the one-time steps below **once**. After that, every push to `main` (or the
manual "Run workflow" button) deploys via `.github/workflows/deploy.yml`.

---

## Part 1 — One-time AWS setup (paste into CloudShell)

Open **CloudShell** (the `>_` icon, top-right of the AWS console; region
**us-east-2**). Paste this whole block:

```bash
set -e
git clone https://github.com/Vkartik-3/ineuronAI.git
cd ineuronAI/infra/aws

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account: $ACCOUNT_ID"

# 1. Create the GitHub OIDC provider (idempotent — ignore 'already exists')
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
  2>/dev/null || echo "OIDC provider already exists — ok"

# 2. Build the trust policy with your real account id
sed "s/AWS_ACCOUNT_ID/${ACCOUNT_ID}/g" github_oidc_trust_policy.json > /tmp/trust.json

# 3. Create the deploy role with that trust policy
aws iam create-role \
  --role-name pm-github-deploy \
  --assume-role-policy-document file:///tmp/trust.json \
  2>/dev/null || aws iam update-assume-role-policy \
  --role-name pm-github-deploy --policy-document file:///tmp/trust.json

# 4. Attach the least-privilege deploy permissions (from the repo)
aws iam put-role-policy \
  --role-name pm-github-deploy \
  --policy-name pm-deploy-permissions \
  --policy-document file://iam_deploy_policy.json

echo "ROLE ARN ->  arn:aws:iam::${ACCOUNT_ID}:role/pm-github-deploy"
```

Copy the printed **ROLE ARN** — you need it in Part 2.

---

## Part 2 — One-time GitHub setup

In **https://github.com/Vkartik-3/ineuronAI → Settings**:

- **Secrets and variables → Actions → Secrets → New secret**
  - `AWS_DEPLOY_ROLE_ARN` = the ROLE ARN from Part 1.
- **Secrets and variables → Actions → Variables → New variable** (add all three)
  - `AWS_REGION` = `us-east-2`
  - `ECS_CLUSTER` = your cluster name (e.g. `predictive-maintenance-cluster`)
  - `ECS_SERVICE` = your service name (e.g. `inference-service`)
- **(optional) Settings → Environments → New environment → `production`**
  add yourself as a required reviewer so deploys need one click of approval.

---

## Part 3 — Prerequisite: the ECS service must exist

`deploy.sh` builds+pushes the image and **updates an existing** ECS service — it
does not create the cluster/ALB/ElastiCache. Before the first deploy you (or an
IaC step) must create, once: the ECR repo (deploy.sh auto-creates this), an ECS
**cluster**, the **task-execution role**, an **ElastiCache Redis**, an **ALB +
target group** (health check `/health`), and the **ECS service** referencing
`infra/aws/ecs_task_definition.json`. See
[`aws_deployment_runbook.md`](aws_deployment_runbook.md) and
[`aws_architecture.md`](aws_architecture.md).

> Until the ECS service exists, the workflow's deploy step will fail at
> "update-service" with a clear not-found error — that's expected, not a
> credentials problem.

---

## Part 4 — Deploy

- Push to `main`, or **Actions → Deploy inference service to AWS ECS → Run
  workflow**.
- The workflow assumes the role (OIDC), builds `inference_service/Dockerfile`,
  pushes to ECR, updates ECS, then runs `verify_deployment.sh`.
- Capture the deployed-endpoint latency into
  `artifacts/deployed_api_latency.json` (run `inference_service/benchmark_http.py
  --url <ALB-URL>`) and fill `artifacts/aws_deployment_evidence.md`.

## Teardown
Follow [`aws_teardown_runbook.md`](aws_teardown_runbook.md) to stop billing.
The `pm-github-deploy` role + OIDC provider are free to keep.
