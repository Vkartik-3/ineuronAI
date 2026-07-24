# AWS Teardown Runbook

Run after any demo deployment to stop standing cost. **Not yet executed** (no
deployment exists).

## Order (dependencies first)

```bash
# 1. Scale the service to zero, then delete it
aws ecs update-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
    --desired-count 0 --region "$AWS_REGION"
aws ecs delete-service --cluster "$ECS_CLUSTER" --service "$ECS_SERVICE" \
    --force --region "$AWS_REGION"

# 2. Deregister task definitions (optional; free to keep)
#    aws ecs deregister-task-definition --task-definition predictive-maintenance-inference:<rev>

# 3. Delete the ALB + target group + listeners
# 4. Delete the ElastiCache Redis cluster
# 5. Delete the ECS cluster (if created only for this)
# 6. Empty + delete the ECR repository images (keeps repo history clean)
#    aws ecr batch-delete-image ... ; aws ecr delete-repository --force ...
# 7. Remove Secrets Manager / SSM entries created for this
# 8. Delete the CloudWatch log group
```

## Confirm teardown
- `aws ecs describe-services` → service `INACTIVE`/absent.
- `aws elasticache describe-cache-clusters` → cluster gone.
- Cost Explorer shows the resources dropping off.
- Record teardown confirmation (timestamp + `describe` outputs) in
  `artifacts/aws_deployment_evidence.md`.

## Note
IAM roles/policies are free to retain and can be reused; delete only if the
account is being decommissioned.
