# Archived Background: IAM Setup Guide For Ursa Monitor

This note is retained for older workset-monitor deployments. Current Ursa runtime work should start from the root [README](../README.md), use `ursa ...` first, and use `tapdb ...`, `daycog ...`, or `daylily-ec ...` only where Ursa delegates that lifecycle.

The active API/GUI surface now centers on worksets, manifests, staging jobs, analysis jobs, cluster operations, linked buckets, user tokens, and Atlas return. This file describes an older monitor/worker permission shape and should not be treated as the full current deployment policy.

## Historical Scope

The older monitor typically needed AWS permissions for:

- S3 read/list access to workset paths and sentinel files
- CloudWatch metrics and logs
- SNS publish access for notifications
- EC2 describe access for cluster discovery/status

The repository includes the historical example policy in [../iam-policy.json](../iam-policy.json).

## Historical Setup Pattern

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --profile "$AWS_PROFILE")

aws iam create-policy \
  --policy-name UrsaWorksetMonitorPolicy \
  --policy-document file://iam-policy.json \
  --description "Permissions for the archived Ursa workset monitor"

aws iam attach-role-policy \
  --role-name YOUR_MONITOR_ROLE \
  --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/UrsaWorksetMonitorPolicy"
```

Review and scope the example policy before any live AWS change. Creating or attaching IAM policies is a destructive/security-sensitive operation and should be explicitly approved before execution.

## Current Notes

- TapDB persistence is Postgres-backed and is not an AWS service API.
- TapDB lifecycle is delegated to the `tapdb` CLI when Ursa explicitly needs DB bootstrap or inspection.
- Cognito lifecycle is delegated to `daycog`.
- Sample staging and workflow execution are delegated to `daylily-ec`.
- Cluster delete operations should use Ursa's delete-plan path before any live delete.
