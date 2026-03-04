# IAM Setup Guide for Ursa Monitor

This guide describes the AWS IAM permissions needed for the **workset monitor/worker** to interact with AWS services (S3, SNS, CloudWatch, EC2).

TapDB persistence is **Postgres-backed**. It is not an AWS “NoSQL table” service, so you should not grant IAM permissions for a fictional `tapdb:*` API.

## Overview

The monitor typically needs AWS permissions for:

- **S3**: read/list workset paths and read/write sentinel files
- **CloudWatch**: publish metrics and write logs (optional, depending on deployment)
- **SNS**: publish notifications (optional)
- **EC2**: describe instances/tags for cluster discovery/status

The repository includes an example policy in [iam-policy.json](/Users/jmajor/projects/daylily/daylily-ursa/iam-policy.json).

## Quick Setup (AWS CLI)

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --profile "$AWS_PROFILE")

# 1. Create the IAM policy
aws iam create-policy \
  --policy-name UrsaWorksetMonitorPolicy \
  --policy-document file://iam-policy.json \
  --description "Permissions for Ursa workset monitor/worker"

# 2. Attach to an existing role
aws iam attach-role-policy \
  --role-name YOUR_MONITOR_ROLE \
  --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/UrsaWorksetMonitorPolicy"
```

## Customizing The Policy

### 1) Update S3 Bucket(s)

Edit `iam-policy.json` and set the bucket ARNs to your workset bucket(s).

### 2) Update SNS Topic Pattern (Optional)

If you use SNS notifications, scope the SNS topic ARN(s) to your environment.

### 3) CloudWatch Namespace

If you restrict `cloudwatch:PutMetricData` by namespace, keep it aligned with Ursa’s metric namespace (`Daylily/WorksetMonitor`).

## TapDB Note (If Using IAM Auth or Secrets Manager)

Depending on your TapDB deployment, the process may additionally need:

- `secretsmanager:GetSecretValue` for the DB secret ARN configured in TapDB, and
- RDS IAM auth permissions (for example `rds-db:connect`) if enabled by your TapDB config.

Those permissions are deployment-specific and intentionally not baked into the example policy.

