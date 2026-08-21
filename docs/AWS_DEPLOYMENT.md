# AWS Deployment Runbook — Visual Inspection Report Generator

Step-by-step to take the project from **cloud-ready** to **live on AWS App Runner**.
Everything here runs on **your** AWS account. Nothing is executed for you; copy the
commands in order. Estimated one-time setup: ~30 min.

> **Cost warning:** App Runner runs a minimum of 1 instance continuously while
> active, so it **bills 24/7** (≈ a few euros/month at 1 vCPU / 2 GB). ECR storage
> and S3 are pennies. **Always run the teardown step (Step 8) when done demoing.**
> App Runner also had a 3-month free trial (90k container-minutes) for new accounts.

---

## Prerequisites

> All commands run in **your Mac's terminal** (first `cd ~/Desktop/visual-inspection-report-generator`).
> The AWS CLI is a "remote control" installed on your machine; it sends commands to AWS,
> while the resulting resources live in your AWS account.

```bash
# 0. Create a free-tier AWS account at https://aws.amazon.com (no charge to start;
#    remember Step 8 to tear down resources and avoid billing)

# 1. Install AWS CLI v2 (Homebrew, already on your Mac)
brew install awscli
aws --version          # should print >= 2.x

# 2. Get credentials:
#    - AWS Console -> IAM -> create user (e.g. mino-deploy), enable "programmatic access"
#    - attach permissions (AdministratorAccess is fine for a personal demo, or the
#      ECR/AppRunner/S3/IAM perms used in Step 3)
#    - save the generated Access Key ID and Secret Access Key
#    then configure in terminal (paste the keys when prompted, region eu-west-1, format json):
aws configure
#    Safer alternative: SSO via `aws configure sso` (recommended; needs IAM Identity Center)

# 3. Set region and confirm identity
export AWS_REGION=eu-west-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account: $AWS_ACCOUNT_ID"   # prints your 12-digit id => login works

# 4. Docker must be running (Colima or Docker Desktop)
docker info >/dev/null && echo "docker OK"

# 5. yolov8m.onnx must be in the project root (gitignored, kept locally)
ls -la yolov8m.onnx
```

---

## Step 1 — Store the API key in AWS Secrets Manager

```bash
aws secretsmanager create-secret \
  --name inspection/dashscope \
  --secret-string "{\"DASHSCOPE_API_KEY\":\"$DASHSCOPE_API_KEY\"}"

export SECRET_ARN=$(aws secretsmanager describe-secret \
  --secret-id inspection/dashscope --query ARN --output text)
echo "Secret ARN: $SECRET_ARN"
```

> Paste your real key instead of `$DASHSCOPE_API_KEY` if it isn't exported.
> This ARN goes into `deploy/apprunner.yaml` → `RuntimeEnvironmentSecrets`.

---

## Step 2 — S3 bucket for the ONNX model (for CI)

The model is gitignored, so GitHub Actions can't build without it. Store it in S3
and the workflow downloads it at build time (`MODEL_S3_URI`).

```bash
aws s3 mb s3://visual-inspection-models-$AWS_ACCOUNT_ID --region $AWS_REGION
aws s3 cp yolov8m.onnx s3://visual-inspection-models-$AWS_ACCOUNT_ID/yolov8m.onnx

export MODEL_S3_URI="s3://visual-inspection-models-$AWS_ACCOUNT_ID/yolov8m.onnx"
echo "Model URI: $MODEL_S3_URI"
```

Set `MODEL_S3_URI` as a **GitHub repository secret** (Settings → Secrets → Actions)
so the workflow file doesn't hardcode your bucket.

---

## Step 3 — GitHub OIDC (no stored AWS keys)

### 3a. Register GitHub as an OIDC provider

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

> If the thumbprint is rejected, fetch the current one:
> `openssl s_client -servername token.actions.githubusercontent.com -showcerts \
>   -connect token.actions.githubusercontent.com:443 2>/dev/null | openssl x509 -fingerprint -noout`

### 3b. Create the IAM role with a trust policy

`gh-oidc-trust.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
      "StringLike":  { "token.actions.githubusercontent.com:sub": "repo:mino19790622-dot/visual-inspection-report-generator:*" }
    }
  }]
}
```
Replace `ACCOUNT_ID` with `$AWS_ACCOUNT_ID`, then:
```bash
sed "s/ACCOUNT_ID/$AWS_ACCOUNT_ID/" gh-oidc-trust.json > /tmp/trust.json
aws iam create-role --role-name github-actions-deploy \
  --assume-role-policy-document file:///tmp/trust.json
```

### 3c. Attach a permissions policy

`gh-oidc-permissions.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": [
        "ecr:GetAuthorizationToken","ecr:CreateRepository","ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer","ecr:BatchGetImage","ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart","ecr:CompleteLayerUpload","ecr:PutImage" ], "Resource": "*" },
    { "Effect": "Allow", "Action": ["apprunner:*"], "Resource": "*" },
    { "Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "arn:aws:s3:::visual-inspection-models-*/*" }
  ]
}
```
```bash
aws iam put-role-policy --role-name github-actions-deploy \
  --policy-name deploy-perms \
  --policy-document file://gh-oidc-permissions.json

export AWS_ROLE_ARN=$(aws iam get-role --role-name github-actions-deploy \
  --query Role.Arn --output text)
echo "Role ARN: $AWS_ROLE_ARN"
```

### 3d. Store the role ARN as a GitHub secret `AWS_ROLE_ARN`

---

## Step 4 — Wire the secret ARN into the service definition

Edit `deploy/apprunner.yaml`:
- Replace `<AWS_ACCOUNT_ID>` in `ImageIdentifier` with your real id.
- Replace the `ValueFrom` placeholder in `RuntimeEnvironmentSecrets` with `$SECRET_ARN`.

```bash
sed -i.bak "s/<AWS_ACCOUNT_ID>/$AWS_ACCOUNT_ID/g" deploy/apprunner.yaml
sed -i.bak "s|arn:aws:secretsmanager:eu-west-1:<AWS_ACCOUNT_ID>:secret:inspection/dashscope-XXXXXX|$SECRET_ARN|" deploy/apprunner.yaml
```

---

## Step 5 — Create the App Runner service

```bash
export APP_RUNNER_SERVICE_ARN=$(aws apprunner create-service \
  --cli-input-yaml file://deploy/apprunner.yaml \
  --query 'Service.ServiceArn' --output text)

echo "Service ARN: $APP_RUNNER_SERVICE_ARN"
```

App Runner now pulls the `:latest` image from ECR. Since the repo/image may not
exist yet, either:
- **Local build first** (recommended for first run): `./scripts/deploy.sh` builds
  and pushes the image, then triggers a deployment; or
- **First push the image**, then create the service.

---

## Step 6 — Deploy (local path)

```bash
export APP_RUNNER_SERVICE_ARN   # from Step 5
./scripts/deploy.sh             # login → build → push ECR → start-deployment
```

---

## Step 7 — Verify it's live

```bash
URL=$(aws apprunner describe-service --service-arn $APP_RUNNER_SERVICE_ARN \
  --query 'Service.ServiceUrl' --output text)
echo "https://$URL"

curl https://$URL/health
curl -F "image=@data/test_images/bus.jpg" https://$URL/inspect
```

Open `https://$URL/docs` for the interactive Swagger UI.

---

## Step 8 — Teardown (stop billing)

```bash
aws apprunner delete-service --service-arn $APP_RUNNER_SERVICE_ARN
# Optional, to fully clean up:
# aws iam delete-role-policy --role-name github-actions-deploy --policy-name deploy-perms
# aws iam delete-role --role-name github-actions-deploy
# aws iam delete-open-id-connect-provider --open-id-connect-provider-arn arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com
# aws s3 rb s3://visual-inspection-models-ACCOUNT_ID --force
```

---

## GitHub secrets summary

| Secret | Value |
|--------|-------|
| `AWS_ROLE_ARN` | from Step 3c |
| `APP_RUNNER_SERVICE_ARN` | from Step 5 |
| `MODEL_S3_URI` | from Step 2 |

---

## Alternative — ECS Fargate (when App Runner is gated)

> **Background**: brand-new AWS accounts can get
> `SubscriptionRequiredException` on **all** App Runner API calls (even
> `list-services`) while SNS/ECR/ECS work fine. This is a known new-account
> activation lag for App Runner — raise a free "Account & billing" support
> case or wait ~72 h. Meanwhile, **ECS Fargate** gets the service live using
> the exact same image, secret, and ECR repo.

### ECS deployment (prereqs: Steps 1/2/3 + `./scripts/deploy.sh` done)

```bash
# 1) Task execution role: pull from ECR + read Secrets Manager (one-time)
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document file://deploy/ecs-execution-trust.json
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam put-role-policy --role-name ecsTaskExecutionRole \
  --policy-name read-dashscope-secret \
  --policy-document file://deploy/ecs-execution-secrets-policy.json

# 2) Log group + cluster + task definition (one-time)
aws logs create-log-group --log-group-name /ecs/visual-inspection-api
aws ecs create-cluster --cluster-name default
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json

# 3) Security group allowing port 8000 (one-time)
SG_ID=$(aws ec2 create-security-group --group-name visual-inspection-api-sg \
  --description "Allow 8000 for visual inspection API" \
  --vpc-id $(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' --output text) --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0

# 4) Run the task (public-IP mode, no load balancer needed)
TASK_ARN=$(aws ecs run-task --cluster default --launch-type FARGATE \
  --task-definition visual-inspection-api:2 --count 1 \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],assignPublicIp=ENABLED,securityGroups=[$SG_ID]}" \
  --query 'tasks[0].taskArn' --output text)

# 5) Get the public IP and verify
ENI=$(aws ecs describe-tasks --cluster default --tasks $TASK_ARN \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value|[0]' --output text)
IP=$(aws ec2 describe-network-interfaces --network-interface-ids $ENI \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
curl http://$IP:8000/health
curl -F "image=@uploads/xxx.jpg" http://$IP:8000/inspect
```

### Gotchas

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| VLM call returns 401 `invalid_api_key` | Secrets Manager stored the full JSON `{"DASHSCOPE_API_KEY": "..."}` (139 chars) instead of the bare key (115 chars); ECS injects it verbatim | `aws secretsmanager put-secret-value --secret-id inspection/dashscope --secret-string "<bare key>"`, then restart the task |
| `Unable to assume the service linked role` | New account missing the ECS service-linked role | Ignored if it exists; otherwise `aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com` |
| Task public IP not visible in `describe-tasks` | Query via the ENI instead | See step 5 above |

### Cost & stopping (important)

Fargate bills per vCPU/memory: 1 vCPU + 2 GB ≈ **$0.05/hour ≈ $1.2/day ≈ $35/month**.
Stop the task once you're done demoing:

```bash
aws ecs stop-task --cluster default --task $TASK_ARN
# The 1.78 GB ECR image costs a few cents/month — negligible
```
