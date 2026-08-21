#!/usr/bin/env bash
#
# Deploy the Visual Inspection API to AWS App Runner.
#
# Prereqs:
#   - AWS CLI v2 configured (`aws configure` or SSO) with permissions for
#     ECR + App Runner in the target region.
#   - Docker running locally (Colima or Docker Desktop).
#   - yolov8m.onnx present in the project root (it is gitignored; keep it locally).
#
# Usage:
#   export AWS_ACCOUNT_ID=123456789012
#   export APP_RUNNER_SERVICE_ARN=arn:aws:apprunner:eu-west-1:123456789012:service/visual-inspection-api/xxxx
#   ./scripts/deploy.sh
#
set -euo pipefail

# ---------- Config (override via env) ----------
AWS_REGION="${AWS_REGION:-eu-west-1}"          # eu-west-1 = Ireland, closest to you
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID (your 12-digit AWS account id)}"
ECR_REPO="visual-inspection-api"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SERVICE_ARN="${APP_RUNNER_SERVICE_ARN:-}"       # optional: leave empty to skip auto-deploy

ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"

# ---------- 1. Authenticate Docker to ECR ----------
echo "-> Logging in to ECR ($AWS_REGION)"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin \
      "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# ---------- 2. Create ECR repo if it does not exist ----------
if ! aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "-> Creating ECR repo: $ECR_REPO"
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null
fi

# ---------- 3. Build & push ----------
echo "-> Building image: $ECR_URI"
docker build -t "$ECR_URI" .
echo "-> Pushing image"
docker push "$ECR_URI"

# ---------- 4. Trigger App Runner deployment (optional) ----------
if [[ -n "$SERVICE_ARN" ]]; then
  echo "-> Triggering App Runner deployment"
  aws apprunner start-deployment --service-arn "$SERVICE_ARN" --region "$AWS_REGION"
  echo "✓ Done. App Runner is rolling out $ECR_URI"
else
  echo "✓ Image pushed to $ECR_URI"
  echo "  Next: create the service once with"
  echo "    aws apprunner create-service --cli-input-yaml file://deploy/apprunner.yaml"
  echo "  or, if it already exists, run:"
  echo "    aws apprunner start-deployment --service-arn <SERVICE_ARN> --region $AWS_REGION"
fi
