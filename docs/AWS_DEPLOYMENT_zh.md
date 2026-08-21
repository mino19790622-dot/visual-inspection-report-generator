# AWS 部署操作手册 — 视觉检测报告生成器

本手册带你把项目从 **cloud-ready（已具备上云条件）** 真正部署到 **AWS App Runner** 上运行。
所有命令都在 **你自己的** AWS 账号下执行，不会替你自动运行。预计一次性配置约 30 分钟。

> **费用提醒：** App Runner 在运行期间至少保持 1 个实例，**不会自动缩到 0**，所以 live 期间
> 是 7×24 计费的（1 vCPU / 2 GB 约几欧元/月）。ECR 镜像存储和 S3 几乎免费。
> **演示完务必执行第 8 步的拆除命令，停止计费。** 新账号通常有 3 个月试用额度
> （9 万容器分钟）。

---

## 前置条件

> 所有命令都在**你 Mac 的终端**里执行（先 `cd ~/Desktop/visual-inspection-report-generator`）。
> AWS CLI 是装在你电脑上的"遥控器"，它把命令发到 AWS；产生的资源落在你的 AWS 账号里。

```bash
# 0. 注册 AWS 账号（免费层）：打开 https://aws.amazon.com 注册，无需付费即可开始
#    （演示完记得用第 8 步拆除资源，避免计费）

# 1. 安装 AWS CLI v2（用 Homebrew，你电脑上已有）
brew install awscli
aws --version          # 应显示 >= 2.x，确认安装成功

# 2. 拿到访问凭证：
#    - 登录 AWS 控制台 → IAM → 新建用户（如 mino-deploy），勾选"编程访问"
#    - 附加权限（个人演示可用 AdministratorAccess，或第 3 步用到的 ECR/AppRunner/S3/IAM 权限）
#    - 保存生成的 Access Key ID 和 Secret Access Key
#    然后在终端配置（按提示粘贴上面的 Key，区域填 eu-west-1，输出格式 json）：
aws configure
#    更安全的方式是 SSO：aws configure sso（推荐，但需先在控制台开 IAM Identity Center）

# 3. 设置区域并确认身份
export AWS_REGION=eu-west-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "账号: $AWS_ACCOUNT_ID"   # 能打印出 12 位账号 ID 即说明登录成功

# 4. Docker 必须在运行（Colima 或 Docker Desktop）
docker info >/dev/null && echo "docker 正常"

# 5. yolov8m.onnx 必须在项目根目录（已被 gitignore，仅本地保留）
ls -la yolov8m.onnx
```

---

## 第 1 步 — 把 API Key 存进 AWS Secrets Manager

```bash
aws secretsmanager create-secret \
  --name inspection/dashscope \
  --secret-string "{\"DASHSCOPE_API_KEY\":\"$DASHSCOPE_API_KEY\"}"

export SECRET_ARN=$(aws secretsmanager describe-secret \
  --secret-id inspection/dashscope --query ARN --output text)
echo "密钥 ARN: $SECRET_ARN"
```

> 如果环境变量里没有 `$DASHSCOPE_API_KEY`，把命令里的 `$DASHSCOPE_API_KEY` 换成真实 key。
> 这个 ARN 要填进 `deploy/apprunner.yaml` 的 `RuntimeEnvironmentSecrets`。

---

## 第 2 步 — 把 ONNX 模型传上 S3（给 CI 用）

模型被 gitignore，所以 GitHub Actions 远程构建时拿不到，要先存到 S3，工作流在构建时下载
（`MODEL_S3_URI`）。

```bash
aws s3 mb s3://visual-inspection-models-$AWS_ACCOUNT_ID --region $AWS_REGION
aws s3 cp yolov8m.onnx s3://visual-inspection-models-$AWS_ACCOUNT_ID/yolov8m.onnx

export MODEL_S3_URI="s3://visual-inspection-models-$AWS_ACCOUNT_ID/yolov8m.onnx"
echo "模型地址: $MODEL_S3_URI"
```

把 `MODEL_S3_URI` 设为 **GitHub 仓库密钥**（Settings → Secrets → Actions），
这样工作流文件里就不用写死你的桶名。

---

## 第 3 步 — 配置 GitHub OIDC（不存任何 AWS 密钥）

### 3a. 把 GitHub 注册为 OIDC 身份提供方

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

> 如果指纹被拒，用下面命令重新获取当前指纹：
> `openssl s_client -servername token.actions.githubusercontent.com -showcerts \
>   -connect token.actions.githubusercontent.com:443 2>/dev/null | openssl x509 -fingerprint -noout`

### 3b. 创建带信任策略的 IAM 角色

`gh-oidc-trust.json`：
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
把里面的 `ACCOUNT_ID` 换成 `$AWS_ACCOUNT_ID`，然后执行：
```bash
sed "s/ACCOUNT_ID/$AWS_ACCOUNT_ID/" gh-oidc-trust.json > /tmp/trust.json
aws iam create-role --role-name github-actions-deploy \
  --assume-role-policy-document file:///tmp/trust.json
```

### 3c. 附加权限策略

`gh-oidc-permissions.json`：
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
echo "角色 ARN: $AWS_ROLE_ARN"
```

### 3d. 把角色 ARN 存成 GitHub 密钥 `AWS_ROLE_ARN`

---

## 第 4 步 — 把密钥 ARN 填进服务定义

编辑 `deploy/apprunner.yaml`：
- 把 `ImageIdentifier` 里的 `<AWS_ACCOUNT_ID>` 换成真实账号 id
- 把 `RuntimeEnvironmentSecrets` 里 `ValueFrom` 的占位符换成第 1 步的 `$SECRET_ARN`

```bash
sed -i.bak "s/<AWS_ACCOUNT_ID>/$AWS_ACCOUNT_ID/g" deploy/apprunner.yaml
sed -i.bak "s|arn:aws:secretsmanager:eu-west-1:<AWS_ACCOUNT_ID>:secret:inspection/dashscope-XXXXXX|$SECRET_ARN|" deploy/apprunner.yaml
```

---

## 第 5 步 — 创建 App Runner 服务

```bash
export APP_RUNNER_SERVICE_ARN=$(aws apprunner create-service \
  --cli-input-yaml file://deploy/apprunner.yaml \
  --query 'Service.ServiceArn' --output text)

echo "服务 ARN: $APP_RUNNER_SERVICE_ARN"
```

App Runner 会从 ECR 拉取 `:latest` 镜像。如果镜像还不存在，可以：
- **先本地构建（首次推荐）**：执行 `./scripts/deploy.sh`，它会构建并推送镜像，再触发一次部署；或
- **先推送镜像**，再创建服务。

---

## 第 6 步 — 部署（本地路径）

```bash
export APP_RUNNER_SERVICE_ARN   # 来自第 5 步
./scripts/deploy.sh             # 登录 → 构建 → 推 ECR → 触发部署
```

---

## 第 7 步 — 验证已经上线

```bash
URL=$(aws apprunner describe-service --service-arn $APP_RUNNER_SERVICE_ARN \
  --query 'Service.ServiceUrl' --output text)
echo "https://$URL"

curl https://$URL/health
curl -F "image=@data/test_images/bus.jpg" https://$URL/inspect
```

浏览器打开 `https://$URL/docs` 是交互式 Swagger 文档。

---

## 第 8 步 — 拆除（停止计费）

```bash
aws apprunner delete-service --service-arn $APP_RUNNER_SERVICE_ARN
# 如需彻底清理，还可执行：
# aws iam delete-role-policy --role-name github-actions-deploy --policy-name deploy-perms
# aws iam delete-role --role-name github-actions-deploy
# aws iam delete-open-id-connect-provider --open-id-connect-provider-arn arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com
# aws s3 rb s3://visual-inspection-models-ACCOUNT_ID --force
```

---

## GitHub 密钥总表

| 密钥名 | 取值 |
|--------|-------|
| `AWS_ROLE_ARN` | 第 3c 步得到 |
| `APP_RUNNER_SERVICE_ARN` | 第 5 步得到 |
| `MODEL_S3_URI` | 第 2 步得到 |

---

## 备选方案 — ECS Fargate（App Runner 被限制时的通道）

> **背景**：新注册的 AWS 账号可能对 App Runner 报
> `SubscriptionRequiredException`（连 `list-services` 读操作都被拒），
> 而 SNS/ECR/ECS 等服务正常。这是 AWS 新账号的已知开通滞后，
> 可提免费 Account & Billing 工单或等 72 小时。
> 期间可以走 **ECS Fargate** 先把服务跑起来——镜像、Secrets Manager、ECR 全部复用。

### ECS 部署 5 条命令（前置：第 1/2/3 步 + `./scripts/deploy.sh` 已完成）

```bash
# 1) 任务执行角色：拉 ECR 镜像 + 读 Secrets Manager（一次性）
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document file://deploy/ecs-execution-trust.json
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam put-role-policy --role-name ecsTaskExecutionRole \
  --policy-name read-dashscope-secret \
  --policy-document file://deploy/ecs-execution-secrets-policy.json

# 2) 日志组 + 集群 + 任务定义（一次性）
aws logs create-log-group --log-group-name /ecs/visual-inspection-api
aws ecs create-cluster --cluster-name default
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json

# 3) 安全组：放行 8000 端口（一次性）
SG_ID=$(aws ec2 create-security-group --group-name visual-inspection-api-sg \
  --description "Allow 8000 for visual inspection API" \
  --vpc-id $(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' --output text) --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0

# 4) 启动任务（公网 IP 模式，无需负载均衡器）
TASK_ARN=$(aws ecs run-task --cluster default --launch-type FARGATE \
  --task-definition visual-inspection-api:2 --count 1 \
  --network-configuration "awsvpcConfiguration={subnets=[<子网ID>],assignPublicIp=ENABLED,securityGroups=[$SG_ID]}" \
  --query 'tasks[0].taskArn' --output text)

# 5) 拿公网 IP 并验证
ENI=$(aws ecs describe-tasks --cluster default --tasks $TASK_ARN \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value|[0]' --output text)
IP=$(aws ec2 describe-network-interfaces --network-interface-ids $ENI \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
curl http://$IP:8000/health
curl -F "image=@uploads/xxx.jpg" http://$IP:8000/inspect
```

### 踩坑记录

| 现象 | 根因 | 修复 |
|------|------|------|
| VLM 调用 401 `invalid_api_key` | Secrets Manager 里存的是 `{"DASHSCOPE_API_KEY": "..."}` 整段 JSON（139 字符）而不是裸 key（115 字符），ECS 原样注入环境变量 | `aws secretsmanager put-secret-value --secret-id inspection/dashscope --secret-string "<裸key>"`，然后重启任务 |
| `Unable to assume the service linked role` | 新账号缺 ECS 服务关联角色 | 已自动存在则忽略；否则 `aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com` |
| 任务公网 IP 在 `describe-tasks` 里看不到 | 要通过 ENI 二跳查询 | 见上面第 5 步 |

### 费用与停止（重要）

Fargate 按 vCPU/内存计费：1 vCPU + 2 GB ≈ **$0.05/小时 ≈ $1.2/天 ≈ $35/月**。
演示验证完建议停掉：

```bash
aws ecs stop-task --cluster default --task $TASK_ARN
# ECR 里的 1.78GB 镜像存储费约几美分/月，可忽略
```
