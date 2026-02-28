# Phase 6: Deployment & Infrastructure

**Goal:** Deploy anywhere with one command.

## Deployment Targets

| Target | Command | Notes |
|--------|---------|-------|
| Local Docker | `docker compose up` | Fastest to start |
| AWS Lambda | `terraform apply` | Serverless, pay-per-request |
| Kubernetes | `kubectl apply -f k8s/` | Production scale |
| Cloudflare Workers | `wrangler deploy` | Edge computing |
| Vercel | `vercel deploy` | Serverless functions |

## Docker Compose (Local Dev)

```yaml
# docker-compose.yaml
version: "3.8"

services:
  agent:
    build: .
    ports: ["3000:3000"]
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/agent
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis]
    volumes: ["./data:/app/data"]

  postgres:
    image: pgvector/pgvector:pg16
    environment: [POSTGRES_PASSWORD=password, POSTGRES_DB=agent]
    volumes: [postgres_data:/var/lib/postgresql/data]
    ports: ["5432:5432"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: [redis_data:/data]

  qdrant:  # Optional vector DB
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: [qdrant_data:/qdrant/storage]

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
```

## Terraform — AWS Lambda + DynamoDB + API Gateway

```hcl
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = var.aws_region }

resource "aws_lambda_function" "agent" {
  filename      = "agent.zip"
  function_name = "universal-agent"
  role          = aws_iam_role.agent_role.arn
  handler       = "index.handler"
  runtime       = "nodejs20.x"
  timeout       = 300
  memory_size   = 1024

  environment {
    variables = {
      MEMORY_TABLE      = aws_dynamodb_table.memory.name
      ANTHROPIC_API_KEY = var.anthropic_api_key
    }
  }
}

resource "aws_dynamodb_table" "memory" {
  name         = "agent-memory"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute { name = "id", type = "S" }

  global_secondary_index {
    name            = "session-index"
    hash_key        = "session_id"
    projection_type = "ALL"
  }
}

resource "aws_apigatewayv2_api" "agent_api" {
  name          = "agent-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id           = aws_apigatewayv2_api.agent_api.id
  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.agent.invoke_arn
}
```

## Kubernetes (Production)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: universal-agent
spec:
  replicas: 3
  selector:
    matchLabels: { app: agent }
  template:
    metadata:
      labels: { app: agent }
    spec:
      containers:
      - name: agent
        image: your-registry/universal-agent:latest
        ports: [{ containerPort: 3000 }]
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef: { name: agent-secrets, key: anthropic-api-key }
        resources:
          requests: { memory: "512Mi", cpu: "500m" }
          limits:   { memory: "2Gi", cpu: "2000m" }
---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector: { app: agent }
  ports: [{ port: 80, targetPort: 3000 }]
  type: LoadBalancer
```

## CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yaml
name: Deploy Agent

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Setup Node
      uses: actions/setup-node@v4
      with: { node-version: "20" }

    - name: Install & Test
      run: |
        npm ci
        npm test
        npm run lint

    - name: Security Scan
      uses: snyk/actions/node@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

    - name: Build Docker Image
      run: |
        docker build -t ${{ secrets.REGISTRY }}/universal-agent:${{ github.sha }} .
        docker push ${{ secrets.REGISTRY }}/universal-agent:${{ github.sha }}

    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/universal-agent \
          agent=${{ secrets.REGISTRY }}/universal-agent:${{ github.sha }}
```

## Deployment Manager Pattern

```typescript
class DeploymentManager {
  async deploy(target: DeploymentTarget, config: DeploymentConfig): Promise<void> {
    await this.build(config);      // 1. Build
    await this.test(config);       // 2. Test (never skip)
    const artifact = await this.package(target, config); // 3. Package
    await this.deployToTarget(target, artifact, config); // 4. Deploy
    await this.smokeTest(target, config); // 5. Validate
  }

  private async deployToTarget(target: DeploymentTarget, artifact: Artifact, config: DeploymentConfig): Promise<void> {
    switch (target.type) {
      case "aws-lambda":          await this.deployToLambda(artifact, config); break;
      case "gcp-cloud-functions": await this.deployToCloudFunctions(artifact, config); break;
      case "azure-functions":     await this.deployToAzureFunctions(artifact, config); break;
      case "docker":              await this.deployDocker(artifact, config); break;
      case "kubernetes":          await this.deployK8s(artifact, config); break;
      case "cloudflare-workers":  await this.deployCloudflareWorkers(artifact, config); break;
      case "vercel":              await this.deployVercel(artifact, config); break;
      default: throw new Error(`Unsupported deployment target: ${target.type}`);
    }
  }
}
```
