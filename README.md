# Retrieval-Augmented Generation on AWS

## Setup

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=...
export TF_VAR_app_name=aws-rag
export TF_VAR_openai_api_key=...

# Run locally
docker build -t aws-rag .
docker run -e OPENAI_API_KEY="$TF_VAR_openai_api_key" -p 8000:8000 aws-rag

# Initialise Terraform/OpenTofu
tofu init

# Create the ECR repo
tofu apply -target=aws_ecr_repository.app --auto-approve

# Get the repo URL
export REPO_URL=$(tofu output -raw repository_url)

# Build and push the image
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO_URL%/*}"
docker build -t "$REPO_URL:latest" .
docker push "$REPO_URL:latest"

# Deploy App Runner
tofu apply --auto-approve

# Get the service URL
tofu output -raw service_url
```

To destroy, use:

```bash
tofu destroy --auto-approve
```
