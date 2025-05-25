# Retrieval-Augmented Generation on AWS

## Setup

```bash
# Install tofu
sudo snap install tofu --classic

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=...
export TF_VAR_app_name=aws-rag
export TF_VAR_openai_api_key=...
export TF_VAR_opensearch_password=...

# Run locally
docker build -t aws-rag .
docker run -e OPENAI_API_KEY="$TF_VAR_openai_api_key" -p 8000:8000 aws-rag

# Deploy
tofu init && tofu apply --auto-approve
```

Test via:

```bash
# Check the AWS runner endpoint
curl "https://$(tofu output -raw service_url)/"

# Check the OpenSearch endpoint
curl -u admin:$TF_VAR_opensearch_password "https://$(tofu output -raw opensearch_endpoint)/_cluster/health?pretty"
```

To destroy, use:

```bash
tofu destroy --auto-approve
```
