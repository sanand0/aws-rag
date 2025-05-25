# Providers
terraform {
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.40" }
    docker = { source = "kreuzwerker/docker", version = "~> 3.0" }
  }
}

provider "aws" {}

# Inputs
variable "app_name" { type = string }
variable "openai_api_key" { sensitive = true }
variable "image_tag" { default = "latest" }
variable "opensearch_user" { default = "admin" }
variable "opensearch_password" { sensitive = true }

# 0. Auth: let Terraform log docker in to our ECR registry
data "aws_ecr_authorization_token" "this" {}
provider "docker" {
  # works for both local `tofu apply` and CI runners that have /var/run/docker.sock
  registry_auth {
    address  = data.aws_ecr_authorization_token.this.proxy_endpoint
    username = data.aws_ecr_authorization_token.this.user_name
    password = data.aws_ecr_authorization_token.this.password
  }
}

# 1. ECR repository (build & push via docker CLI)
resource "aws_ecr_repository" "app" {
  name = var.app_name
  image_scanning_configuration { scan_on_push = true }
}

resource "docker_image" "app" {
  name = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  build {
    context    = path.module # uses the Dockerfile already in root
    dockerfile = "Dockerfile"
    platform   = "linux/amd64"
  }

  # Re-build only when something inside Docker context changes
  triggers = {
    dir_sha1 = filesha1("Dockerfile") # add more files if needed
  }
}

resource "docker_registry_image" "app" {
  name = docker_image.app.name
}

locals {
  image_uri = docker_registry_image.app.name
}

# 2. IAM role that lets App Runner pull from ECR
data "aws_iam_policy_document" "assume_apprunner" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_pull" {
  name               = "${var.app_name}-pull-role"
  assume_role_policy = data.aws_iam_policy_document.assume_apprunner.json
}

resource "aws_iam_role_policy_attachment" "ecr_readonly" {
  role       = aws_iam_role.apprunner_pull.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# 3. App Runner service
resource "aws_apprunner_service" "app" {
  service_name = var.app_name

  source_configuration {
    authentication_configuration { access_role_arn = aws_iam_role.apprunner_pull.arn }

    image_repository {
      image_repository_type = "ECR"
      image_identifier      = local.image_uri
      image_configuration {
        port                          = "8000"
        runtime_environment_variables = { OPENAI_API_KEY = var.openai_api_key }
      }
    }

    auto_deployments_enabled = true
  }
}

# 4. OpenSearch
resource "aws_opensearch_domain" "os" {
  domain_name = "${var.app_name}-os"

  cluster_config {
    instance_type  = "t3.small.search"
    instance_count = 1
  }

  ebs_options {
    ebs_enabled = true
    volume_size = 10    # minimum 10 GiB for t3.small.search
    volume_type = "gp2" # you can also use "gp3" if you prefer
  }

  # Turn on fine-grained access control with an internal user DB
  advanced_security_options {
    enabled                        = true
    internal_user_database_enabled = true # for HTTP basic auth
    master_user_options {
      master_user_name     = var.opensearch_user
      master_user_password = var.opensearch_password
    }
  }

  # Enable node-to-node encryption, enforce HTTPS
  node_to_node_encryption { enabled = true }
  domain_endpoint_options { enforce_https = true }
  encrypt_at_rest { enabled = true }
}

resource "aws_opensearch_domain_policy" "os" {
  domain_name = aws_opensearch_domain.os.domain_name
  access_policies = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = "*"
      Action    = "es:*"
      Resource  = "${aws_opensearch_domain.os.arn}/*"
    }]
  })
}

# Outputs
output "repository_url" { value = aws_ecr_repository.app.repository_url }
output "service_url" { value = aws_apprunner_service.app.service_url }
output "opensearch_endpoint" { value = aws_opensearch_domain.os.endpoint }
