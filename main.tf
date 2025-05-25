# Provider
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.40" }
  }
}

provider "aws" {}

# Inputs
variable "app_name" { type = string }
variable "openai_api_key" {
  type      = string
  sensitive = true
}
variable "image_tag" {
  type    = string
  default = "latest"
}

# 1. ECR repository (build & push via docker CLI)
resource "aws_ecr_repository" "app" {
  name = var.app_name
  image_scanning_configuration { scan_on_push = true }
}

locals { image_uri = "${aws_ecr_repository.app.repository_url}:${var.image_tag}" }

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

# Outputs
output "repository_url" { value = aws_ecr_repository.app.repository_url }
output "service_url" { value = aws_apprunner_service.app.service_url }
