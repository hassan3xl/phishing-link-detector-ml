variable "aws_region" {
  description = "AWS region to deploy into. Pick one close to you to minimize latency; free-tier limits are per-account, not per-region."
  type        = string
  default     = "us-east-1"
}

variable "billing_alarm_threshold_usd" {
  description = "Estimated-charges threshold (USD) that triggers the billing alarm. Kept low since this project should stay in free tier."
  type        = number
  default     = 1
}

variable "alert_email" {
  description = "Email address to notify if the billing alarm fires. Required - set via terraform.tfvars or -var flag, not committed to git."
  type        = string
}

output "api_endpoint" {
  description = "Public invoke URL for the deployed prediction API"
  value       = aws_apigatewayv2_api.phishing_api.api_endpoint
}

output "ecr_repository_url" {
  description = "Push Docker images here before the Lambda function can use them"
  value       = aws_ecr_repository.phishing_model_api.repository_url
}
