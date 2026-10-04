terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# FREE-TIER GUARDRAILS (read this before changing anything below)
#
# - Lambda's 1M requests/month + 400,000 GB-seconds/month free tier is
#   PERMANENT (does not expire after 12 months), unlike EC2. That's why this
#   config is Lambda-only and has no EC2 resources at all.
# - memory_size is pinned at 512 MB (bumped from an initial 256 MB - that
#   was too tight a bet for the xgboost/scikit-learn/pandas/numpy import
#   footprint). Raising this further increases GB-seconds consumed per
#   invocation - keep it as low as the model/app comfortably runs.
# - No provisioned concurrency, no reserved concurrency above 1 - avoids any
#   standing cost from idle warm instances.
# - API Gateway HTTP API (not REST API) - cheaper per-request and the
#   1M-calls/month free tier applies for the account's first 12 months.
# - A CloudWatch billing alarm (billing-alarm.tf) fires if estimated charges
#   exceed $1, regardless of which service is responsible.
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "phishing_model_api" {
  name                 = "phishing-detection-ml-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # allows terraform destroy to clean up images too

  image_scanning_configuration {
    scan_on_push = true
  }
}

data "aws_caller_identity" "current" {}

# Without this, `aws_lambda_function.phishing_predict_api` fails to create
# with "Lambda does not have permission to access the ECR image" - an ECR
# repository doesn't implicitly trust the Lambda service to pull from it,
# even within the same account. The condition is scoped to this specific
# function's ARN (built from known values, not a reference to the Lambda
# resource itself, to avoid a dependency cycle - the policy must exist
# before the function can be created, so it can't depend on the function).
resource "aws_ecr_repository_policy" "allow_lambda_pull" {
  repository = aws_ecr_repository.phishing_model_api.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "AllowLambdaPull"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action = [
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
      ]
      Condition = {
        StringEquals = {
          "aws:sourceArn" = "arn:aws:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:phishing-detection-ml-predict"
        }
      }
    }]
  })
}

resource "aws_iam_role" "lambda_exec" {
  name = "phishing-detection-ml-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "phishing_predict_api" {
  function_name = "phishing-detection-ml-predict"
  role          = aws_iam_role.lambda_exec.arn

  package_type = "Image"
  image_uri    = "${aws_ecr_repository.phishing_model_api.repository_url}:latest"

  # --- free-tier pin: keep this small ---
  # Bumped 256->512 MB: Docker/RIE testing proved the code works but doesn't
  # enforce Lambda's real memory ceiling, and the xgboost+scikit-learn+
  # pandas+numpy import/cold-start footprint is a known risk at 256MB. Cost
  # difference is negligible at low invocation volume (Lambda bills
  # GB-seconds = memory x duration, and free tier covers 400,000 GB-seconds/
  # month regardless).
  memory_size = 512 # MB
  timeout     = 10  # seconds - fail fast rather than idling and billing

  # No reserved/provisioned concurrency set -> Lambda's shared pool, no
  # standing cost.

  environment {
    variables = {
      MODEL_ARTIFACT_PATH = "/var/task/models_saved/final_model.joblib"
      LOG_LEVEL           = "info"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_basic_logs,
    aws_ecr_repository_policy.allow_lambda_pull,
  ]
}

resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${aws_lambda_function.phishing_predict_api.function_name}"
  retention_in_days = 14 # keep logs from growing (and costing) indefinitely
}

# --- API Gateway (HTTP API - cheaper than REST API) ---

resource "aws_apigatewayv2_api" "phishing_api" {
  name          = "phishing-detection-ml-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                 = aws_apigatewayv2_api.phishing_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.phishing_predict_api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "predict_route" {
  api_id    = aws_apigatewayv2_api.phishing_api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_route" "health_route" {
  api_id    = aws_apigatewayv2_api.phishing_api.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.phishing_api.id
  name        = "$default"
  auto_deploy = true

  # basic access logging - useful evidence for the "how would you monitor
  # this in production" interview answer from Phase 9
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gw_logs.arn
    format = jsonencode({
      requestId       = "$context.requestId"
      status          = "$context.status"
      path            = "$context.path"
      responseLatency = "$context.responseLatency"
    })
  }
}

resource "aws_cloudwatch_log_group" "api_gw_logs" {
  name              = "/aws/apigateway/phishing-detection-ml"
  retention_in_days = 14
}

resource "aws_lambda_permission" "apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.phishing_predict_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.phishing_api.execution_arn}/*/*"
}
