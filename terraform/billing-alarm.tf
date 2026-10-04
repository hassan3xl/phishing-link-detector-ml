# Billing alarm - fires if estimated AWS charges exceed the threshold,
# regardless of which service caused it. This is account-wide, not scoped to
# this project, but it's the single cheapest guardrail against "I forgot
# something was running." Must be created in us-east-1 (billing metrics only
# publish there), independent of the main deployment region.

provider "aws" {
  alias  = "billing"
  region = "us-east-1"
}

resource "aws_sns_topic" "billing_alerts" {
  provider = aws.billing
  name     = "phishing-detection-ml-billing-alerts"
}

resource "aws_sns_topic_subscription" "billing_email" {
  provider  = aws.billing
  topic_arn = aws_sns_topic.billing_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "billing_alarm" {
  provider            = aws.billing
  alarm_name          = "phishing-detection-ml-billing-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 21600 # 6 hours - billing metrics don't update faster than this
  statistic           = "Maximum"
  threshold           = var.billing_alarm_threshold_usd
  alarm_description   = "Fires if estimated AWS charges exceed $${var.billing_alarm_threshold_usd}. Requires 'Receive Billing Alerts' enabled in account billing preferences."
  alarm_actions       = [aws_sns_topic.billing_alerts.arn]

  dimensions = {
    Currency = "USD"
  }
}
