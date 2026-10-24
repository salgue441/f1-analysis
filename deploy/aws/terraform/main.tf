# Pitwall on AWS: S3 lakehouse + EMR Serverless (Spark 3.5) with a custom image.
# Nothing here runs a job; see docs/cloud.md for build, push and submit.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { project = var.project }
  }
}

data "aws_caller_identity" "current" {}

# --- Lakehouse bucket -----------------------------------------------------------------------

resource "aws_s3_bucket" "lakehouse" {
  bucket        = var.bucket_name
  force_destroy = false
}

resource "aws_s3_bucket_versioning" "lakehouse" {
  bucket = aws_s3_bucket.lakehouse.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lakehouse" {
  bucket = aws_s3_bucket.lakehouse.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "lakehouse" {
  bucket                  = aws_s3_bucket.lakehouse.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Delta keeps old file versions itself (time travel, VACUUM); expire S3's noncurrent copies
# and job logs so storage doesn't grow without bound.
resource "aws_s3_bucket_lifecycle_configuration" "lakehouse" {
  bucket = aws_s3_bucket.lakehouse.id
  rule {
    id     = "expire-noncurrent"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
  rule {
    id     = "expire-logs"
    status = "Enabled"
    filter {
      prefix = "logs/"
    }
    expiration {
      days = 30
    }
  }
}

# --- Image registry -------------------------------------------------------------------------

resource "aws_ecr_repository" "pitwall" {
  name                 = var.project
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

# EMR Serverless pulls the custom image as a service principal, scoped to this application.
resource "aws_ecr_repository_policy" "emr_pull" {
  repository = aws_ecr_repository.pitwall.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "EmrServerlessPull"
      Effect    = "Allow"
      Principal = { Service = "emr-serverless.amazonaws.com" }
      Action    = ["ecr:BatchGetImage", "ecr:DescribeImages", "ecr:GetDownloadUrlForLayer"]
      Condition = {
        StringEquals = { "aws:SourceArn" = aws_emrserverless_application.pitwall.arn }
      }
    }]
  })
}

# --- EMR Serverless -------------------------------------------------------------------------

resource "aws_emrserverless_application" "pitwall" {
  name          = var.project
  release_label = var.emr_release
  type          = "spark"

  # Hard ceiling on spend: the whole application can never use more than this at once.
  maximum_capacity {
    cpu    = var.max_vcpu
    memory = var.max_memory
  }

  auto_start_configuration {
    enabled = true
  }
  auto_stop_configuration {
    enabled              = true
    idle_timeout_minutes = 15
  }

  image_configuration {
    image_uri = "${aws_ecr_repository.pitwall.repository_url}:${var.image_tag}"
  }
}

data "aws_iam_policy_document" "assume_emr" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["emr-serverless.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_iam_role" "job" {
  name               = "${var.project}-emr-job"
  assume_role_policy = data.aws_iam_policy_document.assume_emr.json
}

# Jobs read and write only the lakehouse bucket (data, code, logs).
data "aws_iam_policy_document" "job" {
  statement {
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.lakehouse.arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.lakehouse.arn}/*"]
  }
}

resource "aws_iam_role_policy" "job" {
  name   = "${var.project}-lakehouse"
  role   = aws_iam_role.job.id
  policy = data.aws_iam_policy_document.job.json
}
