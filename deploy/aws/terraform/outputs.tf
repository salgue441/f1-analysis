output "bucket" {
  value = aws_s3_bucket.lakehouse.bucket
}

output "lakehouse_root" {
  description = "Set as PITWALL_LAKEHOUSE_ROOT (run_job.sh does this)."
  value       = "s3://${aws_s3_bucket.lakehouse.bucket}/lakehouse"
}

output "application_id" {
  value = aws_emrserverless_application.pitwall.id
}

output "job_role_arn" {
  value = aws_iam_role.job.arn
}

output "ecr_repository_url" {
  value = aws_ecr_repository.pitwall.repository_url
}
