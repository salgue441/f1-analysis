variable "region" {
  description = "AWS region for the bucket and the EMR Serverless application."
  type        = string
  default     = "eu-west-1"
}

variable "project" {
  description = "Name prefix for every resource."
  type        = string
  default     = "pitwall"
}

variable "bucket_name" {
  description = "Globally unique S3 bucket name for the lakehouse."
  type        = string
}

variable "emr_release" {
  description = "EMR release; 7.x runs Spark 3.5 like the local stack."
  type        = string
  default     = "emr-7.5.0"
}

variable "image_tag" {
  description = "Tag of the custom image in ECR (deploy/aws/Dockerfile)."
  type        = string
  default     = "latest"
}

variable "max_vcpu" {
  description = "Upper bound on vCPUs the application may use at once (cost ceiling)."
  type        = string
  default     = "32 vCPU"
}

variable "max_memory" {
  description = "Upper bound on memory the application may use at once (cost ceiling)."
  type        = string
  default     = "128 GB"
}
