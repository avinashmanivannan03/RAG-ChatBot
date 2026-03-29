terraform {
  required_version = ">= 1.6.0"
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

variable "aws_region"      { default = "us-east-1" }
variable "environment"     { default = "production" }
variable "app_name"        { description = "Application identifier used across all resource names." }
variable "vpc_cidr"        { default = "10.0.0.0/16" }
variable "instance_type"   { default = "t3.medium" }
variable "min_capacity"    { default = 2 }
variable "max_capacity"    { default = 10 }
variable "opensearch_size" { default = "r6g.large.search" }

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.app_name}-vpc"
  cidr = var.vpc_cidr
  azs              = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets   = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  enable_nat_gateway   = true
  single_nat_gateway   = false
  enable_dns_hostnames = true

  tags = { Environment = var.environment, App = var.app_name }
}

resource "aws_opensearch_domain" "vector_store" {
  domain_name    = "${var.app_name}-vectors"
  engine_version = "OpenSearch_2.13"

  cluster_config {
    instance_type          = var.opensearch_size
    instance_count         = 2
    zone_awareness_enabled = true
  }

  ebs_options {
    ebs_enabled = true
    volume_size = 100
    volume_type = "gp3"
  }

  encrypt_at_rest          { enabled = true }
  node_to_node_encryption  { enabled = true }

  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }

  vpc_options {
    subnet_ids         = [module.vpc.private_subnets[0], module.vpc.private_subnets[1]]
    security_group_ids = [aws_security_group.opensearch.id]
  }

  tags = { Environment = var.environment }
}

resource "aws_security_group" "opensearch" {
  name        = "${var.app_name}-opensearch-sg"
  description = "Allow inbound from ECS tasks only"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ecs_tasks" {
  name        = "${var.app_name}-ecs-sg"
  description = "ECS task security group"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

module "ecs" {
  source  = "terraform-aws-modules/ecs/aws"
  version = "~> 5.0"

  cluster_name = "${var.app_name}-cluster"

  fargate_capacity_providers = {
    FARGATE      = { default_capacity_provider_strategy = { weight = 60 } }
    FARGATE_SPOT = { default_capacity_provider_strategy = { weight = 40 } }
  }
}

resource "aws_s3_bucket" "app_storage" {
  bucket = "${var.app_name}-storage-${var.environment}"
  tags   = { Environment = var.environment }
}

resource "aws_s3_bucket_versioning" "app_storage" {
  bucket = aws_s3_bucket.app_storage.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "app_storage" {
  bucket = aws_s3_bucket.app_storage.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_elasticache_replication_group" "cache" {
  replication_group_id       = "${var.app_name}-cache"
  description                = "Redis cache for RAG pipeline"
  node_type                  = "cache.t3.medium"
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  subnet_group_name          = aws_elasticache_subnet_group.cache.name
  security_group_ids         = [aws_security_group.ecs_tasks.id]
}

resource "aws_elasticache_subnet_group" "cache" {
  name       = "${var.app_name}-cache-subnets"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${module.ecs.cluster_name}/${var.app_name}-service"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_cpu_policy" {
  name               = "${var.app_name}-cpu-autoscaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

output "opensearch_endpoint" {
  value       = aws_opensearch_domain.vector_store.endpoint
  description = "OpenSearch domain endpoint for vector search"
}

output "ecs_cluster_name" {
  value       = module.ecs.cluster_name
  description = "ECS cluster name"
}

output "vpc_id" {
  value       = module.vpc.vpc_id
  description = "VPC ID"
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.app_storage.bucket
  description = "S3 bucket for application storage"
}
