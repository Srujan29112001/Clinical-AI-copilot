# Clinical AI Copilot - AWS EKS Infrastructure
# Terraform configuration for HIPAA-compliant deployment

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }

  backend "s3" {
    bucket         = "clinical-ai-terraform-state"
    key            = "eks/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "clinical-ai-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Clinical-AI-Copilot"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Compliance  = "HIPAA"
    }
  }
}

# VPC for EKS
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "clinical-ai-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = false  # Multiple NAT for HA
  enable_dns_hostnames = true
  enable_dns_support   = true

  # HIPAA: Enable VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "clinical-ai-cluster"
  cluster_version = "1.27"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # HIPAA: Enable encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # Enable control plane logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  # GPU Node Group
  eks_managed_node_groups = {
    gpu_nodes = {
      name           = "gpu-node-group"
      instance_types = ["g4dn.xlarge"]  # Tesla T4 GPU
      ami_type       = "AL2_x86_64_GPU"

      min_size     = 1
      max_size     = 10
      desired_size = 2

      disk_size      = 200
      disk_encrypted = true
      disk_kms_key_id = aws_kms_key.ebs.arn

      labels = {
        gpu = "nvidia"
      }

      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }

    # CPU-only nodes for databases
    cpu_nodes = {
      name           = "cpu-node-group"
      instance_types = ["m5.2xlarge"]

      min_size     = 2
      max_size     = 5
      desired_size = 3

      disk_size      = 100
      disk_encrypted = true
      disk_kms_key_id = aws_kms_key.ebs.arn
    }
  }
}

# KMS Keys for encryption (HIPAA requirement)
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

resource "aws_kms_key" "ebs" {
  description             = "EBS Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

resource "aws_kms_key" "s3" {
  description             = "S3 Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

# S3 Buckets for data storage
resource "aws_s3_bucket" "models" {
  bucket = "clinical-ai-models-${var.environment}"
}

resource "aws_s3_bucket_encryption" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3.arn
    }
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}

# RDS PostgreSQL (HIPAA-compliant)
resource "aws_db_instance" "postgres" {
  identifier = "clinical-ai-db"

  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.r6g.xlarge"

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.ebs.arn

  db_name  = "clinical_ai"
  username = "clinical_admin"
  password = random_password.db_password.result

  multi_az               = true
  publicly_accessible    = false
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = 35  # HIPAA: 7+ years
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  deletion_protection = true
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "aws_db_subnet_group" "main" {
  name       = "clinical-ai-db-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "rds" {
  name   = "clinical-ai-rds-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
  }
}

# Install NVIDIA Device Plugin
resource "kubernetes_daemonset" "nvidia_device_plugin" {
  metadata {
    name      = "nvidia-device-plugin"
    namespace = "kube-system"
  }

  spec {
    selector {
      match_labels = {
        name = "nvidia-device-plugin-ds"
      }
    }

    template {
      metadata {
        labels = {
          name = "nvidia-device-plugin-ds"
        }
      }

      spec {
        container {
          image = "nvcr.io/nvidia/k8s-device-plugin:v0.14.0"
          name  = "nvidia-device-plugin-ctr"

          security_context {
            allow_privilege_escalation = true
            capabilities {
              drop = ["ALL"]
              add  = ["SYS_ADMIN"]
            }
          }

          volume_mount {
            name       = "device-plugin"
            mount_path = "/var/lib/kubelet/device-plugins"
          }
        }

        volume {
          name = "device-plugin"
          host_path {
            path = "/var/lib/kubelet/device-plugins"
          }
        }
      }
    }
  }

  depends_on = [module.eks]
}

# Outputs
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "rds_endpoint" {
  value = aws_db_instance.postgres.endpoint
}
