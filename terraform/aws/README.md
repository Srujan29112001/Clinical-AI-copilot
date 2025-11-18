# Terraform AWS Infrastructure

Deploy Clinical AI Copilot to AWS EKS with HIPAA compliance.

## Prerequisites

- Terraform 1.5+
- AWS CLI configured
- kubectl
- helm

## Infrastructure Components

- **EKS Cluster**: Kubernetes cluster with GPU nodes
- **VPC**: Private network with NAT gateways
- **RDS PostgreSQL**: HIPAA-compliant database
- **S3**: Encrypted model storage
- **KMS**: Encryption key management
- **CloudWatch**: Logging and monitoring

## Usage

### 1. Initialize Terraform

```bash
terraform init
```

### 2. Plan Deployment

```bash
terraform plan
```

### 3. Apply Configuration

```bash
terraform apply
```

### 4. Configure kubectl

```bash
aws eks update-kubeconfig --region us-east-1 --name clinical-ai-cluster
```

### 5. Verify Cluster

```bash
kubectl get nodes
kubectl get pods -n kube-system
```

### 6. Deploy Application

```bash
# Using Helm
helm install clinical-ai ../../helm/clinical-ai-copilot/ \
  --namespace clinical-ai \
  --create-namespace

# Or using Kustomize
kubectl apply -k ../../k8s/base/
```

## Cost Estimation

- EKS Cluster: ~$73/month
- GPU Nodes (2x g4dn.xlarge): ~$1,000/month
- CPU Nodes (3x m5.2xlarge): ~$800/month
- RDS (db.r6g.xlarge Multi-AZ): ~$700/month
- Data Transfer: Variable
- **Total**: ~$2,600/month

## Security Features

- Encrypted EBS volumes (KMS)
- Encrypted RDS database (KMS)
- Encrypted S3 buckets (KMS)
- VPC Flow Logs enabled
- EKS audit logging enabled
- Multi-AZ deployment
- Private subnets for workloads

## Cleanup

```bash
terraform destroy
```

## HIPAA Compliance

This configuration includes HIPAA-required features:
- Encryption at rest and in transit
- Audit logging (CloudWatch, VPC Flow Logs)
- 35-day database backups
- Multi-AZ deployment for HA
- Access controls (Security Groups, IAM)
