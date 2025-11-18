# Clinical AI Copilot - Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Helm Deployment](#helm-deployment)
6. [Vault Setup](#vault-setup)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or better
- **CPU**: 8+ cores
- **RAM**: 32GB+ recommended
- **Storage**: 500GB+ SSD (encrypted for HIPAA compliance)

### Software Requirements
- Docker 24.0+
- Docker Compose 2.20+
- Kubernetes 1.27+ (for K8s deployment)
- Helm 3.12+ (for Helm deployment)
- NVIDIA Container Toolkit (for GPU support)
- HashiCorp Vault (for secrets management)

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-org/Clinical-AI-copilot.git
cd Clinical-AI-copilot
```

### 2. Set Environment Variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start Services (Docker Compose)
```bash
docker-compose up -d
```

### 4. Initialize Database
```bash
docker-compose exec clinical-ai python scripts/init_database.py
```

### 5. Load Medical Ontology
```bash
docker-compose exec clinical-ai python scripts/import_medical_ontology.py \
  --icd10 data/medical_ontology/icd10_comprehensive.json \
  --snomed data/medical_ontology/snomed_ct_comprehensive.json \
  --rxnorm data/medical_ontology/rxnorm_comprehensive.json
```

### 6. Access API
- API: `http://localhost:8000`
- API Docs: `http://localhost:8000/api/docs`
- Grafana: `http://localhost:3000` (admin/admin)
- Kibana: `http://localhost:5601`

---

## Docker Deployment

### Build Image
```bash
docker build -t clinical-ai-copilot:1.0.0 .
```

### Run with GPU Support
```bash
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/models \
  -v $(pwd)/data:/data \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e MODEL_PATH=/models \
  clinical-ai-copilot:1.0.0
```

### Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f clinical-ai

# Scale API instances
docker-compose up -d --scale clinical-ai=3

# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

---

## Kubernetes Deployment

### Prerequisites
1. **Install NVIDIA Device Plugin**
```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

2. **Install Cert-Manager** (for SSL)
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

3. **Install Ingress-NGINX**
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
```

### Deploy with Kustomize

#### 1. Create Namespace
```bash
kubectl apply -f k8s/base/namespace.yaml
```

#### 2. Update Secrets
Edit `k8s/base/secrets.yaml` with production values:
```bash
# Generate encryption keys
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Generate JWT secret
openssl rand -hex 32

# Update secrets.yaml with generated values
```

#### 3. Deploy All Components
```bash
kubectl apply -k k8s/base/
```

#### 4. Deploy ELK Stack
```bash
kubectl apply -k k8s/elk-stack/
```

#### 5. Deploy Vault
```bash
kubectl apply -f k8s/vault/vault-deployment.yaml
```

#### 6. Verify Deployment
```bash
# Check pods
kubectl get pods -n clinical-ai

# Check services
kubectl get svc -n clinical-ai

# Check ingress
kubectl get ingress -n clinical-ai
```

### Initialize Vault

```bash
# Initialize Vault (run once)
kubectl exec -n clinical-ai vault-0 -- vault operator init

# Unseal Vault (repeat for each vault pod)
kubectl exec -n clinical-ai vault-0 -- vault operator unseal <unseal-key-1>
kubectl exec -n clinical-ai vault-0 -- vault operator unseal <unseal-key-2>
kubectl exec -n clinical-ai vault-0 -- vault operator unseal <unseal-key-3>

# Enable KV secrets engine
kubectl exec -n clinical-ai vault-0 -- vault secrets enable -path=secret kv-v2

# Populate secrets
kubectl exec -n clinical-ai vault-0 -- vault kv put secret/clinical-ai/database/postgres \
  username=clinical_admin \
  password=<strong-password> \
  host=postgres-service \
  port=5432
```

---

## Helm Deployment

### Prerequisites
```bash
# Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add neo4j https://helm.neo4j.com/neo4j
helm repo update
```

### Install Chart

#### 1. Update Values
Edit `helm/clinical-ai-copilot/values.yaml`:
- Set image repository
- Configure ingress hostname
- Update resource limits

#### 2. Install
```bash
helm install clinical-ai helm/clinical-ai-copilot/ \
  --namespace clinical-ai \
  --create-namespace \
  --values helm/clinical-ai-copilot/values.yaml
```

#### 3. Verify Installation
```bash
helm status clinical-ai -n clinical-ai
kubectl get pods -n clinical-ai
```

#### 4. Upgrade
```bash
helm upgrade clinical-ai helm/clinical-ai-copilot/ \
  --namespace clinical-ai \
  --values helm/clinical-ai-copilot/values.yaml
```

#### 5. Uninstall
```bash
helm uninstall clinical-ai -n clinical-ai
```

---

## Vault Setup

### Local Development
```bash
# Start Vault in dev mode (NOT for production)
vault server -dev

# Set environment variables
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='dev-token'

# Initialize secrets
python src/utils/vault_manager.py
```

### Production Setup

#### 1. High Availability Cluster
```bash
# Deploy Vault StatefulSet
kubectl apply -f k8s/vault/vault-deployment.yaml

# Initialize cluster
kubectl exec -n clinical-ai vault-0 -- vault operator init \
  -key-shares=5 \
  -key-threshold=3

# Save unseal keys and root token securely!
```

#### 2. Configure Auto-Unsealing (AWS KMS)
```bash
# Update vault.hcl with KMS seal stanza
seal "awskms" {
  region     = "us-east-1"
  kms_key_id = "alias/vault-unseal-key"
}
```

#### 3. Enable Audit Logging
```bash
kubectl exec -n clinical-ai vault-0 -- \
  vault audit enable file file_path=/vault/logs/audit.log
```

#### 4. Configure PKI for Certificates
```bash
# Enable PKI secrets engine
vault secrets enable pki
vault secrets tune -max-lease-ttl=87600h pki

# Generate root CA
vault write -field=certificate pki/root/generate/internal \
  common_name="Clinical AI Root CA" \
  ttl=87600h > CA_cert.crt

# Configure URLs
vault write pki/config/urls \
  issuing_certificates="http://vault-service:8200/v1/pki/ca" \
  crl_distribution_points="http://vault-service:8200/v1/pki/crl"

# Create role
vault write pki/roles/clinical-ai \
  allowed_domains="clinical-ai.example.com" \
  allow_subdomains=true \
  max_ttl=72h
```

---

## Monitoring

### Prometheus Metrics
Access Prometheus:
```bash
kubectl port-forward -n clinical-ai svc/prometheus-service 9090:9090
```
Open: `http://localhost:9090`

### Grafana Dashboards
Access Grafana:
```bash
kubectl port-forward -n clinical-ai svc/grafana-service 3000:3000
```
Open: `http://localhost:3000` (admin / see secrets)

Import dashboard: `monitoring/grafana_dashboards/clinical_ai_dashboard.json`

### Kibana Logs
Access Kibana:
```bash
kubectl port-forward -n clinical-ai svc/kibana-service 5601:5601
```
Open: `http://localhost:5601`

---

## Troubleshooting

### Pod Crashes

#### Check logs
```bash
kubectl logs -n clinical-ai <pod-name>
kubectl describe pod -n clinical-ai <pod-name>
```

#### Common Issues

**Out of GPU Memory**
```bash
# Reduce batch size
kubectl set env deployment/clinical-ai -n clinical-ai MAX_BATCH_SIZE=2

# Enable model quantization
kubectl set env deployment/clinical-ai -n clinical-ai USE_QUANTIZATION=true
```

**Database Connection Fails**
```bash
# Check postgres is running
kubectl get pods -n clinical-ai -l app=postgres

# Test connection
kubectl exec -n clinical-ai postgres-0 -- psql -U clinical_admin -d clinical_ai -c '\l'

# Check secrets
kubectl get secret -n clinical-ai clinical-ai-secrets -o yaml
```

**Kafka Issues**
```bash
# Check Kafka pods
kubectl get pods -n clinical-ai -l app=kafka

# Check topics
kubectl exec -n clinical-ai kafka-0 -- kafka-topics.sh --bootstrap-server localhost:9092 --list

# Create missing topics
kubectl exec -n clinical-ai kafka-0 -- kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic eeg-stream --partitions 3 --replication-factor 1
```

### Performance Issues

#### Check GPU Utilization
```bash
# From within pod
kubectl exec -n clinical-ai <pod-name> -- nvidia-smi

# Metrics
kubectl port-forward -n clinical-ai svc/prometheus-service 9090:9090
# Query: clinical_ai_gpu_utilization_percent
```

#### Check Resource Limits
```bash
kubectl top nodes
kubectl top pods -n clinical-ai
```

#### Enable Horizontal Pod Autoscaling
```bash
kubectl autoscale deployment clinical-ai \
  --namespace clinical-ai \
  --min=2 \
  --max=10 \
  --cpu-percent=70
```

### Network Issues

#### Check Services
```bash
kubectl get svc -n clinical-ai
kubectl get endpoints -n clinical-ai
```

#### Test Internal Connectivity
```bash
# Test from clinical-ai pod
kubectl exec -n clinical-ai <pod-name> -- curl postgres-service:5432
kubectl exec -n clinical-ai <pod-name> -- curl neo4j-service:7687
```

#### Check Ingress
```bash
kubectl get ingress -n clinical-ai
kubectl describe ingress -n clinical-ai clinical-ai-ingress
```

### HIPAA Audit Logs

#### Access Audit Logs
```bash
# From PostgreSQL
kubectl exec -n clinical-ai postgres-0 -- \
  psql -U clinical_admin -d clinical_ai -c 'SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 10;'

# From filesystem
kubectl exec -n clinical-ai <pod-name> -- ls -la /var/log/clinical-ai/

# From Elasticsearch
curl -X GET "localhost:9200/clinical-ai-logs-*/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": { "match_all": {} },
  "size": 10,
  "sort": [{"@timestamp": {"order": "desc"}}]
}
'
```

### Backup and Recovery

#### PostgreSQL Backup
```bash
# Manual backup
kubectl exec -n clinical-ai postgres-0 -- \
  pg_dump -U clinical_admin clinical_ai | gzip > backup_$(date +%Y%m%d).sql.gz

# Restore
gunzip < backup_20240101.sql.gz | \
  kubectl exec -i -n clinical-ai postgres-0 -- \
  psql -U clinical_admin -d clinical_ai
```

#### Neo4j Backup
```bash
# Backup
kubectl exec -n clinical-ai neo4j-0 -- \
  neo4j-admin backup --backup-dir=/backups --database=neo4j

# Copy backup out
kubectl cp clinical-ai/neo4j-0:/backups ./neo4j-backups
```

---

## Security Checklist

- [ ] Change all default passwords in `secrets.yaml`
- [ ] Generate strong encryption keys
- [ ] Configure SSL/TLS certificates
- [ ] Enable network policies
- [ ] Configure RBAC
- [ ] Enable audit logging
- [ ] Set resource limits
- [ ] Enable pod security policies
- [ ] Configure Vault for secrets management
- [ ] Test disaster recovery procedures
- [ ] Configure automated backups
- [ ] Enable monitoring and alerting
- [ ] Review and test HIPAA compliance

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/Clinical-AI-copilot/issues
- Documentation: https://docs.clinical-ai.example.com
- Email: support@clinical-ai.example.com
