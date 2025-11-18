# Clinical AI Copilot - Quick Start & Deployment Guide

This guide provides step-by-step instructions to run, build, and deploy the Clinical AI Copilot project.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running Locally](#running-locally)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Building the Project](#building-the-project)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Hardware
- **CPU**: 4+ cores
- **RAM**: 16GB minimum, 32GB+ recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better recommended)
- **Storage**: 100GB+ free space (SSD recommended)

### Software Dependencies
- **Operating System**: Linux (Ubuntu 20.04/22.04), macOS, or Windows with WSL2
- **Python**: 3.10 or 3.11
- **Docker**: 24.0+ (for containerized deployment)
- **Docker Compose**: 2.20+ (for multi-service deployment)
- **NVIDIA Container Toolkit**: For GPU support in Docker
- **Git**: For version control
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Srujan29112001/Clinical-AI-copilot.git
cd Clinical-AI-copilot
```

### Step 2: Install NVIDIA Container Toolkit (For GPU Support)
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 3: Install Python Dependencies (For Local Development)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Configuration

### Step 1: Set Up Environment Variables
```bash
# Copy the example environment file
cp .env.example .env
```

### Step 2: Edit .env File
Open `.env` in your favorite text editor and configure the following critical settings:

```bash
# Generate encryption key
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Copy the output and paste it in .env as ENCRYPTION_KEY

# Generate JWT secret
python3 -c "import secrets; print(secrets.token_hex(32))"
# Copy the output and paste it in .env as JWT_SECRET_KEY
```

#### Minimum Required Variables
```env
# Security
ENCRYPTION_KEY=<generated-key-from-above>
JWT_SECRET_KEY=<generated-jwt-secret-from-above>

# Database Passwords (Change these!)
POSTGRES_PASSWORD=your-secure-postgres-password
NEO4J_PASSWORD=your-secure-neo4j-password
REDIS_PASSWORD=your-secure-redis-password
GRAFANA_PASSWORD=your-secure-grafana-password

# Model Configuration
MODEL_PATH=/models
CUDA_VISIBLE_DEVICES=0
```

### Step 3: Create Required Directories
```bash
# Create necessary directories
mkdir -p logs models data/medical_ontology monitoring/grafana_dashboards
```

---

## Running Locally

### Option A: Run API Server Directly (Development)
```bash
# Activate virtual environment
source venv/bin/activate

# Start required services (PostgreSQL, Neo4j, Redis, Kafka)
# You can use Docker Compose for just the databases:
docker-compose up -d postgres neo4j redis kafka zookeeper qdrant

# Run the API server
python3 -m src.api.main

# Or use uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: Run with Docker Compose (Recommended)
See [Docker Deployment](#docker-deployment) section below.

---

## Docker Deployment

### Step 1: Ensure Docker is Installed and Running
```bash
# Check Docker installation
docker --version
docker-compose --version

# Test Docker
docker run hello-world
```

### Step 2: Build Docker Images
```bash
# Build the main application image
docker build -t clinical-ai-copilot:latest .

# Or build all services using docker-compose
docker-compose build
```

### Step 3: Start All Services
```bash
# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f clinical-ai
```

### Step 4: Verify Services are Running
```bash
# Check running containers
docker-compose ps

# Check health status
curl http://localhost:8000/health
```

### Step 5: Initialize Database (First Time Only)
```bash
# Create database tables
docker-compose exec clinical-ai python3 -c "
from src.database.models import init_database
init_database()
"
```

### Step 6: Load Medical Ontology (Optional)
```bash
# Import medical ontologies into Neo4j
docker-compose exec clinical-ai python3 scripts/import_medical_ontology.py \
  --icd10 data/medical_ontology/icd10_comprehensive.json \
  --snomed data/medical_ontology/snomed_ct_comprehensive.json \
  --rxnorm data/medical_ontology/rxnorm_comprehensive.json
```

### Step 7: Access Services
- **API**: http://localhost:8000
- **API Documentation (Swagger)**: http://localhost:8000/api/docs
- **API Documentation (ReDoc)**: http://localhost:8000/api/redoc
- **Grafana Dashboard**: http://localhost:3000 (default: admin / check .env for password)
- **Prometheus Metrics**: http://localhost:9090
- **Neo4j Browser**: http://localhost:7474 (neo4j / check .env for password)

### Step 8: Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove all data (WARNING: This deletes all volumes!)
docker-compose down -v
```

---

## Kubernetes Deployment

### Prerequisites
1. **Kubernetes Cluster** (v1.27+)
   - Local: Minikube, Kind, or Docker Desktop
   - Cloud: GKE, EKS, AKS
2. **kubectl** installed and configured
3. **NVIDIA Device Plugin** for GPU support

### Step 1: Install NVIDIA Device Plugin
```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

### Step 2: Create Namespace
```bash
kubectl apply -f k8s/base/namespace.yaml
```

### Step 3: Configure Secrets
```bash
# Generate encryption keys
ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Update secrets.yaml with your values or create secrets directly
kubectl create secret generic clinical-ai-secrets \
  --namespace=clinical-ai \
  --from-literal=encryption-key="$ENCRYPTION_KEY" \
  --from-literal=jwt-secret="$JWT_SECRET" \
  --from-literal=postgres-password="your-postgres-password" \
  --from-literal=neo4j-password="your-neo4j-password" \
  --from-literal=redis-password="your-redis-password"
```

### Step 4: Deploy All Components
```bash
# Deploy using Kustomize
kubectl apply -k k8s/base/

# Or deploy individually
kubectl apply -f k8s/base/storage.yaml
kubectl apply -f k8s/base/postgres-statefulset.yaml
kubectl apply -f k8s/base/neo4j-statefulset.yaml
kubectl apply -f k8s/base/qdrant-deployment.yaml
kubectl apply -f k8s/base/kafka-statefulset.yaml
kubectl apply -f k8s/base/redis-deployment.yaml
kubectl apply -f k8s/base/clinical-ai-deployment.yaml
kubectl apply -f k8s/base/monitoring.yaml
kubectl apply -f k8s/base/ingress.yaml
```

### Step 5: Verify Deployment
```bash
# Check pods
kubectl get pods -n clinical-ai

# Check services
kubectl get svc -n clinical-ai

# Check logs
kubectl logs -n clinical-ai -l app=clinical-ai --tail=100 -f
```

### Step 6: Access the Application
```bash
# Port forward to access locally
kubectl port-forward -n clinical-ai svc/clinical-ai-service 8000:8000

# Or use ingress (configure DNS first)
# Access via: https://clinical-ai.your-domain.com
```

---

## Building the Project

### Build Docker Image
```bash
# Standard build
docker build -t clinical-ai-copilot:1.0.0 .

# Multi-platform build (for ARM and x86)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t clinical-ai-copilot:1.0.0 .

# Build with specific tag
docker build -t your-registry/clinical-ai-copilot:1.0.0 .
```

### Push to Container Registry
```bash
# Docker Hub
docker login
docker push your-username/clinical-ai-copilot:1.0.0

# AWS ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag clinical-ai-copilot:1.0.0 123456789.dkr.ecr.us-east-1.amazonaws.com/clinical-ai:1.0.0
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/clinical-ai:1.0.0

# Google Container Registry
gcloud auth configure-docker
docker tag clinical-ai-copilot:1.0.0 gcr.io/your-project/clinical-ai:1.0.0
docker push gcr.io/your-project/clinical-ai:1.0.0
```

### Build Python Package
```bash
# Install in development mode
pip install -e .

# Build distribution packages
python3 setup.py sdist bdist_wheel

# Install from built package
pip install dist/clinical_ai_copilot-1.0.0-py3-none-any.whl
```

---

## Testing

### Run Unit Tests
```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run tests in parallel
pytest -n auto
```

### Run Integration Tests
```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/test_integration.py

# Cleanup
docker-compose -f docker-compose.test.yml down -v
```

### Run Load Tests
```bash
# Install locust
pip install locust

# Run load test
python tests/load_test.py
```

### Code Quality Checks
```bash
# Format code with Black
black src/ tests/

# Lint with Flake8
flake8 src/ tests/

# Type checking with MyPy
mypy src/

# Security scan with Bandit
bandit -r src/
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Docker Compose Fails to Start
**Problem**: `ERROR: Cannot start service clinical-ai`

**Solutions**:
```bash
# Check Docker daemon is running
sudo systemctl status docker

# Check logs
docker-compose logs clinical-ai

# Rebuild images
docker-compose build --no-cache

# Remove old containers and volumes
docker-compose down -v
docker system prune -a
```

#### 2. GPU Not Detected
**Problem**: CUDA/GPU not available in container

**Solutions**:
```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check NVIDIA drivers
nvidia-smi

# Verify docker-compose GPU configuration
grep -A5 "deploy:" docker-compose.yml
```

#### 3. Database Connection Errors
**Problem**: Cannot connect to PostgreSQL/Neo4j

**Solutions**:
```bash
# Check if database containers are running
docker-compose ps postgres neo4j

# Check database logs
docker-compose logs postgres
docker-compose logs neo4j

# Restart database services
docker-compose restart postgres neo4j

# Verify credentials in .env file
grep -E "POSTGRES|NEO4J" .env
```

#### 4. Out of Memory Errors
**Problem**: Container crashes with OOM

**Solutions**:
```bash
# Increase Docker memory limit (Docker Desktop)
# Settings -> Resources -> Memory -> 16GB+

# Reduce model batch size in .env
echo "MAX_BATCH_SIZE=2" >> .env

# Enable model quantization
echo "MODEL_PRECISION=int8" >> .env

# Use gradient checkpointing
echo "ENABLE_GRADIENT_CHECKPOINTING=true" >> .env
```

#### 5. Port Already in Use
**Problem**: `Error: Port 8000 already in use`

**Solutions**:
```bash
# Find process using port
lsof -i :8000

# Kill the process (replace PID with actual process ID)
kill -9 <PID>

# Or change port in docker-compose.yml
sed -i 's/8000:8000/8080:8000/' docker-compose.yml
```

#### 6. Permission Denied Errors
**Problem**: Permission denied when accessing files/directories

**Solutions**:
```bash
# Fix directory permissions
sudo chown -R $USER:$USER logs/ models/ data/

# Or run with sudo (not recommended for production)
sudo docker-compose up -d
```

#### 7. Python Import Errors
**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solutions**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install in development mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 8. Slow API Response
**Problem**: API endpoints are very slow

**Solutions**:
```bash
# Check GPU utilization
docker-compose exec clinical-ai nvidia-smi

# Enable model caching
echo "ENABLE_MODEL_CACHING=true" >> .env

# Use faster model precision
echo "MODEL_PRECISION=fp16" >> .env

# Scale API instances
docker-compose up -d --scale clinical-ai=3
```

---

## Performance Optimization

### GPU Optimization
```bash
# Use mixed precision (FP16)
MODEL_PRECISION=fp16

# Enable TensorRT optimization
ENABLE_TENSORRT=true

# Adjust GPU memory fraction
GPU_MEMORY_FRACTION=0.9

# Use model quantization (INT8)
MODEL_PRECISION=int8
```

### API Performance
```bash
# Increase worker processes
API_WORKERS=4

# Enable response caching
ENABLE_CACHING=true

# Adjust batch size for inference
MAX_BATCH_SIZE=8
```

### Database Performance
```bash
# Increase PostgreSQL connections
# In docker-compose.yml, add to postgres environment:
POSTGRES_MAX_CONNECTIONS=200

# Enable Redis caching
REDIS_ENABLED=true

# Use connection pooling
DB_POOL_SIZE=20
```

---

## Monitoring and Logging

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service with timestamps
docker-compose logs -f --timestamps clinical-ai

# Last 100 lines
docker-compose logs --tail=100 clinical-ai

# Save logs to file
docker-compose logs clinical-ai > api-logs.txt
```

### Access Monitoring Dashboards
```bash
# Prometheus - Metrics
http://localhost:9090

# Grafana - Visualization
http://localhost:3000
# Default: admin / (check GRAFANA_PASSWORD in .env)

# Import dashboard from:
monitoring/grafana_dashboards/clinical_ai_dashboard.json
```

### Health Checks
```bash
# API Health
curl http://localhost:8000/health

# Detailed metrics
curl http://localhost:8000/metrics

# Database health
docker-compose exec postgres pg_isready -U clinical_user

# Neo4j health
curl http://localhost:7474/db/neo4j/cluster/available
```

---

## Production Deployment Checklist

- [ ] Change all default passwords in `.env`
- [ ] Generate strong encryption keys
- [ ] Configure SSL/TLS certificates
- [ ] Set up external database backups
- [ ] Enable audit logging
- [ ] Configure monitoring alerts (Prometheus/Grafana)
- [ ] Set resource limits in Kubernetes
- [ ] Enable network policies
- [ ] Configure RBAC for Kubernetes
- [ ] Set up secrets management (HashiCorp Vault)
- [ ] Configure automated backups
- [ ] Test disaster recovery procedures
- [ ] Review HIPAA compliance requirements
- [ ] Configure firewall rules
- [ ] Enable rate limiting
- [ ] Set up log aggregation (ELK Stack)
- [ ] Configure CI/CD pipeline
- [ ] Perform security audit
- [ ] Load testing and performance tuning

---

## Additional Resources

- **Full Deployment Guide**: [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
- **API Documentation**: http://localhost:8000/api/docs (when running)
- **Project README**: [README.md](README.md)
- **GitHub Repository**: https://github.com/Srujan29112001/Clinical-AI-copilot
- **Issue Tracker**: https://github.com/Srujan29112001/Clinical-AI-copilot/issues

---

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review logs for error messages
- Consult the troubleshooting section above

---

**Last Updated**: November 2024
**Version**: 1.0.0
