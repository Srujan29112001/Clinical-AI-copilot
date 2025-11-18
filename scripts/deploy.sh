#!/bin/bash
# Deployment Script for Clinical AI Copilot
# Automates deployment to production environment

set -e  # Exit on error

echo "==================================================================="
echo "Clinical AI Copilot - Deployment Script"
echo "==================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check prerequisites
echo ""
echo "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi
print_success "Docker installed"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed"
    exit 1
fi
print_success "Docker Compose installed"

# Check .env file
if [ ! -f .env ]; then
    print_warning ".env file not found"
    echo "Copying .env.example to .env..."
    cp .env.example .env
    print_warning "Please edit .env with your configuration before proceeding"
    read -p "Press enter to continue after editing .env..."
fi
print_success ".env file exists"

# Check for required directories
echo ""
echo "Checking directories..."
mkdir -p logs data models monitoring/grafana/dashboards
print_success "Directories created"

# Generate encryption key if not exists
if ! grep -q "ENCRYPTION_KEY=" .env || grep -q "ENCRYPTION_KEY=generate-with" .env; then
    print_warning "Generating encryption key..."
    python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" > /tmp/enc_key.txt
    ENCRYPTION_KEY=$(cat /tmp/enc_key.txt)
    sed -i "s|ENCRYPTION_KEY=.*|ENCRYPTION_KEY=$ENCRYPTION_KEY|" .env
    rm /tmp/enc_key.txt
    print_success "Encryption key generated"
fi

# Build Docker images
echo ""
echo "Building Docker images..."
docker-compose build
print_success "Docker images built"

# Pull required images
echo ""
echo "Pulling required Docker images..."
docker-compose pull
print_success "Docker images pulled"

# Start services
echo ""
echo "Starting services..."
docker-compose up -d
print_success "Services started"

# Wait for services to be healthy
echo ""
echo "Waiting for services to be healthy..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."

# Check Clinical AI API
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Clinical AI API is healthy"
else
    print_error "Clinical AI API is not responding"
fi

# Check PostgreSQL
if docker-compose ps postgres | grep -q "Up"; then
    print_success "PostgreSQL is running"
else
    print_error "PostgreSQL is not running"
fi

# Check Neo4j
if docker-compose ps neo4j | grep -q "Up"; then
    print_success "Neo4j is running"
else
    print_error "Neo4j is not running"
fi

# Check Qdrant
if docker-compose ps qdrant | grep -q "Up"; then
    print_success "Qdrant is running"
else
    print_error "Qdrant is not running"
fi

# Display service URLs
echo ""
echo "==================================================================="
echo "Deployment Complete!"
echo "==================================================================="
echo ""
echo "Service URLs:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/api/docs"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Neo4j Browser: http://localhost:7474"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f clinical-ai"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "To stop and remove all data:"
echo "  docker-compose down -v"
echo ""
print_warning "Remember to:"
print_warning "  1. Change default passwords in production"
print_warning "  2. Configure SSL/TLS certificates"
print_warning "  3. Set up proper authentication"
print_warning "  4. Configure backup procedures"
print_warning "  5. Review HIPAA compliance checklist"
echo ""
