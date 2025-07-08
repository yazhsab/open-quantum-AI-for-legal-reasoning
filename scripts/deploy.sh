#!/bin/bash

# XQELM Deployment Script
# Automated deployment script for the Quantum Legal AI system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="xqelm"
DOCKER_REGISTRY="ghcr.io"
NAMESPACE="xqelm"
ENVIRONMENT=${1:-"development"}
VERSION=${2:-"latest"}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required tools are installed
    command -v docker >/dev/null 2>&1 || { log_error "Docker is required but not installed. Aborting."; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed. Aborting."; exit 1; }
    command -v helm >/dev/null 2>&1 || { log_error "Helm is required but not installed. Aborting."; exit 1; }
    
    # Check Docker daemon
    docker info >/dev/null 2>&1 || { log_error "Docker daemon is not running. Aborting."; exit 1; }
    
    # Check Kubernetes connection
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to Kubernetes cluster. Aborting."; exit 1; }
    
    log_success "Prerequisites check passed"
}

setup_environment() {
    log_info "Setting up environment: $ENVIRONMENT"
    
    case $ENVIRONMENT in
        "development")
            export DATABASE_URL="postgresql://xqelm_user:xqelm_password@postgres:5432/xqelm_dev"
            export REDIS_URL="redis://:redis_password@redis:6379/0"
            export DEBUG="true"
            export LOG_LEVEL="DEBUG"
            ;;
        "staging")
            export DATABASE_URL="postgresql://xqelm_user:xqelm_password@postgres:5432/xqelm_staging"
            export REDIS_URL="redis://:redis_password@redis:6379/1"
            export DEBUG="false"
            export LOG_LEVEL="INFO"
            ;;
        "production")
            export DATABASE_URL="postgresql://xqelm_user:xqelm_password@postgres:5432/xqelm_prod"
            export REDIS_URL="redis://:redis_password@redis:6379/2"
            export DEBUG="false"
            export LOG_LEVEL="WARNING"
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    log_success "Environment configured for $ENVIRONMENT"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build backend image
    log_info "Building backend image..."
    docker build -t $DOCKER_REGISTRY/$PROJECT_NAME/backend:$VERSION .
    
    # Build frontend image
    log_info "Building frontend image..."
    docker build -t $DOCKER_REGISTRY/$PROJECT_NAME/frontend:$VERSION -f frontend/Dockerfile.frontend frontend/
    
    log_success "Docker images built successfully"
}

push_images() {
    log_info "Pushing Docker images to registry..."
    
    # Login to registry (assumes authentication is already set up)
    docker push $DOCKER_REGISTRY/$PROJECT_NAME/backend:$VERSION
    docker push $DOCKER_REGISTRY/$PROJECT_NAME/frontend:$VERSION
    
    log_success "Docker images pushed successfully"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply ConfigMaps and Secrets
    kubectl apply -f k8s/configmap.yaml -n $NAMESPACE
    kubectl apply -f k8s/secrets.yaml -n $NAMESPACE
    
    # Deploy databases
    log_info "Deploying PostgreSQL..."
    kubectl apply -f k8s/postgres-deployment.yaml -n $NAMESPACE
    
    log_info "Deploying Redis..."
    kubectl apply -f k8s/redis-deployment.yaml -n $NAMESPACE
    
    log_info "Deploying Neo4j..."
    kubectl apply -f k8s/neo4j-deployment.yaml -n $NAMESPACE
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=neo4j -n $NAMESPACE --timeout=300s
    
    log_success "Infrastructure deployed successfully"
}

deploy_application() {
    log_info "Deploying application components..."
    
    # Update image tags in deployment files
    sed -i "s|IMAGE_TAG|$VERSION|g" k8s/backend-deployment.yaml
    sed -i "s|IMAGE_TAG|$VERSION|g" k8s/frontend-deployment.yaml
    
    # Deploy backend
    log_info "Deploying backend..."
    kubectl apply -f k8s/backend-deployment.yaml -n $NAMESPACE
    
    # Deploy frontend
    log_info "Deploying frontend..."
    kubectl apply -f k8s/frontend-deployment.yaml -n $NAMESPACE
    
    # Deploy nginx reverse proxy
    log_info "Deploying nginx..."
    kubectl apply -f k8s/nginx-deployment.yaml -n $NAMESPACE
    
    # Wait for deployments to be ready
    log_info "Waiting for application to be ready..."
    kubectl rollout status deployment/backend-deployment -n $NAMESPACE --timeout=600s
    kubectl rollout status deployment/frontend-deployment -n $NAMESPACE --timeout=600s
    kubectl rollout status deployment/nginx-deployment -n $NAMESPACE --timeout=300s
    
    log_success "Application deployed successfully"
}

deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Deploy Prometheus
    kubectl apply -f k8s/prometheus-deployment.yaml -n $NAMESPACE
    
    # Deploy Grafana
    kubectl apply -f k8s/grafana-deployment.yaml -n $NAMESPACE
    
    # Deploy Elasticsearch and Kibana
    kubectl apply -f k8s/elasticsearch-deployment.yaml -n $NAMESPACE
    kubectl apply -f k8s/kibana-deployment.yaml -n $NAMESPACE
    
    log_success "Monitoring stack deployed successfully"
}

run_database_migrations() {
    log_info "Running database migrations..."
    
    # Get backend pod name
    BACKEND_POD=$(kubectl get pods -n $NAMESPACE -l app=backend -o jsonpath='{.items[0].metadata.name}')
    
    # Run migrations
    kubectl exec -n $NAMESPACE $BACKEND_POD -- python -m alembic upgrade head
    
    log_success "Database migrations completed"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Get service URLs
    BACKEND_URL=$(kubectl get service backend-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    FRONTEND_URL=$(kubectl get service nginx-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    # Check backend health
    log_info "Checking backend health..."
    for i in {1..30}; do
        if curl -f http://$BACKEND_URL:8000/health >/dev/null 2>&1; then
            log_success "Backend is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Backend health check failed"
            exit 1
        fi
        sleep 10
    done
    
    # Check frontend
    log_info "Checking frontend..."
    for i in {1..30}; do
        if curl -f http://$FRONTEND_URL >/dev/null 2>&1; then
            log_success "Frontend is accessible"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Frontend health check failed"
            exit 1
        fi
        sleep 10
    done
    
    log_success "All health checks passed"
}

run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Get backend pod for running tests
    BACKEND_POD=$(kubectl get pods -n $NAMESPACE -l app=backend -o jsonpath='{.items[0].metadata.name}')
    
    # Run smoke tests
    kubectl exec -n $NAMESPACE $BACKEND_POD -- python -m pytest tests/smoke/ -v
    
    log_success "Smoke tests passed"
}

cleanup_old_deployments() {
    log_info "Cleaning up old deployments..."
    
    # Remove old ReplicaSets (keep last 3)
    kubectl get rs -n $NAMESPACE --sort-by=.metadata.creationTimestamp -o name | head -n -3 | xargs -r kubectl delete -n $NAMESPACE
    
    # Remove old pods
    kubectl delete pods -n $NAMESPACE --field-selector=status.phase=Succeeded
    kubectl delete pods -n $NAMESPACE --field-selector=status.phase=Failed
    
    log_success "Cleanup completed"
}

show_deployment_info() {
    log_info "Deployment Information:"
    echo "=========================="
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo "Namespace: $NAMESPACE"
    echo ""
    
    log_info "Service URLs:"
    kubectl get services -n $NAMESPACE -o wide
    echo ""
    
    log_info "Pod Status:"
    kubectl get pods -n $NAMESPACE -o wide
    echo ""
    
    log_info "Deployment Status:"
    kubectl get deployments -n $NAMESPACE
    echo ""
    
    # Get external IPs
    NGINX_IP=$(kubectl get service nginx-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
    GRAFANA_IP=$(kubectl get service grafana-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
    
    echo "Access URLs:"
    echo "  Application: http://$NGINX_IP"
    echo "  API: http://$NGINX_IP/api"
    echo "  Grafana: http://$GRAFANA_IP:3000"
    echo "  Prometheus: http://$NGINX_IP:9090"
    echo ""
}

rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # Rollback backend
    kubectl rollout undo deployment/backend-deployment -n $NAMESPACE
    
    # Rollback frontend
    kubectl rollout undo deployment/frontend-deployment -n $NAMESPACE
    
    # Wait for rollback to complete
    kubectl rollout status deployment/backend-deployment -n $NAMESPACE --timeout=300s
    kubectl rollout status deployment/frontend-deployment -n $NAMESPACE --timeout=300s
    
    log_success "Rollback completed"
}

# Main deployment function
main() {
    log_info "Starting XQELM deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    # Trap errors and rollback if needed
    trap 'log_error "Deployment failed. Rolling back..."; rollback_deployment; exit 1' ERR
    
    check_prerequisites
    setup_environment
    
    if [ "$ENVIRONMENT" != "development" ]; then
        build_images
        push_images
    fi
    
    deploy_infrastructure
    deploy_application
    
    if [ "$ENVIRONMENT" = "production" ]; then
        deploy_monitoring
    fi
    
    run_database_migrations
    run_health_checks
    
    if [ "$ENVIRONMENT" != "development" ]; then
        run_smoke_tests
    fi
    
    cleanup_old_deployments
    show_deployment_info
    
    log_success "XQELM deployment completed successfully!"
}

# Script options
case "${1:-}" in
    "development"|"staging"|"production")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "status")
        show_deployment_info
        ;;
    "health")
        run_health_checks
        ;;
    "cleanup")
        cleanup_old_deployments
        ;;
    *)
        echo "Usage: $0 {development|staging|production|rollback|status|health|cleanup} [version]"
        echo ""
        echo "Commands:"
        echo "  development  - Deploy to development environment"
        echo "  staging      - Deploy to staging environment"
        echo "  production   - Deploy to production environment"
        echo "  rollback     - Rollback to previous deployment"
        echo "  status       - Show deployment status"
        echo "  health       - Run health checks"
        echo "  cleanup      - Clean up old resources"
        echo ""
        echo "Examples:"
        echo "  $0 development"
        echo "  $0 production v1.2.0"
        echo "  $0 rollback"
        exit 1
        ;;
esac