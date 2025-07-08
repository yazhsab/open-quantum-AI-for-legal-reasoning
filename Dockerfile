# Multi-stage Dockerfile for XQELM (Explainable Quantum-Enhanced Language Models for Legal Reasoning)

# Stage 1: Base image with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r xqelm && useradd -r -g xqelm xqelm

# Set working directory
WORKDIR /app

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to non-root user
RUN chown -R xqelm:xqelm /app

# Switch to non-root user
USER xqelm

# Expose port
EXPOSE 8000

# Default command for development
CMD ["uvicorn", "src.xqelm.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Production image
FROM base as production

# Copy requirements
COPY requirements.txt ./

# Install only production dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip cache purge

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY pyproject.toml ./

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/temp && \
    chown -R xqelm:xqelm /app

# Switch to non-root user
USER xqelm

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "src.xqelm.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Stage 4: Testing image
FROM development as testing

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini ./

# Install test dependencies
RUN pip install pytest pytest-cov pytest-asyncio pytest-mock

# Run tests
CMD ["pytest", "--cov=src/xqelm", "--cov-report=html", "--cov-report=term"]