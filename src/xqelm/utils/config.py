"""
Configuration Management

This module handles configuration settings for the XQELM system,
including quantum backend settings, legal domain configurations,
and system parameters.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
from loguru import logger


@dataclass
class QuantumConfig:
    """Quantum computing configuration."""
    backend: str = "default.qubit"
    n_qubits: int = 20
    n_layers: int = 4
    shots: int = 1024
    noise_model: Optional[str] = None
    device_options: Dict[str, Any] = field(default_factory=dict)
    
    # Hardware-specific settings
    ibm_token: Optional[str] = None
    aws_access_key: Optional[str] = None
    google_credentials: Optional[str] = None
    
    # Circuit optimization
    optimization_level: int = 1
    transpiler_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalConfig:
    """Legal domain configuration."""
    supported_jurisdictions: List[str] = field(default_factory=lambda: [
        "india", "supreme_court", "high_courts", "district_courts"
    ])
    
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa"
    ])
    
    legal_databases: Dict[str, str] = field(default_factory=lambda: {
        "manupatra": "https://api.manupatra.com",
        "sci": "https://main.sci.gov.in",
        "indiankanoon": "https://api.indiankanoon.org",
        "judis": "https://judis.nic.in"
    })
    
    citation_formats: List[str] = field(default_factory=lambda: [
        "neutral", "traditional", "short"
    ])
    
    # Use case configurations
    use_case_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "bail_application": {
            "precedent_weight": 0.7,
            "statute_weight": 0.8,
            "fact_weight": 0.6
        },
        "cheque_bounce": {
            "precedent_weight": 0.5,
            "statute_weight": 0.9,
            "fact_weight": 0.7
        },
        "property_dispute": {
            "precedent_weight": 0.8,
            "statute_weight": 0.6,
            "fact_weight": 0.8
        }
    })


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    # Classical ML settings
    embedding_dim: int = 512
    hidden_dim: int = 256
    num_attention_heads: int = 8
    num_layers: int = 6
    dropout_rate: float = 0.1
    
    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Quantum-classical hybrid settings
    quantum_classical_ratio: float = 0.5
    gradient_clipping: float = 1.0
    
    # Model paths
    pretrained_model_path: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    
    # Evaluation settings
    validation_split: float = 0.2
    test_split: float = 0.1


@dataclass
class DatabaseConfig:
    """Database configuration."""
    # Primary database (PostgreSQL)
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "xqelm"
    db_user: str = "xqelm_user"
    db_password: str = "xqelm_password"
    db_ssl_mode: str = "prefer"
    
    # Graph database (Neo4j)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"
    
    # Cache database (Redis)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Search engine (Elasticsearch)
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_index: str = "legal_documents"
    
    # Connection pooling
    max_connections: int = 20
    connection_timeout: int = 30


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    
    # CORS
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8080"
    ])
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # API versioning
    api_version: str = "v1"
    api_prefix: str = "/api/v1"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    
    # File logging
    log_file: Optional[str] = "logs/xqelm.log"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"
    
    # Structured logging
    json_logs: bool = False
    
    # External logging
    sentry_dsn: Optional[str] = None
    elasticsearch_logging: bool = False


@dataclass
class SecurityConfig:
    """Security configuration."""
    # Encryption
    encryption_key: Optional[str] = None
    hash_algorithm: str = "bcrypt"
    
    # Authentication
    auth_providers: List[str] = field(default_factory=lambda: ["local", "oauth2"])
    oauth2_providers: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Data protection
    anonymize_logs: bool = True
    data_retention_days: int = 365
    
    # Network security
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    ssl_required: bool = False


class XQELMConfig:
    """Main configuration class for XQELM system."""
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        environment: str = "development"
    ):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file
            environment: Environment name (development, staging, production)
        """
        self.environment = environment
        self.config_file = config_file
        
        # Initialize sub-configurations
        self.quantum = QuantumConfig()
        self.legal = LegalConfig()
        self.model = ModelConfig()
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        
        # Load configuration
        self._load_configuration()
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"Configuration loaded for environment: {environment}")
    
    def _load_configuration(self) -> None:
        """Load configuration from various sources."""
        # 1. Load from environment variables
        self._load_from_environment()
        
        # 2. Load from configuration file
        if self.config_file:
            self._load_from_file(self.config_file)
        else:
            # Try default configuration files
            default_configs = [
                f"config/{self.environment}.yaml",
                f"config/{self.environment}.yml",
                f"config/{self.environment}.json",
                "config/default.yaml",
                "config/default.yml"
            ]
            
            for config_path in default_configs:
                if os.path.exists(config_path):
                    self._load_from_file(config_path)
                    break
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # Quantum configuration
        if os.getenv("QUANTUM_BACKEND"):
            self.quantum.backend = os.getenv("QUANTUM_BACKEND")
        
        if os.getenv("QUANTUM_QUBITS"):
            self.quantum.n_qubits = int(os.getenv("QUANTUM_QUBITS"))
        
        if os.getenv("IBM_QUANTUM_TOKEN"):
            self.quantum.ibm_token = os.getenv("IBM_QUANTUM_TOKEN")
        
        # Database configuration
        if os.getenv("DATABASE_URL"):
            self._parse_database_url(os.getenv("DATABASE_URL"))
        
        if os.getenv("REDIS_URL"):
            self._parse_redis_url(os.getenv("REDIS_URL"))
        
        if os.getenv("NEO4J_URI"):
            self.database.neo4j_uri = os.getenv("NEO4J_URI")
        
        # API configuration
        if os.getenv("API_HOST"):
            self.api.host = os.getenv("API_HOST")
        
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))
        
        if os.getenv("SECRET_KEY"):
            self.api.secret_key = os.getenv("SECRET_KEY")
        
        # Security configuration
        if os.getenv("ENCRYPTION_KEY"):
            self.security.encryption_key = os.getenv("ENCRYPTION_KEY")
        
        # Logging configuration
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL")
        
        if os.getenv("SENTRY_DSN"):
            self.logging.sentry_dsn = os.getenv("SENTRY_DSN")
    
    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                    return
            
            # Update configuration with loaded data
            self._update_from_dict(config_data)
            
            logger.info(f"Configuration loaded from: {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file {config_file}: {e}")
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'quantum' in config_data:
            self._update_dataclass(self.quantum, config_data['quantum'])
        
        if 'legal' in config_data:
            self._update_dataclass(self.legal, config_data['legal'])
        
        if 'model' in config_data:
            self._update_dataclass(self.model, config_data['model'])
        
        if 'database' in config_data:
            self._update_dataclass(self.database, config_data['database'])
        
        if 'api' in config_data:
            self._update_dataclass(self.api, config_data['api'])
        
        if 'logging' in config_data:
            self._update_dataclass(self.logging, config_data['logging'])
        
        if 'security' in config_data:
            self._update_dataclass(self.security, config_data['security'])
    
    def _update_dataclass(self, dataclass_instance: Any, update_dict: Dict[str, Any]) -> None:
        """Update dataclass instance with dictionary values."""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def _parse_database_url(self, database_url: str) -> None:
        """Parse database URL and update configuration."""
        # Simple URL parsing for PostgreSQL
        # Format: postgresql://user:password@host:port/database
        try:
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            
            if parsed.hostname:
                self.database.db_host = parsed.hostname
            if parsed.port:
                self.database.db_port = parsed.port
            if parsed.username:
                self.database.db_user = parsed.username
            if parsed.password:
                self.database.db_password = parsed.password
            if parsed.path and len(parsed.path) > 1:
                self.database.db_name = parsed.path[1:]  # Remove leading '/'
                
        except Exception as e:
            logger.error(f"Error parsing database URL: {e}")
    
    def _parse_redis_url(self, redis_url: str) -> None:
        """Parse Redis URL and update configuration."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(redis_url)
            
            if parsed.hostname:
                self.database.redis_host = parsed.hostname
            if parsed.port:
                self.database.redis_port = parsed.port
            if parsed.password:
                self.database.redis_password = parsed.password
            if parsed.path and len(parsed.path) > 1:
                self.database.redis_db = int(parsed.path[1:])
                
        except Exception as e:
            logger.error(f"Error parsing Redis URL: {e}")
    
    def _validate_configuration(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Validate quantum configuration
        if self.quantum.n_qubits < 1:
            errors.append("Number of qubits must be positive")
        
        if self.quantum.n_qubits > 50:
            logger.warning("Large number of qubits may cause performance issues")
        
        # Validate model configuration
        if self.model.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.model.batch_size < 1:
            errors.append("Batch size must be positive")
        
        # Validate API configuration
        if not (1 <= self.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        
        # Validate database configuration
        if not self.database.db_name:
            errors.append("Database name is required")
        
        # Security validations for production
        if self.environment == "production":
            if self.api.secret_key == "your-secret-key-change-in-production":
                errors.append("Secret key must be changed in production")
            
            if self.api.debug:
                errors.append("Debug mode should be disabled in production")
            
            if not self.security.ssl_required:
                logger.warning("SSL should be enabled in production")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return (
            f"postgresql://{self.database.db_user}:{self.database.db_password}"
            f"@{self.database.db_host}:{self.database.db_port}/{self.database.db_name}"
        )
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.database.redis_password}@" if self.database.redis_password else ""
        return f"redis://{auth}{self.database.redis_host}:{self.database.redis_port}/{self.database.redis_db}"
    
    def get_neo4j_url(self) -> str:
        """Get Neo4j connection URL."""
        return self.database.neo4j_uri
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment,
            'quantum': self.quantum.__dict__,
            'legal': self.legal.__dict__,
            'model': self.model.__dict__,
            'database': self.database.__dict__,
            'api': self.api.__dict__,
            'logging': self.logging.__dict__,
            'security': self.security.__dict__
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to file."""
        config_dict = self.to_dict()
        
        # Remove sensitive information
        if 'db_password' in config_dict['database']:
            config_dict['database']['db_password'] = '***'
        if 'secret_key' in config_dict['api']:
            config_dict['api']['secret_key'] = '***'
        if 'encryption_key' in config_dict['security']:
            config_dict['security']['encryption_key'] = '***'
        
        filepath = Path(filepath)
        
        try:
            with open(filepath, 'w') as f:
                if filepath.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif filepath.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            logger.info(f"Configuration saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {filepath}: {e}")
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"XQELMConfig(environment={self.environment}, qubits={self.quantum.n_qubits})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"XQELMConfig(\n"
            f"  environment={self.environment},\n"
            f"  quantum_backend={self.quantum.backend},\n"
            f"  n_qubits={self.quantum.n_qubits},\n"
            f"  api_port={self.api.port},\n"
            f"  database={self.database.db_name}\n"
            f")"
        )


# Global configuration instance
_config_instance: Optional[XQELMConfig] = None


def get_config() -> XQELMConfig:
    """Get global configuration instance."""
    global _config_instance
    
    if _config_instance is None:
        environment = os.getenv("XQELM_ENV", "development")
        _config_instance = XQELMConfig(environment=environment)
    
    return _config_instance


def set_config(config: XQELMConfig) -> None:
    """Set global configuration instance."""
    global _config_instance
    _config_instance = config


def reset_config() -> None:
    """Reset global configuration instance."""
    global _config_instance
    _config_instance = None