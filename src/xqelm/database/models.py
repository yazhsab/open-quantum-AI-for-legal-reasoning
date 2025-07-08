"""
Database Models

SQLAlchemy models for the XQELM system.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import json

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, 
    ForeignKey, JSON, LargeBinary, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid

from loguru import logger


Base = declarative_base()


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    organization = Column(String(255))
    role = Column(String(50), default="user")  # user, admin, researcher
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # API access
    api_key = Column(String(255), unique=True, index=True)
    api_key_created_at = Column(DateTime)
    rate_limit_tier = Column(String(20), default="basic")  # basic, premium, enterprise
    
    # Usage tracking
    total_queries = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    last_login = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    legal_cases = relationship("LegalCase", back_populates="user")
    query_logs = relationship("QueryLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "organization": self.organization,
            "role": self.role,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "rate_limit_tier": self.rate_limit_tier,
            "total_queries": self.total_queries,
            "total_tokens_used": self.total_tokens_used,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class LegalCase(Base):
    """Legal case model for storing case information."""
    
    __tablename__ = "legal_cases"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Case identification
    case_number = Column(String(100), index=True)
    case_title = Column(String(500))
    case_type = Column(String(100))  # bail_application, cheque_bounce, etc.
    court_name = Column(String(255))
    jurisdiction = Column(String(100))
    
    # Case details
    parties_involved = Column(JSON)  # List of parties
    case_summary = Column(Text)
    legal_issues = Column(ARRAY(String))
    applicable_laws = Column(ARRAY(String))
    precedents_cited = Column(JSON)  # List of precedent cases
    
    # Case status
    status = Column(String(50), default="active")  # active, closed, pending
    filing_date = Column(DateTime)
    hearing_date = Column(DateTime)
    judgment_date = Column(DateTime)
    
    # Quantum analysis results
    quantum_analysis = Column(JSON)
    confidence_score = Column(Float)
    risk_assessment = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="legal_cases")
    documents = relationship("LegalDocument", back_populates="legal_case")
    query_logs = relationship("QueryLog", back_populates="legal_case")
    
    # Indexes
    __table_args__ = (
        Index("idx_case_type_status", "case_type", "status"),
        Index("idx_filing_date", "filing_date"),
        Index("idx_user_case_type", "user_id", "case_type"),
    )
    
    def __repr__(self):
        return f"<LegalCase(case_number='{self.case_number}', case_type='{self.case_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert legal case to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "case_number": self.case_number,
            "case_title": self.case_title,
            "case_type": self.case_type,
            "court_name": self.court_name,
            "jurisdiction": self.jurisdiction,
            "parties_involved": self.parties_involved,
            "case_summary": self.case_summary,
            "legal_issues": self.legal_issues,
            "applicable_laws": self.applicable_laws,
            "precedents_cited": self.precedents_cited,
            "status": self.status,
            "filing_date": self.filing_date.isoformat() if self.filing_date else None,
            "hearing_date": self.hearing_date.isoformat() if self.hearing_date else None,
            "judgment_date": self.judgment_date.isoformat() if self.judgment_date else None,
            "quantum_analysis": self.quantum_analysis,
            "confidence_score": self.confidence_score,
            "risk_assessment": self.risk_assessment,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class LegalDocument(Base):
    """Legal document model for storing case documents."""
    
    __tablename__ = "legal_documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    legal_case_id = Column(UUID(as_uuid=True), ForeignKey("legal_cases.id"), nullable=False)
    
    # Document metadata
    document_name = Column(String(255), nullable=False)
    document_type = Column(String(100))  # petition, judgment, evidence, etc.
    file_path = Column(String(500))
    file_size = Column(Integer)
    mime_type = Column(String(100))
    
    # Document content
    content = Column(Text)
    extracted_text = Column(Text)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default="pending")
    
    # Embeddings and analysis
    embeddings = Column(LargeBinary)  # Serialized numpy array
    quantum_features = Column(JSON)
    legal_entities = Column(JSON)  # Extracted legal entities
    key_phrases = Column(ARRAY(String))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    legal_case = relationship("LegalCase", back_populates="documents")
    
    # Indexes
    __table_args__ = (
        Index("idx_document_type", "document_type"),
        Index("idx_processing_status", "processing_status"),
        Index("idx_case_document", "legal_case_id", "document_type"),
    )
    
    def __repr__(self):
        return f"<LegalDocument(name='{self.document_name}', type='{self.document_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert legal document to dictionary."""
        return {
            "id": str(self.id),
            "legal_case_id": str(self.legal_case_id),
            "document_name": self.document_name,
            "document_type": self.document_type,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "content": self.content,
            "is_processed": self.is_processed,
            "processing_status": self.processing_status,
            "quantum_features": self.quantum_features,
            "legal_entities": self.legal_entities,
            "key_phrases": self.key_phrases,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class QueryLog(Base):
    """Query log model for tracking user queries and responses."""
    
    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    legal_case_id = Column(UUID(as_uuid=True), ForeignKey("legal_cases.id"), nullable=True)
    
    # Query details
    query_text = Column(Text, nullable=False)
    query_type = Column(String(100))  # legal_analysis, precedent_search, etc.
    query_parameters = Column(JSON)
    
    # Response details
    response_text = Column(Text)
    response_metadata = Column(JSON)
    confidence_score = Column(Float)
    
    # Processing metrics
    processing_time_ms = Column(Integer)
    quantum_circuit_depth = Column(Integer)
    classical_tokens_used = Column(Integer)
    quantum_gates_used = Column(Integer)
    
    # Quantum analysis
    quantum_state_vector = Column(LargeBinary)  # Serialized quantum state
    quantum_measurements = Column(JSON)
    explainability_data = Column(JSON)
    
    # Feedback and evaluation
    user_rating = Column(Integer)  # 1-5 rating
    user_feedback = Column(Text)
    is_helpful = Column(Boolean)
    
    # System metadata
    model_version = Column(String(50))
    api_endpoint = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="query_logs")
    legal_case = relationship("LegalCase", back_populates="query_logs")
    
    # Indexes
    __table_args__ = (
        Index("idx_user_query_type", "user_id", "query_type"),
        Index("idx_created_at", "created_at"),
        Index("idx_confidence_score", "confidence_score"),
        Index("idx_processing_time", "processing_time_ms"),
    )
    
    def __repr__(self):
        return f"<QueryLog(user_id='{self.user_id}', query_type='{self.query_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query log to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "legal_case_id": str(self.legal_case_id) if self.legal_case_id else None,
            "query_text": self.query_text,
            "query_type": self.query_type,
            "query_parameters": self.query_parameters,
            "response_text": self.response_text,
            "response_metadata": self.response_metadata,
            "confidence_score": self.confidence_score,
            "processing_time_ms": self.processing_time_ms,
            "quantum_circuit_depth": self.quantum_circuit_depth,
            "classical_tokens_used": self.classical_tokens_used,
            "quantum_gates_used": self.quantum_gates_used,
            "quantum_measurements": self.quantum_measurements,
            "explainability_data": self.explainability_data,
            "user_rating": self.user_rating,
            "user_feedback": self.user_feedback,
            "is_helpful": self.is_helpful,
            "model_version": self.model_version,
            "api_endpoint": self.api_endpoint,
            "created_at": self.created_at.isoformat()
        }


class ModelTraining(Base):
    """Model training log for tracking quantum model training sessions."""
    
    __tablename__ = "model_training"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Training session details
    training_name = Column(String(255), nullable=False)
    training_type = Column(String(100))  # supervised, unsupervised, reinforcement
    dataset_name = Column(String(255))
    dataset_size = Column(Integer)
    
    # Model configuration
    model_architecture = Column(JSON)
    hyperparameters = Column(JSON)
    quantum_circuit_config = Column(JSON)
    
    # Training progress
    status = Column(String(50), default="running")  # running, completed, failed, paused
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer)
    current_loss = Column(Float)
    best_loss = Column(Float)
    
    # Training metrics
    training_metrics = Column(JSON)  # Loss, accuracy, etc. over time
    validation_metrics = Column(JSON)
    quantum_metrics = Column(JSON)  # Quantum-specific metrics
    
    # Resource usage
    training_time_seconds = Column(Integer)
    quantum_shots_used = Column(Integer)
    classical_compute_hours = Column(Float)
    
    # Model artifacts
    model_checkpoint_path = Column(String(500))
    final_model_path = Column(String(500))
    training_logs_path = Column(String(500))
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index("idx_training_status", "status"),
        Index("idx_training_type", "training_type"),
        Index("idx_started_at", "started_at"),
    )
    
    def __repr__(self):
        return f"<ModelTraining(name='{self.training_name}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model training to dictionary."""
        return {
            "id": str(self.id),
            "training_name": self.training_name,
            "training_type": self.training_type,
            "dataset_name": self.dataset_name,
            "dataset_size": self.dataset_size,
            "model_architecture": self.model_architecture,
            "hyperparameters": self.hyperparameters,
            "quantum_circuit_config": self.quantum_circuit_config,
            "status": self.status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_loss": self.current_loss,
            "best_loss": self.best_loss,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "quantum_metrics": self.quantum_metrics,
            "training_time_seconds": self.training_time_seconds,
            "quantum_shots_used": self.quantum_shots_used,
            "classical_compute_hours": self.classical_compute_hours,
            "model_checkpoint_path": self.model_checkpoint_path,
            "final_model_path": self.final_model_path,
            "training_logs_path": self.training_logs_path,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class ExplanationLog(Base):
    """Explanation log for tracking quantum explainability results."""
    
    __tablename__ = "explanation_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_log_id = Column(UUID(as_uuid=True), ForeignKey("query_logs.id"), nullable=False)
    
    # Explanation details
    explanation_type = Column(String(100))  # quantum_tomography, classical_shadows, etc.
    explanation_method = Column(String(100))
    
    # Quantum state analysis
    quantum_state_fidelity = Column(Float)
    entanglement_measures = Column(JSON)
    quantum_coherence = Column(JSON)
    
    # Feature importance
    classical_feature_importance = Column(JSON)
    quantum_feature_importance = Column(JSON)
    legal_concept_weights = Column(JSON)
    
    # Visualization data
    circuit_visualization = Column(JSON)
    state_visualization = Column(JSON)
    attention_maps = Column(JSON)
    
    # Human-readable explanations
    natural_language_explanation = Column(Text)
    key_reasoning_steps = Column(JSON)
    confidence_intervals = Column(JSON)
    
    # Validation metrics
    explanation_quality_score = Column(Float)
    human_interpretability_score = Column(Float)
    technical_accuracy_score = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    query_log = relationship("QueryLog")
    
    # Indexes
    __table_args__ = (
        Index("idx_explanation_type", "explanation_type"),
        Index("idx_quality_score", "explanation_quality_score"),
        Index("idx_created_at", "created_at"),
    )
    
    def __repr__(self):
        return f"<ExplanationLog(type='{self.explanation_type}', method='{self.explanation_method}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation log to dictionary."""
        return {
            "id": str(self.id),
            "query_log_id": str(self.query_log_id),
            "explanation_type": self.explanation_type,
            "explanation_method": self.explanation_method,
            "quantum_state_fidelity": self.quantum_state_fidelity,
            "entanglement_measures": self.entanglement_measures,
            "quantum_coherence": self.quantum_coherence,
            "classical_feature_importance": self.classical_feature_importance,
            "quantum_feature_importance": self.quantum_feature_importance,
            "legal_concept_weights": self.legal_concept_weights,
            "circuit_visualization": self.circuit_visualization,
            "state_visualization": self.state_visualization,
            "attention_maps": self.attention_maps,
            "natural_language_explanation": self.natural_language_explanation,
            "key_reasoning_steps": self.key_reasoning_steps,
            "confidence_intervals": self.confidence_intervals,
            "explanation_quality_score": self.explanation_quality_score,
            "human_interpretability_score": self.human_interpretability_score,
            "technical_accuracy_score": self.technical_accuracy_score,
            "created_at": self.created_at.isoformat()
        }


# Database utility functions
def create_tables(engine):
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def drop_tables(engine):
    """Drop all database tables."""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise


def get_table_info(engine) -> Dict[str, Any]:
    """Get information about database tables."""
    from sqlalchemy import inspect
    
    inspector = inspect(engine)
    tables_info = {}
    
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        indexes = inspector.get_indexes(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        
        tables_info[table_name] = {
            "columns": [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"],
                    "primary_key": col.get("primary_key", False)
                }
                for col in columns
            ],
            "indexes": [
                {
                    "name": idx["name"],
                    "columns": idx["column_names"],
                    "unique": idx["unique"]
                }
                for idx in indexes
            ],
            "foreign_keys": [
                {
                    "name": fk["name"],
                    "constrained_columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"]
                }
                for fk in foreign_keys
            ]
        }
    
    return tables_info