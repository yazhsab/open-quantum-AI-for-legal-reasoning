"""
Explainable Quantum-Enhanced Language Models for Legal Reasoning (XQELM)

A comprehensive framework for quantum-enhanced legal AI that combines quantum computing
with large language models to revolutionize legal reasoning and decision-making.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

__version__ = "1.0.0"
__author__ = "XQELM Research Team"
__email__ = "research@xqelm.org"
__license__ = "Apache-2.0"

from .core.quantum_legal_model import QuantumLegalModel
from .core.explainability import QuantumExplainabilityModule
from .quantum.embeddings import QuantumLegalEmbedding
from .quantum.attention import QuantumAttentionMechanism
from .quantum.reasoning import QuantumLegalReasoningCircuit
from .legal.use_cases import LegalUseCaseManager
from .api.client import XQELMClient

# Core exports for easy access
__all__ = [
    # Core Components
    "QuantumLegalModel",
    "QuantumExplainabilityModule",
    
    # Quantum Components
    "QuantumLegalEmbedding",
    "QuantumAttentionMechanism", 
    "QuantumLegalReasoningCircuit",
    
    # Legal Domain
    "LegalUseCaseManager",
    
    # API Client
    "XQELMClient",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Version information
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable",
    "build": "2024.01.01"
}

# Quantum backend configuration
SUPPORTED_QUANTUM_BACKENDS = [
    "pennylane.numpy",
    "pennylane.torch", 
    "qiskit.aer",
    "qiskit.ibm_runtime",
    "cirq.simulator",
    "braket.local",
]

# Legal domain configuration
SUPPORTED_LEGAL_DOMAINS = [
    "indian_law",
    "constitutional_law",
    "criminal_law",
    "civil_law",
    "commercial_law",
    "tax_law",
    "family_law",
    "labor_law",
    "environmental_law",
    "intellectual_property",
]

# Language support
SUPPORTED_LANGUAGES = [
    "en",  # English
    "hi",  # Hindi
    "bn",  # Bengali
    "te",  # Telugu
    "mr",  # Marathi
    "ta",  # Tamil
    "gu",  # Gujarati
    "kn",  # Kannada
    "ml",  # Malayalam
    "pa",  # Punjabi
]

def get_version() -> str:
    """Get the current version string."""
    return __version__

def get_version_info() -> dict:
    """Get detailed version information."""
    return VERSION_INFO.copy()

def check_dependencies() -> dict:
    """Check if all required dependencies are available."""
    dependencies = {}
    
    try:
        import pennylane as qml
        dependencies["pennylane"] = qml.__version__
    except ImportError:
        dependencies["pennylane"] = "Not installed"
    
    try:
        import qiskit
        dependencies["qiskit"] = qiskit.__version__
    except ImportError:
        dependencies["qiskit"] = "Not installed"
    
    try:
        import torch
        dependencies["torch"] = torch.__version__
    except ImportError:
        dependencies["torch"] = "Not installed"
    
    try:
        import transformers
        dependencies["transformers"] = transformers.__version__
    except ImportError:
        dependencies["transformers"] = "Not installed"
    
    try:
        import fastapi
        dependencies["fastapi"] = fastapi.__version__
    except ImportError:
        dependencies["fastapi"] = "Not installed"
    
    return dependencies

def configure_logging(level: str = "INFO") -> None:
    """Configure logging for the XQELM package."""
    import logging
    from loguru import logger
    
    # Remove default handler
    logger.remove()
    
    # Add custom handler with formatting
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("pennylane").setLevel(logging.WARNING)
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# Initialize logging on import
configure_logging()