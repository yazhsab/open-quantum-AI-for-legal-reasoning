"""
Pytest configuration and shared fixtures for XQELM tests

This module provides common test fixtures and configuration for the entire test suite.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime, date
import numpy as np
import torch

# Set test environment variables
os.environ["XQELM_ENV"] = "test"
os.environ["XQELM_LOG_LEVEL"] = "DEBUG"
os.environ["XQELM_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["XQELM_REDIS_URL"] = "redis://localhost:6379/15"  # Test database


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        "quantum": {
            "backend": "default.qubit",
            "n_qubits": 4,
            "n_layers": 2,
            "shots": 1000
        },
        "legal_system": {
            "jurisdiction": "india",
            "language": "en",
            "court_hierarchy": ["supreme", "high", "district", "magistrate"]
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": True
        },
        "database": {
            "url": "sqlite:///:memory:",
            "echo": False
        },
        "redis": {
            "url": "redis://localhost:6379/15",
            "decode_responses": True
        }
    }


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp(prefix="xqelm_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_quantum_device():
    """Mock quantum device for testing"""
    device = Mock()
    device.name = "default.qubit"
    device.num_wires = 4
    device.shots = 1000
    device.execute.return_value = np.array([0.6, 0.4])  # Mock measurement results
    return device


@pytest.fixture
def mock_quantum_circuit():
    """Mock quantum circuit for testing"""
    circuit = Mock()
    circuit.num_params = 8
    circuit.interface = "torch"
    circuit.return_value = torch.tensor([0.6, 0.4])
    return circuit


@pytest.fixture
def mock_legal_knowledge_base():
    """Mock legal knowledge base for testing"""
    kb = Mock()
    kb.search_cases.return_value = [
        {
            "case_id": "CASE-001",
            "title": "Sample Legal Case",
            "court": "Supreme Court",
            "year": 2023,
            "relevance_score": 0.85
        }
    ]
    kb.get_legal_provisions.return_value = [
        {
            "section": "Section 138",
            "act": "Negotiable Instruments Act",
            "description": "Dishonour of cheque for insufficiency of funds"
        }
    ]
    return kb


@pytest.fixture
def mock_preprocessor():
    """Mock text preprocessor for testing"""
    preprocessor = Mock()
    preprocessor.preprocess.return_value = {
        "tokens": ["legal", "case", "analysis"],
        "embeddings": np.random.rand(512),
        "features": {
            "case_type": "civil",
            "urgency": "medium",
            "complexity": 0.7
        }
    }
    return preprocessor


@pytest.fixture
def sample_legal_case():
    """Sample legal case data for testing"""
    return {
        "case_id": "TEST-001",
        "title": "Test Legal Case",
        "description": "This is a test case for legal analysis",
        "case_type": "civil",
        "court": "District Court",
        "filing_date": datetime.now(),
        "parties": {
            "plaintiff": "John Doe",
            "defendant": "Jane Smith"
        },
        "facts": [
            "Plaintiff filed a suit for recovery of money",
            "Defendant failed to repay the loan amount",
            "Contract was signed on 2023-01-15"
        ],
        "legal_issues": [
            "Breach of contract",
            "Recovery of debt"
        ],
        "amount_involved": 500000.0,
        "urgency": "medium"
    }


@pytest.fixture
def sample_gst_case():
    """Sample GST dispute case for testing"""
    return {
        "case_id": "GST-TEST-001",
        "taxpayer_gstin": "29ABCDE1234F1Z5",
        "taxpayer_name": "Test Enterprises",
        "dispute_type": "input_tax_credit",
        "dispute_amount": 150000.0,
        "tax_period": "2024-01",
        "description": "ITC claim rejected for interstate purchases",
        "supporting_documents": [
            "purchase_invoices.pdf",
            "transport_receipts.pdf"
        ],
        "urgency_level": "medium",
        "business_impact": "cash_flow_issues"
    }


@pytest.fixture
def sample_legal_aid_applicant():
    """Sample legal aid applicant for testing"""
    return {
        "applicant_id": "LA-TEST-001",
        "name": "Test Applicant",
        "age": 35,
        "gender": "Female",
        "address": "123 Test Street, Test City",
        "phone": "9876543210",
        "email": "test@example.com",
        "annual_income": 150000.0,
        "family_size": 4,
        "occupation": "Domestic Worker",
        "education_level": "Primary",
        "social_category": "SC",
        "is_disabled": False,
        "is_senior_citizen": False,
        "is_single_woman": True,
        "is_victim_of_crime": True,
        "assets_value": 50000.0,
        "monthly_expenses": 12000.0,
        "dependents": 3
    }


@pytest.fixture
def sample_legal_aid_case(sample_legal_aid_applicant):
    """Sample legal aid case for testing"""
    return {
        "case_id": "LA-CASE-001",
        "applicant": sample_legal_aid_applicant,
        "case_category": "women_rights",
        "legal_aid_type": "legal_representation",
        "case_description": "Domestic violence case requiring legal representation",
        "urgency_level": "high",
        "opposing_party": "Husband",
        "court_involved": "Family Court, Test City",
        "case_stage": "pre_litigation",
        "lawyer_required": True,
        "court_representation": True,
        "document_assistance": True,
        "application_date": datetime.now(),
        "emergency_case": True,
        "pro_bono_eligible": True
    }


@pytest.fixture
def mock_database_session():
    """Mock database session for testing"""
    session = Mock()
    session.query.return_value.filter.return_value.first.return_value = None
    session.add.return_value = None
    session.commit.return_value = None
    session.rollback.return_value = None
    session.close.return_value = None
    return session


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    redis_client = Mock()
    redis_client.get.return_value = None
    redis_client.set.return_value = True
    redis_client.delete.return_value = 1
    redis_client.exists.return_value = False
    redis_client.expire.return_value = True
    return redis_client


@pytest.fixture
def mock_api_client():
    """Mock API client for external services"""
    client = Mock()
    client.get.return_value.status_code = 200
    client.get.return_value.json.return_value = {"status": "success"}
    client.post.return_value.status_code = 200
    client.post.return_value.json.return_value = {"result": "processed"}
    return client


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables"""
    test_env_vars = {
        "XQELM_ENV": "test",
        "XQELM_LOG_LEVEL": "DEBUG",
        "XQELM_DATABASE_URL": "sqlite:///:memory:",
        "XQELM_REDIS_URL": "redis://localhost:6379/15",
        "XQELM_SECRET_KEY": "test-secret-key",
        "XQELM_API_HOST": "127.0.0.1",
        "XQELM_API_PORT": "8000",
        "XQELM_QUANTUM_BACKEND": "default.qubit",
        "XQELM_QUANTUM_SHOTS": "1000"
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def mock_file_system(tmp_path):
    """Mock file system for testing file operations"""
    # Create test directory structure
    test_dirs = [
        "data/datasets",
        "data/models",
        "config",
        "logs",
        "temp"
    ]
    
    for dir_path in test_dirs:
        (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create test files
    test_files = {
        "config/test.yaml": """
quantum:
  backend: default.qubit
  n_qubits: 4
legal_system:
  jurisdiction: india
""",
        "data/datasets/test_cases.json": """
[
  {
    "case_id": "TEST-001",
    "description": "Test case"
  }
]
""",
        "data/models/test_model.pkl": "mock_model_data"
    }
    
    for file_path, content in test_files.items():
        file_obj = tmp_path / file_path
        file_obj.write_text(content)
    
    return tmp_path


@pytest.fixture
def mock_logging():
    """Mock logging for tests"""
    with patch('logging.getLogger') as mock_logger:
        logger_instance = Mock()
        mock_logger.return_value = logger_instance
        yield logger_instance


@pytest.fixture
def quantum_test_data():
    """Generate test data for quantum computations"""
    return {
        "input_vectors": np.random.rand(10, 4),
        "target_outputs": np.random.rand(10, 2),
        "quantum_params": np.random.rand(8),
        "classical_features": np.random.rand(10, 16),
        "expected_probabilities": np.array([0.6, 0.4]),
        "measurement_results": np.array([1, 0, 1, 1, 0])
    }


@pytest.fixture
def legal_test_data():
    """Generate test data for legal analysis"""
    return {
        "case_texts": [
            "The plaintiff filed a suit for breach of contract",
            "The defendant failed to perform contractual obligations",
            "The court awarded damages to the plaintiff"
        ],
        "legal_categories": ["contract", "tort", "criminal"],
        "case_outcomes": ["plaintiff_wins", "defendant_wins", "settlement"],
        "legal_provisions": [
            "Section 73 of Indian Contract Act",
            "Section 138 of Negotiable Instruments Act"
        ],
        "precedent_cases": [
            {
                "citation": "AIR 2023 SC 1234",
                "ratio": "Breach of contract requires proof of damages"
            }
        ]
    }


# Pytest hooks for custom behavior
def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "quantum: mark test as quantum-related"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API-related"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location"""
    for item in items:
        # Add markers based on test file location
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        if "test_quantum" in item.nodeid:
            item.add_marker(pytest.mark.quantum)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging for tests"""
    import logging
    
    # Configure test logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tests.log'),
            logging.StreamHandler()
        ]
    )
    
    # Suppress noisy loggers during tests
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('pennylane').setLevel(logging.WARNING)


@pytest.fixture
def cleanup_test_data():
    """Cleanup test data after tests"""
    yield
    # Cleanup code here if needed
    pass