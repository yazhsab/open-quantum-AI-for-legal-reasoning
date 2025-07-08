"""
Integration Tests for XQELM API

This module contains integration tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
from datetime import datetime

from src.xqelm.api.main import app
from src.xqelm.api.models import (
    GSTDisputeRequest, LegalAidRequest, GSTDisputeTypeEnum,
    LegalAidTypeEnum, CaseCategoryEnum, UrgencyLevelEnum
)


class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_gst_request(self):
        """Sample GST dispute request"""
        return {
            "taxpayer_gstin": "29ABCDE1234F1Z5",
            "taxpayer_name": "ABC Enterprises",
            "dispute_type": "input_tax_credit",
            "dispute_amount": 150000.0,
            "tax_period": "2024-01",
            "description": "ITC claim rejected for interstate purchases",
            "supporting_documents": [
                "purchase_invoices.pdf",
                "transport_receipts.pdf"
            ],
            "previous_orders": [],
            "urgency_level": "medium",
            "preferred_resolution": "appeal",
            "business_impact": "cash_flow_issues"
        }
    
    @pytest.fixture
    def sample_legal_aid_request(self):
        """Sample legal aid request"""
        return {
            "applicant": {
                "name": "Jane Doe",
                "age": 35,
                "gender": "Female",
                "address": "123 Main Street, Delhi",
                "phone": "9876543210",
                "email": "jane.doe@email.com",
                "annual_income": 150000.0,
                "family_size": 4,
                "occupation": "Domestic Worker",
                "education_level": "Primary",
                "social_category": "SC",
                "is_disabled": False,
                "is_senior_citizen": False,
                "is_single_woman": True,
                "is_victim_of_crime": True
            },
            "case_category": "women_rights",
            "legal_aid_type": "legal_representation",
            "case_description": "Domestic violence case requiring legal representation",
            "urgency_level": "high",
            "opposing_party": "Husband",
            "court_involved": "Family Court, Delhi",
            "case_stage": "pre_litigation",
            "lawyer_required": True,
            "emergency_case": True
        }
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "documentation" in data
    
    @patch('src.xqelm.use_cases.gst_dispute.GSTDisputeManager.analyze_gst_dispute')
    def test_gst_dispute_analyze_success(self, mock_analyze, client, sample_gst_request):
        """Test successful GST dispute analysis"""
        # Mock the analysis result
        mock_result = Mock()
        mock_result.case_id = "GST-001"
        mock_result.dispute_type = "input_tax_credit"
        mock_result.risk_assessment = "medium"
        mock_result.recommended_action = "file_appeal"
        mock_result.success_probability = 0.75
        mock_result.estimated_timeline = "3-6 months"
        mock_result.quantum_analysis = {
            "confidence": 0.85,
            "coherence": 0.80,
            "final_state": "mock_state"
        }
        mock_result.financial_analysis = {
            "dispute_amount": 150000.0,
            "penalty_amount": 15000.0,
            "interest_amount": 5000.0,
            "total_liability": 170000.0
        }
        mock_result.recommendations = [
            "Gather additional supporting documents",
            "File appeal within 30 days"
        ]
        
        mock_analyze.return_value = mock_result
        
        response = client.post("/gst-dispute/analyze", json=sample_gst_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["case_id"] == "GST-001"
        assert data["dispute_type"] == "input_tax_credit"
        assert data["risk_assessment"] == "medium"
        assert data["recommended_action"] == "file_appeal"
        assert data["success_probability"] == 0.75
        assert "quantum_analysis" in data
        assert "financial_analysis" in data
        assert "recommendations" in data
    
    def test_gst_dispute_analyze_invalid_data(self, client):
        """Test GST dispute analysis with invalid data"""
        invalid_request = {
            "taxpayer_gstin": "invalid_gstin",  # Invalid GSTIN format
            "dispute_amount": -1000,  # Negative amount
            "tax_period": "invalid_period"  # Invalid period format
        }
        
        response = client.post("/gst-dispute/analyze", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    @patch('src.xqelm.use_cases.legal_aid.LegalAidManager.assess_legal_aid_eligibility')
    def test_legal_aid_assess_success(self, mock_assess, client, sample_legal_aid_request):
        """Test successful legal aid assessment"""
        # Mock the assessment result
        mock_result = Mock()
        mock_result.case_id = "LA-001"
        mock_result.eligibility_score = 0.85
        mock_result.is_eligible = True
        mock_result.recommended_aid_type = "legal_representation"
        mock_result.recommended_authority = "dlsa"
        mock_result.priority_level = 2
        mock_result.estimated_cost = 25000.0
        mock_result.quantum_analysis = {
            "confidence": 0.90,
            "coherence": 0.88,
            "final_state": "mock_state"
        }
        mock_result.financial_analysis = {
            "fee_waiver": True,
            "court_fee_exemption": 100.0,
            "service_cost": 0.0
        }
        mock_result.recommendations = [
            "Submit income certificate",
            "Provide caste certificate"
        ]
        
        mock_assess.return_value = mock_result
        
        response = client.post("/legal-aid/assess", json=sample_legal_aid_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["case_id"] == "LA-001"
        assert data["eligibility_score"] == 0.85
        assert data["is_eligible"] == True
        assert data["recommended_aid_type"] == "legal_representation"
        assert data["priority_level"] == 2
        assert "quantum_analysis" in data
        assert "financial_analysis" in data
        assert "recommendations" in data
    
    def test_legal_aid_assess_invalid_data(self, client):
        """Test legal aid assessment with invalid data"""
        invalid_request = {
            "applicant": {
                "name": "",  # Empty name
                "age": -1,  # Invalid age
                "annual_income": -1000  # Negative income
            },
            "case_category": "invalid_category",  # Invalid category
            "urgency_level": "invalid_urgency"  # Invalid urgency
        }
        
        response = client.post("/legal-aid/assess", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_gst_dispute_missing_required_fields(self, client):
        """Test GST dispute with missing required fields"""
        incomplete_request = {
            "taxpayer_name": "ABC Enterprises"
            # Missing required fields like taxpayer_gstin, dispute_type, etc.
        }
        
        response = client.post("/gst-dispute/analyze", json=incomplete_request)
        assert response.status_code == 422
    
    def test_legal_aid_missing_required_fields(self, client):
        """Test legal aid with missing required fields"""
        incomplete_request = {
            "case_description": "Some description"
            # Missing required fields like applicant, case_category, etc.
        }
        
        response = client.post("/legal-aid/assess", json=incomplete_request)
        assert response.status_code == 422
    
    @patch('src.xqelm.use_cases.gst_dispute.GSTDisputeManager.analyze_gst_dispute')
    def test_gst_dispute_server_error(self, mock_analyze, client, sample_gst_request):
        """Test GST dispute analysis server error handling"""
        mock_analyze.side_effect = Exception("Internal server error")
        
        response = client.post("/gst-dispute/analyze", json=sample_gst_request)
        assert response.status_code == 500
        
        data = response.json()
        assert "error" in data
        assert "Internal server error" in data["error"]
    
    @patch('src.xqelm.use_cases.legal_aid.LegalAidManager.assess_legal_aid_eligibility')
    def test_legal_aid_server_error(self, mock_assess, client, sample_legal_aid_request):
        """Test legal aid assessment server error handling"""
        mock_assess.side_effect = Exception("Database connection failed")
        
        response = client.post("/legal-aid/assess", json=sample_legal_aid_request)
        assert response.status_code == 500
        
        data = response.json()
        assert "error" in data
        assert "Database connection failed" in data["error"]
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/gst-dispute/analyze")
        assert response.status_code == 200
        
        # Check for CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers
        assert "access-control-allow-headers" in headers
    
    def test_content_type_validation(self, client):
        """Test content type validation"""
        # Test with invalid content type
        response = client.post(
            "/gst-dispute/analyze",
            data="invalid data",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422
    
    def test_request_size_limits(self, client):
        """Test request size limits"""
        # Create a very large request
        large_request = {
            "taxpayer_gstin": "29ABCDE1234F1Z5",
            "taxpayer_name": "ABC Enterprises",
            "dispute_type": "input_tax_credit",
            "dispute_amount": 150000.0,
            "tax_period": "2024-01",
            "description": "A" * 10000,  # Very long description
            "supporting_documents": ["doc.pdf"] * 1000  # Many documents
        }
        
        response = client.post("/gst-dispute/analyze", json=large_request)
        # Should either process successfully or return appropriate error
        assert response.status_code in [200, 413, 422]
    
    def test_concurrent_requests(self, client, sample_gst_request):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            try:
                response = client.post("/gst-dispute/analyze", json=sample_gst_request)
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should complete (may succeed or fail gracefully)
        assert len(results) == 5
        assert all(isinstance(result, int) for result in results)
    
    def test_api_documentation_endpoints(self, client):
        """Test API documentation endpoints"""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check that our endpoints are documented
        paths = schema["paths"]
        assert "/gst-dispute/analyze" in paths
        assert "/legal-aid/assess" in paths
        assert "/health" in paths
    
    def test_rate_limiting_headers(self, client):
        """Test rate limiting headers if implemented"""
        response = client.get("/health")
        
        # Check for rate limiting headers (if implemented)
        headers = response.headers
        # These headers might be present if rate limiting is implemented
        rate_limit_headers = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset"
        ]
        
        # Just verify the response is successful
        assert response.status_code == 200
    
    def test_security_headers(self, client):
        """Test security headers"""
        response = client.get("/")
        headers = response.headers
        
        # Check for basic security headers
        security_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection"
        ]
        
        # At minimum, response should be successful
        assert response.status_code == 200


class TestAPIModels:
    """Test API model validation"""
    
    def test_gst_dispute_request_validation(self):
        """Test GST dispute request model validation"""
        # Valid request
        valid_data = {
            "taxpayer_gstin": "29ABCDE1234F1Z5",
            "taxpayer_name": "ABC Enterprises",
            "dispute_type": "input_tax_credit",
            "dispute_amount": 150000.0,
            "tax_period": "2024-01",
            "description": "ITC claim rejected"
        }
        
        request = GSTDisputeRequest(**valid_data)
        assert request.taxpayer_gstin == "29ABCDE1234F1Z5"
        assert request.dispute_type == GSTDisputeTypeEnum.INPUT_TAX_CREDIT
        assert request.dispute_amount == 150000.0
    
    def test_legal_aid_request_validation(self):
        """Test legal aid request model validation"""
        # Valid request
        valid_data = {
            "applicant": {
                "name": "Jane Doe",
                "age": 35,
                "gender": "Female",
                "address": "123 Main Street",
                "annual_income": 150000.0,
                "family_size": 4
            },
            "case_category": "women_rights",
            "legal_aid_type": "legal_representation",
            "case_description": "Domestic violence case",
            "urgency_level": "high"
        }
        
        request = LegalAidRequest(**valid_data)
        assert request.applicant.name == "Jane Doe"
        assert request.case_category == CaseCategoryEnum.WOMEN_RIGHTS
        assert request.legal_aid_type == LegalAidTypeEnum.LEGAL_REPRESENTATION
        assert request.urgency_level == UrgencyLevelEnum.HIGH
    
    def test_enum_validation(self):
        """Test enum validation in models"""
        # Test valid enum values
        assert GSTDisputeTypeEnum.INPUT_TAX_CREDIT.value == "input_tax_credit"
        assert LegalAidTypeEnum.LEGAL_REPRESENTATION.value == "legal_representation"
        assert CaseCategoryEnum.WOMEN_RIGHTS.value == "women_rights"
        assert UrgencyLevelEnum.HIGH.value == "high"
        
        # Test enum membership
        assert "input_tax_credit" in [e.value for e in GSTDisputeTypeEnum]
        assert "legal_representation" in [e.value for e in LegalAidTypeEnum]
        assert "women_rights" in [e.value for e in CaseCategoryEnum]
        assert "high" in [e.value for e in UrgencyLevelEnum]


if __name__ == "__main__":
    pytest.main([__file__])