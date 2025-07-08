# XQELM API Documentation

The XQELM API provides programmatic access to quantum-enhanced legal reasoning capabilities. This RESTful API enables integration with legal software, web applications, and custom tools.

## 🌐 Base URL

```
Production: https://api.xqelm.org/v1
Staging: https://staging-api.xqelm.org/v1
Development: http://localhost:8000
```

## 🔐 Authentication

XQELM API uses API key authentication. Include your API key in the request headers:

```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

### Getting an API Key

1. Register at [XQELM Portal](https://portal.xqelm.org)
2. Create a new application
3. Generate API key from the dashboard
4. Configure rate limits and permissions

## 📊 Rate Limits

| Plan | Requests/Hour | Requests/Day | Concurrent |
|------|---------------|--------------|------------|
| Free | 100 | 1,000 | 2 |
| Basic | 1,000 | 10,000 | 5 |
| Pro | 10,000 | 100,000 | 20 |
| Enterprise | Unlimited | Unlimited | 100 |

## 🎯 Core Endpoints

### Health Check

Check API status and version information.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime": "72h30m15s",
  "quantum_backend": "default.qubit",
  "database": "connected",
  "redis": "connected"
}
```

### System Information

Get detailed system information and capabilities.

```http
GET /info
```

**Response:**
```json
{
  "version": "1.0.0",
  "build": "2024.01.15.1030",
  "quantum": {
    "backend": "default.qubit",
    "qubits": 4,
    "shots": 1000,
    "available_backends": ["default.qubit", "qiskit.aer", "cirq.simulator"]
  },
  "legal_system": {
    "jurisdiction": "india",
    "supported_languages": ["en", "hi", "ta", "te"],
    "use_cases": 7,
    "total_cases": 50000,
    "last_updated": "2024-01-15T00:00:00Z"
  },
  "features": {
    "quantum_reasoning": true,
    "explainable_ai": true,
    "multi_language": true,
    "real_time_analysis": true
  }
}
```

## 🏛️ Legal Use Cases

### GST Dispute Analysis

Analyze GST disputes and provide recommendations.

```http
POST /gst-dispute/analyze
```

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "case_id": "GST-2024-001",
  "analysis_timestamp": "2024-01-15T10:30:00Z",
  "dispute_type": "input_tax_credit",
  "risk_assessment": "medium",
  "recommended_action": "file_appeal",
  "success_probability": 0.75,
  "estimated_timeline": "3-6 months",
  "quantum_analysis": {
    "confidence": 0.85,
    "coherence": 0.80,
    "entanglement_measure": 0.65,
    "final_state": [0.6, 0.4],
    "measurement_count": 1000
  },
  "financial_analysis": {
    "dispute_amount": 150000.0,
    "penalty_amount": 15000.0,
    "interest_amount": 5000.0,
    "total_liability": 170000.0,
    "potential_savings": 140000.0,
    "cost_benefit_ratio": 9.33
  },
  "legal_analysis": {
    "applicable_sections": [
      "Section 16 of CGST Act",
      "Rule 36 of CGST Rules"
    ],
    "precedent_cases": [
      {
        "citation": "2023 (12) GSTL 456 (AAR)",
        "relevance": 0.85,
        "outcome": "favorable"
      }
    ],
    "key_arguments": [
      "Valid tax invoices available",
      "Goods received and accounted",
      "Interstate transaction properly documented"
    ]
  },
  "recommendations": [
    "Gather additional supporting documents",
    "File appeal within 30 days",
    "Prepare detailed written submissions",
    "Consider alternative dispute resolution"
  ],
  "next_steps": [
    {
      "action": "Document Review",
      "deadline": "2024-01-20",
      "priority": "high"
    },
    {
      "action": "File Appeal",
      "deadline": "2024-02-15",
      "priority": "critical"
    }
  ]
}
```

### Legal Aid Assessment

Assess eligibility for legal aid and recommend appropriate assistance.

```http
POST /legal-aid/assess
```

**Request Body:**
```json
{
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
    "is_disabled": false,
    "is_senior_citizen": false,
    "is_single_woman": true,
    "is_victim_of_crime": true,
    "assets_value": 50000.0,
    "monthly_expenses": 12000.0,
    "dependents": 3
  },
  "case_category": "women_rights",
  "legal_aid_type": "legal_representation",
  "case_description": "Domestic violence case requiring legal representation",
  "urgency_level": "high",
  "opposing_party": "Husband",
  "court_involved": "Family Court, Delhi",
  "case_stage": "pre_litigation",
  "lawyer_required": true,
  "emergency_case": true
}
```

**Response:**
```json
{
  "case_id": "LA-2024-001",
  "assessment_timestamp": "2024-01-15T10:30:00Z",
  "eligibility_score": 0.85,
  "is_eligible": true,
  "recommended_aid_type": "legal_representation",
  "recommended_authority": "dlsa",
  "priority_level": 2,
  "estimated_cost": 25000.0,
  "quantum_analysis": {
    "confidence": 0.90,
    "coherence": 0.88,
    "vulnerability_score": 0.75,
    "social_impact": 0.80
  },
  "eligibility_analysis": {
    "income_eligible": true,
    "category_eligible": true,
    "case_merit": 0.85,
    "urgency_score": 0.90,
    "social_vulnerability": 0.75
  },
  "financial_analysis": {
    "fee_waiver": true,
    "court_fee_exemption": 100.0,
    "service_cost": 0.0,
    "total_savings": 25000.0
  },
  "resource_allocation": {
    "lawyer": {
      "assigned": true,
      "lawyer_id": "LAW-001",
      "specialization": "Family Law",
      "experience": "8 years",
      "success_rate": 0.82
    },
    "authority": "District Legal Services Authority, Delhi",
    "estimated_duration": "6-12 months",
    "support_services": [
      "Counseling",
      "Document assistance",
      "Court representation"
    ]
  },
  "recommendations": [
    "Submit income certificate",
    "Provide caste certificate",
    "File police complaint if not done",
    "Gather evidence of domestic violence"
  ],
  "required_documents": [
    "Income certificate",
    "Caste certificate",
    "Identity proof",
    "Address proof",
    "Medical reports (if any)",
    "Police complaint copy"
  ],
  "next_steps": [
    {
      "action": "Document Submission",
      "deadline": "2024-01-20",
      "priority": "high"
    },
    {
      "action": "Legal Consultation",
      "deadline": "2024-01-22",
      "priority": "critical"
    }
  ]
}
```

## 🔍 Query and Search

### Case Search

Search for similar legal cases and precedents.

```http
POST /search/cases
```

**Request Body:**
```json
{
  "query": "property dispute boundary wall",
  "case_type": "civil",
  "jurisdiction": "delhi",
  "date_range": {
    "start": "2020-01-01",
    "end": "2024-01-01"
  },
  "limit": 10,
  "include_summary": true
}
```

**Response:**
```json
{
  "total_results": 156,
  "returned_results": 10,
  "search_time": "0.45s",
  "quantum_enhanced": true,
  "cases": [
    {
      "case_id": "CASE-001",
      "title": "Ram vs. Shyam - Property Boundary Dispute",
      "court": "Delhi High Court",
      "year": 2023,
      "citation": "2023 DLT 456",
      "relevance_score": 0.92,
      "summary": "Dispute regarding boundary wall construction...",
      "key_points": [
        "Survey settlement records are primary evidence",
        "Adverse possession requires 12 years continuous possession"
      ],
      "outcome": "Plaintiff successful",
      "quantum_similarity": 0.88
    }
  ]
}
```

### Legal Provision Search

Search for relevant legal provisions and statutes.

```http
POST /search/provisions
```

**Request Body:**
```json
{
  "query": "cheque bounce dishonour",
  "act": "negotiable_instruments_act",
  "section": null,
  "include_amendments": true
}
```

**Response:**
```json
{
  "total_results": 5,
  "provisions": [
    {
      "section": "Section 138",
      "act": "Negotiable Instruments Act, 1881",
      "title": "Dishonour of cheque for insufficiency of funds",
      "text": "Where any cheque drawn by a person...",
      "amendments": [
        {
          "year": 2018,
          "description": "Interim compensation provision added"
        }
      ],
      "relevance_score": 0.95,
      "related_sections": ["Section 139", "Section 140"]
    }
  ]
}
```

## 📈 Analytics and Insights

### Case Analytics

Get analytics and insights for legal cases.

```http
GET /analytics/cases
```

**Query Parameters:**
- `period`: Time period (day, week, month, year)
- `case_type`: Filter by case type
- `jurisdiction`: Filter by jurisdiction

**Response:**
```json
{
  "period": "month",
  "total_cases": 1250,
  "success_rate": 0.78,
  "average_duration": "4.5 months",
  "case_distribution": {
    "civil": 45,
    "criminal": 25,
    "family": 20,
    "commercial": 10
  },
  "quantum_performance": {
    "average_confidence": 0.85,
    "coherence_score": 0.82,
    "processing_time": "2.3s"
  }
}
```

## 🚨 Error Handling

The API uses standard HTTP status codes and returns detailed error information.

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "taxpayer_gstin",
      "issue": "Invalid GSTIN format"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `QUANTUM_ERROR` | 503 | Quantum backend unavailable |

## 📚 SDKs and Libraries

### Python SDK

```bash
pip install xqelm-python
```

```python
from xqelm import XQELMClient

client = XQELMClient(api_key="your_api_key")

# Analyze GST dispute
result = client.gst_dispute.analyze({
    "taxpayer_gstin": "29ABCDE1234F1Z5",
    "dispute_type": "input_tax_credit",
    "dispute_amount": 150000.0
})

print(f"Recommended action: {result.recommended_action}")
```

### JavaScript SDK

```bash
npm install @xqelm/client
```

```javascript
import { XQELMClient } from '@xqelm/client';

const client = new XQELMClient({
  apiKey: 'your_api_key'
});

// Assess legal aid eligibility
const result = await client.legalAid.assess({
  applicant: {
    name: "Jane Doe",
    annual_income: 150000
  },
  case_category: "women_rights"
});

console.log(`Eligibility: ${result.is_eligible}`);
```

## 🔧 Testing

### Sandbox Environment

Use the sandbox environment for testing:

```
Sandbox URL: https://sandbox-api.xqelm.org/v1
Sandbox API Key: sk_test_...
```

### Postman Collection

Download our [Postman Collection](https://api.xqelm.org/postman/collection.json) for easy API testing.

### OpenAPI Specification

Access the complete OpenAPI specification:
- [OpenAPI 3.0 JSON](https://api.xqelm.org/openapi.json)
- [Swagger UI](https://api.xqelm.org/docs)
- [ReDoc](https://api.xqelm.org/redoc)

---

## 📞 Support

- **Documentation**: [docs.xqelm.org](https://docs.xqelm.org)
- **API Status**: [status.xqelm.org](https://status.xqelm.org)
- **Support Email**: api-support@xqelm.org
- **Community Forum**: [community.xqelm.org](https://community.xqelm.org)

---

**Last Updated**: January 2024  
**API Version**: v1.0.0