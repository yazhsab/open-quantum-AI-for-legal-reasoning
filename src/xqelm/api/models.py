"""
API Models

Pydantic models for request and response schemas in the XQELM API.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from enum import Enum

from pydantic import BaseModel, Field, validator
from pydantic.types import EmailStr


# Enums for API
class ResponseTypeEnum(str, Enum):
    """Response types for API."""
    CASE_ANALYSIS = "case_analysis"
    LEGAL_ADVICE = "legal_advice"
    PROCEDURAL_GUIDANCE = "procedural_guidance"
    BAIL_APPLICATION = "bail_application"
    CHEQUE_BOUNCE = "cheque_bounce"
    PROPERTY_DISPUTE = "property_dispute"
    MOTOR_VEHICLE_CLAIM = "motor_vehicle_claim"
    CONSUMER_DISPUTE = "consumer_dispute"
    GST_DISPUTE = "gst_dispute"
    LEGAL_AID = "legal_aid"
    CONTRACT_ANALYSIS = "contract_analysis"


class BailTypeEnum(str, Enum):
    """Bail types for API."""
    ANTICIPATORY_BAIL = "anticipatory_bail"
    REGULAR_BAIL = "regular_bail"
    INTERIM_BAIL = "interim_bail"
    STATUTORY_BAIL = "statutory_bail"
    DEFAULT_BAIL = "default_bail"


class OffenseCategoryEnum(str, Enum):
    """Offense categories for API."""
    BAILABLE = "bailable"
    NON_BAILABLE = "non_bailable"
    COGNIZABLE = "cognizable"
    NON_COGNIZABLE = "non_cognizable"
    HEINOUS = "heinous"
    ECONOMIC = "economic"
    WHITE_COLLAR = "white_collar"


class ChequeTypeEnum(str, Enum):
    """Cheque types for API."""
    ACCOUNT_PAYEE = "account_payee"
    BEARER = "bearer"
    ORDER = "order"
    CROSSED = "crossed"
    POST_DATED = "post_dated"
    STALE = "stale"


class DishonorReasonEnum(str, Enum):
    """Dishonor reasons for API."""
    INSUFFICIENT_FUNDS = "insufficient_funds"
    ACCOUNT_CLOSED = "account_closed"
    SIGNATURE_MISMATCH = "signature_mismatch"
    AMOUNT_MISMATCH = "amount_mismatch"
    POST_DATED = "post_dated"
    STOP_PAYMENT = "stop_payment"
    FROZEN_ACCOUNT = "frozen_account"
    TECHNICAL_REASON = "technical_reason"


class CaseStageEnum(str, Enum):
    """Case stages for API."""
    PRE_LITIGATION = "pre_litigation"
    LEGAL_NOTICE = "legal_notice"
    COMPLAINT_FILED = "complaint_filed"
    TRIAL = "trial"
    JUDGMENT = "judgment"
    APPEAL = "appeal"


# Authentication models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# General query models
class QueryRequest(BaseModel):
    """General query request model."""
    query: str = Field(..., min_length=10, max_length=10000, description="Legal query text")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")
    use_case: Optional[str] = Field(None, description="Specific use case for specialized processing")
    response_type: ResponseTypeEnum = Field(ResponseTypeEnum.CASE_ANALYSIS, description="Type of response required")
    include_explanations: bool = Field(True, description="Include quantum explanations in response")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What are the chances of getting bail in a cheque bounce case under Section 138?",
                "context": {"jurisdiction": "india", "court_type": "magistrate"},
                "use_case": "bail_application",
                "response_type": "legal_advice",
                "include_explanations": True
            }
        }


class QueryResponse(BaseModel):
    """General query response model."""
    query_id: str = Field(..., description="Unique identifier for the query")
    response: str = Field(..., description="Generated legal response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the analysis")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the response")
    
    class Config:
        schema_extra = {
            "example": {
                "query_id": "xqelm_1234567890",
                "response": "Based on quantum-enhanced legal analysis...",
                "confidence": 0.85,
                "processing_time": 2.34,
                "metadata": {
                    "quantum_coherence": 0.92,
                    "citations_count": 5,
                    "recommendations_count": 3
                }
            }
        }


# Bail application models
class BailApplicationRequest(BaseModel):
    """Bail application analysis request."""
    applicant_name: str = Field(..., min_length=2, max_length=100)
    case_number: Optional[str] = Field(None, max_length=50)
    court: str = Field(..., min_length=5, max_length=100)
    offense_details: str = Field(..., min_length=10, max_length=2000)
    sections_charged: List[str] = Field(..., min_items=1, max_items=20)
    offense_category: OffenseCategoryEnum
    bail_type: BailTypeEnum
    
    # Applicant details
    age: Optional[int] = Field(None, ge=18, le=100)
    occupation: Optional[str] = Field(None, max_length=100)
    address: str = Field(..., min_length=10, max_length=500)
    family_details: Optional[str] = Field(None, max_length=1000)
    previous_convictions: Optional[List[str]] = Field(None, max_items=10)
    
    # Case details
    date_of_offense: Optional[date] = None
    date_of_arrest: Optional[date] = None
    investigation_status: str = Field(..., max_length=200)
    evidence_status: str = Field(..., max_length=200)
    
    # Supporting information
    supporting_documents: Optional[List[str]] = Field(None, max_items=20)
    character_witnesses: Optional[List[str]] = Field(None, max_items=10)
    medical_condition: Optional[str] = Field(None, max_length=500)
    employment_verification: Optional[str] = Field(None, max_length=200)
    property_details: Optional[str] = Field(None, max_length=500)
    
    # Additional context
    additional_context: Optional[str] = Field(None, max_length=2000)
    
    @validator('date_of_arrest')
    def validate_arrest_date(cls, v, values):
        if v and 'date_of_offense' in values and values['date_of_offense']:
            if v < values['date_of_offense']:
                raise ValueError('Arrest date cannot be before offense date')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "applicant_name": "John Doe",
                "case_number": "CR-123/2024",
                "court": "Additional Sessions Court, Delhi",
                "offense_details": "Accused of cheating under Section 420 IPC",
                "sections_charged": ["420", "406"],
                "offense_category": "non_bailable",
                "bail_type": "regular_bail",
                "age": 35,
                "occupation": "Software Engineer",
                "address": "123 Main Street, New Delhi",
                "investigation_status": "Ongoing",
                "evidence_status": "Under collection"
            }
        }


class BailAnalysisResponse(BaseModel):
    """Bail application analysis response."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    recommendation: str = Field(..., description="Bail recommendation (grant/deny/conditional)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the recommendation")
    reasoning: str = Field(..., description="Detailed reasoning for the recommendation")
    conditions: List[str] = Field(..., description="Recommended bail conditions")
    risk_assessment: Dict[str, float] = Field(..., description="Risk factor assessments")
    precedents: List[str] = Field(..., description="Relevant legal precedents")
    statutory_basis: List[str] = Field(..., description="Applicable statutory provisions")
    detailed_response: str = Field(..., description="Complete formatted legal response")
    processing_time: float = Field(..., description="Processing time in seconds")
    quantum_metrics: Dict[str, Any] = Field(..., description="Quantum analysis metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "bail_1234567890",
                "recommendation": "conditional",
                "confidence": 0.78,
                "reasoning": "The quantum analysis indicates moderate risk factors...",
                "conditions": [
                    "Furnish personal bond of ₹50,000",
                    "Not to leave the jurisdiction without permission"
                ],
                "risk_assessment": {
                    "flight_risk": 0.3,
                    "tampering_evidence": 0.4
                },
                "precedents": ["Sanjay Chandra v. CBI"],
                "statutory_basis": ["Section 437 CrPC"],
                "processing_time": 3.45
            }
        }


# Cheque bounce models
class ChequeDetailsModel(BaseModel):
    """Cheque details model."""
    cheque_number: str = Field(..., min_length=1, max_length=50)
    cheque_date: date
    amount: float = Field(..., gt=0, description="Cheque amount")
    bank_name: str = Field(..., min_length=2, max_length=100)
    account_number: str = Field(..., min_length=5, max_length=30)
    drawer_name: str = Field(..., min_length=2, max_length=100)
    payee_name: str = Field(..., min_length=2, max_length=100)
    cheque_type: ChequeTypeEnum
    dishonor_date: date
    dishonor_reason: DishonorReasonEnum
    bank_memo: str = Field(..., min_length=5, max_length=500)
    first_presentation_date: Optional[date] = None
    second_presentation_date: Optional[date] = None
    purpose_of_cheque: Optional[str] = Field(None, max_length=200)
    underlying_transaction: Optional[str] = Field(None, max_length=500)
    consideration: Optional[str] = Field(None, max_length=500)


class LegalNoticeModel(BaseModel):
    """Legal notice details model."""
    notice_date: Optional[date] = None
    notice_served_date: Optional[date] = None
    service_method: Optional[str] = Field(None, max_length=100)
    response_received: bool = False
    response_date: Optional[date] = None
    response_details: Optional[str] = Field(None, max_length=1000)
    within_30_days: Optional[bool] = None
    proper_service: Optional[bool] = None
    adequate_content: Optional[bool] = None


class ChequeBounceRequest(BaseModel):
    """Cheque bounce case analysis request."""
    case_id: str = Field(..., min_length=1, max_length=50)
    cheque_details: ChequeDetailsModel
    legal_notice: LegalNoticeModel
    
    # Parties
    complainant_name: str = Field(..., min_length=2, max_length=100)
    complainant_address: str = Field(..., min_length=10, max_length=500)
    accused_name: str = Field(..., min_length=2, max_length=100)
    accused_address: str = Field(..., min_length=10, max_length=500)
    
    # Case details
    case_stage: CaseStageEnum
    court: Optional[str] = Field(None, max_length=100)
    case_number: Optional[str] = Field(None, max_length=50)
    filing_date: Optional[date] = None
    
    # Evidence and financial details
    supporting_documents: Optional[List[str]] = Field(None, max_items=20)
    witness_details: Optional[List[str]] = Field(None, max_items=10)
    interest_claimed: Optional[float] = Field(None, ge=0)
    compensation_claimed: Optional[float] = Field(None, ge=0)
    
    # Additional information
    previous_transactions: Optional[List[str]] = Field(None, max_items=10)
    relationship_between_parties: Optional[str] = Field(None, max_length=200)
    additional_context: Optional[str] = Field(None, max_length=2000)
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "CB-001-2024",
                "cheque_details": {
                    "cheque_number": "123456",
                    "cheque_date": "2024-01-15",
                    "amount": 100000.0,
                    "bank_name": "State Bank of India",
                    "account_number": "12345678901",
                    "drawer_name": "ABC Company",
                    "payee_name": "XYZ Enterprises",
                    "cheque_type": "account_payee",
                    "dishonor_date": "2024-01-20",
                    "dishonor_reason": "insufficient_funds",
                    "bank_memo": "Insufficient funds in account"
                },
                "complainant_name": "XYZ Enterprises",
                "accused_name": "ABC Company",
                "case_stage": "legal_notice"
            }
        }


class ChequeBounceAnalysisResponse(BaseModel):
    """Cheque bounce case analysis response."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    liability_assessment: str = Field(..., description="Liability assessment (criminal/civil/both/none)")
    conviction_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of conviction")
    compensation_estimate: float = Field(..., ge=0.0, description="Estimated compensation amount")
    statutory_compliance: Dict[str, bool] = Field(..., description="Section 138 compliance check")
    available_defenses: List[str] = Field(..., description="Available defense strategies")
    defense_strength: Dict[str, float] = Field(..., description="Strength of each defense")
    recommendations: List[str] = Field(..., description="Legal recommendations")
    next_steps: List[str] = Field(..., description="Recommended next steps")
    similar_cases: List[Dict[str, Any]] = Field(..., description="Similar case precedents")
    detailed_response: str = Field(..., description="Complete formatted legal response")
    processing_time: float = Field(..., description="Processing time in seconds")
    quantum_metrics: Dict[str, Any] = Field(..., description="Quantum analysis metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "cb_1234567890",
                "liability_assessment": "criminal",
                "conviction_probability": 0.82,
                "compensation_estimate": 200000.0,
                "statutory_compliance": {
                    "dishonor_of_cheque": True,
                    "insufficient_funds": True,
                    "legal_notice_timely": True,
                    "failure_to_pay": True
                },
                "available_defenses": ["improper_legal_notice"],
                "defense_strength": {"improper_legal_notice": 0.3},
                "processing_time": 2.87
            }
        }


# Knowledge base models
class KnowledgeSearchResponse(BaseModel):
    """Knowledge base search response."""
    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    processing_time: float = Field(..., description="Search processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "bail application procedure",
                "results": [
                    {
                        "document_id": "doc_123",
                        "title": "Sanjay Chandra v. CBI",
                        "document_type": "case_law",
                        "citation": "(2012) 1 SCC 40",
                        "summary": "Economic offenses and bail considerations",
                        "similarity_score": 0.89,
                        "relevance_score": 0.92,
                        "match_type": "semantic",
                        "matched_terms": ["bail", "economic", "offense"]
                    }
                ],
                "total_results": 1,
                "processing_time": 0.45
            }
        }


# Error models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid request parameters",
                "status_code": 400,
                "timestamp": "2024-01-15T10:30:00Z",
                "details": {"field": "query", "message": "Query too short"}
            }
        }


# Statistics models
class SystemStatistics(BaseModel):
    """System statistics model."""
    timestamp: str = Field(..., description="Statistics timestamp")
    components: Dict[str, Any] = Field(..., description="Component-wise statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "components": {
                    "quantum_model": {
                        "queries_processed": 1250,
                        "average_processing_time": 2.34,
                        "success_rate": 0.98
                    },
                    "knowledge_base": {
                        "total_documents": 50000,
                        "search_queries": 3400,
                        "cache_hits": 2100
                    }
                }
            }
        }


# Health check models
class HealthStatus(BaseModel):
    """Health status model."""
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    components: Optional[Dict[str, Any]] = Field(None, description="Component health status")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "components": {
                    "quantum_model": "healthy",
                    "knowledge_base": {"status": "healthy", "documents": 50000},
                    "bail_manager": "healthy",
                    "cheque_bounce_manager": "healthy"
                }
            }
        }


# Validation models
class ValidationError(BaseModel):
    """Validation error model."""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(..., description="Invalid value")


class ValidationResponse(BaseModel):
    """Validation response model."""
    valid: bool = Field(..., description="Whether the input is valid")
    errors: List[ValidationError] = Field(..., description="List of validation errors")
    
    class Config:
        schema_extra = {
            "example": {
                "valid": False,
                "errors": [
                    {
                        "field": "query",
                        "message": "Query must be at least 10 characters long",
                        "value": "short"
                    }
                ]
            }
        }


# Batch processing models
class BatchQueryRequest(BaseModel):
    """Batch query processing request."""
    queries: List[QueryRequest] = Field(..., min_items=1, max_items=100)
    parallel_processing: bool = Field(True, description="Process queries in parallel")
    
    class Config:
        schema_extra = {
            "example": {
                "queries": [
                    {
                        "query": "What are the bail provisions under CrPC?",
                        "response_type": "legal_advice"
                    },
                    {
                        "query": "Section 138 NI Act requirements",
                        "response_type": "case_analysis"
                    }
                ],
                "parallel_processing": True
            }
        }


class BatchQueryResponse(BaseModel):
    """Batch query processing response."""
    batch_id: str = Field(..., description="Unique batch identifier")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successfully processed queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    results: List[Union[QueryResponse, ErrorResponse]] = Field(..., description="Query results")
    total_processing_time: float = Field(..., description="Total batch processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_1234567890",
                "total_queries": 2,
                "successful_queries": 2,
                "failed_queries": 0,
                "results": [],
                "total_processing_time": 4.56
            }
        }


# Property Dispute models
class PropertyTypeEnum(str, Enum):
    """Property types for API."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    AGRICULTURAL = "agricultural"
    INDUSTRIAL = "industrial"
    PLOT = "plot"
    APARTMENT = "apartment"
    VILLA = "villa"
    OFFICE = "office"
    SHOP = "shop"
    WAREHOUSE = "warehouse"


class DisputeTypeEnum(str, Enum):
    """Property dispute types for API."""
    TITLE_DISPUTE = "title_dispute"
    BOUNDARY_DISPUTE = "boundary_dispute"
    POSSESSION_DISPUTE = "possession_dispute"
    PARTITION_SUIT = "partition_suit"
    EASEMENT_RIGHTS = "easement_rights"
    LANDLORD_TENANT = "landlord_tenant"
    SALE_DEED_DISPUTE = "sale_deed_dispute"
    INHERITANCE_DISPUTE = "inheritance_dispute"
    ENCROACHMENT = "encroachment"
    MORTGAGE_DISPUTE = "mortgage_dispute"


class PropertyDisputeRequest(BaseModel):
    """Property dispute analysis request."""
    case_id: str = Field(..., min_length=1, max_length=50)
    dispute_type: DisputeTypeEnum
    property_type: PropertyTypeEnum
    
    # Property details
    property_address: str = Field(..., min_length=10, max_length=500)
    property_area: Optional[float] = Field(None, gt=0, description="Property area in sq ft")
    property_value: Optional[float] = Field(None, gt=0, description="Property value in INR")
    survey_number: Optional[str] = Field(None, max_length=50)
    
    # Parties
    plaintiff_name: str = Field(..., min_length=2, max_length=100)
    plaintiff_address: str = Field(..., min_length=10, max_length=500)
    defendant_name: str = Field(..., min_length=2, max_length=100)
    defendant_address: str = Field(..., min_length=10, max_length=500)
    
    # Title documents
    title_documents: List[str] = Field(..., min_items=1, max_items=20)
    registration_details: Optional[str] = Field(None, max_length=500)
    mutation_records: Optional[str] = Field(None, max_length=500)
    
    # Dispute details
    dispute_description: str = Field(..., min_length=20, max_length=2000)
    possession_status: str = Field(..., max_length=200)
    disputed_area: Optional[float] = Field(None, gt=0)
    
    # Case details
    case_stage: CaseStageEnum
    court: Optional[str] = Field(None, max_length=100)
    case_number: Optional[str] = Field(None, max_length=50)
    filing_date: Optional[date] = None
    
    # Relief sought
    possession_sought: bool = False
    damages_claimed: Optional[float] = Field(None, ge=0)
    injunction_sought: bool = False
    declaration_sought: bool = False
    
    # Additional context
    previous_litigation: Optional[List[str]] = Field(None, max_items=10)
    survey_reports: Optional[List[str]] = Field(None, max_items=5)
    additional_context: Optional[str] = Field(None, max_length=2000)
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "PD-001-2024",
                "dispute_type": "title_dispute",
                "property_type": "residential",
                "property_address": "Plot No. 123, Sector 45, Gurgaon",
                "property_area": 2000.0,
                "property_value": 5000000.0,
                "plaintiff_name": "John Doe",
                "plaintiff_address": "123 Main Street, Delhi",
                "defendant_name": "Jane Smith",
                "defendant_address": "456 Park Avenue, Gurgaon",
                "title_documents": ["Sale Deed", "Registry"],
                "dispute_description": "Dispute over ownership of residential plot",
                "possession_status": "Plaintiff in possession",
                "case_stage": "trial"
            }
        }


class PropertyDisputeAnalysisResponse(BaseModel):
    """Property dispute analysis response."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    title_strength: Dict[str, float] = Field(..., description="Title strength assessment")
    possession_rights: Dict[str, float] = Field(..., description="Possession rights analysis")
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Case success probability")
    compensation_estimate: Dict[str, float] = Field(..., description="Compensation estimates")
    recommendations: List[str] = Field(..., description="Legal recommendations")
    evidence_gaps: List[str] = Field(..., description="Evidence gaps identified")
    settlement_options: List[str] = Field(..., description="Settlement possibilities")
    case_duration_estimate: Dict[str, int] = Field(..., description="Case duration estimates")
    litigation_cost_estimate: float = Field(..., description="Estimated litigation costs")
    detailed_response: str = Field(..., description="Complete formatted legal response")
    processing_time: float = Field(..., description="Processing time in seconds")
    quantum_metrics: Dict[str, Any] = Field(..., description="Quantum analysis metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "pd_1234567890",
                "title_strength": {"plaintiff": 0.75, "defendant": 0.25},
                "possession_rights": {"current_possession": 0.8, "legal_possession": 0.7},
                "success_probability": 0.72,
                "compensation_estimate": {"damages": 500000.0, "costs": 50000.0},
                "recommendations": ["Strengthen title documents", "Conduct survey"],
                "processing_time": 3.21
            }
        }


# Motor Vehicle Claims models
class AccidentTypeEnum(str, Enum):
    """Motor vehicle accident types for API."""
    HEAD_ON_COLLISION = "head_on_collision"
    REAR_END_COLLISION = "rear_end_collision"
    SIDE_IMPACT = "side_impact"
    ROLLOVER = "rollover"
    HIT_AND_RUN = "hit_and_run"
    PEDESTRIAN_ACCIDENT = "pedestrian_accident"
    MULTI_VEHICLE = "multi_vehicle"
    SINGLE_VEHICLE = "single_vehicle"


class InjuryTypeEnum(str, Enum):
    """Injury types for API."""
    FATAL = "fatal"
    GRIEVOUS = "grievous"
    SIMPLE = "simple"
    PERMANENT_DISABILITY = "permanent_disability"
    TEMPORARY_DISABILITY = "temporary_disability"
    NO_INJURY = "no_injury"


class VehicleTypeEnum(str, Enum):
    """Vehicle types for API."""
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    TRUCK = "truck"
    BUS = "bus"
    AUTO_RICKSHAW = "auto_rickshaw"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    COMMERCIAL_VEHICLE = "commercial_vehicle"


class MotorVehicleClaimRequest(BaseModel):
    """Motor vehicle claim analysis request."""
    case_id: str = Field(..., min_length=1, max_length=50)
    accident_type: AccidentTypeEnum
    
    # Accident details
    accident_date: date
    accident_location: str = Field(..., min_length=10, max_length=500)
    accident_description: str = Field(..., min_length=20, max_length=2000)
    police_complaint_number: Optional[str] = Field(None, max_length=50)
    
    # Vehicle details
    claimant_vehicle_type: VehicleTypeEnum
    claimant_vehicle_number: Optional[str] = Field(None, max_length=20)
    opposite_vehicle_type: VehicleTypeEnum
    opposite_vehicle_number: Optional[str] = Field(None, max_length=20)
    
    # Parties
    claimant_name: str = Field(..., min_length=2, max_length=100)
    claimant_address: str = Field(..., min_length=10, max_length=500)
    claimant_age: Optional[int] = Field(None, ge=0, le=120)
    opposite_party_name: str = Field(..., min_length=2, max_length=100)
    opposite_party_address: str = Field(..., min_length=10, max_length=500)
    
    # Injury and damage details
    injury_type: InjuryTypeEnum
    injury_description: Optional[str] = Field(None, max_length=1000)
    medical_expenses: Optional[float] = Field(None, ge=0)
    vehicle_damage_cost: Optional[float] = Field(None, ge=0)
    
    # Income details (for compensation calculation)
    claimant_monthly_income: Optional[float] = Field(None, ge=0)
    claimant_occupation: Optional[str] = Field(None, max_length=100)
    loss_of_earning_period: Optional[int] = Field(None, ge=0, description="Days of earning loss")
    
    # Insurance details
    claimant_insurance_policy: Optional[str] = Field(None, max_length=100)
    opposite_party_insurance: Optional[str] = Field(None, max_length=100)
    insurance_claim_amount: Optional[float] = Field(None, ge=0)
    
    # Case details
    case_stage: CaseStageEnum
    tribunal: Optional[str] = Field(None, max_length=100)
    case_number: Optional[str] = Field(None, max_length=50)
    filing_date: Optional[date] = None
    
    # Evidence
    supporting_documents: Optional[List[str]] = Field(None, max_items=20)
    witness_details: Optional[List[str]] = Field(None, max_items=10)
    
    # Additional context
    fault_percentage: Optional[Dict[str, float]] = Field(None, description="Fault distribution")
    additional_context: Optional[str] = Field(None, max_length=2000)
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "MVC-001-2024",
                "accident_type": "rear_end_collision",
                "accident_date": "2024-01-15",
                "accident_location": "NH-8, Gurgaon",
                "accident_description": "Rear-end collision at traffic signal",
                "claimant_vehicle_type": "car",
                "opposite_vehicle_type": "truck",
                "claimant_name": "John Doe",
                "claimant_address": "123 Main Street, Delhi",
                "opposite_party_name": "ABC Transport",
                "injury_type": "grievous",
                "medical_expenses": 200000.0,
                "case_stage": "complaint_filed"
            }
        }


class MotorVehicleClaimAnalysisResponse(BaseModel):
    """Motor vehicle claim analysis response."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    liability_assessment: Dict[str, float] = Field(..., description="Liability distribution")
    compensation_estimate: Dict[str, float] = Field(..., description="Compensation breakdown")
    total_compensation: float = Field(..., ge=0.0, description="Total estimated compensation")
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Claim success probability")
    insurance_coverage: Dict[str, Any] = Field(..., description="Insurance coverage analysis")
    recommendations: List[str] = Field(..., description="Legal recommendations")
    evidence_gaps: List[str] = Field(..., description="Evidence gaps identified")
    settlement_estimate: Dict[str, float] = Field(..., description="Settlement amount estimates")
    case_duration_estimate: Dict[str, int] = Field(..., description="Case duration estimates")
    detailed_response: str = Field(..., description="Complete formatted legal response")
    processing_time: float = Field(..., description="Processing time in seconds")
    quantum_metrics: Dict[str, Any] = Field(..., description="Quantum analysis metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "mvc_1234567890",
                "liability_assessment": {"claimant": 0.2, "opposite_party": 0.8},
                "compensation_estimate": {
                    "medical_expenses": 200000.0,
                    "vehicle_damage": 150000.0,
                    "loss_of_income": 100000.0,
                    "pain_suffering": 50000.0
                },
                "total_compensation": 500000.0,
                "success_probability": 0.78,
                "processing_time": 2.95
            }
        }


# Consumer Dispute models
class ComplaintTypeEnum(str, Enum):
    """Consumer complaint types for API."""
    DEFECTIVE_GOODS = "defective_goods"
    DEFICIENCY_IN_SERVICE = "deficiency_in_service"
    UNFAIR_TRADE_PRACTICE = "unfair_trade_practice"
    OVERCHARGING = "overcharging"
    FALSE_ADVERTISEMENT = "false_advertisement"
    PRODUCT_LIABILITY = "product_liability"
    E_COMMERCE_DISPUTE = "e_commerce_dispute"
    INSURANCE_DISPUTE = "insurance_dispute"
    BANKING_SERVICE = "banking_service"
    TELECOM_SERVICE = "telecom_service"


class ServiceTypeEnum(str, Enum):
    """Service types for API."""
    BANKING = "banking"
    INSURANCE = "insurance"
    TELECOM = "telecom"
    ELECTRICITY = "electricity"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    TRANSPORT = "transport"
    HOSPITALITY = "hospitality"
    E_COMMERCE = "e_commerce"
    REAL_ESTATE = "real_estate"


class ForumTypeEnum(str, Enum):
    """Consumer forum types for API."""
    DISTRICT_FORUM = "district_forum"
    STATE_COMMISSION = "state_commission"
    NATIONAL_COMMISSION = "national_commission"
    ONLINE_DISPUTE_RESOLUTION = "online_dispute_resolution"


class ConsumerDisputeRequest(BaseModel):
    """Consumer dispute analysis request."""
    case_id: str = Field(..., min_length=1, max_length=50)
    complaint_type: ComplaintTypeEnum
    forum_type: ForumTypeEnum
    
    # Consumer details
    consumer_name: str = Field(..., min_length=2, max_length=100)
    consumer_address: str = Field(..., min_length=10, max_length=500)
    consumer_phone: Optional[str] = Field(None, max_length=15)
    consumer_email: Optional[EmailStr] = None
    is_senior_citizen: bool = False
    
    # Opposite party details
    opposite_party_name: str = Field(..., min_length=2, max_length=100)
    opposite_party_address: str = Field(..., min_length=10, max_length=500)
    business_type: str = Field(..., max_length=100)
    
    # Product/Service details
    product_name: Optional[str] = Field(None, max_length=200)
    service_type: Optional[ServiceTypeEnum] = None
    purchase_date: Optional[date] = None
    purchase_amount: Optional[float] = Field(None, ge=0)
    
    # Complaint details
    complaint_description: str = Field(..., min_length=20, max_length=2000)
    deficiency_description: Optional[str] = Field(None, max_length=1000)
    financial_loss: Optional[float] = Field(None, ge=0)
    
    # Case details
    case_stage: CaseStageEnum
    case_number: Optional[str] = Field(None, max_length=50)
    filing_date: Optional[date] = None
    
    # Relief sought
    compensation_claimed: Optional[float] = Field(None, ge=0)
    replacement_sought: bool = False
    refund_sought: bool = False
    service_improvement_sought: bool = False
    
    # Evidence
    supporting_documents: Optional[List[str]] = Field(None, max_items=20)
    correspondence_history: Optional[List[str]] = Field(None, max_items=10)
    
    # Additional context
    mediation_attempted: bool = False
    previous_complaints: Optional[List[str]] = Field(None, max_items=5)
    additional_context: Optional[str] = Field(None, max_length=2000)
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "CD-001-2024",
                "complaint_type": "defective_goods",
                "forum_type": "district_forum",
                "consumer_name": "John Doe",
                "consumer_address": "123 Main Street, Delhi",
                "opposite_party_name": "ABC Electronics",
                "opposite_party_address": "456 Market Street, Delhi",
                "business_type": "Electronics Retailer",
                "product_name": "Smartphone XYZ",
                "purchase_date": "2024-01-15",
                "purchase_amount": 25000.0,
                "complaint_description": "Mobile phone stopped working within warranty period",
                "financial_loss": 25000.0,
                "case_stage": "pre_litigation",
                "compensation_claimed": 30000.0
            }
        }


class ConsumerDisputeAnalysisResponse(BaseModel):
    """Consumer dispute analysis response."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    case_merit: float = Field(..., ge=0.0, le=1.0, description="Case merit assessment")
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Success probability")
    jurisdiction_validity: bool = Field(..., description="Forum jurisdiction validity")
    limitation_compliance: bool = Field(..., description="Limitation period compliance")
    evidence_strength: Dict[str, float] = Field(..., description="Evidence strength assessment")
    compensation_estimate: Dict[str, float] = Field(..., description="Compensation estimates")
    relief_likelihood: Dict[str, float] = Field(..., description="Relief likelihood assessment")
    recommendations: List[str] = Field(..., description="Legal recommendations")
    evidence_gaps: List[str] = Field(..., description="Evidence gaps identified")
    settlement_options: List[str] = Field(..., description="Settlement possibilities")
    case_duration_estimate: Dict[str, int] = Field(..., description="Case duration estimates")
    litigation_cost_estimate: float = Field(..., description="Estimated litigation costs")
    detailed_response: str = Field(..., description="Complete formatted legal response")
    processing_time: float = Field(..., description="Processing time in seconds")
    quantum_metrics: Dict[str, Any] = Field(..., description="Quantum analysis metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "cd_1234567890",
                "case_merit": 0.75,
                "success_probability": 0.68,
                "jurisdiction_validity": True,
                "limitation_compliance": True,
                "evidence_strength": {"overall_strength": 0.7},
                "compensation_estimate": {
                    "financial_loss": 25000.0,
                    "mental_agony": 7500.0,
                    "litigation_cost": 2500.0,
                    "total_estimated": 35000.0
                },
                "relief_likelihood": {
                    "compensation": 0.7,
                    "replacement": 0.8,
                    "refund": 0.6
                },
                "processing_time": 2.67
            }
        }


# GST Dispute models
class GSTDisputeTypeEnum(str, Enum):
    """GST dispute types for API."""
    INPUT_TAX_CREDIT = "input_tax_credit"
    CLASSIFICATION = "classification"
    VALUATION = "valuation"
    PLACE_OF_SUPPLY = "place_of_supply"
    REFUND = "refund"
    PENALTY = "penalty"
    INTEREST = "interest"
    REVERSE_CHARGE = "reverse_charge"
    COMPOSITION_SCHEME = "composition_scheme"
    REGISTRATION = "registration"


class GSTTaxRateEnum(str, Enum):
    """GST tax rates for API."""
    EXEMPT = "0.0"
    GST_5 = "5.0"
    GST_12 = "12.0"
    GST_18 = "18.0"
    GST_28 = "28.0"


class BusinessTypeEnum(str, Enum):
    """Business types for API."""
    MANUFACTURER = "manufacturer"
    TRADER = "trader"
    SERVICE_PROVIDER = "service_provider"
    E_COMMERCE = "e_commerce"
    IMPORTER = "importer"
    EXPORTER = "exporter"
    COMPOSITION_DEALER = "composition_dealer"
    REGULAR_DEALER = "regular_dealer"


class GSTDisputeRequest(BaseModel):
    """GST dispute analysis request."""
    case_id: str = Field(..., min_length=1, max_length=50)
    taxpayer_name: str = Field(..., min_length=2, max_length=100)
    gstin: str = Field(..., min_length=15, max_length=15, description="15-digit GSTIN")
    dispute_type: GSTDisputeTypeEnum
    business_type: BusinessTypeEnum
    dispute_description: str = Field(..., min_length=10, max_length=2000)
    disputed_amount: float = Field(..., gt=0, description="Disputed tax amount")
    tax_period: str = Field(..., description="Tax period (MM/YYYY)")
    
    # Transaction details
    transaction_value: Optional[float] = Field(None, gt=0)
    tax_rate_claimed: Optional[GSTTaxRateEnum] = None
    tax_rate_demanded: Optional[GSTTaxRateEnum] = None
    
    # Timeline
    notice_date: Optional[date] = None
    response_deadline: Optional[date] = None
    
    # Supporting documents
    evidence_documents: Optional[List[str]] = Field(None, max_items=20)
    invoices_provided: bool = False
    books_of_accounts: bool = False
    contracts_agreements: bool = False
    
    # Previous proceedings
    show_cause_notice: bool = False
    personal_hearing_attended: bool = False
    written_submissions: bool = False
    
    # Legal representation
    has_legal_counsel: bool = False
    authorized_representative: bool = False
    
    # Additional context
    similar_cases_precedent: Optional[List[str]] = Field(None, max_items=10)
    department_position: Optional[str] = Field(None, max_length=1000)
    taxpayer_position: Optional[str] = Field(None, max_length=1000)
    additional_context: Optional[str] = Field(None, max_length=2000)
    
    @validator('gstin')
    def validate_gstin(cls, v):
        if len(v) != 15:
            raise ValueError('GSTIN must be exactly 15 characters')
        return v.upper()
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "GST-001-2024",
                "taxpayer_name": "ABC Enterprises",
                "gstin": "29ABCDE1234F1Z5",
                "dispute_type": "input_tax_credit",
                "business_type": "manufacturer",
                "dispute_description": "Denial of input tax credit on capital goods",
                "disputed_amount": 500000.0,
                "tax_period": "03/2024",
                "transaction_value": 2500000.0,
                "tax_rate_claimed": "18.0",
                "invoices_provided": True,
                "books_of_accounts": True
            }
        }


class GSTDisputeAnalysisResponse(BaseModel):
    """GST dispute analysis response."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Success probability")
    legal_position_strength: Dict[str, float] = Field(..., description="Legal position strength scores")
    applicable_provisions: List[str] = Field(..., description="Applicable GST provisions")
    relevant_precedents: List[Dict[str, Any]] = Field(..., description="Relevant legal precedents")
    
    # Financial analysis
    estimated_liability: Dict[str, float] = Field(..., description="Estimated tax liability")
    penalty_assessment: Dict[str, float] = Field(..., description="Penalty calculation")
    interest_calculation: Dict[str, float] = Field(..., description="Interest calculation")
    
    # Procedural analysis
    limitation_compliance: bool = Field(..., description="Limitation period compliance")
    forum_jurisdiction: str = Field(..., description="Appropriate forum")
    appeal_options: List[str] = Field(..., description="Available appeal options")
    
    # Recommendations
    recommended_strategy: List[str] = Field(..., description="Recommended legal strategy")
    settlement_options: List[Dict[str, Any]] = Field(..., description="Settlement options")
    documentation_requirements: List[str] = Field(..., description="Required documentation")
    
    # Risk assessment
    litigation_risk: float = Field(..., ge=0.0, le=1.0, description="Litigation risk")
    compliance_risk: float = Field(..., ge=0.0, le=1.0, description="Compliance risk")
    financial_risk: float = Field(..., ge=0.0, le=1.0, description="Financial risk")
    
    # Timeline
    estimated_resolution_time: int = Field(..., description="Estimated resolution time in days")
    critical_deadlines: List[Dict[str, Any]] = Field(..., description="Critical deadlines")
    
    # Quantum analysis
    quantum_confidence: float = Field(..., ge=0.0, le=1.0, description="Quantum analysis confidence")
    quantum_explanation: Dict[str, Any] = Field(..., description="Quantum analysis explanation")
    
    detailed_response: str = Field(..., description="Complete formatted legal response")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "gst_1234567890",
                "success_probability": 0.75,
                "legal_position_strength": {
                    "statutory_compliance": 0.8,
                    "precedent_support": 0.7,
                    "documentation_quality": 0.9
                },
                "applicable_provisions": ["Section 16 CGST Act", "Rule 36 CGST Rules"],
                "estimated_liability": {"principal_tax": 500000.0, "total_liability": 500000.0},
                "penalty_assessment": {"penalty_rate": 1.0, "penalty_amount": 500000.0},
                "limitation_compliance": True,
                "forum_jurisdiction": "appellate_authority",
                "litigation_risk": 0.25,
                "estimated_resolution_time": 180,
                "quantum_confidence": 0.82,
                "processing_time": 3.21
            }
        }


# Legal Aid models
class LegalAidTypeEnum(str, Enum):
    """Legal aid types for API."""
    FREE_LEGAL_ADVICE = "free_legal_advice"
    LEGAL_REPRESENTATION = "legal_representation"
    DOCUMENT_DRAFTING = "document_drafting"
    MEDIATION_SERVICES = "mediation_services"
    LEGAL_LITERACY = "legal_literacy"
    PARALEGAL_SERVICES = "paralegal_services"
    COURT_FEE_WAIVER = "court_fee_waiver"
    BAIL_ASSISTANCE = "bail_assistance"


class CaseCategoryEnum(str, Enum):
    """Legal case categories for API."""
    CRIMINAL = "criminal"
    CIVIL = "civil"
    FAMILY = "family"
    LABOR = "labor"
    CONSUMER = "consumer"
    CONSTITUTIONAL = "constitutional"
    ENVIRONMENTAL = "environmental"
    HUMAN_RIGHTS = "human_rights"
    WOMEN_RIGHTS = "women_rights"
    CHILD_RIGHTS = "child_rights"


class LegalAidAuthorityEnum(str, Enum):
    """Legal aid authorities for API."""
    NALSA = "nalsa"
    SLSA = "slsa"
    DLSA = "dlsa"
    TLSC = "tlsc"
    HCLSC = "hclsc"
    SCLSC = "sclsc"


class LegalAidApplicantModel(BaseModel):
    """Legal aid applicant model."""
    name: str = Field(..., min_length=2, max_length=100)
    age: int = Field(..., ge=18, le=100)
    gender: str = Field(..., max_length=20)
    address: str = Field(..., min_length=10, max_length=500)
    phone: Optional[str] = Field(None, max_length=15)
    email: Optional[EmailStr] = None
    
    # Socio-economic details
    annual_income: float = Field(..., ge=0, description="Annual income in INR")
    family_size: int = Field(..., ge=1, le=20)
    occupation: Optional[str] = Field(None, max_length=100)
    education_level: Optional[str] = Field(None, max_length=50)
    
    # Eligibility factors
    social_category: Optional[str] = Field(None, max_length=20, description="SC/ST/OBC/General")
    is_disabled: bool = False
    disability_type: Optional[str] = Field(None, max_length=100)
    is_senior_citizen: bool = False
    is_single_woman: bool = False
    is_victim_of_crime: bool = False
    
    # Financial details
    assets_value: Optional[float] = Field(None, ge=0)
    monthly_expenses: Optional[float] = Field(None, ge=0)
    dependents: int = Field(0, ge=0, le=20)
    
    # Supporting documents
    income_certificate: bool = False
    caste_certificate: bool = False
    disability_certificate: bool = False
    identity_proof: bool = False
    address_proof: bool = False


class LegalAidRequest(BaseModel):
    """Legal aid assessment request."""
    case_id: str = Field(..., min_length=1, max_length=50)
    applicant: LegalAidApplicantModel
    case_category: CaseCategoryEnum
    legal_aid_type: LegalAidTypeEnum
    case_description: str = Field(..., min_length=10, max_length=2000)
    urgency_level: str = Field(..., regex="^(high|medium|low)$")
    
    # Case specifics
    opposing_party: Optional[str] = Field(None, max_length=100)
    court_involved: Optional[str] = Field(None, max_length=100)
    case_number: Optional[str] = Field(None, max_length=50)
    case_stage: Optional[str] = Field(None, max_length=50)
    
    # Legal requirements
    lawyer_required: bool = False
    court_representation: bool = False
    document_assistance: bool = False
    legal_advice_only: bool = False
    
    # Timeline
    application_date: Optional[date] = None
    required_by_date: Optional[date] = None
    
    # Previous legal aid
    previous_aid_received: bool = False
    previous_aid_details: Optional[str] = Field(None, max_length=500)
    
    # Special circumstances
    emergency_case: bool = False
    pro_bono_eligible: bool = False
    media_attention: bool = False
    
    additional_context: Optional[str] = Field(None, max_length=2000)
    
    class Config:
        schema_extra = {
            "example": {
                "case_id": "LA-001-2024",
                "applicant": {
                    "name": "Jane Doe",
                    "age": 35,
                    "gender": "Female",
                    "address": "123 Main Street, Delhi",
                    "annual_income": 150000.0,
                    "family_size": 4,
                    "social_category": "SC",
                    "is_single_woman": True,
                    "income_certificate": True,
                    "identity_proof": True
                },
                "case_category": "family",
                "legal_aid_type": "legal_representation",
                "case_description": "Domestic violence case requiring legal representation",
                "urgency_level": "high",
                "lawyer_required": True,
                "court_representation": True,
                "emergency_case": True
            }
        }


class LegalAidAssessmentResponse(BaseModel):
    """Legal aid assessment response."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    eligibility_score: float = Field(..., ge=0.0, le=1.0, description="Eligibility score")
    
    # Eligibility analysis
    income_eligibility: bool = Field(..., description="Income eligibility status")
    category_eligibility: bool = Field(..., description="Category eligibility status")
    case_merit: float = Field(..., ge=0.0, le=1.0, description="Case merit score")
    urgency_assessment: float = Field(..., ge=0.0, le=1.0, description="Urgency assessment")
    
    # Recommended aid
    recommended_aid_type: LegalAidTypeEnum = Field(..., description="Recommended aid type")
    recommended_authority: LegalAidAuthorityEnum = Field(..., description="Recommended authority")
    estimated_cost: float = Field(..., ge=0.0, description="Estimated cost")
    
    # Resource allocation
    lawyer_assignment: Dict[str, Any] = Field(..., description="Lawyer assignment details")
    priority_level: int = Field(..., ge=1, le=5, description="Priority level (1-5, 1 highest)")
    estimated_duration: int = Field(..., ge=0, description="Estimated duration in days")
    
    # Requirements
    additional_documents: List[str] = Field(..., description="Additional documents required")
    verification_needed: List[str] = Field(..., description="Verification requirements")
    conditions: List[str] = Field(..., description="Conditions for aid")
    
    # Financial analysis
    fee_waiver_eligible: bool = Field(..., description="Fee waiver eligibility")
    court_fee_exemption: float = Field(..., ge=0.0, description="Court fee exemption amount")
    legal_service_cost: float = Field(..., ge=0.0, description="Legal service cost")
    
    # Success factors
    case_strength: float = Field(..., ge=0.0, le=1.0, description="Case strength assessment")
    likelihood_of_success: float = Field(..., ge=0.0, le=1.0, description="Likelihood of success")
    social_impact: float = Field(..., ge=0.0, le=1.0, description="Social impact score")
    
    # Quantum analysis
    quantum_confidence: float = Field(..., ge=0.0, le=1.0, description="Quantum analysis confidence")
    quantum_explanation: Dict[str, Any] = Field(..., description="Quantum analysis explanation")
    
    # Recommendations
    next_steps: List[str] = Field(..., description="Recommended next steps")
    alternative_options: List[str] = Field(..., description="Alternative options")
    referral_suggestions: List[str] = Field(..., description="Referral suggestions")
    
    detailed_response: str = Field(..., description="Complete formatted legal response")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "la_1234567890",
                "eligibility_score": 0.85,
                "income_eligibility": True,
                "category_eligibility": True,
                "case_merit": 0.8,
                "urgency_assessment": 0.9,
                "recommended_aid_type": "legal_representation",
                "recommended_authority": "dlsa",
                "estimated_cost": 15000.0,
                "lawyer_assignment": {
                    "assigned": True,
                    "lawyer_id": "LAW001",
                    "specialization": "family"
                },
                "priority_level": 1,
                "estimated_duration": 90,
                "fee_waiver_eligible": True,
                "court_fee_exemption": 5000.0,
                "case_strength": 0.75,
                "likelihood_of_success": 0.8,
                "social_impact": 0.9,
                "quantum_confidence": 0.87,
                "processing_time": 2.45
            }
        }