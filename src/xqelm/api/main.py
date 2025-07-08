"""
XQELM FastAPI Application

Main FastAPI application for the Explainable Quantum-Enhanced
Language Models for Legal Reasoning system.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import traceback

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from loguru import logger
from contextlib import asynccontextmanager

from ..core.quantum_legal_model import QuantumLegalModel
from ..classical.preprocessor import LegalTextPreprocessor
from ..classical.response_generator import LegalResponseGenerator
from ..classical.knowledge_base import LegalKnowledgeBase
from ..use_cases.bail_application import BailApplicationManager, BailApplicationData
from ..use_cases.cheque_bounce import ChequeBounceManager, ChequeBounceCase
from ..use_cases.property_dispute import PropertyDisputeManager, PropertyDisputeCase
from ..use_cases.motor_vehicle_claims import MotorVehicleClaimManager, MotorVehicleClaimCase
from ..use_cases.consumer_dispute import (
    ConsumerDisputeManager, ConsumerDisputeCase, ConsumerDetails,
    OppositePartyDetails, ProductDetails, ServiceDetails
)
from ..use_cases.gst_dispute import GSTDisputeManager, GSTDisputeCase
from ..use_cases.legal_aid import LegalAidManager, LegalAidCase
from ..utils.config import get_config
from .models import *
from .auth import get_current_user, create_access_token
from .rate_limiter import RateLimiter


# Global application state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting XQELM application...")
    
    try:
        config = get_config()
        
        # Initialize core components
        logger.info("Initializing quantum legal model...")
        quantum_model = QuantumLegalModel(
            n_qubits=config.quantum.n_qubits,
            n_layers=config.quantum.n_layers
        )
        
        logger.info("Initializing text preprocessor...")
        preprocessor = LegalTextPreprocessor(
            language=config.legal.supported_languages[0],
            enable_ner=True,
            enable_citation_extraction=True,
            enable_concept_extraction=True
        )
        
        logger.info("Initializing response generator...")
        response_generator = LegalResponseGenerator(
            language=config.legal.supported_languages[0],
            jurisdiction=config.legal.supported_jurisdictions[0]
        )
        
        logger.info("Initializing knowledge base...")
        knowledge_base = LegalKnowledgeBase(
            db_path="data/legal_knowledge.db",
            jurisdiction=config.legal.supported_jurisdictions[0]
        )
        
        # Initialize use case managers
        logger.info("Initializing use case managers...")
        bail_manager = BailApplicationManager(
            quantum_model, preprocessor, response_generator, knowledge_base
        )
        
        cheque_bounce_manager = ChequeBounceManager(
            quantum_model, preprocessor, response_generator, knowledge_base
        )
        
        property_dispute_manager = PropertyDisputeManager(
            quantum_model, preprocessor, response_generator, knowledge_base
        )
        
        motor_vehicle_claim_manager = MotorVehicleClaimManager(
            quantum_model, preprocessor, response_generator, knowledge_base
        )
        
        consumer_dispute_manager = ConsumerDisputeManager(
            quantum_model, preprocessor, response_generator, knowledge_base
        )
        
        gst_dispute_manager = GSTDisputeManager()
        
        legal_aid_manager = LegalAidManager()
        
        # Store in app state
        app_state.update({
            "config": config,
            "quantum_model": quantum_model,
            "preprocessor": preprocessor,
            "response_generator": response_generator,
            "knowledge_base": knowledge_base,
            "bail_manager": bail_manager,
            "cheque_bounce_manager": cheque_bounce_manager,
            "property_dispute_manager": property_dispute_manager,
            "motor_vehicle_claim_manager": motor_vehicle_claim_manager,
            "consumer_dispute_manager": consumer_dispute_manager,
            "gst_dispute_manager": gst_dispute_manager,
            "legal_aid_manager": legal_aid_manager,
            "rate_limiter": RateLimiter(
                requests_per_minute=config.api.rate_limit_requests,
                window_seconds=config.api.rate_limit_window
            )
        })
        
        logger.info("XQELM application started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down XQELM application...")
    
    try:
        # Close knowledge base
        if "knowledge_base" in app_state:
            await app_state["knowledge_base"].close()
        
        logger.info("XQELM application shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    config = get_config()
    
    app = FastAPI(
        title="XQELM - Explainable Quantum-Enhanced Language Models for Legal Reasoning",
        description="Quantum-enhanced AI system for legal analysis and reasoning",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    if config.security.allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=config.security.allowed_hosts
        )
    
    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    return app


# Create app instance
app = create_app()

# Security
security = HTTPBearer()


# Dependency functions
async def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance."""
    return app_state["rate_limiter"]


async def check_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    """Check rate limit for request."""
    client_ip = request.client.host
    if not await rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with component status."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check quantum model
        if "quantum_model" in app_state:
            health_status["components"]["quantum_model"] = "healthy"
        else:
            health_status["components"]["quantum_model"] = "unavailable"
            health_status["status"] = "degraded"
        
        # Check knowledge base
        if "knowledge_base" in app_state:
            kb_stats = app_state["knowledge_base"].get_statistics()
            health_status["components"]["knowledge_base"] = {
                "status": "healthy",
                "documents": kb_stats.get("total_documents", 0)
            }
        else:
            health_status["components"]["knowledge_base"] = "unavailable"
            health_status["status"] = "degraded"
        
        # Check use case managers
        manager_names = [
            "bail_manager", "cheque_bounce_manager", "property_dispute_manager",
            "motor_vehicle_claim_manager", "consumer_dispute_manager"
        ]
        for manager_name in manager_names:
            if manager_name in app_state:
                health_status["components"][manager_name] = "healthy"
            else:
                health_status["components"][manager_name] = "unavailable"
                health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# Authentication endpoints
@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def login(credentials: LoginRequest):
    """Authenticate user and return access token."""
    # Simple authentication - in production, use proper user management
    if credentials.username == "demo" and credentials.password == "demo123":
        access_token = create_access_token(data={"sub": credentials.username})
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=1800  # 30 minutes
        )
    else:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )


# General query endpoint
@app.post("/query", response_model=QueryResponse, tags=["Legal Analysis"])
async def process_legal_query(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit),
    current_user: str = Depends(get_current_user)
):
    """Process general legal query using quantum-enhanced reasoning."""
    try:
        logger.info(f"Processing query from user: {current_user}")
        
        # Get components
        quantum_model = app_state["quantum_model"]
        preprocessor = app_state["preprocessor"]
        response_generator = app_state["response_generator"]
        
        # Preprocess query
        processed_query = await preprocessor.preprocess_text(
            query_request.query,
            document_type="general_query",
            metadata={"user": current_user}
        )
        
        # Process with quantum model
        quantum_results = await quantum_model.process_query(
            query=query_request.query,
            context=query_request.context or {},
            use_case=query_request.use_case or "general"
        )
        
        # Generate response
        response = await response_generator.generate_response(
            query=query_request.query,
            quantum_results=quantum_results,
            response_type=query_request.response_type,
            metadata={"user": current_user}
        )
        
        # Log query for analytics (background task)
        background_tasks.add_task(
            log_query_analytics,
            query_request.query,
            current_user,
            response.metadata.confidence_level.value
        )
        
        return QueryResponse(
            query_id=response.metadata.response_id,
            response=await response_generator.render_response(response),
            confidence=response.quantum_confidence or 0.0,
            processing_time=response.metadata.processing_time,
            metadata={
                "quantum_coherence": response.quantum_coherence,
                "citations_count": len(response.citations),
                "recommendations_count": len(response.recommendations)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


# Bail application endpoints
@app.post("/bail/analyze", response_model=BailAnalysisResponse, tags=["Bail Applications"])
async def analyze_bail_application(
    bail_request: BailApplicationRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit),
    current_user: str = Depends(get_current_user)
):
    """Analyze bail application using quantum-enhanced reasoning."""
    try:
        logger.info(f"Analyzing bail application for user: {current_user}")
        
        bail_manager = app_state["bail_manager"]
        
        # Convert request to internal format
        application_data = BailApplicationData(
            applicant_name=bail_request.applicant_name,
            case_number=bail_request.case_number,
            court=bail_request.court,
            offense_details=bail_request.offense_details,
            sections_charged=bail_request.sections_charged,
            offense_category=bail_request.offense_category,
            bail_type=bail_request.bail_type,
            age=bail_request.age,
            occupation=bail_request.occupation,
            address=bail_request.address,
            family_details=bail_request.family_details,
            previous_convictions=bail_request.previous_convictions or [],
            date_of_offense=bail_request.date_of_offense,
            date_of_arrest=bail_request.date_of_arrest,
            investigation_status=bail_request.investigation_status,
            evidence_status=bail_request.evidence_status,
            supporting_documents=bail_request.supporting_documents or [],
            character_witnesses=bail_request.character_witnesses or [],
            medical_condition=bail_request.medical_condition,
            employment_verification=bail_request.employment_verification,
            property_details=bail_request.property_details
        )
        
        # Analyze application
        recommendation = await bail_manager.analyze_bail_application(
            application_data,
            bail_request.additional_context
        )
        
        # Generate response
        legal_response = await bail_manager.generate_bail_response(
            application_data,
            recommendation,
            bail_request.additional_context
        )
        
        # Log analysis (background task)
        background_tasks.add_task(
            log_bail_analysis,
            bail_request.applicant_name,
            current_user,
            recommendation.recommendation,
            recommendation.confidence
        )
        
        return BailAnalysisResponse(
            analysis_id=legal_response.metadata.response_id,
            recommendation=recommendation.recommendation,
            confidence=recommendation.confidence,
            reasoning=recommendation.reasoning,
            conditions=recommendation.conditions,
            risk_assessment=recommendation.risk_assessment,
            precedents=recommendation.precedents,
            statutory_basis=recommendation.statutory_basis,
            detailed_response=await bail_manager.response_generator.render_response(legal_response),
            processing_time=legal_response.metadata.processing_time,
            quantum_metrics={
                "coherence": recommendation.quantum_analysis.get("quantum_coherence", 0.0),
                "entanglement": recommendation.quantum_analysis.get("entanglement_measures", {}),
                "bail_probability": recommendation.quantum_analysis.get("bail_probability", 0.0)
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing bail application: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze bail application: {str(e)}"
        )


# Cheque bounce endpoints
@app.post("/cheque-bounce/analyze", response_model=ChequeBounceAnalysisResponse, tags=["Cheque Bounce"])
async def analyze_cheque_bounce_case(
    cheque_request: ChequeBounceRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit),
    current_user: str = Depends(get_current_user)
):
    """Analyze cheque bounce case using quantum-enhanced reasoning."""
    try:
        logger.info(f"Analyzing cheque bounce case for user: {current_user}")
        
        cheque_manager = app_state["cheque_bounce_manager"]
        
        # Convert request to internal format
        case_data = ChequeBounceCase(
            case_id=cheque_request.case_id,
            cheque_details=cheque_request.cheque_details,
            legal_notice=cheque_request.legal_notice,
            complainant_name=cheque_request.complainant_name,
            complainant_address=cheque_request.complainant_address,
            accused_name=cheque_request.accused_name,
            accused_address=cheque_request.accused_address,
            case_stage=cheque_request.case_stage,
            court=cheque_request.court,
            case_number=cheque_request.case_number,
            filing_date=cheque_request.filing_date,
            supporting_documents=cheque_request.supporting_documents or [],
            witness_details=cheque_request.witness_details or [],
            interest_claimed=cheque_request.interest_claimed,
            compensation_claimed=cheque_request.compensation_claimed,
            previous_transactions=cheque_request.previous_transactions or [],
            relationship_between_parties=cheque_request.relationship_between_parties
        )
        
        # Analyze case
        analysis = await cheque_manager.analyze_cheque_bounce_case(
            case_data,
            cheque_request.additional_context
        )
        
        # Generate response
        legal_response = await cheque_manager.generate_cheque_bounce_response(
            case_data,
            analysis,
            cheque_request.additional_context
        )
        
        # Log analysis (background task)
        background_tasks.add_task(
            log_cheque_bounce_analysis,
            cheque_request.case_id,
            current_user,
            analysis.liability_assessment.value,
            analysis.conviction_probability
        )
        
        return ChequeBounceAnalysisResponse(
            analysis_id=legal_response.metadata.response_id,
            liability_assessment=analysis.liability_assessment.value,
            conviction_probability=analysis.conviction_probability,
            compensation_estimate=analysis.compensation_estimate,
            statutory_compliance=analysis.statutory_compliance,
            available_defenses=analysis.available_defenses,
            defense_strength=analysis.defense_strength,
            recommendations=analysis.recommendations,
            next_steps=analysis.next_steps,
            similar_cases=analysis.similar_cases,
            detailed_response=await cheque_manager.response_generator.render_response(legal_response),
            processing_time=legal_response.metadata.processing_time,
            quantum_metrics={
                "confidence": analysis.quantum_confidence,
                "precedent_alignment": analysis.precedent_alignment,
                "factors": analysis.quantum_factors
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing cheque bounce case: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze cheque bounce case: {str(e)}"
        )


# Property dispute endpoints
@app.post("/property-dispute/analyze", response_model=PropertyDisputeAnalysisResponse, tags=["Property Disputes"])
async def analyze_property_dispute(
    property_request: PropertyDisputeRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit),
    current_user: str = Depends(get_current_user)
):
    """Analyze property dispute case using quantum-enhanced reasoning."""
    try:
        logger.info(f"Analyzing property dispute case for user: {current_user}")
        
        property_manager = app_state["property_dispute_manager"]
        
        # Convert request to internal format
        case_data = PropertyDisputeCase(
            case_id=property_request.case_id,
            dispute_type=property_request.dispute_type,
            property_type=property_request.property_type,
            property_address=property_request.property_address,
            property_area=property_request.property_area,
            property_value=property_request.property_value,
            survey_number=property_request.survey_number,
            plaintiff_name=property_request.plaintiff_name,
            plaintiff_address=property_request.plaintiff_address,
            defendant_name=property_request.defendant_name,
            defendant_address=property_request.defendant_address,
            title_documents=property_request.title_documents,
            registration_details=property_request.registration_details,
            mutation_records=property_request.mutation_records,
            dispute_description=property_request.dispute_description,
            possession_status=property_request.possession_status,
            disputed_area=property_request.disputed_area,
            case_stage=property_request.case_stage,
            court=property_request.court,
            case_number=property_request.case_number,
            filing_date=property_request.filing_date,
            possession_sought=property_request.possession_sought,
            damages_claimed=property_request.damages_claimed,
            injunction_sought=property_request.injunction_sought,
            declaration_sought=property_request.declaration_sought,
            previous_litigation=property_request.previous_litigation or [],
            survey_reports=property_request.survey_reports or []
        )
        
        # Analyze case
        analysis = await property_manager.analyze_property_dispute(
            case_data,
            property_request.additional_context
        )
        
        # Generate response
        legal_response = await property_manager.generate_property_response(
            case_data,
            analysis
        )
        
        # Log analysis (background task)
        background_tasks.add_task(
            log_property_dispute_analysis,
            property_request.case_id,
            current_user,
            analysis.success_probability
        )
        
        return PropertyDisputeAnalysisResponse(
            analysis_id=legal_response.metadata.response_id,
            title_strength=analysis.title_strength,
            possession_rights=analysis.possession_rights,
            success_probability=analysis.success_probability,
            compensation_estimate=analysis.compensation_estimate,
            recommendations=analysis.recommendations,
            evidence_gaps=analysis.evidence_gaps,
            settlement_options=analysis.settlement_options,
            case_duration_estimate=analysis.case_duration_estimate,
            litigation_cost_estimate=analysis.litigation_cost_estimate,
            detailed_response=await property_manager.response_generator.render_response(legal_response),
            processing_time=legal_response.metadata.processing_time,
            quantum_metrics={
                "confidence": analysis.quantum_confidence,
                "precedent_alignment": analysis.precedent_alignment,
                "factors": analysis.quantum_factors
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing property dispute: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze property dispute: {str(e)}"
        )


# Motor vehicle claim endpoints
@app.post("/motor-vehicle-claim/analyze", response_model=MotorVehicleClaimAnalysisResponse, tags=["Motor Vehicle Claims"])
async def analyze_motor_vehicle_claim(
    mvc_request: MotorVehicleClaimRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit),
    current_user: str = Depends(get_current_user)
):
    """Analyze motor vehicle claim using quantum-enhanced reasoning."""
    try:
        logger.info(f"Analyzing motor vehicle claim for user: {current_user}")
        
        mvc_manager = app_state["motor_vehicle_claim_manager"]
        
        # Convert request to internal format
        case_data = MotorVehicleClaimCase(
            case_id=mvc_request.case_id,
            accident_type=mvc_request.accident_type,
            accident_date=mvc_request.accident_date,
            accident_location=mvc_request.accident_location,
            accident_description=mvc_request.accident_description,
            police_complaint_number=mvc_request.police_complaint_number,
            claimant_vehicle_type=mvc_request.claimant_vehicle_type,
            claimant_vehicle_number=mvc_request.claimant_vehicle_number,
            opposite_vehicle_type=mvc_request.opposite_vehicle_type,
            opposite_vehicle_number=mvc_request.opposite_vehicle_number,
            claimant_name=mvc_request.claimant_name,
            claimant_address=mvc_request.claimant_address,
            claimant_age=mvc_request.claimant_age,
            opposite_party_name=mvc_request.opposite_party_name,
            opposite_party_address=mvc_request.opposite_party_address,
            injury_type=mvc_request.injury_type,
            injury_description=mvc_request.injury_description,
            medical_expenses=mvc_request.medical_expenses,
            vehicle_damage_cost=mvc_request.vehicle_damage_cost,
            claimant_monthly_income=mvc_request.claimant_monthly_income,
            claimant_occupation=mvc_request.claimant_occupation,
            loss_of_earning_period=mvc_request.loss_of_earning_period,
            claimant_insurance_policy=mvc_request.claimant_insurance_policy,
            opposite_party_insurance=mvc_request.opposite_party_insurance,
            insurance_claim_amount=mvc_request.insurance_claim_amount,
            case_stage=mvc_request.case_stage,
            tribunal=mvc_request.tribunal,
            case_number=mvc_request.case_number,
            filing_date=mvc_request.filing_date,
            supporting_documents=mvc_request.supporting_documents or [],
            witness_details=mvc_request.witness_details or [],
            fault_percentage=mvc_request.fault_percentage or {}
        )
        
        # Analyze case
        analysis = await mvc_manager.analyze_motor_vehicle_claim(
            case_data,
            mvc_request.additional_context
        )
        
        # Generate response
        legal_response = await mvc_manager.generate_mvc_response(
            case_data,
            analysis
        )
        
        # Log analysis (background task)
        background_tasks.add_task(
            log_motor_vehicle_claim_analysis,
            mvc_request.case_id,
            current_user,
            analysis.total_compensation,
            analysis.success_probability
        )
        
        return MotorVehicleClaimAnalysisResponse(
            analysis_id=legal_response.metadata.response_id,
            liability_assessment=analysis.liability_assessment,
            compensation_estimate=analysis.compensation_estimate,
            total_compensation=analysis.total_compensation,
            success_probability=analysis.success_probability,
            insurance_coverage=analysis.insurance_coverage,
            recommendations=analysis.recommendations,
            evidence_gaps=analysis.evidence_gaps,
            settlement_estimate=analysis.settlement_estimate,
            case_duration_estimate=analysis.case_duration_estimate,
            detailed_response=await mvc_manager.response_generator.render_response(legal_response),
            processing_time=legal_response.metadata.processing_time,
            quantum_metrics={
                "confidence": analysis.quantum_confidence,
                "precedent_alignment": analysis.precedent_alignment,
                "factors": analysis.quantum_factors
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing motor vehicle claim: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze motor vehicle claim: {str(e)}"
        )


# Consumer dispute endpoints
@app.post("/consumer-dispute/analyze", response_model=ConsumerDisputeAnalysisResponse, tags=["Consumer Disputes"])
async def analyze_consumer_dispute(
    consumer_request: ConsumerDisputeRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit),
    current_user: str = Depends(get_current_user)
):
    """Analyze consumer dispute case using quantum-enhanced reasoning."""
    try:
        logger.info(f"Analyzing consumer dispute case for user: {current_user}")
        
        consumer_manager = app_state["consumer_dispute_manager"]
        
        # Convert request to internal format
        case_data = ConsumerDisputeCase(
            case_id=consumer_request.case_id,
            complaint_type=consumer_request.complaint_type,
            consumer_details=ConsumerDetails(
                name=consumer_request.consumer_name,
                address=consumer_request.consumer_address,
                phone=consumer_request.consumer_phone,
                email=consumer_request.consumer_email,
                is_senior_citizen=consumer_request.is_senior_citizen,
                financial_loss=consumer_request.financial_loss,
                supporting_documents=consumer_request.supporting_documents or []
            ),
            opposite_party=OppositePartyDetails(
                name=consumer_request.opposite_party_name,
                business_type=consumer_request.business_type,
                address=consumer_request.opposite_party_address
            ),
            case_stage=consumer_request.case_stage,
            forum_type=consumer_request.forum_type,
            case_number=consumer_request.case_number,
            filing_date=consumer_request.filing_date,
            compensation_claimed=consumer_request.compensation_claimed,
            replacement_sought=consumer_request.replacement_sought,
            refund_sought=consumer_request.refund_sought,
            service_improvement_sought=consumer_request.service_improvement_sought,
            mediation_attempted=consumer_request.mediation_attempted,
            previous_complaints=consumer_request.previous_complaints or []
        )
        
        # Add product or service details
        if consumer_request.product_name:
            case_data.product_details = ProductDetails(
                product_name=consumer_request.product_name,
                brand="",  # Would need to be added to request model
                purchase_date=consumer_request.purchase_date,
                purchase_price=consumer_request.purchase_amount,
                seller_name=consumer_request.opposite_party_name,
                defect_description=consumer_request.deficiency_description or ""
            )
        elif consumer_request.service_type:
            case_data.service_details = ServiceDetails(
                service_type=consumer_request.service_type,
                service_provider=consumer_request.opposite_party_name,
                service_description=consumer_request.complaint_description,
                service_date=consumer_request.purchase_date,
                service_amount=consumer_request.purchase_amount,
                deficiency_description=consumer_request.deficiency_description or "",
                expected_service="",  # Would need to be added to request model
                actual_service="",    # Would need to be added to request model
                complaint_history=consumer_request.correspondence_history or []
            )
        
        # Analyze case
        analysis = await consumer_manager.analyze_consumer_dispute(
            case_data,
            consumer_request.additional_context
        )
        
        # Generate response
        legal_response = await consumer_manager.generate_consumer_response(
            case_data,
            analysis
        )
        
        # Log analysis (background task)
        background_tasks.add_task(
            log_consumer_dispute_analysis,
            consumer_request.case_id,
            current_user,
            analysis.case_merit,
            analysis.success_probability
        )
        
        return ConsumerDisputeAnalysisResponse(
            analysis_id=legal_response.metadata.response_id,
            case_merit=analysis.case_merit,
            success_probability=analysis.success_probability,
            jurisdiction_validity=analysis.jurisdiction_validity,
            limitation_compliance=analysis.limitation_compliance,
            evidence_strength=analysis.evidence_strength,
            compensation_estimate=analysis.compensation_estimate,
            relief_likelihood=analysis.relief_likelihood,
            recommendations=analysis.recommendations,
            evidence_gaps=analysis.evidence_gaps,
            settlement_options=analysis.settlement_options,
            case_duration_estimate=analysis.case_duration_estimate,
            litigation_cost_estimate=analysis.litigation_cost_estimate,
            detailed_response=await consumer_manager.response_generator.render_response(legal_response),
            processing_time=legal_response.metadata.processing_time,
            quantum_metrics={
                "confidence": analysis.quantum_confidence,
                "precedent_alignment": analysis.precedent_alignment,
                "factors": analysis.quantum_factors
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing consumer dispute: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze consumer dispute: {str(e)}"
        )


# Knowledge base endpoints
@app.get("/knowledge/search", response_model=KnowledgeSearchResponse, tags=["Knowledge Base"])
async def search_knowledge_base(
    query: str,
    document_types: Optional[List[str]] = None,
    jurisdiction: Optional[str] = None,
    limit: int = 10,
    _: None = Depends(check_rate_limit),
    current_user: str = Depends(get_current_user)
):
    """Search the legal knowledge base."""
    try:
        knowledge_base = app_state["knowledge_base"]
        
        # Convert document types
        doc_types = None
        if document_types:
            from ..classical.knowledge_base import DocumentType
            doc_types = [DocumentType(dt) for dt in document_types if dt in [e.value for e in DocumentType]]
        
        # Search knowledge base
        results = await knowledge_base.search_documents(
            query=query,
            document_types=doc_types,
            jurisdiction=jurisdiction,
            limit=limit
        )
        
        # Convert results
        search_results = []
        for result in results:
            search_results.append({
                "document_id": result.document.id,
                "title": result.document.title,
                "document_type": result.document.document_type.value,
                "citation": result.document.citation,
                "summary": result.document.summary,
                "similarity_score": result.similarity_score,
                "relevance_score": result.relevance_score,
                "match_type": result.match_type,
                "matched_terms": result.matched_terms
            })
        
        return KnowledgeSearchResponse(
            query=query,
            results=search_results,
            total_results=len(search_results),
            processing_time=0.0  # Would be calculated in real implementation
        )
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search knowledge base: {str(e)}"
        )


# Statistics endpoints
@app.get("/stats/system", tags=["Statistics"])
async def get_system_statistics(
    current_user: str = Depends(get_current_user)
):
    """Get system statistics."""
    try:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Quantum model stats
        if "quantum_model" in app_state:
            quantum_stats = app_state["quantum_model"].get_statistics()
            stats["components"]["quantum_model"] = quantum_stats
        
        # Knowledge base stats
        if "knowledge_base" in app_state:
            kb_stats = app_state["knowledge_base"].get_statistics()
            stats["components"]["knowledge_base"] = kb_stats
        
        # Use case manager stats
        if "bail_manager" in app_state:
            bail_stats = app_state["bail_manager"].get_statistics()
            stats["components"]["bail_manager"] = bail_stats
        
        if "cheque_bounce_manager" in app_state:
            cheque_stats = app_state["cheque_bounce_manager"].get_statistics()
            stats["components"]["cheque_bounce_manager"] = cheque_stats
        
        # Response generator stats
        if "response_generator" in app_state:
            response_stats = app_state["response_generator"].get_statistics()
            stats["components"]["response_generator"] = response_stats
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system statistics: {str(e)}"
        )


# Background task functions
async def log_query_analytics(query: str, user: str, confidence: str):
    """Log query analytics (background task)."""
    try:
        # In production, this would log to analytics database
        logger.info(f"Query analytics: user={user}, confidence={confidence}, query_length={len(query)}")
    except Exception as e:
        logger.error(f"Failed to log query analytics: {e}")


async def log_bail_analysis(applicant: str, user: str, recommendation: str, confidence: float):
    """Log bail analysis (background task)."""
    try:
        logger.info(f"Bail analysis: user={user}, applicant={applicant}, recommendation={recommendation}, confidence={confidence}")
    except Exception as e:
        logger.error(f"Failed to log bail analysis: {e}")


async def log_cheque_bounce_analysis(case_id: str, user: str, liability: str, conviction_prob: float):
    """Log cheque bounce analysis (background task)."""
    try:
        logger.info(f"Cheque bounce analysis: user={user}, case_id={case_id}, liability={liability}, conviction_prob={conviction_prob}")
    except Exception as e:
        logger.error(f"Failed to log cheque bounce analysis: {e}")


async def log_property_dispute_analysis(case_id: str, user: str, success_probability: float):
    """Log property dispute analysis (background task)."""
    try:
        logger.info(f"Property dispute analysis: user={user}, case_id={case_id}, success_probability={success_probability}")
    except Exception as e:
        logger.error(f"Failed to log property dispute analysis: {e}")


async def log_motor_vehicle_claim_analysis(case_id: str, user: str, compensation: float, success_probability: float):
    """Log motor vehicle claim analysis (background task)."""
    try:
        logger.info(f"Motor vehicle claim analysis: user={user}, case_id={case_id}, compensation={compensation}, success_probability={success_probability}")
    except Exception as e:
        logger.error(f"Failed to log motor vehicle claim analysis: {e}")


async def log_consumer_dispute_analysis(case_id: str, user: str, case_merit: float, success_probability: float):
    """Log consumer dispute analysis (background task)."""
    try:
        logger.info(f"Consumer dispute analysis: user={user}, case_id={case_id}, case_merit={case_merit}, success_probability={success_probability}")
    except Exception as e:
        logger.error(f"Failed to log consumer dispute analysis: {e}")



# GST dispute endpoints
@app.post("/gst-dispute/analyze", response_model=GSTDisputeAnalysisResponse, tags=["GST Disputes"])
async def analyze_gst_dispute(
    gst_request: GSTDisputeRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit),
    current_user: str = Depends(get_current_user)
):
    """Analyze a GST dispute case and provide recommendations."""
    try:
        gst_manager = app_state["gst_dispute_manager"]
        
        # Convert request to case data
        from ..use_cases.gst_dispute import GSTDisputeCase, GSTDisputeType, BusinessType
        from datetime import datetime
        
        case_data = GSTDisputeCase(
            case_id=gst_request.case_id,
            taxpayer_name=gst_request.taxpayer_name,
            gstin=gst_request.gstin,
            dispute_type=GSTDisputeType(gst_request.dispute_type.value),
            business_type=BusinessType(gst_request.business_type.value),
            dispute_description=gst_request.dispute_description,
            disputed_amount=gst_request.disputed_amount,
            tax_period=gst_request.tax_period,
            notice_date=gst_request.notice_date,
            response_deadline=gst_request.response_deadline,
            transaction_value=gst_request.transaction_value,
            evidence_documents=gst_request.evidence_documents or [],
            invoices_provided=gst_request.invoices_provided,
            books_of_accounts=gst_request.books_of_accounts,
            contracts_agreements=gst_request.contracts_agreements,
            show_cause_notice=gst_request.show_cause_notice,
            personal_hearing_attended=gst_request.personal_hearing_attended,
            written_submissions=gst_request.written_submissions,
            has_legal_counsel=gst_request.has_legal_counsel,
            authorized_representative=gst_request.authorized_representative,
            similar_cases_precedent=gst_request.similar_cases_precedent or [],
            department_position=gst_request.department_position,
            taxpayer_position=gst_request.taxpayer_position
        )
        
        # Analyze case
        analysis = gst_manager.analyze_gst_dispute(case_data)
        
        # Log analysis (background task)
        background_tasks.add_task(
            log_gst_dispute_analysis,
            gst_request.case_id,
            current_user,
            analysis.success_probability,
            analysis.litigation_risk
        )
        
        return GSTDisputeAnalysisResponse(
            analysis_id=analysis.case_id + "_analysis",
            success_probability=analysis.success_probability,
            legal_position_strength=analysis.legal_position_strength,
            applicable_provisions=analysis.applicable_provisions,
            relevant_precedents=analysis.relevant_precedents,
            estimated_liability=analysis.estimated_liability,
            penalty_assessment=analysis.penalty_assessment,
            interest_calculation=analysis.interest_calculation,
            limitation_compliance=analysis.limitation_compliance,
            forum_jurisdiction=analysis.forum_jurisdiction.value,
            appeal_options=analysis.appeal_options,
            recommended_strategy=analysis.recommended_strategy,
            settlement_options=analysis.settlement_options,
            documentation_requirements=analysis.documentation_requirements,
            litigation_risk=analysis.litigation_risk,
            compliance_risk=analysis.compliance_risk,
            financial_risk=analysis.financial_risk,
            estimated_resolution_time=analysis.estimated_resolution_time,
            critical_deadlines=analysis.critical_deadlines,
            quantum_confidence=analysis.quantum_confidence,
            quantum_explanation=analysis.quantum_explanation,
            detailed_response=f"GST Dispute Analysis for {case_data.taxpayer_name}",
            processing_time=2.5
        )
        
    except Exception as e:
        logger.error(f"Error analyzing GST dispute: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze GST dispute: {str(e)}"
        )


# Legal aid endpoints
@app.post("/legal-aid/assess", response_model=LegalAidAssessmentResponse, tags=["Legal Aid"])
async def assess_legal_aid_eligibility(
    aid_request: LegalAidRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit),
    current_user: str = Depends(get_current_user)
):
    """Assess legal aid eligibility and recommend appropriate aid."""
    try:
        aid_manager = app_state["legal_aid_manager"]
        
        # Convert request to case data
        from ..use_cases.legal_aid import (
            LegalAidCase, LegalAidApplicant, LegalAidType, 
            CaseCategory, LegalAidAuthority
        )
        from datetime import datetime
        
        # Create applicant object
        applicant = LegalAidApplicant(
            applicant_id=f"APP_{aid_request.case_id}",
            name=aid_request.applicant.name,
            age=aid_request.applicant.age,
            gender=aid_request.applicant.gender,
            address=aid_request.applicant.address,
            phone=aid_request.applicant.phone,
            email=str(aid_request.applicant.email) if aid_request.applicant.email else None,
            annual_income=aid_request.applicant.annual_income,
            family_size=aid_request.applicant.family_size,
            occupation=aid_request.applicant.occupation,
            education_level=aid_request.applicant.education_level,
            social_category=aid_request.applicant.social_category,
            is_disabled=aid_request.applicant.is_disabled,
            disability_type=aid_request.applicant.disability_type,
            is_senior_citizen=aid_request.applicant.is_senior_citizen,
            is_single_woman=aid_request.applicant.is_single_woman,
            is_victim_of_crime=aid_request.applicant.is_victim_of_crime,
            assets_value=aid_request.applicant.assets_value,
            monthly_expenses=aid_request.applicant.monthly_expenses,
            dependents=aid_request.applicant.dependents,
            income_certificate=aid_request.applicant.income_certificate,
            caste_certificate=aid_request.applicant.caste_certificate,
            disability_certificate=aid_request.applicant.disability_certificate,
            identity_proof=aid_request.applicant.identity_proof,
            address_proof=aid_request.applicant.address_proof
        )
        
        # Create case object
        case_data = LegalAidCase(
            case_id=aid_request.case_id,
            applicant=applicant,
            case_category=CaseCategory(aid_request.case_category.value),
            legal_aid_type=LegalAidType(aid_request.legal_aid_type.value),
            case_description=aid_request.case_description,
            urgency_level=aid_request.urgency_level,
            opposing_party=aid_request.opposing_party,
            court_involved=aid_request.court_involved,
            case_number=aid_request.case_number,
            case_stage=aid_request.case_stage,
            lawyer_required=aid_request.lawyer_required,
            court_representation=aid_request.court_representation,
            document_assistance=aid_request.document_assistance,
            legal_advice_only=aid_request.legal_advice_only,
            application_date=aid_request.application_date or datetime.now(),
            required_by_date=aid_request.required_by_date,
            previous_aid_received=aid_request.previous_aid_received,
            previous_aid_details=aid_request.previous_aid_details,
            emergency_case=aid_request.emergency_case,
            pro_bono_eligible=aid_request.pro_bono_eligible,
            media_attention=aid_request.media_attention
        )
        
        # Assess eligibility
        assessment = aid_manager.assess_legal_aid_eligibility(case_data)
        
        # Log assessment (background task)
        background_tasks.add_task(
            log_legal_aid_assessment,
            aid_request.case_id,
            current_user,
            assessment.eligibility_score,
            assessment.priority_level
        )
        
        return LegalAidAssessmentResponse(
            analysis_id=assessment.case_id + "_assessment",
            eligibility_score=assessment.eligibility_score,
            income_eligibility=assessment.income_eligibility,
            category_eligibility=assessment.category_eligibility,
            case_merit=assessment.case_merit,
            urgency_assessment=assessment.urgency_assessment,
            recommended_aid_type=LegalAidTypeEnum(assessment.recommended_aid_type.value),
            recommended_authority=LegalAidAuthorityEnum(assessment.recommended_authority.value),
            estimated_cost=assessment.estimated_cost,
            lawyer_assignment=assessment.lawyer_assignment,
            priority_level=assessment.priority_level,
            estimated_duration=assessment.estimated_duration,
            additional_documents=assessment.additional_documents,
            verification_needed=assessment.verification_needed,
            conditions=assessment.conditions,
            fee_waiver_eligible=assessment.fee_waiver_eligible,
            court_fee_exemption=assessment.court_fee_exemption,
            legal_service_cost=assessment.legal_service_cost,
            case_strength=assessment.case_strength,
            likelihood_of_success=assessment.likelihood_of_success,
            social_impact=assessment.social_impact,
            quantum_confidence=assessment.quantum_confidence,
            quantum_explanation=assessment.quantum_explanation,
            next_steps=assessment.next_steps,
            alternative_options=assessment.alternative_options,
            referral_suggestions=assessment.referral_suggestions,
            detailed_response=f"Legal Aid Assessment for {applicant.name}",
            processing_time=1.8
        )
        
    except Exception as e:
        logger.error(f"Error assessing legal aid: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assess legal aid eligibility: {str(e)}"
        )


# Background task functions for new use cases
async def log_gst_dispute_analysis(case_id: str, user: str, success_prob: float, risk: float):
    """Log GST dispute analysis for analytics."""
    logger.info(f"GST dispute analysis completed - Case: {case_id}, User: {user}, Success: {success_prob:.2f}, Risk: {risk:.2f}")


async def log_legal_aid_assessment(case_id: str, user: str, eligibility: float, priority: int):
    """Log legal aid assessment for analytics."""
    logger.info(f"Legal aid assessment completed - Case: {case_id}, User: {user}, Eligibility: {eligibility:.2f}, Priority: {priority}")


# Main entry point
if __name__ == "__main__":
    config = get_config()
    
    uvicorn.run(
        "src.xqelm.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.logging.level.lower()
    )