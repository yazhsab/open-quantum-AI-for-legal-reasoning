"""
Consumer Dispute Use Case Manager

This module handles the specific logic for consumer dispute cases
under the Consumer Protection Act using quantum-enhanced legal reasoning.
It implements the specialized workflow for consumer complaints,
deficiency in service, and unfair trade practices.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date
import asyncio

import numpy as np
from loguru import logger

from ..core.quantum_legal_model import QuantumLegalModel
from ..classical.preprocessor import LegalTextPreprocessor, PreprocessedText
from ..classical.response_generator import LegalResponseGenerator, ResponseType, LegalResponse
from ..classical.knowledge_base import LegalKnowledgeBase, DocumentType
from ..utils.config import get_config


class ComplaintType(Enum):
    """Types of consumer complaints."""
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


class ServiceType(Enum):
    """Types of services."""
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


class ForumType(Enum):
    """Types of consumer forums."""
    DISTRICT_FORUM = "district_forum"
    STATE_COMMISSION = "state_commission"
    NATIONAL_COMMISSION = "national_commission"
    ONLINE_DISPUTE_RESOLUTION = "online_dispute_resolution"


class CaseStage(Enum):
    """Stages of consumer dispute case."""
    PRE_COMPLAINT = "pre_complaint"
    COMPLAINT_FILED = "complaint_filed"
    NOTICE_ISSUED = "notice_issued"
    REPLY_FILED = "reply_filed"
    EVIDENCE_STAGE = "evidence_stage"
    ARGUMENTS = "arguments"
    ORDER_PASSED = "order_passed"
    APPEAL = "appeal"
    EXECUTION = "execution"


@dataclass
class ProductDetails:
    """Details of the product involved in dispute."""
    product_name: str
    brand: str
    model: Optional[str] = None
    purchase_date: Optional[date] = None
    purchase_price: Optional[float] = None
    warranty_period: Optional[int] = None  # months
    
    # Purchase details
    seller_name: str = ""
    purchase_location: str = ""
    bill_number: Optional[str] = None
    
    # Defect details
    defect_description: Optional[str] = None
    defect_discovered_date: Optional[date] = None
    
    # Repair attempts
    repair_attempts: List[str] = None
    
    def __post_init__(self):
        if self.repair_attempts is None:
            self.repair_attempts = []


@dataclass
class ServiceDetails:
    """Details of the service involved in dispute."""
    service_type: ServiceType
    service_provider: str
    service_description: str
    service_date: Optional[date] = None
    service_amount: Optional[float] = None
    
    # Service agreement
    agreement_terms: Optional[str] = None
    service_level_agreement: Optional[str] = None
    
    # Deficiency details
    deficiency_description: str = ""
    expected_service: str = ""
    actual_service: str = ""
    
    # Communication history
    complaint_history: List[str] = None
    
    def __post_init__(self):
        if self.complaint_history is None:
            self.complaint_history = []


@dataclass
class ConsumerDetails:
    """Details of the consumer."""
    name: str
    address: str
    phone: Optional[str] = None
    email: Optional[str] = None
    
    # Consumer status
    is_senior_citizen: bool = False
    is_disabled: bool = False
    
    # Financial impact
    financial_loss: Optional[float] = None
    mental_agony: Optional[str] = None
    
    # Supporting documents
    supporting_documents: List[str] = None
    
    def __post_init__(self):
        if self.supporting_documents is None:
            self.supporting_documents = []


@dataclass
class OppositePartyDetails:
    """Details of the opposite party (seller/service provider)."""
    name: str
    business_type: str
    address: str
    
    # Business details
    registration_number: Optional[str] = None
    license_number: Optional[str] = None
    
    # Response to complaint
    response_provided: bool = False
    response_details: Optional[str] = None
    
    # Settlement offers
    settlement_offers: List[str] = None
    
    def __post_init__(self):
        if self.settlement_offers is None:
            self.settlement_offers = []


@dataclass
class ConsumerDisputeCase:
    """Complete consumer dispute case data."""
    case_id: str
    complaint_type: ComplaintType
    consumer_details: ConsumerDetails
    opposite_party: OppositePartyDetails
    
    # Case details
    case_stage: CaseStage
    forum_type: ForumType
    case_number: Optional[str] = None
    filing_date: Optional[date] = None
    
    # Dispute specifics
    product_details: Optional[ProductDetails] = None
    service_details: Optional[ServiceDetails] = None
    
    # Relief sought
    compensation_claimed: Optional[float] = None
    replacement_sought: bool = False
    refund_sought: bool = False
    service_improvement_sought: bool = False
    
    # Additional context
    mediation_attempted: bool = False
    previous_complaints: List[str] = None
    
    def __post_init__(self):
        if self.previous_complaints is None:
            self.previous_complaints = []


@dataclass
class ConsumerDisputeAnalysis:
    """Analysis result for consumer dispute case."""
    case_merit: float  # 0-1 scale
    success_probability: float
    
    # Legal analysis
    jurisdiction_validity: bool
    limitation_compliance: bool
    evidence_strength: Dict[str, float]
    
    # Relief analysis
    compensation_estimate: Dict[str, float]
    relief_likelihood: Dict[str, float]
    
    # Recommendations
    recommendations: List[str]
    evidence_gaps: List[str]
    settlement_options: List[str]
    
    # Timeline and costs
    case_duration_estimate: Dict[str, int]
    litigation_cost_estimate: float
    
    # Quantum analysis
    quantum_confidence: float
    quantum_factors: Dict[str, float]
    
    # Precedent analysis
    similar_cases: List[Dict[str, Any]]
    precedent_alignment: float


class ConsumerDisputeManager:
    """
    Specialized manager for consumer dispute cases under Consumer Protection Act.
    
    This class implements the complete workflow for analyzing consumer
    disputes in the Indian legal context, considering consumer rights,
    forum jurisdiction, and relief mechanisms.
    """
    
    def __init__(
        self,
        quantum_model: QuantumLegalModel,
        preprocessor: LegalTextPreprocessor,
        response_generator: LegalResponseGenerator,
        knowledge_base: LegalKnowledgeBase
    ):
        """
        Initialize the consumer dispute manager.
        
        Args:
            quantum_model: Quantum legal reasoning model
            preprocessor: Legal text preprocessor
            response_generator: Response generator
            knowledge_base: Legal knowledge base
        """
        self.quantum_model = quantum_model
        self.preprocessor = preprocessor
        self.response_generator = response_generator
        self.knowledge_base = knowledge_base
        self.config = get_config()
        
        # Load consumer protection law framework
        self._load_legal_framework()
        
        # Initialize consumer rights and remedies
        self._initialize_consumer_rights()
        
        # Load precedent patterns
        self._load_precedent_patterns()
        
        # Statistics
        self.stats = {
            "disputes_analyzed": 0,
            "defective_goods_cases": 0,
            "service_deficiency_cases": 0,
            "average_compensation": 0.0,
            "success_rate": 0.0,
            "settlement_rate": 0.0
        }
        
        logger.info("Consumer dispute manager initialized")
    
    def _load_legal_framework(self) -> None:
        """Load consumer protection law framework."""
        # Consumer Protection Act 2019 provisions
        self.statutory_provisions = {
            "section_2_7": {
                "title": "Consumer defined",
                "description": "Person who buys goods or hires services for consideration"
            },
            "section_2_11": {
                "title": "Deficiency defined", 
                "description": "Any fault, imperfection, shortcoming in quality, nature and manner of performance"
            },
            "section_2_47": {
                "title": "Unfair trade practice defined",
                "description": "Trade practice which adopts unfair method or deceptive practice"
            },
            "section_34": {
                "title": "Jurisdiction of District Commission",
                "description": "Complaints where value does not exceed ₹1 crore"
            },
            "section_47": {
                "title": "Jurisdiction of State Commission", 
                "description": "Complaints where value exceeds ₹1 crore but does not exceed ₹10 crore"
            },
            "section_58": {
                "title": "Jurisdiction of National Commission",
                "description": "Complaints where value exceeds ₹10 crore"
            }
        }
        
        # Consumer rights under the Act
        self.consumer_rights = {
            "right_to_safety": "Protection against hazardous goods and services",
            "right_to_information": "Right to be informed about quality, quantity, potency, purity, standard and price",
            "right_to_choice": "Right to be assured access to variety of goods and services at competitive prices",
            "right_to_be_heard": "Right to be heard and assured that consumer interests receive due consideration",
            "right_to_redressal": "Right to seek redressal against unfair trade practices",
            "right_to_education": "Right to acquire knowledge and skill to be informed consumer"
        }
        
        # Relief available under the Act
        self.relief_mechanisms = {
            "replacement": "Replace goods with new goods free from defects",
            "refund": "Return price paid along with compensation",
            "repair": "Remove defect or deficiency in goods or services",
            "compensation": "Pay compensation for loss or injury suffered",
            "punitive_damages": "Pay punitive damages in appropriate cases",
            "discontinue_practice": "Discontinue unfair trade practice and not repeat"
        }
    
    def _initialize_consumer_rights(self) -> None:
        """Initialize consumer rights and compensation principles."""
        # Compensation factors
        self.compensation_factors = {
            "financial_loss": 1.0,
            "mental_agony": 0.3,  # 30% of financial loss
            "harassment": 0.2,    # 20% of financial loss
            "litigation_cost": 0.1,  # 10% of financial loss
            "interest": 0.12      # 12% per annum
        }
        
        # Forum jurisdiction limits (in rupees)
        self.jurisdiction_limits = {
            ForumType.DISTRICT_FORUM: 10000000,      # ₹1 crore
            ForumType.STATE_COMMISSION: 100000000,    # ₹10 crore
            ForumType.NATIONAL_COMMISSION: float('inf')  # Above ₹10 crore
        }
        
        # Limitation period (in years)
        self.limitation_periods = {
            "goods": 2,      # 2 years from date of purchase
            "services": 2,   # 2 years from date of service
            "continuing_cause": 2  # 2 years from cessation of cause
        }
        
        # Evidence strength factors
        self.evidence_factors = {
            "purchase_bill": 0.9,
            "warranty_card": 0.8,
            "service_agreement": 0.8,
            "correspondence": 0.7,
            "witness_testimony": 0.6,
            "expert_opinion": 0.8,
            "photographs": 0.7,
            "medical_reports": 0.9
        }
    
    def _load_precedent_patterns(self) -> None:
        """Load patterns from consumer law precedents."""
        self.precedent_patterns = {
            "lucknow_development_authority_v_mk_gupta": {
                "principle": "Deficiency in service includes delay in delivery",
                "factors": ["service_deficiency", "delay", "compensation"],
                "quantum_pattern": "service_deficiency_pattern"
            },
            "spring_meadows_v_haryana_urban": {
                "principle": "Mental agony compensation in consumer cases",
                "factors": ["mental_agony", "harassment", "compensation"],
                "quantum_pattern": "mental_agony_pattern"
            },
            "indian_medical_association_v_vp_shantha": {
                "principle": "Medical services covered under Consumer Protection Act",
                "factors": ["medical_service", "professional_service", "consumer_protection"],
                "quantum_pattern": "medical_service_pattern"
            }
        }
    
    async def analyze_consumer_dispute(
        self,
        case_data: ConsumerDisputeCase,
        additional_context: Optional[str] = None
    ) -> ConsumerDisputeAnalysis:
        """
        Analyze consumer dispute case using quantum-enhanced reasoning.
        
        Args:
            case_data: Consumer dispute case data
            additional_context: Additional context or arguments
            
        Returns:
            Comprehensive analysis of the consumer dispute
        """
        logger.info(f"Analyzing consumer dispute case: {case_data.case_id}")
        
        try:
            # Step 1: Preprocess case data
            processed_data = await self._preprocess_case_data(case_data, additional_context)
            
            # Step 2: Check jurisdiction and limitation
            jurisdiction_analysis = await self._check_jurisdiction_and_limitation(case_data)
            
            # Step 3: Assess evidence strength
            evidence_analysis = await self._assess_evidence_strength(case_data)
            
            # Step 4: Calculate compensation
            compensation_analysis = await self._calculate_compensation(case_data)
            
            # Step 5: Retrieve relevant precedents
            legal_context = await self._retrieve_legal_context(case_data)
            
            # Step 6: Perform quantum analysis
            quantum_results = await self._perform_quantum_analysis(
                processed_data, legal_context, case_data
            )
            
            # Step 7: Generate comprehensive analysis
            analysis = await self._generate_dispute_analysis(
                case_data, jurisdiction_analysis, evidence_analysis,
                compensation_analysis, quantum_results, legal_context
            )
            
            # Step 8: Update statistics
            self._update_statistics(case_data, analysis)
            
            logger.info(f"Consumer dispute analysis completed: {case_data.case_id}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing consumer dispute: {e}")
            # Return default analysis
            return ConsumerDisputeAnalysis(
                case_merit=0.0,
                success_probability=0.0,
                jurisdiction_validity=False,
                limitation_compliance=False,
                evidence_strength={},
                compensation_estimate={},
                relief_likelihood={},
                recommendations=["Case analysis failed - manual review required"],
                evidence_gaps=["Complete case review needed"],
                settlement_options=[],
                case_duration_estimate={"min_months": 6, "max_months": 18},
                litigation_cost_estimate=0.0,
                quantum_confidence=0.0,
                quantum_factors={},
                similar_cases=[],
                precedent_alignment=0.0
            )
    
    async def _preprocess_case_data(
        self,
        case_data: ConsumerDisputeCase,
        additional_context: Optional[str]
    ) -> PreprocessedText:
        """Preprocess consumer dispute case data."""
        # Combine all case information
        text_content = f"""
        Consumer Dispute Case Analysis: {case_data.case_id}
        
        Complaint Type: {case_data.complaint_type.value}
        Forum: {case_data.forum_type.value}
        Case Stage: {case_data.case_stage.value}
        Filing Date: {case_data.filing_date or 'Not specified'}
        
        Consumer Details:
        Name: {case_data.consumer_details.name}
        Address: {case_data.consumer_details.address}
        Financial Loss: ₹{case_data.consumer_details.financial_loss:,.2f if case_data.consumer_details.financial_loss else 0}
        Senior Citizen: {case_data.consumer_details.is_senior_citizen}
        
        Opposite Party:
        Name: {case_data.opposite_party.name}
        Business Type: {case_data.opposite_party.business_type}
        Response Provided: {case_data.opposite_party.response_provided}
        
        {self._format_product_service_details(case_data)}
        
        Relief Sought:
        Compensation: ₹{case_data.compensation_claimed:,.2f if case_data.compensation_claimed else 0}
        Replacement: {case_data.replacement_sought}
        Refund: {case_data.refund_sought}
        Service Improvement: {case_data.service_improvement_sought}
        
        Mediation Attempted: {case_data.mediation_attempted}
        
        {additional_context or ''}
        """
        
        # Preprocess the text
        processed = await self.preprocessor.preprocess_text(
            text_content,
            document_type="consumer_dispute",
            metadata={
                "case_id": case_data.case_id,
                "complaint_type": case_data.complaint_type.value,
                "forum_type": case_data.forum_type.value,
                "compensation_claimed": case_data.compensation_claimed or 0
            }
        )
        
        return processed
    
    def _format_product_service_details(self, case_data: ConsumerDisputeCase) -> str:
        """Format product or service details for preprocessing."""
        if case_data.product_details:
            product = case_data.product_details
            return f"""
            Product Details:
            Name: {product.product_name}
            Brand: {product.brand}
            Purchase Date: {product.purchase_date or 'Not specified'}
            Purchase Price: ₹{product.purchase_price:,.2f if product.purchase_price else 0}
            Seller: {product.seller_name}
            Defect: {product.defect_description or 'Not specified'}
            """
        elif case_data.service_details:
            service = case_data.service_details
            return f"""
            Service Details:
            Type: {service.service_type.value}
            Provider: {service.service_provider}
            Description: {service.service_description}
            Service Date: {service.service_date or 'Not specified'}
            Amount: ₹{service.service_amount:,.2f if service.service_amount else 0}
            Deficiency: {service.deficiency_description}
            """
        else:
            return "No product or service details provided"
    
    async def _check_jurisdiction_and_limitation(
        self,
        case_data: ConsumerDisputeCase
    ) -> Dict[str, Any]:
        """Check jurisdiction and limitation compliance."""
        analysis = {
            "jurisdiction_valid": False,
            "limitation_complied": False,
            "appropriate_forum": None,
            "limitation_period_remaining": 0
        }
        
        # Check jurisdiction based on compensation claimed
        compensation = case_data.compensation_claimed or case_data.consumer_details.financial_loss or 0
        
        for forum, limit in self.jurisdiction_limits.items():
            if compensation <= limit:
                analysis["appropriate_forum"] = forum
                analysis["jurisdiction_valid"] = (case_data.forum_type == forum)
                break
        
        # Check limitation period
        if case_data.product_details and case_data.product_details.purchase_date:
            purchase_date = case_data.product_details.purchase_date
            filing_date = case_data.filing_date or date.today()
            
            days_elapsed = (filing_date - purchase_date).days
            limitation_days = self.limitation_periods["goods"] * 365
            
            analysis["limitation_complied"] = days_elapsed <= limitation_days
            analysis["limitation_period_remaining"] = max(0, limitation_days - days_elapsed)
        
        elif case_data.service_details and case_data.service_details.service_date:
            service_date = case_data.service_details.service_date
            filing_date = case_data.filing_date or date.today()
            
            days_elapsed = (filing_date - service_date).days
            limitation_days = self.limitation_periods["services"] * 365
            
            analysis["limitation_complied"] = days_elapsed <= limitation_days
            analysis["limitation_period_remaining"] = max(0, limitation_days - days_elapsed)
        
        return analysis
    
    async def _assess_evidence_strength(
        self,
        case_data: ConsumerDisputeCase
    ) -> Dict[str, float]:
        """Assess strength of evidence available."""
        evidence_strength = {}
        
        # Check consumer's supporting documents
        for doc in case_data.consumer_details.supporting_documents:
            doc_lower = doc.lower()
            for evidence_type, strength in self.evidence_factors.items():
                if evidence_type.replace("_", " ") in doc_lower:
                    evidence_strength[evidence_type] = strength
        
        # Product-specific evidence
        if case_data.product_details:
            if case_data.product_details.bill_number:
                evidence_strength["purchase_bill"] = 0.9
            if case_data.product_details.warranty_period:
                evidence_strength["warranty_card"] = 0.8
        
        # Service-specific evidence
        if case_data.service_details:
            if case_data.service_details.agreement_terms:
                evidence_strength["service_agreement"] = 0.8
            if case_data.service_details.complaint_history:
                evidence_strength["correspondence"] = 0.7
        
        # Calculate overall evidence strength
        if evidence_strength:
            overall_strength = sum(evidence_strength.values()) / len(evidence_strength)
            evidence_strength["overall_strength"] = overall_strength
        else:
            evidence_strength["overall_strength"] = 0.3  # Weak evidence
        
        return evidence_strength
    
    async def _calculate_compensation(
        self,
        case_data: ConsumerDisputeCase
    ) -> Dict[str, float]:
        """Calculate estimated compensation."""
        compensation = {}
        
        # Financial loss
        financial_loss = case_data.consumer_details.financial_loss or 0
        if case_data.product_details and case_data.product_details.purchase_price:
            financial_loss = max(financial_loss, case_data.product_details.purchase_price)
        elif case_data.service_details and case_data.service_details.service_amount:
            financial_loss = max(financial_loss, case_data.service_details.service_amount)
        
        compensation["financial_loss"] = financial_loss
        
        # Mental agony and harassment
        if case_data.consumer_details.mental_agony:
            mental_agony_amount = financial_loss * self.compensation_factors["mental_agony"]
            compensation["mental_agony"] = mental_agony_amount
        
        # Litigation cost
        litigation_cost = financial_loss * self.compensation_factors["litigation_cost"]
        compensation["litigation_cost"] = litigation_cost
        
        # Interest (if applicable)
        if case_data.filing_date and case_data.product_details and case_data.product_details.purchase_date:
            months_elapsed = (case_data.filing_date - case_data.product_details.purchase_date).days / 30
            interest = financial_loss * self.compensation_factors["interest"] * (months_elapsed / 12)
            compensation["interest"] = interest
        
        # Total compensation
        compensation["total_estimated"] = sum(compensation.values())
        
        return compensation
    
    async def _retrieve_legal_context(
        self,
        case_data: ConsumerDisputeCase
    ) -> Dict[str, Any]:
        """Retrieve relevant legal context."""
        legal_context = {
            "precedents": [],
            "statutes": [],
            "similar_cases": []
        }
        
        # Search for relevant precedents
        precedent_query = f"""
        consumer protection {case_data.complaint_type.value}
        {case_data.service_details.service_type.value if case_data.service_details else 'goods'}
        compensation deficiency unfair trade practice
        """
        
        precedent_results = await self.knowledge_base.search_documents(
            query=precedent_query,
            document_types=[DocumentType.CASE_LAW],
            jurisdiction="india",
            limit=10
        )
        
        legal_context["precedents"] = [
            {
                "case_name": result.document.title,
                "citation": result.document.citation,
                "summary": result.document.summary,
                "holding": result.document.holding,
                "relevance_score": result.relevance_score
            }
            for result in precedent_results
        ]
        
        # Add statutory framework
        legal_context["statutory_provisions"] = self.statutory_provisions
        legal_context["consumer_rights"] = self.consumer_rights
        legal_context["relief_mechanisms"] = self.relief_mechanisms
        
        return legal_context
    
    async def _perform_quantum_analysis(
        self,
        processed_data: PreprocessedText,
        legal_context: Dict[str, Any],
        case_data: ConsumerDisputeCase
    ) -> Dict[str, Any]:
        """Perform quantum analysis of consumer dispute."""
        # Prepare quantum input
        quantum_input = {
            "query": processed_data.cleaned_text,
            "legal_concepts": processed_data.legal_concepts,
            "entities": [entity.text for entity in processed_data.entities],
            "precedents": legal_context["precedents"],
            "complaint_type": case_data.complaint_type.value,
            "forum_type": case_data.forum_type.value,
            "compensation_claimed": case_data.compensation_claimed or 0
        }
        
        # Perform quantum reasoning
        quantum_results = await self.quantum_model.process_query(
            query=processed_data.cleaned_text,
            context=quantum_input,
            use_case="consumer_dispute"
        )
        
        # Extract consumer dispute specific metrics
        quantum_analysis = {
            "success_probability": quantum_results.get("predictions", {}).get("success_probability", 0.5),
            "compensation_likelihood": quantum_results.get("compensation_probability", 0.5),
            "settlement_probability": quantum_results.get("settlement_likelihood", 0.5),
            "case_merit": quantum_results.get("case_strength", 0.5),
            "precedent_similarity": quantum_results.get("precedent_similarity", 0.0),
            "statutory_alignment": quantum_results.get("statutory_alignment", 0.0),
            "quantum_coherence": quantum_results.get("coherence", 0.0),
            "entanglement_measures": quantum_results.get("entanglement", {}),
            "risk_factors": quantum_results.get("risk_assessment", {})
        }
        
        return quantum_analysis
    
    async def _generate_dispute_analysis(
        self,
        case_data: ConsumerDisputeCase,
        jurisdiction_analysis: Dict[str, Any],
        evidence_analysis: Dict[str, float],
        compensation_analysis: Dict[str, float],
        quantum_results: Dict[str, Any],
        legal_context: Dict[str, Any]
    ) -> ConsumerDisputeAnalysis:
        """Generate comprehensive dispute analysis."""
        
        # Calculate case merit
        case_merit = 0.0
        if jurisdiction_analysis["jurisdiction_valid"]:
            case_merit += 0.3
        if jurisdiction_analysis["limitation_complied"]:
            case_merit += 0.3
        case_merit += evidence_analysis.get("overall_strength", 0.0) * 0.4
        
        # Calculate success probability
        success_probability = quantum_results.get("success_probability", case_merit)
        
        # Relief likelihood
        relief_likelihood = {
            "compensation": quantum_results.get("compensation_likelihood", 0.6),
            "replacement": 0.7 if case_data.replacement_sought else 0.0,
            "refund": 0.8 if case_data.refund_sought else 0.0,
            "service_improvement": 0.5 if case_data.service_improvement_sought else 0.0
        }
        
        # Generate recommendations
        recommendations = []
        
        if not jurisdiction_analysis["jurisdiction_valid"]:
            recommendations.append(f"File complaint in {jurisdiction_analysis['appropriate_forum'].value}")
        
        if not jurisdiction_analysis["limitation_complied"]:
            recommendations.append("Complaint may be time-barred - check for continuing cause of action")
        
        if evidence_analysis.get("overall_strength", 0) < 0.5:
            recommendations.append("Strengthen evidence with additional documentation")
        
        if case_data.case_stage == CaseStage.PRE_COMPLAINT:
            recommendations.extend([
                "Send legal notice to opposite party before filing complaint",
                "Collect all purchase/service documents",
                "Document all communication with opposite party"
            ])
        
        # Evidence gaps
        evidence_gaps = []
        required_evidence = ["purchase_bill", "correspondence", "witness_testimony"]
        for evidence in required_evidence:
            if evidence not in evidence_analysis:
                evidence_gaps.append(evidence.replace("_", " ").title())
        
        # Settlement options
        settlement_options = [
            "Direct negotiation with opposite party",
            "Mediation through consumer forum",
            "Compromise settlement with partial compensation",
            "Product replacement with additional warranty"
        ]
        
        # Case duration estimate
        duration_estimate = {"min_months": 4, "max_months": 12}
        if case_data.forum_type == ForumType.STATE_COMMISSION:
            duration_estimate = {"min_months": 6, "max_months": 18}
        elif case_data.forum_type == ForumType.NATIONAL_COMMISSION:
            duration_estimate = {"min_months": 12, "max_months": 36}
        
        # Litigation cost estimate
        compensation_claimed = case_data.compensation_claimed or compensation_analysis.get("total_estimated", 0)
        litigation_cost = compensation_claimed * 0.05  # 5% of claim value
        
        # Extract similar cases
        similar_cases = [
            {
                "case_name": prec["case_name"],
                "similarity_score": prec["relevance_score"],
                "citation": prec["citation"],
                "summary": prec["summary"]
            }
            for prec in legal_context["precedents"][:5]
        ]
        
        return ConsumerDisputeAnalysis(
            case_merit=case_merit,
            success_probability=success_probability,
            jurisdiction_validity=jurisdiction_analysis["jurisdiction_valid"],
            limitation_compliance=jurisdiction_analysis["limitation_complied"],
            evidence_strength=evidence_analysis,
            compensation_estimate=compensation_analysis,
            relief_likelihood=relief_likelihood,
            recommendations=recommendations,
            evidence_gaps=evidence_gaps,
            settlement_options=settlement_options,
            case_duration_estimate=duration_estimate,
            litigation_cost_estimate=litigation_cost,
            quantum_confidence=quantum_results.get("quantum_coherence", 0.0),
            quantum_factors=quantum_results.get("entanglement_measures", {}),
            similar_cases=similar_cases,
            precedent_alignment=quantum_results.get("precedent_similarity", 0.0)
        )
    
    def _update_statistics(
        self,
        case_data: ConsumerDisputeCase,
        analysis: ConsumerDisputeAnalysis
    ) -> None:
        """Update consumer dispute statistics."""
        self.stats["disputes_analyzed"] += 1
        
        # Update case type statistics
        if case_data.complaint_type in [ComplaintType.DEFECTIVE_GOODS, ComplaintType.PRODUCT_LIABILITY]:
            self.stats["defective_goods_cases"] += 1
        elif case_data.complaint_type in [ComplaintType.DEFICIENCY_IN_SERVICE, ComplaintType.BANKING_SERVICE,
                                         ComplaintType.TELECOM_SERVICE]:
            self.stats["service_deficiency_cases"] += 1
        
        # Update average compensation
        if analysis.compensation_estimate.get("total_estimated", 0) > 0:
            current_avg = self.stats["average_compensation"]
            new_compensation = analysis.compensation_estimate["total_estimated"]
            total_cases = self.stats["disputes_analyzed"]
            
            self.stats["average_compensation"] = (
                (current_avg * (total_cases - 1) + new_compensation) / total_cases
            )
        
        # Update success rate
        if analysis.success_probability > 0.6:
            current_success = self.stats["success_rate"]
            total_cases = self.stats["disputes_analyzed"]
            
            self.stats["success_rate"] = (
                (current_success * (total_cases - 1) + analysis.success_probability) / total_cases
            )
        
        # Update settlement rate
        settlement_prob = analysis.quantum_factors.get("settlement_probability", 0.0)
        if settlement_prob > 0:
            current_settlement = self.stats["settlement_rate"]
            total_cases = self.stats["disputes_analyzed"]
            
            self.stats["settlement_rate"] = (
                (current_settlement * (total_cases - 1) + settlement_prob) / total_cases
            )
    
    async def generate_consumer_response(
        self,
        case_data: ConsumerDisputeCase,
        analysis: ConsumerDisputeAnalysis,
        response_type: str = "comprehensive"
    ) -> LegalResponse:
        """
        Generate legal response for consumer dispute case.
        
        Args:
            case_data: Consumer dispute case data
            analysis: Analysis results
            response_type: Type of response to generate
            
        Returns:
            Formatted legal response
        """
        # Prepare context for response generation
        context = {
            "case_type": "consumer_dispute",
            "complaint_type": case_data.complaint_type.value,
            "forum_type": case_data.forum_type.value,
            "case_merit": analysis.case_merit,
            "success_probability": analysis.success_probability,
            "compensation_estimate": analysis.compensation_estimate,
            "recommendations": analysis.recommendations,
            "evidence_gaps": analysis.evidence_gaps,
            "settlement_options": analysis.settlement_options,
            "quantum_confidence": analysis.quantum_confidence,
            "similar_cases": analysis.similar_cases
        }
        
        # Generate response using the response generator
        response = await self.response_generator.generate_response(
            query=f"Consumer dispute analysis for {case_data.case_id}",
            context=context,
            response_type=ResponseType.CONSUMER_DISPUTE,
            legal_framework="consumer_protection_act_2019"
        )
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consumer dispute manager statistics."""
        return {
            **self.stats,
            "manager_type": "consumer_dispute",
            "legal_framework": "Consumer Protection Act 2019",
            "jurisdiction_coverage": "India",
            "supported_complaint_types": [ct.value for ct in ComplaintType],
            "supported_forums": [ft.value for ft in ForumType]
        }