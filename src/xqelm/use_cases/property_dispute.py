"""
Property Dispute Use Case Manager

This module handles the specific logic for property dispute cases
using quantum-enhanced legal reasoning. It implements the specialized
workflow for property-related legal proceedings including title disputes,
partition suits, possession matters, and property transactions.

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


class PropertyType(Enum):
    """Types of property."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    AGRICULTURAL = "agricultural"
    INDUSTRIAL = "industrial"
    VACANT_LAND = "vacant_land"
    APARTMENT = "apartment"
    PLOT = "plot"


class DisputeType(Enum):
    """Types of property disputes."""
    TITLE_DISPUTE = "title_dispute"
    PARTITION_SUIT = "partition_suit"
    POSSESSION_MATTER = "possession_matter"
    BOUNDARY_DISPUTE = "boundary_dispute"
    EASEMENT_RIGHTS = "easement_rights"
    RENT_CONTROL = "rent_control"
    SALE_DEED_CHALLENGE = "sale_deed_challenge"
    INHERITANCE_DISPUTE = "inheritance_dispute"
    ENCROACHMENT = "encroachment"
    SPECIFIC_PERFORMANCE = "specific_performance"


class PropertyStatus(Enum):
    """Status of property ownership."""
    CLEAR_TITLE = "clear_title"
    DISPUTED_TITLE = "disputed_title"
    JOINT_OWNERSHIP = "joint_ownership"
    ANCESTRAL_PROPERTY = "ancestral_property"
    SELF_ACQUIRED = "self_acquired"
    GOVERNMENT_LAND = "government_land"
    LEASEHOLD = "leasehold"
    FREEHOLD = "freehold"


class CaseStage(Enum):
    """Stages of property dispute case."""
    PRE_LITIGATION = "pre_litigation"
    NOTICE_SENT = "notice_sent"
    SUIT_FILED = "suit_filed"
    WRITTEN_STATEMENT = "written_statement"
    EVIDENCE_STAGE = "evidence_stage"
    ARGUMENTS = "arguments"
    JUDGMENT = "judgment"
    APPEAL = "appeal"
    EXECUTION = "execution"


@dataclass
class PropertyDetails:
    """Details of the disputed property."""
    property_id: str
    property_type: PropertyType
    location: str
    area: float  # in square feet/meters
    market_value: float
    survey_number: Optional[str] = None
    sub_division: Optional[str] = None
    village: Optional[str] = None
    district: str = ""
    state: str = ""
    
    # Registration details
    registration_date: Optional[date] = None
    registration_number: Optional[str] = None
    registrar_office: Optional[str] = None
    
    # Current status
    current_possession: Optional[str] = None
    property_status: PropertyStatus = PropertyStatus.CLEAR_TITLE
    encumbrances: List[str] = None
    
    def __post_init__(self):
        if self.encumbrances is None:
            self.encumbrances = []


@dataclass
class PartyDetails:
    """Details of parties in property dispute."""
    name: str
    relationship_to_property: str  # owner, heir, purchaser, etc.
    share_claimed: Optional[float] = None  # percentage or fraction
    possession_period: Optional[str] = None
    basis_of_claim: str = ""
    supporting_documents: List[str] = None
    
    def __post_init__(self):
        if self.supporting_documents is None:
            self.supporting_documents = []


@dataclass
class PropertyDisputeCase:
    """Complete property dispute case data."""
    case_id: str
    property_details: PropertyDetails
    dispute_type: DisputeType
    
    # Parties
    plaintiff_details: PartyDetails
    defendant_details: List[PartyDetails]
    
    # Case details
    case_stage: CaseStage
    court: Optional[str] = None
    case_number: Optional[str] = None
    filing_date: Optional[date] = None
    
    # Dispute specifics
    dispute_description: str = ""
    relief_sought: str = ""
    disputed_area: Optional[float] = None
    disputed_value: Optional[float] = None
    
    # Legal documents
    title_documents: List[str] = None
    revenue_records: List[str] = None
    survey_reports: List[str] = None
    witness_statements: List[str] = None
    
    # Timeline
    dispute_start_date: Optional[date] = None
    last_transaction_date: Optional[date] = None
    
    # Additional context
    family_tree: Optional[str] = None
    previous_litigation: List[str] = None
    settlement_attempts: List[str] = None
    
    def __post_init__(self):
        if self.defendant_details is None:
            self.defendant_details = []
        if self.title_documents is None:
            self.title_documents = []
        if self.revenue_records is None:
            self.revenue_records = []
        if self.survey_reports is None:
            self.survey_reports = []
        if self.witness_statements is None:
            self.witness_statements = []
        if self.previous_litigation is None:
            self.previous_litigation = []
        if self.settlement_attempts is None:
            self.settlement_attempts = []


@dataclass
class PropertyDisputeAnalysis:
    """Analysis result for property dispute case."""
    title_strength: Dict[str, float]  # strength of each party's title
    possession_rights: Dict[str, float]  # possession rights assessment
    legal_validity: Dict[str, bool]  # validity of various claims
    
    # Outcome predictions
    case_outcome_probability: Dict[str, float]  # probability for each party
    settlement_probability: float
    case_duration_estimate: Dict[str, int]  # min/max months
    
    # Recommendations
    recommendations: List[str]
    settlement_options: List[str]
    evidence_gaps: List[str]
    legal_strategies: List[str]
    
    # Financial analysis
    litigation_cost_estimate: float
    property_valuation: float
    potential_compensation: Dict[str, float]
    
    # Quantum analysis
    quantum_confidence: float
    quantum_factors: Dict[str, float]
    
    # Precedent analysis
    similar_cases: List[Dict[str, Any]]
    precedent_alignment: float


class PropertyDisputeManager:
    """
    Specialized manager for property dispute cases.
    
    This class implements the complete workflow for analyzing property
    disputes in the Indian legal context, considering property laws,
    revenue records, and case-specific factors.
    """
    
    def __init__(
        self,
        quantum_model: QuantumLegalModel,
        preprocessor: LegalTextPreprocessor,
        response_generator: LegalResponseGenerator,
        knowledge_base: LegalKnowledgeBase
    ):
        """
        Initialize the property dispute manager.
        
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
        
        # Load property law framework
        self._load_legal_framework()
        
        # Initialize property law principles
        self._initialize_property_principles()
        
        # Load precedent patterns
        self._load_precedent_patterns()
        
        # Statistics
        self.stats = {
            "cases_analyzed": 0,
            "title_disputes": 0,
            "partition_suits": 0,
            "possession_matters": 0,
            "average_case_value": 0.0,
            "settlement_rate": 0.0
        }
        
        logger.info("Property dispute manager initialized")
    
    def _load_legal_framework(self) -> None:
        """Load property law legal framework."""
        # Key statutory provisions
        self.statutory_provisions = {
            "transfer_of_property_act": {
                "section_5": "Transfer of property defined",
                "section_17": "Documents of which registration is compulsory",
                "section_53A": "Part performance of contract for transfer of immovable property",
                "section_54": "Sale defined"
            },
            "registration_act": {
                "section_17": "Documents of which registration is compulsory",
                "section_49": "Effect of non-registration"
            },
            "limitation_act": {
                "article_65": "Suit for possession of immovable property based on title",
                "article_144": "Any other suit for which no period of limitation is provided"
            },
            "specific_relief_act": {
                "section_12": "Specific performance of contracts",
                "section_16": "Personal bars to relief"
            },
            "partition_act": {
                "section_4": "Partition by sale",
                "section_11": "Preliminary decree in partition suit"
            }
        }
        
        # Constitutional provisions
        self.constitutional_provisions = {
            "article_300A": {
                "title": "Right to property",
                "text": "No person shall be deprived of his property save by authority of law"
            }
        }
        
        # Property law principles
        self.legal_principles = {
            "nemo_dat_quod_non_habet": "No one can give what he does not have",
            "prior_tempore_potior_jure": "First in time, stronger in right",
            "possession_follows_title": "Possession should follow legal title",
            "adverse_possession": "Continuous possession can create title",
            "doctrine_of_part_performance": "Part performance can enforce oral contracts"
        }
    
    def _initialize_property_principles(self) -> None:
        """Initialize property law principles and factors."""
        # Title strength factors
        self.title_factors = {
            "registered_sale_deed": 0.9,
            "gift_deed": 0.8,
            "inheritance_document": 0.7,
            "partition_deed": 0.8,
            "court_decree": 0.95,
            "revenue_records": 0.6,
            "possession_certificate": 0.5,
            "oral_agreement": 0.2
        }
        
        # Possession factors
        self.possession_factors = {
            "continuous_possession": 0.8,
            "peaceful_possession": 0.7,
            "open_possession": 0.6,
            "adverse_possession_12_years": 0.9,
            "permissive_possession": 0.3,
            "disputed_possession": 0.2
        }
        
        # Evidence strength mapping
        self.evidence_strength = {
            "original_documents": 0.9,
            "certified_copies": 0.8,
            "revenue_records": 0.7,
            "survey_settlement": 0.8,
            "mutation_entries": 0.6,
            "tax_receipts": 0.5,
            "witness_testimony": 0.4,
            "circumstantial_evidence": 0.3
        }
    
    def _load_precedent_patterns(self) -> None:
        """Load patterns from property law precedents."""
        self.precedent_patterns = {
            "sarla_ahuja_v_united_india": {
                "principle": "Adverse possession requires continuous, open, and hostile possession",
                "factors": ["adverse_possession", "continuous_possession", "animus_possidendi"],
                "quantum_pattern": "adverse_possession_pattern"
            },
            "kiran_singh_v_chaman_paswan": {
                "principle": "Revenue entries are not conclusive proof of title",
                "factors": ["revenue_records", "title_documents", "burden_of_proof"],
                "quantum_pattern": "revenue_records_pattern"
            },
            "suraj_lamp_v_state_of_haryana": {
                "principle": "Partition by metes and bounds preferred over sale",
                "factors": ["partition_suit", "family_property", "equitable_division"],
                "quantum_pattern": "partition_pattern"
            }
        }
    
    async def analyze_property_dispute(
        self,
        case_data: PropertyDisputeCase,
        additional_context: Optional[str] = None
    ) -> PropertyDisputeAnalysis:
        """
        Analyze property dispute case using quantum-enhanced reasoning.
        
        Args:
            case_data: Property dispute case data
            additional_context: Additional context or arguments
            
        Returns:
            Comprehensive analysis of the property dispute
        """
        logger.info(f"Analyzing property dispute case: {case_data.case_id}")
        
        try:
            # Step 1: Preprocess case data
            processed_data = await self._preprocess_case_data(case_data, additional_context)
            
            # Step 2: Analyze title strength
            title_analysis = await self._analyze_title_strength(case_data)
            
            # Step 3: Assess possession rights
            possession_analysis = await self._assess_possession_rights(case_data)
            
            # Step 4: Retrieve relevant precedents
            legal_context = await self._retrieve_legal_context(case_data)
            
            # Step 5: Perform quantum analysis
            quantum_results = await self._perform_quantum_analysis(
                processed_data, legal_context, case_data
            )
            
            # Step 6: Generate comprehensive analysis
            analysis = await self._generate_dispute_analysis(
                case_data, title_analysis, possession_analysis,
                quantum_results, legal_context
            )
            
            # Step 7: Update statistics
            self._update_statistics(case_data, analysis)
            
            logger.info(f"Property dispute analysis completed for case: {case_data.case_id}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing property dispute: {e}")
            # Return default analysis
            return PropertyDisputeAnalysis(
                title_strength={},
                possession_rights={},
                legal_validity={},
                case_outcome_probability={},
                settlement_probability=0.0,
                case_duration_estimate={"min_months": 12, "max_months": 60},
                recommendations=["Case analysis failed - manual review required"],
                settlement_options=[],
                evidence_gaps=["Complete case review needed"],
                legal_strategies=[],
                litigation_cost_estimate=0.0,
                property_valuation=case_data.property_details.market_value,
                potential_compensation={},
                quantum_confidence=0.0,
                quantum_factors={},
                similar_cases=[],
                precedent_alignment=0.0
            )
    
    async def _preprocess_case_data(
        self,
        case_data: PropertyDisputeCase,
        additional_context: Optional[str]
    ) -> PreprocessedText:
        """Preprocess property dispute case data."""
        # Combine all case information
        text_content = f"""
        Property Dispute Case Analysis: {case_data.case_id}
        
        Property Details:
        Property Type: {case_data.property_details.property_type.value}
        Location: {case_data.property_details.location}
        Area: {case_data.property_details.area} sq ft
        Market Value: ₹{case_data.property_details.market_value:,.2f}
        Survey Number: {case_data.property_details.survey_number or 'Not specified'}
        District: {case_data.property_details.district}
        State: {case_data.property_details.state}
        Property Status: {case_data.property_details.property_status.value}
        
        Dispute Details:
        Dispute Type: {case_data.dispute_type.value}
        Case Stage: {case_data.case_stage.value}
        Court: {case_data.court or 'Not specified'}
        Filing Date: {case_data.filing_date or 'Not specified'}
        
        Plaintiff Details:
        Name: {case_data.plaintiff_details.name}
        Relationship: {case_data.plaintiff_details.relationship_to_property}
        Share Claimed: {case_data.plaintiff_details.share_claimed or 'Not specified'}
        Basis of Claim: {case_data.plaintiff_details.basis_of_claim}
        
        Defendants:
        {self._format_defendants(case_data.defendant_details)}
        
        Dispute Description: {case_data.dispute_description}
        Relief Sought: {case_data.relief_sought}
        
        Title Documents: {', '.join(case_data.title_documents)}
        Revenue Records: {', '.join(case_data.revenue_records)}
        
        {additional_context or ''}
        """
        
        # Preprocess the text
        processed = await self.preprocessor.preprocess_text(
            text_content,
            document_type="property_dispute",
            metadata={
                "case_id": case_data.case_id,
                "dispute_type": case_data.dispute_type.value,
                "property_value": case_data.property_details.market_value,
                "property_type": case_data.property_details.property_type.value
            }
        )
        
        return processed
    
    def _format_defendants(self, defendants: List[PartyDetails]) -> str:
        """Format defendant details for preprocessing."""
        if not defendants:
            return "No defendants specified"
        
        formatted = []
        for i, defendant in enumerate(defendants, 1):
            formatted.append(f"""
            Defendant {i}:
            Name: {defendant.name}
            Relationship: {defendant.relationship_to_property}
            Share Claimed: {defendant.share_claimed or 'Not specified'}
            Basis of Claim: {defendant.basis_of_claim}
            """)
        
        return "\n".join(formatted)
    
    async def _analyze_title_strength(
        self,
        case_data: PropertyDisputeCase
    ) -> Dict[str, float]:
        """Analyze title strength for each party."""
        title_strength = {}
        
        # Analyze plaintiff's title
        plaintiff_strength = 0.0
        for doc in case_data.plaintiff_details.supporting_documents:
            doc_lower = doc.lower()
            for doc_type, strength in self.title_factors.items():
                if doc_type.replace("_", " ") in doc_lower:
                    plaintiff_strength = max(plaintiff_strength, strength)
        
        title_strength[case_data.plaintiff_details.name] = plaintiff_strength
        
        # Analyze defendants' titles
        for defendant in case_data.defendant_details:
            defendant_strength = 0.0
            for doc in defendant.supporting_documents:
                doc_lower = doc.lower()
                for doc_type, strength in self.title_factors.items():
                    if doc_type.replace("_", " ") in doc_lower:
                        defendant_strength = max(defendant_strength, strength)
            
            title_strength[defendant.name] = defendant_strength
        
        return title_strength
    
    async def _assess_possession_rights(
        self,
        case_data: PropertyDisputeCase
    ) -> Dict[str, float]:
        """Assess possession rights for each party."""
        possession_rights = {}
        
        # Assess based on current possession and duration
        current_possessor = case_data.property_details.current_possession
        
        if current_possessor:
            # Find the party in possession
            all_parties = [case_data.plaintiff_details] + case_data.defendant_details
            
            for party in all_parties:
                if party.name.lower() in current_possessor.lower():
                    possession_score = 0.7  # Base score for current possession
                    
                    # Enhance based on possession period
                    if party.possession_period:
                        period_lower = party.possession_period.lower()
                        if "12 years" in period_lower or "adverse" in period_lower:
                            possession_score = 0.9
                        elif "continuous" in period_lower:
                            possession_score = 0.8
                    
                    possession_rights[party.name] = possession_score
                else:
                    possession_rights[party.name] = 0.2  # Not in possession
        else:
            # No clear possession information
            all_parties = [case_data.plaintiff_details] + case_data.defendant_details
            for party in all_parties:
                possession_rights[party.name] = 0.5  # Neutral
        
        return possession_rights
    
    async def _retrieve_legal_context(
        self,
        case_data: PropertyDisputeCase
    ) -> Dict[str, Any]:
        """Retrieve relevant legal context."""
        legal_context = {
            "precedents": [],
            "statutes": [],
            "similar_cases": []
        }
        
        # Search for relevant precedents
        precedent_query = f"""
        property dispute {case_data.dispute_type.value} 
        {case_data.property_details.property_type.value}
        title possession {case_data.property_details.state}
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
        legal_context["constitutional_provisions"] = self.constitutional_provisions
        legal_context["legal_principles"] = self.legal_principles
        
        return legal_context
    
    async def _perform_quantum_analysis(
        self,
        processed_data: PreprocessedText,
        legal_context: Dict[str, Any],
        case_data: PropertyDisputeCase
    ) -> Dict[str, Any]:
        """Perform quantum analysis of property dispute."""
        # Prepare quantum input
        quantum_input = {
            "query": processed_data.cleaned_text,
            "legal_concepts": processed_data.legal_concepts,
            "entities": [entity.text for entity in processed_data.entities],
            "precedents": legal_context["precedents"],
            "dispute_type": case_data.dispute_type.value,
            "property_value": case_data.property_details.market_value,
            "property_type": case_data.property_details.property_type.value
        }
        
        # Perform quantum reasoning
        quantum_results = await self.quantum_model.process_query(
            query=processed_data.cleaned_text,
            context=quantum_input,
            use_case="property_dispute"
        )
        
        # Extract property-specific quantum metrics
        quantum_analysis = {
            "outcome_probabilities": quantum_results.get("predictions", {}),
            "settlement_probability": quantum_results.get("settlement_likelihood", 0.5),
            "case_complexity": quantum_results.get("complexity_score", 0.5),
            "precedent_similarity": quantum_results.get("precedent_similarity", 0.0),
            "statutory_alignment": quantum_results.get("statutory_alignment", 0.0),
            "quantum_coherence": quantum_results.get("coherence", 0.0),
            "entanglement_measures": quantum_results.get("entanglement", {}),
            "risk_factors": quantum_results.get("risk_assessment", {})
        }
        
        return quantum_analysis
    
    async def _generate_dispute_analysis(
        self,
        case_data: PropertyDisputeCase,
        title_analysis: Dict[str, float],
        possession_analysis: Dict[str, float],
        quantum_results: Dict[str, Any],
        legal_context: Dict[str, Any]
    ) -> PropertyDisputeAnalysis:
        """Generate comprehensive dispute analysis."""
        
        # Calculate case outcome probabilities
        outcome_probabilities = {}
        all_parties = [case_data.plaintiff_details.name] + [d.name for d in case_data.defendant_details]
        
        for party in all_parties:
            title_strength = title_analysis.get(party, 0.0)
            possession_strength = possession_analysis.get(party, 0.0)
            
            # Combine title and possession (weighted)
            combined_strength = 0.6 * title_strength + 0.4 * possession_strength
            outcome_probabilities[party] = combined_strength
        
        # Normalize probabilities
        total_strength = sum(outcome_probabilities.values())
        if total_strength > 0:
            outcome_probabilities = {
                party: strength / total_strength 
                for party, strength in outcome_probabilities.items()
            }
        
        # Generate recommendations
        recommendations = []
        
        if case_data.case_stage == CaseStage.PRE_LITIGATION:
            recommendations.extend([
                "Conduct thorough title verification",
                "Collect all relevant revenue records",
                "Consider sending legal notice before filing suit",
                "Explore settlement possibilities"
            ])
        elif case_data.case_stage == CaseStage.SUIT_FILED:
            recommendations.extend([
                "File comprehensive written statement",
                "Prepare evidence list systematically",
                "Consider court-annexed mediation"
            ])
        
        # Add specific recommendations based on dispute type
        if case_data.dispute_type == DisputeType.TITLE_DISPUTE:
            recommendations.append("Focus on documentary evidence of title")
        elif case_data.dispute_type == DisputeType.PARTITION_SUIT:
            recommendations.append("Prepare detailed family tree and property genealogy")
        elif case_data.dispute_type == DisputeType.POSSESSION_MATTER:
            recommendations.append("Document current possession status with evidence")
        
        # Settlement options
        settlement_options = [
            "Monetary compensation for disputed share",
            "Partition by metes and bounds",
            "Sale and division of proceeds",
            "Exchange of properties",
            "Long-term lease arrangement"
        ]
        
        # Evidence gaps analysis
        evidence_gaps = []
        if not case_data.title_documents:
            evidence_gaps.append("Title documents missing")
        if not case_data.revenue_records:
            evidence_gaps.append("Revenue records not provided")
        if not case_data.survey_reports:
            evidence_gaps.append("Survey reports needed")
        
        # Estimate case duration
        base_duration = {"min_months": 12, "max_months": 36}
        if case_data.dispute_type in [DisputeType.TITLE_DISPUTE, DisputeType.PARTITION_SUIT]:
            base_duration = {"min_months": 18, "max_months": 60}
        
        # Estimate litigation costs
        property_value = case_data.property_details.market_value
        litigation_cost = property_value * 0.05  # 5% of property value as rough estimate
        
        # Extract similar cases
        similar_cases = [
            {
                "case_name": prec["case_name"],
                "similarity_score": prec["relevance_score"],
                "key_principle": prec.get("summary", "")
            }
            for prec in legal_context["precedents"][:5]
        ]
        
        return PropertyDisputeAnalysis(
            title_strength=title_analysis,
            possession_rights=possession_analysis,
            legal_validity={"title_documents_valid": True, "possession_lawful": True},
            case_outcome_probability=outcome_probabilities,
            settlement_probability=quantum_results.get("settlement_probability", 0.6),
            case_duration_estimate=base_duration,
            recommendations=recommendations,
            settlement_options=settlement_options,
            evidence_gaps=evidence_gaps,
            legal_strategies=["Documentary evidence strategy", "Witness examination plan"],
            litigation_cost_estimate=litigation_cost,
            property_valuation=property_value,
            potential_compensation=outcome_probabilities,
            quantum_confidence=quantum_results.get("quantum_coherence", 0.0),
            quantum_factors=quantum_results.get("risk_factors", {}),
            similar_cases=similar_cases,
            precedent_alignment=quantum_results.get("precedent_similarity", 0.0)
        )
    
    def _update_statistics(
        self,
        case_data: PropertyDisputeCase,
        analysis: PropertyDisputeAnalysis
    ) -> None:
        """Update processing statistics."""
        self.stats["cases_analyzed"] += 1
        
        # Update dispute type counts
        if case_data.dispute_type == DisputeType.TITLE_DISPUTE:
            self.stats["title_disputes"] += 1
        elif case_data.dispute_type == DisputeType.PARTITION_SUIT:
            self.stats["partition_suits"] += 1
        elif case_data.dispute_type == DisputeType.POSSESSION_MATTER:
            self.stats["possession_matters"] += 1
        
        # Update average case value
        current_avg = self.stats["average_case_value"]
        count = self.stats["cases_analyzed"]
        self.stats["average_case_value"] = (
            (current_avg * (count - 1) + case_data.property_details.market_value) / count
        )
        
        # Update settlement rate
        settlement_indicator = 1.0 if analysis.settlement_probability > 0.7 else 0.0
        current_settlement_rate = self.stats["settlement_rate"]
        self.stats["settlement_rate"] = (
            (current_settlement_rate * (count - 1) + settlement_indicator) / count
        )
    
    async def generate_property_dispute_response(
        self,
        case_data: PropertyDisputeCase,
        analysis: PropertyDisputeAnalysis,
        additional_context: Optional[str] = None
    ) -> LegalResponse:
        """Generate comprehensive property dispute response."""
        # Prepare quantum results for response generation
        quantum_results = {
            "predictions": [
                {
                    "party": party,
                    "success_probability": prob,
                    "title_strength": analysis.title_strength.get(party, 0.0),
                    "possession_rights": analysis.possession_rights.get(party, 0.0)
                }
                for party, prob in analysis.case_outcome_probability.items()
            ],
            "precedents": [
                {
                    "case_name": case["case_name"],
                    "summary": case["key_principle"],
                    "relevance_score": case["similarity_score"]
                }
                for case in analysis.similar_cases
            ],
            "statutes": [
                {
                    "name": "Transfer of Property Act, 1882",
                    "interpretation": "Governs property transfers and title"
                },
                {
                    "name": "Registration Act, 1908",
                    "interpretation": "Mandatory registration requirements"
                }
            ],
            "legal_concepts": list(analysis.quantum_factors.keys()),
            "confidence": analysis.quantum_confidence,
            "coherence": analysis.quantum_confidence,
            "metrics": {
                "settlement_probability": analysis.settlement_probability,
                "case_duration": analysis.case_duration_estimate,
                "litigation_cost": analysis.litigation_cost_estimate,
                **analysis.quantum_factors
            },
            "explanations": {
                "quantum_superposition": "Multiple property ownership scenarios evaluated simultaneously",
                "quantum_entanglement": "Complex relationships between title, possession, and legal rights analyzed"
            }
        }
        
        # Create query text
        query = f"Property dispute analysis for {case_data.property_details.property_type.value} property worth ₹{case_data.property_details.market_value:,.2f}"
        
        # Generate response
        response = await self.response_generator.generate_response(
            query=query,
            quantum_results=quantum_results,
            response_type=ResponseType.PROPERTY_DISPUTE,
            metadata={
                "case_id": case_data.case_id,
                "dispute_type": case_data.dispute_type.value,
                "property_value": case_data.property_details.market_value,
                "settlement_probability": analysis.settlement_probability,
                "strongest_party": max(analysis.case_outcome_probability.items(), key=lambda x: x[1])[0] if analysis.case_outcome_probability else "Unknown"
            }
        )
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get property dispute processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            "cases_analyzed": 0,
            "title_disputes": 0,
            "partition_suits": 0,
            "possession_matters": 0,
            "average_case_value": 0.0,
            "settlement_rate": 0.0
        }
    
    def estimate_property_value(
        self,
        property_details: PropertyDetails,
        market_factors: Optional[Dict[str, float]] = None
    ) -> float:
        """Estimate property value based on various factors."""
        base_value = property_details.market_value
        
        if not market_factors:
            return base_value
        
        # Apply market factors
        adjusted_value = base_value
        
        # Location factor
        location_factor = market_factors.get("location_premium", 1.0)
        adjusted_value *= location_factor
        
        # Property type factor
        type_factors = {
            PropertyType.COMMERCIAL: 1.2,
            PropertyType.RESIDENTIAL: 1.0,
            PropertyType.AGRICULTURAL: 0.8,
            PropertyType.INDUSTRIAL: 1.1,
            PropertyType.VACANT_LAND: 0.9
        }
        
        type_factor = type_factors.get(property_details.property_type, 1.0)
        adjusted_value *= type_factor
        
        # Market conditions
        market_condition = market_factors.get("market_condition", 1.0)
        adjusted_value *= market_condition
        
        return adjusted_value
    
    def calculate_stamp_duty(
        self,
        property_value: float,
        state: str,
        property_type: PropertyType
    ) -> Dict[str, float]:
        """Calculate stamp duty and registration charges."""
        # Basic stamp duty rates (varies by state)
        base_rates = {
            "delhi": 0.06,
            "mumbai": 0.05,
            "bangalore": 0.05,
            "chennai": 0.07,
            "hyderabad": 0.05,
            "pune": 0.05
        }
        
        state_lower = state.lower()
        base_rate = base_rates.get(state_lower, 0.05)  # Default 5%
        
        # Property type adjustments
        if property_type == PropertyType.AGRICULTURAL:
            base_rate *= 0.5  # Lower rate for agricultural land
        elif property_type == PropertyType.COMMERCIAL:
            base_rate *= 1.2  # Higher rate for commercial
        
        stamp_duty = property_value * base_rate
        registration_charges = property_value * 0.01  # 1% registration
        
        return {
            "stamp_duty": stamp_duty,
            "registration_charges": registration_charges,
            "total_charges": stamp_duty + registration_charges,
            "effective_rate": (stamp_duty + registration_charges) / property_value
        }