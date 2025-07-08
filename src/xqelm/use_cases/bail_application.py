"""
Bail Application Use Case Manager

This module handles the specific logic for bail application analysis
using quantum-enhanced legal reasoning. It implements the specialized
workflow for Indian bail law including anticipatory bail, regular bail,
and interim bail applications.

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


class BailType(Enum):
    """Types of bail applications."""
    ANTICIPATORY_BAIL = "anticipatory_bail"
    REGULAR_BAIL = "regular_bail"
    INTERIM_BAIL = "interim_bail"
    STATUTORY_BAIL = "statutory_bail"
    DEFAULT_BAIL = "default_bail"


class OffenseCategory(Enum):
    """Categories of offenses for bail consideration."""
    BAILABLE = "bailable"
    NON_BAILABLE = "non_bailable"
    COGNIZABLE = "cognizable"
    NON_COGNIZABLE = "non_cognizable"
    HEINOUS = "heinous"
    ECONOMIC = "economic"
    WHITE_COLLAR = "white_collar"


class BailFactor(Enum):
    """Factors considered in bail decisions."""
    FLIGHT_RISK = "flight_risk"
    TAMPERING_EVIDENCE = "tampering_evidence"
    INFLUENCING_WITNESSES = "influencing_witnesses"
    REPEAT_OFFENSE = "repeat_offense"
    SEVERITY_OF_OFFENSE = "severity_of_offense"
    ROOTS_IN_SOCIETY = "roots_in_society"
    EMPLOYMENT_STATUS = "employment_status"
    FAMILY_TIES = "family_ties"
    HEALTH_CONDITION = "health_condition"
    AGE_FACTOR = "age_factor"
    COOPERATION_WITH_INVESTIGATION = "cooperation_with_investigation"


@dataclass
class BailApplicationData:
    """Data structure for bail application analysis."""
    applicant_name: str
    case_number: Optional[str]
    court: str
    offense_details: str
    sections_charged: List[str]
    offense_category: OffenseCategory
    bail_type: BailType
    
    # Applicant details
    age: Optional[int]
    occupation: Optional[str]
    address: str
    family_details: Optional[str]
    previous_convictions: List[str]
    
    # Case details
    date_of_offense: Optional[date]
    date_of_arrest: Optional[date]
    investigation_status: str
    evidence_status: str
    
    # Bail factors
    flight_risk_assessment: Optional[float]
    tampering_risk: Optional[float]
    influence_risk: Optional[float]
    
    # Supporting documents
    supporting_documents: List[str]
    character_witnesses: List[str]
    
    # Additional information
    medical_condition: Optional[str]
    employment_verification: Optional[str]
    property_details: Optional[str]
    
    # Quantum analysis results
    quantum_factors: Optional[Dict[str, float]] = None
    precedent_similarity: Optional[float] = None


@dataclass
class BailRecommendation:
    """Bail recommendation with reasoning."""
    recommendation: str  # "grant", "deny", "conditional"
    confidence: float
    reasoning: str
    conditions: List[str]
    precedents: List[str]
    statutory_basis: List[str]
    risk_assessment: Dict[str, float]
    quantum_analysis: Dict[str, Any]


class BailApplicationManager:
    """
    Specialized manager for bail application analysis using quantum-enhanced reasoning.
    
    This class implements the complete workflow for analyzing bail applications
    in the Indian legal context, considering statutory provisions, precedents,
    and case-specific factors.
    """
    
    def __init__(
        self,
        quantum_model: QuantumLegalModel,
        preprocessor: LegalTextPreprocessor,
        response_generator: LegalResponseGenerator,
        knowledge_base: LegalKnowledgeBase
    ):
        """
        Initialize the bail application manager.
        
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
        
        # Load bail-specific legal framework
        self._load_bail_legal_framework()
        
        # Initialize bail factors and weights
        self._initialize_bail_factors()
        
        # Load precedent patterns
        self._load_precedent_patterns()
        
        # Statistics
        self.stats = {
            "applications_processed": 0,
            "recommendations_generated": 0,
            "average_confidence": 0.0,
            "grant_rate": 0.0,
            "denial_rate": 0.0
        }
        
        logger.info("Bail application manager initialized")
    
    def _load_bail_legal_framework(self) -> None:
        """Load bail-specific legal framework."""
        # Key statutory provisions for bail in India
        self.statutory_provisions = {
            "crpc_section_436": {
                "title": "In what cases bail to be taken",
                "text": "When any person other than a person accused of a non-bailable offence is arrested or detained without warrant...",
                "applicability": "bailable_offenses"
            },
            "crpc_section_437": {
                "title": "When bail may be taken in case of non-bailable offence",
                "text": "When any person accused of a non-bailable offence is arrested or detained without warrant...",
                "applicability": "non_bailable_offenses"
            },
            "crpc_section_438": {
                "title": "Direction for grant of bail to person apprehending arrest",
                "text": "When any person has reason to believe that he may be arrested on accusation of having committed a non-bailable offence...",
                "applicability": "anticipatory_bail"
            },
            "crpc_section_439": {
                "title": "Special powers of High Court or Court of Session regarding bail",
                "text": "A High Court or Court of Session may direct that any person accused of an offence and in custody be released on bail...",
                "applicability": "special_powers"
            }
        }
        
        # Constitutional provisions
        self.constitutional_provisions = {
            "article_21": {
                "title": "Protection of life and personal liberty",
                "text": "No person shall be deprived of his life or personal liberty except according to procedure established by law",
                "relevance": "fundamental_right_to_liberty"
            },
            "article_22": {
                "title": "Protection against arrest and detention in certain cases",
                "text": "No person who is arrested shall be detained in custody without being informed...",
                "relevance": "arrest_and_detention_safeguards"
            }
        }
        
        # Key legal principles
        self.legal_principles = {
            "bail_not_jail": "Bail is the rule, jail is the exception",
            "presumption_of_innocence": "Every person is presumed innocent until proven guilty",
            "right_to_speedy_trial": "Prolonged detention without trial violates fundamental rights",
            "triple_test": "Bail decisions must consider flight risk, tampering with evidence, and influencing witnesses"
        }
    
    def _initialize_bail_factors(self) -> None:
        """Initialize bail factors and their weights."""
        # Factor weights for different types of bail
        self.factor_weights = {
            BailType.ANTICIPATORY_BAIL: {
                BailFactor.FLIGHT_RISK: 0.25,
                BailFactor.TAMPERING_EVIDENCE: 0.20,
                BailFactor.INFLUENCING_WITNESSES: 0.15,
                BailFactor.SEVERITY_OF_OFFENSE: 0.15,
                BailFactor.ROOTS_IN_SOCIETY: 0.10,
                BailFactor.COOPERATION_WITH_INVESTIGATION: 0.10,
                BailFactor.REPEAT_OFFENSE: 0.05
            },
            BailType.REGULAR_BAIL: {
                BailFactor.FLIGHT_RISK: 0.20,
                BailFactor.TAMPERING_EVIDENCE: 0.18,
                BailFactor.INFLUENCING_WITNESSES: 0.15,
                BailFactor.SEVERITY_OF_OFFENSE: 0.12,
                BailFactor.ROOTS_IN_SOCIETY: 0.12,
                BailFactor.EMPLOYMENT_STATUS: 0.08,
                BailFactor.FAMILY_TIES: 0.08,
                BailFactor.HEALTH_CONDITION: 0.07
            },
            BailType.INTERIM_BAIL: {
                BailFactor.HEALTH_CONDITION: 0.30,
                BailFactor.AGE_FACTOR: 0.20,
                BailFactor.FAMILY_TIES: 0.15,
                BailFactor.FLIGHT_RISK: 0.15,
                BailFactor.SEVERITY_OF_OFFENSE: 0.10,
                BailFactor.COOPERATION_WITH_INVESTIGATION: 0.10
            }
        }
        
        # Offense severity mapping
        self.offense_severity = {
            "murder": 0.9,
            "rape": 0.9,
            "kidnapping": 0.8,
            "robbery": 0.7,
            "cheating": 0.4,
            "theft": 0.3,
            "assault": 0.5,
            "fraud": 0.6,
            "corruption": 0.7,
            "drug_trafficking": 0.8,
            "terrorism": 1.0,
            "economic_offense": 0.6
        }
    
    def _load_precedent_patterns(self) -> None:
        """Load patterns from bail precedents."""
        # Key precedent patterns for quantum analysis
        self.precedent_patterns = {
            "sanjay_chandra_v_cbi": {
                "principle": "Economic offenses require different bail considerations",
                "factors": ["white_collar_crime", "economic_impact", "flight_risk"],
                "quantum_pattern": "economic_offense_pattern"
            },
            "arnesh_kumar_v_state": {
                "principle": "Arrest should not be automatic in cognizable cases",
                "factors": ["necessity_of_arrest", "gravity_of_offense"],
                "quantum_pattern": "arrest_necessity_pattern"
            },
            "dataram_singh_v_state": {
                "principle": "Prolonged detention without trial violates Article 21",
                "factors": ["detention_period", "trial_delay", "fundamental_rights"],
                "quantum_pattern": "prolonged_detention_pattern"
            }
        }
    
    async def analyze_bail_application(
        self,
        application_data: BailApplicationData,
        additional_context: Optional[str] = None
    ) -> BailRecommendation:
        """
        Analyze bail application using quantum-enhanced reasoning.
        
        Args:
            application_data: Bail application data
            additional_context: Additional context or arguments
            
        Returns:
            Bail recommendation with detailed analysis
        """
        logger.info(f"Analyzing bail application for {application_data.applicant_name}")
        
        try:
            # Step 1: Preprocess application data
            processed_data = await self._preprocess_application_data(
                application_data, additional_context
            )
            
            # Step 2: Retrieve relevant precedents and statutes
            legal_context = await self._retrieve_legal_context(application_data)
            
            # Step 3: Perform quantum analysis
            quantum_results = await self._perform_quantum_analysis(
                processed_data, legal_context, application_data
            )
            
            # Step 4: Assess bail factors
            factor_assessment = await self._assess_bail_factors(
                application_data, quantum_results
            )
            
            # Step 5: Generate recommendation
            recommendation = await self._generate_bail_recommendation(
                application_data, quantum_results, factor_assessment, legal_context
            )
            
            # Step 6: Update statistics
            self._update_statistics(recommendation)
            
            logger.info(f"Bail analysis completed: {recommendation.recommendation}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error analyzing bail application: {e}")
            # Return default recommendation
            return BailRecommendation(
                recommendation="review_required",
                confidence=0.0,
                reasoning="Analysis failed due to technical error",
                conditions=[],
                precedents=[],
                statutory_basis=[],
                risk_assessment={},
                quantum_analysis={}
            )
    
    async def _preprocess_application_data(
        self,
        application_data: BailApplicationData,
        additional_context: Optional[str]
    ) -> PreprocessedText:
        """Preprocess bail application data."""
        # Combine all textual information
        text_content = f"""
        Bail Application for {application_data.applicant_name}
        
        Case Details:
        Case Number: {application_data.case_number or 'Not specified'}
        Court: {application_data.court}
        Offense: {application_data.offense_details}
        Sections: {', '.join(application_data.sections_charged)}
        Offense Category: {application_data.offense_category.value}
        Bail Type: {application_data.bail_type.value}
        
        Applicant Details:
        Age: {application_data.age or 'Not specified'}
        Occupation: {application_data.occupation or 'Not specified'}
        Address: {application_data.address}
        Family Details: {application_data.family_details or 'Not specified'}
        Previous Convictions: {', '.join(application_data.previous_convictions) if application_data.previous_convictions else 'None'}
        
        Case Timeline:
        Date of Offense: {application_data.date_of_offense or 'Not specified'}
        Date of Arrest: {application_data.date_of_arrest or 'Not specified'}
        Investigation Status: {application_data.investigation_status}
        Evidence Status: {application_data.evidence_status}
        
        Supporting Information:
        Supporting Documents: {', '.join(application_data.supporting_documents)}
        Character Witnesses: {', '.join(application_data.character_witnesses)}
        Medical Condition: {application_data.medical_condition or 'None'}
        Employment Verification: {application_data.employment_verification or 'Not provided'}
        Property Details: {application_data.property_details or 'Not provided'}
        
        {additional_context or ''}
        """
        
        # Preprocess the text
        processed = await self.preprocessor.preprocess_text(
            text_content,
            document_type="bail_application",
            metadata={
                "applicant": application_data.applicant_name,
                "bail_type": application_data.bail_type.value,
                "offense_category": application_data.offense_category.value
            }
        )
        
        return processed
    
    async def _retrieve_legal_context(
        self,
        application_data: BailApplicationData
    ) -> Dict[str, Any]:
        """Retrieve relevant legal context for bail analysis."""
        legal_context = {
            "precedents": [],
            "statutes": [],
            "principles": [],
            "similar_cases": []
        }
        
        # Search for relevant precedents
        precedent_query = f"""
        bail {application_data.bail_type.value} {application_data.offense_category.value}
        {' '.join(application_data.sections_charged)} {application_data.offense_details}
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
                "relevance_score": result.relevance_score,
                "similarity_score": result.similarity_score
            }
            for result in precedent_results
        ]
        
        # Search for relevant statutes
        statute_query = f"bail {' '.join(application_data.sections_charged)} CrPC"
        
        statute_results = await self.knowledge_base.search_documents(
            query=statute_query,
            document_types=[DocumentType.STATUTE],
            jurisdiction="india",
            limit=5
        )
        
        legal_context["statutes"] = [
            {
                "title": result.document.title,
                "content": result.document.content,
                "relevance_score": result.relevance_score
            }
            for result in statute_results
        ]
        
        # Add statutory provisions
        legal_context["statutory_provisions"] = self.statutory_provisions
        legal_context["constitutional_provisions"] = self.constitutional_provisions
        legal_context["legal_principles"] = self.legal_principles
        
        return legal_context
    
    async def _perform_quantum_analysis(
        self,
        processed_data: PreprocessedText,
        legal_context: Dict[str, Any],
        application_data: BailApplicationData
    ) -> Dict[str, Any]:
        """Perform quantum analysis of bail application."""
        # Prepare quantum input
        quantum_input = {
            "query": processed_data.cleaned_text,
            "legal_concepts": processed_data.legal_concepts,
            "entities": [entity.text for entity in processed_data.entities],
            "precedents": legal_context["precedents"],
            "statutes": legal_context["statutes"],
            "bail_type": application_data.bail_type.value,
            "offense_category": application_data.offense_category.value
        }
        
        # Perform quantum reasoning
        quantum_results = await self.quantum_model.process_query(
            query=processed_data.cleaned_text,
            context=quantum_input,
            use_case="bail_application"
        )
        
        # Extract bail-specific quantum metrics
        quantum_analysis = {
            "bail_probability": quantum_results.get("predictions", {}).get("bail_grant_probability", 0.5),
            "risk_factors": quantum_results.get("risk_assessment", {}),
            "precedent_alignment": quantum_results.get("precedent_similarity", 0.0),
            "statutory_compliance": quantum_results.get("statutory_alignment", 0.0),
            "quantum_coherence": quantum_results.get("coherence", 0.0),
            "entanglement_measures": quantum_results.get("entanglement", {}),
            "interference_patterns": quantum_results.get("interference", {})
        }
        
        return quantum_analysis
    
    async def _assess_bail_factors(
        self,
        application_data: BailApplicationData,
        quantum_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess individual bail factors."""
        factor_scores = {}
        weights = self.factor_weights.get(application_data.bail_type, {})
        
        # Flight risk assessment
        flight_risk = 0.5  # Default neutral
        if application_data.roots_in_society:
            flight_risk -= 0.2
        if application_data.employment_verification:
            flight_risk -= 0.1
        if application_data.property_details:
            flight_risk -= 0.1
        if application_data.previous_convictions:
            flight_risk += 0.2 * len(application_data.previous_convictions)
        
        factor_scores[BailFactor.FLIGHT_RISK.value] = max(0.0, min(1.0, flight_risk))
        
        # Evidence tampering risk
        tampering_risk = 0.5
        if application_data.investigation_status == "completed":
            tampering_risk -= 0.3
        elif application_data.investigation_status == "ongoing":
            tampering_risk += 0.2
        
        factor_scores[BailFactor.TAMPERING_EVIDENCE.value] = max(0.0, min(1.0, tampering_risk))
        
        # Witness influence risk
        influence_risk = 0.5
        if "witness_protection" in application_data.evidence_status.lower():
            influence_risk -= 0.2
        
        factor_scores[BailFactor.INFLUENCING_WITNESSES.value] = max(0.0, min(1.0, influence_risk))
        
        # Severity of offense
        offense_severity = 0.5
        for offense_type, severity in self.offense_severity.items():
            if offense_type in application_data.offense_details.lower():
                offense_severity = max(offense_severity, severity)
        
        factor_scores[BailFactor.SEVERITY_OF_OFFENSE.value] = offense_severity
        
        # Roots in society
        roots_score = 0.5
        if application_data.family_details:
            roots_score += 0.2
        if application_data.employment_verification:
            roots_score += 0.2
        if application_data.property_details:
            roots_score += 0.1
        
        factor_scores[BailFactor.ROOTS_IN_SOCIETY.value] = min(1.0, roots_score)
        
        # Age factor
        age_factor = 0.5
        if application_data.age:
            if application_data.age < 25:
                age_factor = 0.3  # Young age favors bail
            elif application_data.age > 65:
                age_factor = 0.2  # Senior citizen favors bail
        
        factor_scores[BailFactor.AGE_FACTOR.value] = age_factor
        
        # Health condition
        health_factor = 0.5
        if application_data.medical_condition:
            if any(condition in application_data.medical_condition.lower() 
                   for condition in ["serious", "chronic", "terminal", "critical"]):
                health_factor = 0.1  # Serious health issues favor bail
        
        factor_scores[BailFactor.HEALTH_CONDITION.value] = health_factor
        
        # Incorporate quantum insights
        quantum_risk_factors = quantum_results.get("risk_factors", {})
        for factor, quantum_score in quantum_risk_factors.items():
            if factor in factor_scores:
                # Blend classical and quantum assessments
                classical_score = factor_scores[factor]
                blended_score = 0.7 * classical_score + 0.3 * quantum_score
                factor_scores[factor] = blended_score
        
        return factor_scores
    
    async def _generate_bail_recommendation(
        self,
        application_data: BailApplicationData,
        quantum_results: Dict[str, Any],
        factor_assessment: Dict[str, float],
        legal_context: Dict[str, Any]
    ) -> BailRecommendation:
        """Generate final bail recommendation."""
        # Calculate overall bail score
        weights = self.factor_weights.get(application_data.bail_type, {})
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor, weight in weights.items():
            if factor.value in factor_assessment:
                weighted_score += weight * (1.0 - factor_assessment[factor.value])  # Lower risk = higher bail score
                total_weight += weight
        
        if total_weight > 0:
            bail_score = weighted_score / total_weight
        else:
            bail_score = 0.5
        
        # Incorporate quantum probability
        quantum_bail_prob = quantum_results.get("bail_probability", 0.5)
        final_score = 0.6 * bail_score + 0.4 * quantum_bail_prob
        
        # Determine recommendation
        if final_score >= 0.7:
            recommendation = "grant"
        elif final_score >= 0.4:
            recommendation = "conditional"
        else:
            recommendation = "deny"
        
        # Generate conditions for conditional bail
        conditions = []
        if recommendation in ["grant", "conditional"]:
            if factor_assessment.get(BailFactor.FLIGHT_RISK.value, 0) > 0.6:
                conditions.append("Surrender passport and not to leave the jurisdiction")
            
            if factor_assessment.get(BailFactor.TAMPERING_EVIDENCE.value, 0) > 0.6:
                conditions.append("Not to tamper with evidence or contact witnesses")
            
            if factor_assessment.get(BailFactor.INFLUENCING_WITNESSES.value, 0) > 0.6:
                conditions.append("Not to directly or indirectly contact prosecution witnesses")
            
            # Standard conditions
            conditions.extend([
                "Furnish personal bond and surety as directed by the court",
                "Appear before the investigating officer as and when required",
                "Not to commit any offense while on bail"
            ])
        
        # Generate reasoning
        reasoning_parts = []
        
        if recommendation == "grant":
            reasoning_parts.append(f"The quantum analysis indicates a {final_score:.1%} probability favoring bail grant.")
            reasoning_parts.append("The applicant demonstrates strong roots in society and low flight risk.")
        elif recommendation == "conditional":
            reasoning_parts.append(f"While there are some concerns, conditional bail is recommended with {final_score:.1%} confidence.")
            reasoning_parts.append("Appropriate conditions can mitigate identified risks.")
        else:
            reasoning_parts.append(f"The analysis indicates significant risks with only {final_score:.1%} confidence for bail.")
            reasoning_parts.append("The severity of offense and risk factors weigh against bail grant.")
        
        # Add quantum insights
        if quantum_results.get("quantum_coherence", 0) > 0.8:
            reasoning_parts.append("The quantum analysis shows high coherence in legal reasoning patterns.")
        
        reasoning = " ".join(reasoning_parts)
        
        # Extract precedents and statutory basis
        precedents = [
            f"{prec['case_name']} - {prec.get('summary', 'Relevant precedent')}"
            for prec in legal_context["precedents"][:3]
        ]
        
        statutory_basis = [
            "Section 436, CrPC - Bail in bailable offenses",
            "Section 437, CrPC - Bail in non-bailable offenses",
            "Article 21, Constitution - Right to life and liberty"
        ]
        
        if application_data.bail_type == BailType.ANTICIPATORY_BAIL:
            statutory_basis.append("Section 438, CrPC - Anticipatory bail")
        
        return BailRecommendation(
            recommendation=recommendation,
            confidence=final_score,
            reasoning=reasoning,
            conditions=conditions,
            precedents=precedents,
            statutory_basis=statutory_basis,
            risk_assessment=factor_assessment,
            quantum_analysis=quantum_results
        )
    
    def _update_statistics(self, recommendation: BailRecommendation) -> None:
        """Update processing statistics."""
        self.stats["applications_processed"] += 1
        self.stats["recommendations_generated"] += 1
        
        # Update average confidence
        current_avg = self.stats["average_confidence"]
        count = self.stats["applications_processed"]
        self.stats["average_confidence"] = (
            (current_avg * (count - 1) + recommendation.confidence) / count
        )
        
        # Update grant/denial rates
        if recommendation.recommendation == "grant":
            self.stats["grant_rate"] = (
                (self.stats["grant_rate"] * (count - 1) + 1.0) / count
            )
        elif recommendation.recommendation == "deny":
            self.stats["denial_rate"] = (
                (self.stats["denial_rate"] * (count - 1) + 1.0) / count
            )
    
    async def generate_bail_response(
        self,
        application_data: BailApplicationData,
        recommendation: BailRecommendation,
        additional_context: Optional[str] = None
    ) -> LegalResponse:
        """Generate comprehensive bail application response."""
        # Prepare quantum results for response generation
        quantum_results = {
            "predictions": [{"outcome": recommendation.recommendation, "confidence": recommendation.confidence}],
            "precedents": [{"case_name": prec.split(" - ")[0], "summary": prec.split(" - ")[1] if " - " in prec else prec} for prec in recommendation.precedents],
            "statutes": [{"name": statute, "interpretation": "Applicable to bail proceedings"} for statute in recommendation.statutory_basis],
            "legal_concepts": list(recommendation.risk_assessment.keys()),
            "confidence": recommendation.confidence,
            "coherence": recommendation.quantum_analysis.get("quantum_coherence", 0.0),
            "metrics": recommendation.quantum_analysis,
            "explanations": {
                "quantum_superposition": "Multiple bail scenarios evaluated simultaneously",
                "quantum_entanglement": "Complex relationships between legal factors analyzed"
            }
        }
        
        # Create query text
        query = f"Bail application analysis for {application_data.applicant_name} charged under {', '.join(application_data.sections_charged)}"
        
        # Generate response
        response = await self.response_generator.generate_response(
            query=query,
            quantum_results=quantum_results,
            response_type=ResponseType.BAIL_APPLICATION,
            metadata={
                "applicant": application_data.applicant_name,
                "bail_type": application_data.bail_type.value,
                "recommendation": recommendation.recommendation,
                "conditions": recommendation.conditions
            }
        )
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bail application processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            "applications_processed": 0,
            "recommendations_generated": 0,
            "average_confidence": 0.0,
            "grant_rate": 0.0,
            "denial_rate": 0.0
        }