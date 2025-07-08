"""
Cheque Bounce Use Case Manager

This module handles the specific logic for cheque bounce cases under
Section 138 of the Negotiable Instruments Act using quantum-enhanced
legal reasoning. It implements the specialized workflow for dishonor
of cheques and related legal proceedings.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date, timedelta
import asyncio

import numpy as np
from loguru import logger

from ..core.quantum_legal_model import QuantumLegalModel
from ..classical.preprocessor import LegalTextPreprocessor, PreprocessedText
from ..classical.response_generator import LegalResponseGenerator, ResponseType, LegalResponse
from ..classical.knowledge_base import LegalKnowledgeBase, DocumentType
from ..utils.config import get_config


class ChequeType(Enum):
    """Types of cheques."""
    ACCOUNT_PAYEE = "account_payee"
    BEARER = "bearer"
    ORDER = "order"
    CROSSED = "crossed"
    POST_DATED = "post_dated"
    STALE = "stale"


class DishonorReason(Enum):
    """Reasons for cheque dishonor."""
    INSUFFICIENT_FUNDS = "insufficient_funds"
    ACCOUNT_CLOSED = "account_closed"
    SIGNATURE_MISMATCH = "signature_mismatch"
    AMOUNT_MISMATCH = "amount_mismatch"
    POST_DATED = "post_dated"
    STOP_PAYMENT = "stop_payment"
    FROZEN_ACCOUNT = "frozen_account"
    TECHNICAL_REASON = "technical_reason"


class CaseStage(Enum):
    """Stages of cheque bounce case."""
    PRE_LITIGATION = "pre_litigation"
    LEGAL_NOTICE = "legal_notice"
    COMPLAINT_FILED = "complaint_filed"
    TRIAL = "trial"
    JUDGMENT = "judgment"
    APPEAL = "appeal"


class LiabilityType(Enum):
    """Types of liability in cheque bounce."""
    CRIMINAL = "criminal"
    CIVIL = "civil"
    BOTH = "both"
    NONE = "none"


@dataclass
class ChequeDetails:
    """Details of the dishonored cheque."""
    cheque_number: str
    cheque_date: date
    amount: float
    bank_name: str
    account_number: str
    drawer_name: str
    payee_name: str
    cheque_type: ChequeType
    
    # Dishonor details
    dishonor_date: date
    dishonor_reason: DishonorReason
    bank_memo: str
    
    # Presentation details
    first_presentation_date: Optional[date] = None
    second_presentation_date: Optional[date] = None
    
    # Additional details
    purpose_of_cheque: Optional[str] = None
    underlying_transaction: Optional[str] = None
    consideration: Optional[str] = None


@dataclass
class LegalNoticeDetails:
    """Details of legal notice under Section 138."""
    notice_date: Optional[date] = None
    notice_served_date: Optional[date] = None
    service_method: Optional[str] = None
    response_received: bool = False
    response_date: Optional[date] = None
    response_details: Optional[str] = None
    
    # Statutory compliance
    within_30_days: Optional[bool] = None
    proper_service: Optional[bool] = None
    adequate_content: Optional[bool] = None


@dataclass
class ChequeBounceCase:
    """Complete cheque bounce case data."""
    case_id: str
    cheque_details: ChequeDetails
    legal_notice: LegalNoticeDetails
    
    # Parties
    complainant_name: str
    complainant_address: str
    accused_name: str
    accused_address: str
    
    # Case details
    case_stage: CaseStage
    court: Optional[str] = None
    case_number: Optional[str] = None
    filing_date: Optional[date] = None
    
    # Evidence
    supporting_documents: List[str] = None
    witness_details: List[str] = None
    
    # Financial details
    interest_claimed: Optional[float] = None
    compensation_claimed: Optional[float] = None
    
    # Legal representation
    complainant_lawyer: Optional[str] = None
    accused_lawyer: Optional[str] = None
    
    # Additional context
    previous_transactions: List[str] = None
    relationship_between_parties: Optional[str] = None
    
    def __post_init__(self):
        if self.supporting_documents is None:
            self.supporting_documents = []
        if self.witness_details is None:
            self.witness_details = []
        if self.previous_transactions is None:
            self.previous_transactions = []


@dataclass
class ChequeBounceAnalysis:
    """Analysis result for cheque bounce case."""
    liability_assessment: LiabilityType
    conviction_probability: float
    compensation_estimate: float
    
    # Legal compliance
    statutory_compliance: Dict[str, bool]
    procedural_compliance: Dict[str, bool]
    
    # Defenses available
    available_defenses: List[str]
    defense_strength: Dict[str, float]
    
    # Recommendations
    recommendations: List[str]
    next_steps: List[str]
    
    # Quantum analysis
    quantum_confidence: float
    quantum_factors: Dict[str, float]
    
    # Precedent analysis
    similar_cases: List[Dict[str, Any]]
    precedent_alignment: float


class ChequeBounceManager:
    """
    Specialized manager for cheque bounce cases under Section 138 NI Act.
    
    This class implements the complete workflow for analyzing cheque bounce
    cases in the Indian legal context, considering statutory requirements,
    procedural compliance, and case-specific factors.
    """
    
    def __init__(
        self,
        quantum_model: QuantumLegalModel,
        preprocessor: LegalTextPreprocessor,
        response_generator: LegalResponseGenerator,
        knowledge_base: LegalKnowledgeBase
    ):
        """
        Initialize the cheque bounce manager.
        
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
        
        # Load cheque bounce legal framework
        self._load_legal_framework()
        
        # Initialize statutory requirements
        self._initialize_statutory_requirements()
        
        # Load precedent patterns
        self._load_precedent_patterns()
        
        # Statistics
        self.stats = {
            "cases_analyzed": 0,
            "conviction_rate": 0.0,
            "average_compensation": 0.0,
            "compliance_rate": 0.0,
            "defense_success_rate": 0.0
        }
        
        logger.info("Cheque bounce manager initialized")
    
    def _load_legal_framework(self) -> None:
        """Load cheque bounce legal framework."""
        # Section 138 NI Act requirements
        self.section_138_elements = {
            "dishonor_of_cheque": {
                "description": "Cheque drawn by a person on an account maintained by him with a banker for payment of any amount of money to another person from out of that account for the discharge, in whole or in part, of any debt or other liability, is returned by the bank unpaid",
                "essential": True
            },
            "insufficient_funds": {
                "description": "Either because the amount of money standing to the credit of that account is insufficient to honour the cheque or that it exceeds the amount arranged to be paid from that account by an agreement made with that bank",
                "essential": True
            },
            "legal_notice": {
                "description": "The payee or the holder in due course of the cheque makes a demand for the payment of the said amount of money by giving a notice in writing, to the drawer of the cheque, within thirty days of the receipt of the information by him from the bank regarding the return of the cheque as unpaid",
                "essential": True
            },
            "failure_to_pay": {
                "description": "The drawer of such cheque fails to make the payment of the said amount of money to the payee or, as the case may be, to the holder in due course of the cheque, within fifteen days of the receipt of the said notice",
                "essential": True
            }
        }
        
        # Statutory provisions
        self.statutory_provisions = {
            "section_138": {
                "title": "Dishonour of cheque for insufficiency, etc., of funds in the account",
                "punishment": "Imprisonment up to two years or fine up to twice the amount of cheque or both",
                "nature": "Cognizable, bailable, non-compoundable"
            },
            "section_139": {
                "title": "Presumption in favour of holder",
                "description": "It shall be presumed, unless the contrary is proved, that the holder of a cheque received the cheque of the nature referred to in section 138 for the discharge, in whole or in part, of any debt or other liability"
            },
            "section_142": {
                "title": "Cognizance of offence",
                "description": "No court shall take cognizance of any offence punishable under section 138 except upon a complaint, in writing, made by the payee or, as the case may be, the holder in due course of the cheque"
            }
        }
        
        # Common defenses
        self.common_defenses = {
            "no_legally_enforceable_debt": {
                "description": "The cheque was not issued for discharge of any debt or liability",
                "strength": "high",
                "precedents": ["Kusum Ingots v. Pennar Peterson", "MSR Leathers v. S. Palaniappan"]
            },
            "improper_legal_notice": {
                "description": "Legal notice not served properly or within statutory period",
                "strength": "medium",
                "precedents": ["C.C. Alavi Haji v. Palapetty Muhammed", "Jagdish Singh v. Natthu Singh"]
            },
            "technical_dishonor": {
                "description": "Cheque dishonored for technical reasons not covered under Section 138",
                "strength": "medium",
                "precedents": ["Electronics Trade v. Indian Technologists", "Laxmi Dyechem v. State of Gujarat"]
            },
            "limitation": {
                "description": "Complaint filed beyond the limitation period",
                "strength": "high",
                "precedents": ["Saketh India v. India Securities", "Meters and Instruments v. Kanchan Mehta"]
            }
        }
    
    def _initialize_statutory_requirements(self) -> None:
        """Initialize statutory requirements and timelines."""
        self.statutory_timelines = {
            "legal_notice_period": 30,  # days from dishonor
            "payment_period": 15,  # days from notice receipt
            "complaint_filing_period": 30,  # days from payment period expiry
            "total_limitation_period": 365  # days from cause of action
        }
        
        # Compliance checklist
        self.compliance_checklist = {
            "cheque_requirements": [
                "Cheque properly filled and signed",
                "Cheque presented within validity period",
                "Sufficient consideration for cheque issuance"
            ],
            "dishonor_requirements": [
                "Dishonor for insufficient funds or exceeding arrangement",
                "Bank memo clearly stating reason for dishonor",
                "Dishonor memo received by payee"
            ],
            "notice_requirements": [
                "Legal notice served within 30 days of dishonor",
                "Notice contains all essential elements",
                "Proper service of notice with acknowledgment"
            ],
            "complaint_requirements": [
                "Complaint filed within limitation period",
                "All essential documents attached",
                "Proper court jurisdiction"
            ]
        }
    
    def _load_precedent_patterns(self) -> None:
        """Load patterns from cheque bounce precedents."""
        self.precedent_patterns = {
            "kumar_exports_v_sharma": {
                "principle": "Burden of proof shifts to accused under Section 139",
                "factors": ["presumption_of_liability", "burden_of_proof"],
                "quantum_pattern": "presumption_pattern"
            },
            "rangappa_v_mohan": {
                "principle": "Technical dishonor not covered under Section 138",
                "factors": ["dishonor_reason", "technical_issues"],
                "quantum_pattern": "technical_dishonor_pattern"
            },
            "meters_instruments_v_kanchan": {
                "principle": "Limitation period strictly enforced",
                "factors": ["limitation", "filing_delay"],
                "quantum_pattern": "limitation_pattern"
            }
        }
    
    async def analyze_cheque_bounce_case(
        self,
        case_data: ChequeBounceCase,
        additional_context: Optional[str] = None
    ) -> ChequeBounceAnalysis:
        """
        Analyze cheque bounce case using quantum-enhanced reasoning.
        
        Args:
            case_data: Cheque bounce case data
            additional_context: Additional context or arguments
            
        Returns:
            Comprehensive analysis of the case
        """
        logger.info(f"Analyzing cheque bounce case: {case_data.case_id}")
        
        try:
            # Step 1: Preprocess case data
            processed_data = await self._preprocess_case_data(case_data, additional_context)
            
            # Step 2: Check statutory compliance
            compliance_analysis = await self._check_statutory_compliance(case_data)
            
            # Step 3: Retrieve relevant precedents
            legal_context = await self._retrieve_legal_context(case_data)
            
            # Step 4: Perform quantum analysis
            quantum_results = await self._perform_quantum_analysis(
                processed_data, legal_context, case_data
            )
            
            # Step 5: Assess liability and defenses
            liability_analysis = await self._assess_liability_and_defenses(
                case_data, quantum_results, compliance_analysis
            )
            
            # Step 6: Generate comprehensive analysis
            analysis = await self._generate_case_analysis(
                case_data, quantum_results, compliance_analysis, 
                liability_analysis, legal_context
            )
            
            # Step 7: Update statistics
            self._update_statistics(analysis)
            
            logger.info(f"Cheque bounce analysis completed: {analysis.liability_assessment.value}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cheque bounce case: {e}")
            # Return default analysis
            return ChequeBounceAnalysis(
                liability_assessment=LiabilityType.NONE,
                conviction_probability=0.0,
                compensation_estimate=0.0,
                statutory_compliance={},
                procedural_compliance={},
                available_defenses=[],
                defense_strength={},
                recommendations=["Case analysis failed - manual review required"],
                next_steps=["Consult legal expert"],
                quantum_confidence=0.0,
                quantum_factors={},
                similar_cases=[],
                precedent_alignment=0.0
            )
    
    async def _preprocess_case_data(
        self,
        case_data: ChequeBounceCase,
        additional_context: Optional[str]
    ) -> PreprocessedText:
        """Preprocess cheque bounce case data."""
        # Combine all case information
        text_content = f"""
        Cheque Bounce Case Analysis: {case_data.case_id}
        
        Cheque Details:
        Cheque Number: {case_data.cheque_details.cheque_number}
        Date: {case_data.cheque_details.cheque_date}
        Amount: ₹{case_data.cheque_details.amount:,.2f}
        Bank: {case_data.cheque_details.bank_name}
        Drawer: {case_data.cheque_details.drawer_name}
        Payee: {case_data.cheque_details.payee_name}
        Cheque Type: {case_data.cheque_details.cheque_type.value}
        
        Dishonor Details:
        Dishonor Date: {case_data.cheque_details.dishonor_date}
        Reason: {case_data.cheque_details.dishonor_reason.value}
        Bank Memo: {case_data.cheque_details.bank_memo}
        
        Legal Notice:
        Notice Date: {case_data.legal_notice.notice_date or 'Not specified'}
        Service Date: {case_data.legal_notice.notice_served_date or 'Not specified'}
        Response Received: {case_data.legal_notice.response_received}
        
        Case Information:
        Stage: {case_data.case_stage.value}
        Court: {case_data.court or 'Not specified'}
        Filing Date: {case_data.filing_date or 'Not specified'}
        
        Parties:
        Complainant: {case_data.complainant_name}
        Accused: {case_data.accused_name}
        
        Purpose: {case_data.cheque_details.purpose_of_cheque or 'Not specified'}
        Underlying Transaction: {case_data.cheque_details.underlying_transaction or 'Not specified'}
        
        Supporting Documents: {', '.join(case_data.supporting_documents)}
        
        {additional_context or ''}
        """
        
        # Preprocess the text
        processed = await self.preprocessor.preprocess_text(
            text_content,
            document_type="cheque_bounce",
            metadata={
                "case_id": case_data.case_id,
                "amount": case_data.cheque_details.amount,
                "dishonor_reason": case_data.cheque_details.dishonor_reason.value,
                "case_stage": case_data.case_stage.value
            }
        )
        
        return processed
    
    async def _check_statutory_compliance(
        self,
        case_data: ChequeBounceCase
    ) -> Dict[str, Any]:
        """Check compliance with statutory requirements."""
        compliance = {
            "section_138_elements": {},
            "procedural_compliance": {},
            "timeline_compliance": {},
            "overall_compliance": 0.0
        }
        
        # Check Section 138 elements
        elements = compliance["section_138_elements"]
        
        # Element 1: Dishonor of cheque
        elements["dishonor_of_cheque"] = True  # Assumed if case exists
        
        # Element 2: Insufficient funds reason
        elements["insufficient_funds"] = case_data.cheque_details.dishonor_reason in [
            DishonorReason.INSUFFICIENT_FUNDS,
            DishonorReason.ACCOUNT_CLOSED
        ]
        
        # Element 3: Legal notice within 30 days
        if case_data.legal_notice.notice_date and case_data.cheque_details.dishonor_date:
            notice_delay = (case_data.legal_notice.notice_date - case_data.cheque_details.dishonor_date).days
            elements["legal_notice_timely"] = notice_delay <= 30
        else:
            elements["legal_notice_timely"] = False
        
        # Element 4: Failure to pay within 15 days
        if (case_data.legal_notice.notice_served_date and 
            not case_data.legal_notice.response_received):
            elements["failure_to_pay"] = True
        else:
            elements["failure_to_pay"] = case_data.legal_notice.response_received == False
        
        # Check procedural compliance
        procedural = compliance["procedural_compliance"]
        
        procedural["proper_legal_notice"] = case_data.legal_notice.proper_service or False
        procedural["adequate_notice_content"] = case_data.legal_notice.adequate_content or False
        procedural["proper_court_jurisdiction"] = True  # Assumed
        
        # Check timeline compliance
        timeline = compliance["timeline_compliance"]
        
        if case_data.filing_date and case_data.cheque_details.dishonor_date:
            filing_delay = (case_data.filing_date - case_data.cheque_details.dishonor_date).days
            timeline["within_limitation"] = filing_delay <= 365
        else:
            timeline["within_limitation"] = None
        
        # Calculate overall compliance
        all_checks = []
        for category in [elements, procedural, timeline]:
            for key, value in category.items():
                if isinstance(value, bool):
                    all_checks.append(value)
        
        if all_checks:
            compliance["overall_compliance"] = sum(all_checks) / len(all_checks)
        
        return compliance
    
    async def _retrieve_legal_context(
        self,
        case_data: ChequeBounceCase
    ) -> Dict[str, Any]:
        """Retrieve relevant legal context."""
        legal_context = {
            "precedents": [],
            "statutes": [],
            "similar_cases": []
        }
        
        # Search for relevant precedents
        precedent_query = f"""
        cheque bounce section 138 negotiable instruments act
        dishonor {case_data.cheque_details.dishonor_reason.value}
        amount {case_data.cheque_details.amount}
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
        legal_context["section_138_elements"] = self.section_138_elements
        legal_context["common_defenses"] = self.common_defenses
        
        return legal_context
    
    async def _perform_quantum_analysis(
        self,
        processed_data: PreprocessedText,
        legal_context: Dict[str, Any],
        case_data: ChequeBounceCase
    ) -> Dict[str, Any]:
        """Perform quantum analysis of cheque bounce case."""
        # Prepare quantum input
        quantum_input = {
            "query": processed_data.cleaned_text,
            "legal_concepts": processed_data.legal_concepts,
            "entities": [entity.text for entity in processed_data.entities],
            "precedents": legal_context["precedents"],
            "amount": case_data.cheque_details.amount,
            "dishonor_reason": case_data.cheque_details.dishonor_reason.value,
            "case_stage": case_data.case_stage.value
        }
        
        # Perform quantum reasoning
        quantum_results = await self.quantum_model.process_query(
            query=processed_data.cleaned_text,
            context=quantum_input,
            use_case="cheque_bounce"
        )
        
        # Extract cheque bounce specific metrics
        quantum_analysis = {
            "conviction_probability": quantum_results.get("predictions", {}).get("conviction_probability", 0.5),
            "compensation_estimate": quantum_results.get("predictions", {}).get("compensation_amount", 0.0),
            "defense_effectiveness": quantum_results.get("defense_analysis", {}),
            "precedent_similarity": quantum_results.get("precedent_similarity", 0.0),
            "statutory_alignment": quantum_results.get("statutory_alignment", 0.0),
            "quantum_coherence": quantum_results.get("coherence", 0.0),
            "entanglement_measures": quantum_results.get("entanglement", {}),
            "risk_factors": quantum_results.get("risk_assessment", {})
        }
        
        return quantum_analysis
    
    async def _assess_liability_and_defenses(
        self,
        case_data: ChequeBounceCase,
        quantum_results: Dict[str, Any],
        compliance_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess liability and available defenses."""
        liability_analysis = {
            "liability_type": LiabilityType.NONE,
            "conviction_probability": 0.0,
            "available_defenses": [],
            "defense_strength": {}
        }
        
        # Assess liability based on compliance
        compliance_score = compliance_analysis["overall_compliance"]
        
        if compliance_score >= 0.8:
            liability_analysis["liability_type"] = LiabilityType.CRIMINAL
            liability_analysis["conviction_probability"] = 0.8
        elif compliance_score >= 0.6:
            liability_analysis["liability_type"] = LiabilityType.CRIMINAL
            liability_analysis["conviction_probability"] = 0.6
        elif compliance_score >= 0.4:
            liability_analysis["liability_type"] = LiabilityType.CIVIL
            liability_analysis["conviction_probability"] = 0.3
        else:
            liability_analysis["liability_type"] = LiabilityType.NONE
            liability_analysis["conviction_probability"] = 0.1
        
        # Incorporate quantum probability
        quantum_conviction_prob = quantum_results.get("conviction_probability", 0.5)
        blended_probability = 0.6 * liability_analysis["conviction_probability"] + 0.4 * quantum_conviction_prob
        liability_analysis["conviction_probability"] = blended_probability
        
        # Assess available defenses
        defenses = []
        defense_strength = {}
        
        # Check for common defenses
        if not compliance_analysis["section_138_elements"].get("insufficient_funds", True):
            defenses.append("technical_dishonor")
            defense_strength["technical_dishonor"] = 0.8
        
        if not compliance_analysis["section_138_elements"].get("legal_notice_timely", True):
            defenses.append("improper_legal_notice")
            defense_strength["improper_legal_notice"] = 0.7
        
        if not compliance_analysis["timeline_compliance"].get("within_limitation", True):
            defenses.append("limitation")
            defense_strength["limitation"] = 0.9
        
        if not case_data.cheque_details.consideration:
            defenses.append("no_legally_enforceable_debt")
            defense_strength["no_legally_enforceable_debt"] = 0.6
        
        # Incorporate quantum defense analysis
        quantum_defenses = quantum_results.get("defense_effectiveness", {})
        for defense, strength in quantum_defenses.items():
            if defense not in defense_strength:
                defenses.append(defense)
            # Blend classical and quantum assessments
            classical_strength = defense_strength.get(defense, 0.5)
            defense_strength[defense] = 0.7 * classical_strength + 0.3 * strength
        
        liability_analysis["available_defenses"] = defenses
        liability_analysis["defense_strength"] = defense_strength
        
        return liability_analysis
    
    async def _generate_case_analysis(
        self,
        case_data: ChequeBounceCase,
        quantum_results: Dict[str, Any],
        compliance_analysis: Dict[str, Any],
        liability_analysis: Dict[str, Any],
        legal_context: Dict[str, Any]
    ) -> ChequeBounceAnalysis:
        """Generate comprehensive case analysis."""
        # Calculate compensation estimate
        base_amount = case_data.cheque_details.amount
        compensation_multiplier = 1.0
        
        if liability_analysis["conviction_probability"] > 0.7:
            compensation_multiplier = 2.0  # Maximum under Section 138
        elif liability_analysis["conviction_probability"] > 0.5:
            compensation_multiplier = 1.5
        
        compensation_estimate = base_amount * compensation_multiplier
        
        # Incorporate quantum compensation estimate
        quantum_compensation = quantum_results.get("compensation_estimate", 0.0)
        if quantum_compensation > 0:
            compensation_estimate = 0.6 * compensation_estimate + 0.4 * quantum_compensation
        
        # Generate recommendations
        recommendations = []
        
        if case_data.case_stage == CaseStage.PRE_LITIGATION:
            recommendations.extend([
                "Send legal notice under Section 138 within 30 days of dishonor",
                "Ensure proper service of notice with acknowledgment",
                "Maintain all original documents and bank memos"
            ])
        elif case_data.case_stage == CaseStage.LEGAL_NOTICE:
            recommendations.extend([
                "Wait for 15-day response period to expire",
                "File complaint within 30 days of non-payment",
                "Prepare comprehensive evidence list"
            ])
        elif case_data.case_stage == CaseStage.COMPLAINT_FILED:
            recommendations.extend([
                "Ensure all documents are properly filed",
                "Prepare for trial with witness statements",
                "Consider settlement negotiations"
            ])
        
        # Add defense-specific recommendations
        if liability_analysis["available_defenses"]:
            strongest_defense = max(
                liability_analysis["defense_strength"].items(),
                key=lambda x: x[1]
            )[0]
            recommendations.append(f"Focus on {strongest_defense} defense strategy")
        
        # Generate next steps
        next_steps = []
        
        if case_data.case_stage == CaseStage.PRE_LITIGATION:
            next_steps.extend([
                "Draft and serve legal notice",
                "Collect additional supporting documents",
                "Prepare for potential litigation"
            ])
        elif case_data.case_stage == CaseStage.TRIAL:
            next_steps.extend([
                "Present evidence systematically",
                "Examine witnesses effectively",
                "Address court queries promptly"
            ])
        
        # Extract similar cases
        similar_cases = [
            {
                "case_name": prec["case_name"],
                "similarity_score": prec["relevance_score"],
                "key_principle": prec.get("summary", "")
            }
            for prec in legal_context["precedents"][:5]
        ]
        
        return ChequeBounceAnalysis(
            liability_assessment=liability_analysis["liability_type"],
            conviction_probability=liability_analysis["conviction_probability"],
            compensation_estimate=compensation_estimate,
            statutory_compliance=compliance_analysis["section_138_elements"],
            procedural_compliance=compliance_analysis["procedural_compliance"],
            available_defenses=liability_analysis["available_defenses"],
            defense_strength=liability_analysis["defense_strength"],
            recommendations=recommendations,
            next_steps=next_steps,
            quantum_confidence=quantum_results.get("quantum_coherence", 0.0),
            quantum_factors=quantum_results.get("risk_factors", {}),
            similar_cases=similar_cases,
            precedent_alignment=quantum_results.get("precedent_similarity", 0.0)
        )
    
    def _update_statistics(self, analysis: ChequeBounceAnalysis) -> None:
        """Update processing statistics."""
        self.stats["cases_analyzed"] += 1
        
        # Update conviction rate
        current_rate = self.stats["conviction_rate"]
        count = self.stats["cases_analyzed"]
        conviction_indicator = 1.0 if analysis.conviction_probability > 0.5 else 0.0
        self.stats["conviction_rate"] = (
            (current_rate * (count - 1) + conviction_indicator) / count
        )
        
        # Update average compensation
        current_avg = self.stats["average_compensation"]
        self.stats["average_compensation"] = (
            (current_avg * (count - 1) + analysis.compensation_estimate) / count
        )
        
        # Update compliance rate
        compliance_score = sum(analysis.statutory_compliance.values()) / len(analysis.statutory_compliance) if analysis.statutory_compliance else 0.0
        current_compliance = self.stats["compliance_rate"]
        self.stats["compliance_rate"] = (
            (current_compliance * (count - 1) + compliance_score) / count
        )
        
        # Update defense success rate
        defense_success = 1.0 if analysis.available_defenses and max(analysis.defense_strength.values(), default=0.0) > 0.7 else 0.0
        current_defense_rate = self.stats["defense_success_rate"]
        self.stats["defense_success_rate"] = (
            (current_defense_rate * (count - 1) + defense_success) / count
        )
    
    async def generate_cheque_bounce_response(
        self,
        case_data: ChequeBounceCase,
        analysis: ChequeBounceAnalysis,
        additional_context: Optional[str] = None
    ) -> LegalResponse:
        """Generate comprehensive cheque bounce case response."""
        # Prepare quantum results for response generation
        quantum_results = {
            "predictions": [
                {
                    "outcome": analysis.liability_assessment.value,
                    "conviction_probability": analysis.conviction_probability,
                    "compensation_estimate": analysis.compensation_estimate
                }
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
                    "name": "Negotiable Instruments Act, 1881 - Section 138",
                    "interpretation": "Dishonour of cheque for insufficiency of funds"
                },
                {
                    "name": "Negotiable Instruments Act, 1881 - Section 139",
                    "interpretation": "Presumption in favour of holder"
                }
            ],
            "legal_concepts": list(analysis.quantum_factors.keys()),
            "confidence": analysis.quantum_confidence,
            "coherence": analysis.quantum_confidence,
            "metrics": {
                "conviction_probability": analysis.conviction_probability,
                "compensation_estimate": analysis.compensation_estimate,
                "precedent_alignment": analysis.precedent_alignment,
                **analysis.quantum_factors
            },
            "explanations": {
                "quantum_superposition": "Multiple case outcomes evaluated simultaneously",
                "quantum_entanglement": "Complex relationships between statutory elements analyzed"
            }
        }
        
        # Create query text
        query = f"Cheque bounce case analysis under Section 138 NI Act for amount ₹{case_data.cheque_details.amount:,.2f}"
        
        # Generate response
        response = await self.response_generator.generate_response(
            query=query,
            quantum_results=quantum_results,
            response_type=ResponseType.CHEQUE_BOUNCE,
            metadata={
                "case_id": case_data.case_id,
                "amount": case_data.cheque_details.amount,
                "dishonor_reason": case_data.cheque_details.dishonor_reason.value,
                "liability_assessment": analysis.liability_assessment.value,
                "conviction_probability": analysis.conviction_probability,
                "compensation_estimate": analysis.compensation_estimate
            }
        )
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cheque bounce case processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            "cases_analyzed": 0,
            "conviction_rate": 0.0,
            "average_compensation": 0.0,
            "compliance_rate": 0.0,
            "defense_success_rate": 0.0
        }
    
    def calculate_limitation_period(
        self,
        dishonor_date: date,
        notice_date: Optional[date] = None,
        payment_deadline: Optional[date] = None
    ) -> Dict[str, Any]:
        """Calculate limitation periods for cheque bounce case."""
        limitation_info = {
            "dishonor_date": dishonor_date,
            "notice_deadline": dishonor_date + timedelta(days=30),
            "payment_deadline": None,
            "complaint_deadline": None,
            "overall_limitation": dishonor_date + timedelta(days=365),
            "status": {}
        }
        
        # Calculate payment deadline
        if notice_date:
            limitation_info["payment_deadline"] = notice_date + timedelta(days=15)
            limitation_info["complaint_deadline"] = limitation_info["payment_deadline"] + timedelta(days=30)
        
        # Check current status
        today = date.today()
        
        limitation_info["status"]["notice_period_expired"] = today > limitation_info["notice_deadline"]
        
        if limitation_info["payment_deadline"]:
            limitation_info["status"]["payment_period_expired"] = today > limitation_info["payment_deadline"]
        
        if limitation_info["complaint_deadline"]:
            limitation_info["status"]["complaint_period_expired"] = today > limitation_info["complaint_deadline"]
        
        limitation_info["status"]["overall_limitation_expired"] = today > limitation_info["overall_limitation"]
        
        return limitation_info
    
    def estimate_case_duration(
        self,
        case_stage: CaseStage,
        court_type: str = "magistrate"
    ) -> Dict[str, Any]:
        """Estimate case duration based on current stage and court."""
        duration_estimates = {
            "magistrate": {
                CaseStage.PRE_LITIGATION: {"min_months": 1, "max_months": 3},
                CaseStage.LEGAL_NOTICE: {"min_months": 1, "max_months": 2},
                CaseStage.COMPLAINT_FILED: {"min_months": 6, "max_months": 18},
                CaseStage.TRIAL: {"min_months": 12, "max_months": 36},
                CaseStage.JUDGMENT: {"min_months": 1, "max_months": 3}
            },
            "sessions": {
                CaseStage.COMPLAINT_FILED: {"min_months": 8, "max_months": 24},
                CaseStage.TRIAL: {"min_months": 18, "max_months": 48},
                CaseStage.JUDGMENT: {"min_months": 2, "max_months": 6}
            }
        }
        
        court_estimates = duration_estimates.get(court_type.lower(), duration_estimates["magistrate"])
        stage_estimate = court_estimates.get(case_stage, {"min_months": 6, "max_months": 24})
        
        return {
            "current_stage": case_stage.value,
            "court_type": court_type,
            "estimated_duration": stage_estimate,
            "factors_affecting_duration": [
                "Court workload and backlog",
                "Complexity of case",
                "Availability of parties and witnesses",
                "Settlement negotiations",
                "Appeals and revisions"
            ]
        }