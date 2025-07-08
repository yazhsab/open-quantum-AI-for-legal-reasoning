"""
GST Dispute Resolution Use Case Manager

This module handles GST (Goods and Services Tax) dispute analysis and resolution
recommendations for the Indian tax system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging
from datetime import datetime, timedelta

from ..core.quantum_legal_model import QuantumLegalModel
from ..classical.preprocessor import LegalTextPreprocessor
from ..classical.knowledge_base import LegalKnowledgeBase
from ..quantum.embeddings import QuantumLegalEmbedding
from ..quantum.reasoning import QuantumLegalReasoningCircuit
from ..core.explainability import QuantumExplainabilityModule

logger = logging.getLogger(__name__)


class GSTDisputeType(Enum):
    """Types of GST disputes"""
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


class GSTTaxRate(Enum):
    """GST tax rates"""
    EXEMPT = 0.0
    GST_5 = 5.0
    GST_12 = 12.0
    GST_18 = 18.0
    GST_28 = 28.0


class GSTForum(Enum):
    """GST dispute resolution forums"""
    ADJUDICATING_AUTHORITY = "adjudicating_authority"
    APPELLATE_AUTHORITY = "appellate_authority"
    APPELLATE_TRIBUNAL = "appellate_tribunal"
    HIGH_COURT = "high_court"
    SUPREME_COURT = "supreme_court"


class BusinessType(Enum):
    """Types of businesses"""
    MANUFACTURER = "manufacturer"
    TRADER = "trader"
    SERVICE_PROVIDER = "service_provider"
    E_COMMERCE = "e_commerce"
    IMPORTER = "importer"
    EXPORTER = "exporter"
    COMPOSITION_DEALER = "composition_dealer"
    REGULAR_DEALER = "regular_dealer"


@dataclass
class GSTDisputeCase:
    """GST dispute case data structure"""
    case_id: str
    taxpayer_name: str
    gstin: str
    dispute_type: GSTDisputeType
    business_type: BusinessType
    dispute_description: str
    disputed_amount: float
    tax_period: str
    notice_date: Optional[datetime] = None
    response_deadline: Optional[datetime] = None
    
    # Transaction details
    transaction_value: Optional[float] = None
    tax_rate_claimed: Optional[GSTTaxRate] = None
    tax_rate_demanded: Optional[GSTTaxRate] = None
    
    # Supporting documents
    evidence_documents: List[str] = field(default_factory=list)
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
    similar_cases_precedent: List[str] = field(default_factory=list)
    department_position: Optional[str] = None
    taxpayer_position: Optional[str] = None


@dataclass
class GSTDisputeAnalysis:
    """GST dispute analysis results"""
    case_id: str
    success_probability: float
    
    # Legal analysis
    legal_position_strength: Dict[str, float]
    applicable_provisions: List[str]
    relevant_precedents: List[Dict[str, Any]]
    
    # Financial analysis
    estimated_liability: Dict[str, float]
    penalty_assessment: Dict[str, float]
    interest_calculation: Dict[str, float]
    
    # Procedural analysis
    limitation_compliance: bool
    forum_jurisdiction: GSTForum
    appeal_options: List[str]
    
    # Recommendations
    recommended_strategy: List[str]
    settlement_options: List[Dict[str, Any]]
    documentation_requirements: List[str]
    
    # Risk assessment
    litigation_risk: float
    compliance_risk: float
    financial_risk: float
    
    # Timeline
    estimated_resolution_time: int  # in days
    critical_deadlines: List[Dict[str, Any]]
    
    # Quantum analysis
    quantum_confidence: float
    quantum_explanation: Dict[str, Any]


class GSTDisputeManager:
    """Manager for GST dispute analysis and resolution"""
    
    def __init__(self):
        self.quantum_model = QuantumLegalModel()
        self.preprocessor = LegalTextPreprocessor()
        self.knowledge_base = LegalKnowledgeBase()
        self.quantum_embedding = QuantumLegalEmbedding()
        self.quantum_reasoning = QuantumLegalReasoningCircuit()
        self.explainability = QuantumExplainabilityModule()
        
        # GST-specific knowledge
        self.gst_provisions = self._load_gst_provisions()
        self.gst_rates = self._load_gst_rates()
        self.precedent_database = self._load_gst_precedents()
    
    def analyze_gst_dispute(self, case: GSTDisputeCase) -> GSTDisputeAnalysis:
        """
        Analyze a GST dispute case and provide comprehensive recommendations
        
        Args:
            case: GSTDisputeCase object containing dispute details
            
        Returns:
            GSTDisputeAnalysis with detailed analysis and recommendations
        """
        logger.info(f"Analyzing GST dispute case: {case.case_id}")
        
        try:
            # Preprocess case data
            processed_data = self._preprocess_case_data(case)
            
            # Quantum analysis
            quantum_results = self._perform_quantum_analysis(processed_data)
            
            # Legal position analysis
            legal_strength = self._analyze_legal_position(case)
            
            # Financial analysis
            financial_analysis = self._analyze_financial_impact(case)
            
            # Procedural analysis
            procedural_analysis = self._analyze_procedural_aspects(case)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(case, quantum_results)
            
            # Risk assessment
            risk_assessment = self._assess_risks(case, quantum_results)
            
            # Timeline analysis
            timeline = self._analyze_timeline(case)
            
            # Explainability
            explanation = self.explainability.explain_quantum_decision(
                quantum_results, "gst_dispute_analysis"
            )
            
            return GSTDisputeAnalysis(
                case_id=case.case_id,
                success_probability=quantum_results.get('success_probability', 0.0),
                legal_position_strength=legal_strength,
                applicable_provisions=self._get_applicable_provisions(case),
                relevant_precedents=self._find_relevant_precedents(case),
                estimated_liability=financial_analysis['liability'],
                penalty_assessment=financial_analysis['penalty'],
                interest_calculation=financial_analysis['interest'],
                limitation_compliance=procedural_analysis['limitation_compliance'],
                forum_jurisdiction=procedural_analysis['forum'],
                appeal_options=procedural_analysis['appeal_options'],
                recommended_strategy=recommendations['strategy'],
                settlement_options=recommendations['settlement'],
                documentation_requirements=recommendations['documentation'],
                litigation_risk=risk_assessment['litigation'],
                compliance_risk=risk_assessment['compliance'],
                financial_risk=risk_assessment['financial'],
                estimated_resolution_time=timeline['resolution_days'],
                critical_deadlines=timeline['deadlines'],
                quantum_confidence=quantum_results.get('confidence', 0.0),
                quantum_explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing GST dispute case {case.case_id}: {str(e)}")
            raise
    
    def _preprocess_case_data(self, case: GSTDisputeCase) -> Dict[str, Any]:
        """Preprocess GST dispute case data for quantum analysis"""
        
        # Extract and clean text data
        dispute_text = self.preprocessor.preprocess_legal_text(case.dispute_description)
        
        # Create feature vector
        features = {
            'dispute_type': case.dispute_type.value,
            'business_type': case.business_type.value,
            'disputed_amount': case.disputed_amount,
            'transaction_value': case.transaction_value or 0,
            'evidence_strength': self._calculate_evidence_strength(case),
            'procedural_compliance': self._assess_procedural_compliance(case),
            'text_features': dispute_text
        }
        
        return features
    
    def _perform_quantum_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform quantum-enhanced analysis of GST dispute"""
        
        # Create quantum embeddings
        text_embedding = self.quantum_embedding.encode_legal_text(
            processed_data['text_features']
        )
        
        # Quantum reasoning
        reasoning_result = self.quantum_reasoning.process_legal_reasoning(
            text_embedding,
            context_type="gst_dispute",
            legal_domain="tax_law"
        )
        
        # Calculate success probability using quantum interference
        success_prob = self._calculate_quantum_success_probability(
            reasoning_result, processed_data
        )
        
        return {
            'success_probability': success_prob,
            'confidence': reasoning_result.get('confidence', 0.0),
            'quantum_state': reasoning_result.get('final_state'),
            'coherence': reasoning_result.get('coherence', 0.0)
        }
    
    def _analyze_legal_position(self, case: GSTDisputeCase) -> Dict[str, float]:
        """Analyze the legal position strength"""
        
        strengths = {
            'statutory_compliance': 0.0,
            'precedent_support': 0.0,
            'documentation_quality': 0.0,
            'procedural_adherence': 0.0
        }
        
        # Statutory compliance analysis
        if case.dispute_type == GSTDisputeType.INPUT_TAX_CREDIT:
            strengths['statutory_compliance'] = self._assess_itc_compliance(case)
        elif case.dispute_type == GSTDisputeType.CLASSIFICATION:
            strengths['statutory_compliance'] = self._assess_classification_compliance(case)
        elif case.dispute_type == GSTDisputeType.VALUATION:
            strengths['statutory_compliance'] = self._assess_valuation_compliance(case)
        
        # Precedent support
        precedents = self._find_relevant_precedents(case)
        strengths['precedent_support'] = len(precedents) * 0.1  # Simple scoring
        
        # Documentation quality
        doc_score = 0.0
        if case.invoices_provided:
            doc_score += 0.3
        if case.books_of_accounts:
            doc_score += 0.3
        if case.contracts_agreements:
            doc_score += 0.2
        if case.evidence_documents:
            doc_score += len(case.evidence_documents) * 0.05
        
        strengths['documentation_quality'] = min(doc_score, 1.0)
        
        # Procedural adherence
        proc_score = 0.0
        if case.personal_hearing_attended:
            proc_score += 0.3
        if case.written_submissions:
            proc_score += 0.3
        if case.has_legal_counsel:
            proc_score += 0.2
        
        strengths['procedural_adherence'] = min(proc_score, 1.0)
        
        return strengths
    
    def _analyze_financial_impact(self, case: GSTDisputeCase) -> Dict[str, Dict[str, float]]:
        """Analyze financial impact of the dispute"""
        
        disputed_amount = case.disputed_amount
        
        # Liability calculation
        liability = {
            'principal_tax': disputed_amount,
            'additional_tax': 0.0,
            'total_liability': disputed_amount
        }
        
        # Penalty calculation (Section 74/75 of CGST Act)
        penalty_rate = 1.0 if case.show_cause_notice else 0.5  # 100% or 50%
        penalty = {
            'penalty_rate': penalty_rate,
            'penalty_amount': disputed_amount * penalty_rate,
            'minimum_penalty': min(10000, disputed_amount * 0.1),
            'maximum_penalty': disputed_amount * penalty_rate
        }
        
        # Interest calculation (Section 50 of CGST Act)
        interest_rate = 0.18 / 12  # 18% per annum, monthly
        months_delayed = self._calculate_delay_months(case)
        interest = {
            'interest_rate': 0.18,
            'months_delayed': months_delayed,
            'interest_amount': disputed_amount * interest_rate * months_delayed,
            'daily_interest': disputed_amount * 0.18 / 365
        }
        
        return {
            'liability': liability,
            'penalty': penalty,
            'interest': interest
        }
    
    def _analyze_procedural_aspects(self, case: GSTDisputeCase) -> Dict[str, Any]:
        """Analyze procedural aspects of the dispute"""
        
        # Limitation period check
        limitation_compliance = True
        if case.notice_date:
            limitation_period = timedelta(days=30)  # Standard appeal period
            if datetime.now() - case.notice_date > limitation_period:
                limitation_compliance = False
        
        # Forum determination
        forum = self._determine_appropriate_forum(case)
        
        # Appeal options
        appeal_options = self._get_appeal_options(case, forum)
        
        return {
            'limitation_compliance': limitation_compliance,
            'forum': forum,
            'appeal_options': appeal_options
        }
    
    def _generate_recommendations(self, case: GSTDisputeCase, quantum_results: Dict[str, float]) -> Dict[str, List]:
        """Generate strategic recommendations"""
        
        success_prob = quantum_results.get('success_probability', 0.0)
        
        strategy = []
        settlement = []
        documentation = []
        
        # Strategy recommendations based on success probability
        if success_prob > 0.7:
            strategy.extend([
                "Strong case for appeal - proceed with formal proceedings",
                "Gather additional supporting documentation",
                "Consider seeking stay of demand during appeal"
            ])
        elif success_prob > 0.4:
            strategy.extend([
                "Moderate chances - consider settlement negotiations",
                "Strengthen legal position with expert opinions",
                "Explore alternative dispute resolution mechanisms"
            ])
        else:
            strategy.extend([
                "Weak case - consider voluntary compliance",
                "Negotiate for penalty waiver",
                "Focus on minimizing interest liability"
            ])
        
        # Settlement options
        if case.disputed_amount > 1000000:  # For high-value disputes
            settlement.append({
                'type': 'Voluntary Compliance',
                'tax_percentage': 100,
                'penalty_percentage': 25,
                'interest_applicable': True
            })
            settlement.append({
                'type': 'Settlement Commission',
                'tax_percentage': 100,
                'penalty_percentage': 50,
                'interest_applicable': True
            })
        
        # Documentation requirements
        documentation.extend([
            "Complete set of invoices and supporting documents",
            "Books of accounts and financial statements",
            "Legal opinions on disputed provisions",
            "Comparative analysis with similar cases"
        ])
        
        return {
            'strategy': strategy,
            'settlement': settlement,
            'documentation': documentation
        }
    
    def _assess_risks(self, case: GSTDisputeCase, quantum_results: Dict[str, float]) -> Dict[str, float]:
        """Assess various risks associated with the dispute"""
        
        # Litigation risk
        litigation_risk = 1.0 - quantum_results.get('success_probability', 0.5)
        
        # Compliance risk
        compliance_risk = 0.5
        if case.business_type in [BusinessType.COMPOSITION_DEALER, BusinessType.E_COMMERCE]:
            compliance_risk += 0.2
        
        # Financial risk
        financial_risk = min(case.disputed_amount / 10000000, 1.0)  # Normalize to 1 crore
        
        return {
            'litigation': litigation_risk,
            'compliance': compliance_risk,
            'financial': financial_risk
        }
    
    def _analyze_timeline(self, case: GSTDisputeCase) -> Dict[str, Any]:
        """Analyze timeline and critical deadlines"""
        
        resolution_days = 180  # Default 6 months
        
        # Adjust based on dispute type and amount
        if case.disputed_amount > 5000000:
            resolution_days += 90
        
        if case.dispute_type in [GSTDisputeType.CLASSIFICATION, GSTDisputeType.VALUATION]:
            resolution_days += 60
        
        deadlines = []
        if case.response_deadline:
            deadlines.append({
                'type': 'Response Deadline',
                'date': case.response_deadline,
                'days_remaining': (case.response_deadline - datetime.now()).days
            })
        
        return {
            'resolution_days': resolution_days,
            'deadlines': deadlines
        }
    
    # Helper methods
    def _load_gst_provisions(self) -> Dict[str, Any]:
        """Load GST Act provisions"""
        return {
            'cgst_act': {
                'section_15': 'Value of taxable supply',
                'section_16': 'Eligibility and conditions for taking input tax credit',
                'section_17': 'Apportionment of credit and blocked credits',
                'section_50': 'Interest on delayed payment of tax',
                'section_74': 'Determination of tax not paid or short paid or erroneously refunded',
                'section_75': 'Determination of tax not paid or short paid or input tax credit wrongly availed'
            },
            'sgst_act': {
                'section_15': 'Value of taxable supply',
                'section_16': 'Eligibility and conditions for taking input tax credit'
            },
            'igst_act': {
                'section_5': 'Levy and collection of integrated tax',
                'section_16': 'Place of supply of services'
            },
            'gst_rules': {
                'rule_36': 'Documents to be furnished by the recipient for taking input tax credit',
                'rule_89': 'Refund of tax paid on zero-rated supplies'
            }
        }
    
    def _load_gst_rates(self) -> Dict[str, float]:
        """Load GST rate structure"""
        return {
            'goods': {
                'essential_items': 0.0,
                'basic_necessities': 5.0,
                'standard_goods': 12.0,
                'luxury_goods': 18.0,
                'sin_goods': 28.0
            },
            'services': {
                'essential_services': 0.0,
                'basic_services': 5.0,
                'standard_services': 18.0,
                'luxury_services': 28.0
            }
        }
    
    def _load_gst_precedents(self) -> List[Dict[str, Any]]:
        """Load GST precedent database"""
        return [
            {
                'case_name': 'Union of India vs. Mohit Minerals Pvt. Ltd.',
                'citation': '2021 (47) G.S.T.L. 433 (SC)',
                'principle': 'Input tax credit eligibility on capital goods',
                'dispute_type': 'input_tax_credit'
            },
            {
                'case_name': 'Safari Retreats Pvt. Ltd. vs. Commissioner of CGST',
                'citation': '2019 (29) G.S.T.L. 257 (AAR)',
                'principle': 'Classification of accommodation services',
                'dispute_type': 'classification'
            },
            {
                'case_name': 'Kellogg India Pvt. Ltd. vs. Commissioner of CGST',
                'citation': '2020 (35) G.S.T.L. 145 (CESTAT)',
                'principle': 'Valuation of related party transactions',
                'dispute_type': 'valuation'
            }
        ]
    
    def _calculate_evidence_strength(self, case: GSTDisputeCase) -> float:
        """Calculate evidence strength score"""
        score = 0.0
        if case.invoices_provided:
            score += 0.3
        if case.books_of_accounts:
            score += 0.3
        if case.contracts_agreements:
            score += 0.2
        score += len(case.evidence_documents) * 0.05
        return min(score, 1.0)
    
    def _assess_procedural_compliance(self, case: GSTDisputeCase) -> float:
        """Assess procedural compliance"""
        score = 0.5  # Base score
        if case.personal_hearing_attended:
            score += 0.2
        if case.written_submissions:
            score += 0.2
        if case.has_legal_counsel:
            score += 0.1
        return min(score, 1.0)
    
    def _calculate_quantum_success_probability(self, reasoning_result: Dict, processed_data: Dict) -> float:
        """Calculate success probability using quantum analysis"""
        base_prob = reasoning_result.get('confidence', 0.5)
        
        # Adjust based on evidence strength
        evidence_factor = processed_data.get('evidence_strength', 0.5)
        adjusted_prob = base_prob * (0.5 + evidence_factor * 0.5)
        
        return min(max(adjusted_prob, 0.0), 1.0)
    
    def _get_applicable_provisions(self, case: GSTDisputeCase) -> List[str]:
        """Get applicable GST provisions"""
        provisions = []
        
        if case.dispute_type == GSTDisputeType.INPUT_TAX_CREDIT:
            provisions.extend(["Section 16 CGST Act", "Section 17 CGST Act", "Rule 36 CGST Rules"])
        elif case.dispute_type == GSTDisputeType.CLASSIFICATION:
            provisions.extend(["Section 2(52) CGST Act", "HSN Classification"])
        elif case.dispute_type == GSTDisputeType.VALUATION:
            provisions.extend(["Section 15 CGST Act", "Valuation Rules"])
        
        return provisions
    
    def _find_relevant_precedents(self, case: GSTDisputeCase) -> List[Dict[str, Any]]:
        """Find relevant legal precedents"""
        relevant_precedents = []
        
        for precedent in self.precedent_database:
            if precedent['dispute_type'] == case.dispute_type.value:
                relevant_precedents.append({
                    'case_name': precedent['case_name'],
                    'citation': precedent['citation'],
                    'principle': precedent['principle'],
                    'relevance_score': 0.8,
                    'supporting': True
                })
        
        return relevant_precedents[:5]  # Return top 5 relevant cases
    
    def _assess_itc_compliance(self, case: GSTDisputeCase) -> float:
        """Assess Input Tax Credit compliance"""
        compliance_score = 0.5  # Base score
        
        # Check documentation
        if case.invoices_provided:
            compliance_score += 0.2
        if case.books_of_accounts:
            compliance_score += 0.1
        
        # Check business type compliance
        if case.business_type in [BusinessType.MANUFACTURER, BusinessType.TRADER]:
            compliance_score += 0.1
        
        # Check if goods/services received
        if case.contracts_agreements:
            compliance_score += 0.1
        
        return min(compliance_score, 1.0)
    
    def _assess_classification_compliance(self, case: GSTDisputeCase) -> float:
        """Assess classification compliance"""
        compliance_score = 0.5  # Base score
        
        # Check if proper HSN/SAC code used
        if case.evidence_documents:
            compliance_score += 0.2
        
        # Check transaction nature
        if case.transaction_value and case.transaction_value > 0:
            compliance_score += 0.1
        
        # Check business understanding
        if case.taxpayer_position:
            compliance_score += 0.2
        
        return min(compliance_score, 1.0)
    
    def _assess_valuation_compliance(self, case: GSTDisputeCase) -> float:
        """Assess valuation compliance"""
        compliance_score = 0.5  # Base score
        
        # Check if transaction value is reasonable
        if case.transaction_value and case.disputed_amount:
            ratio = case.disputed_amount / case.transaction_value
            if ratio < 0.1:  # Dispute is less than 10% of transaction
                compliance_score += 0.2
        
        # Check related party transactions
        if case.contracts_agreements:
            compliance_score += 0.2
        
        # Check market value evidence
        if len(case.evidence_documents) > 2:
            compliance_score += 0.1
        
        return min(compliance_score, 1.0)
    
    def _calculate_delay_months(self, case: GSTDisputeCase) -> int:
        """Calculate delay in months"""
        if case.notice_date:
            return max(1, (datetime.now() - case.notice_date).days // 30)
        return 1
    
    def _determine_appropriate_forum(self, case: GSTDisputeCase) -> GSTForum:
        """Determine appropriate forum for dispute"""
        if case.disputed_amount < 2000000:
            return GSTForum.ADJUDICATING_AUTHORITY
        elif case.disputed_amount < 10000000:
            return GSTForum.APPELLATE_AUTHORITY
        else:
            return GSTForum.APPELLATE_TRIBUNAL
    
    def _get_appeal_options(self, case: GSTDisputeCase, forum: GSTForum) -> List[str]:
        """Get available appeal options"""
        options = []
        
        if forum == GSTForum.ADJUDICATING_AUTHORITY:
            options.append("Appeal to Appellate Authority")
        elif forum == GSTForum.APPELLATE_AUTHORITY:
            options.append("Appeal to Appellate Tribunal")
        elif forum == GSTForum.APPELLATE_TRIBUNAL:
            options.append("Appeal to High Court")
        
        options.append("Writ Petition to High Court")
        options.append("Settlement through Voluntary Compliance")
        
        return options