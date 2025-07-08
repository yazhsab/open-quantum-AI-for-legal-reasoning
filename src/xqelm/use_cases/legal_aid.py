"""
Legal Aid Distribution Use Case Manager

This module handles legal aid eligibility assessment and distribution
recommendations for the Indian legal aid system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging
from datetime import datetime

from ..core.quantum_legal_model import QuantumLegalModel
from ..classical.preprocessor import LegalTextPreprocessor
from ..classical.knowledge_base import LegalKnowledgeBase
from ..quantum.embeddings import QuantumLegalEmbedding
from ..quantum.reasoning import QuantumLegalReasoningCircuit
from ..core.explainability import QuantumExplainabilityModule

logger = logging.getLogger(__name__)


class LegalAidType(Enum):
    """Types of legal aid services"""
    FREE_LEGAL_ADVICE = "free_legal_advice"
    LEGAL_REPRESENTATION = "legal_representation"
    DOCUMENT_DRAFTING = "document_drafting"
    MEDIATION_SERVICES = "mediation_services"
    LEGAL_LITERACY = "legal_literacy"
    PARALEGAL_SERVICES = "paralegal_services"
    COURT_FEE_WAIVER = "court_fee_waiver"
    BAIL_ASSISTANCE = "bail_assistance"


class CaseCategory(Enum):
    """Categories of legal cases for aid"""
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


class EligibilityCriteria(Enum):
    """Eligibility criteria for legal aid"""
    INCOME_BASED = "income_based"
    SOCIAL_CATEGORY = "social_category"
    DISABILITY = "disability"
    SENIOR_CITIZEN = "senior_citizen"
    WOMEN = "women"
    CHILD = "child"
    VICTIM_OF_CRIME = "victim_of_crime"
    MARGINALIZED_COMMUNITY = "marginalized_community"


class LegalAidAuthority(Enum):
    """Legal aid providing authorities"""
    NATIONAL_LEGAL_SERVICES_AUTHORITY = "nalsa"
    STATE_LEGAL_SERVICES_AUTHORITY = "slsa"
    DISTRICT_LEGAL_SERVICES_AUTHORITY = "dlsa"
    TALUK_LEGAL_SERVICES_COMMITTEE = "tlsc"
    HIGH_COURT_LEGAL_SERVICES = "hclsc"
    SUPREME_COURT_LEGAL_SERVICES = "sclsc"


@dataclass
class LegalAidApplicant:
    """Legal aid applicant information"""
    applicant_id: str
    name: str
    age: int
    gender: str
    
    # Contact information
    address: str
    phone: Optional[str] = None
    email: Optional[str] = None
    
    # Socio-economic details
    annual_income: float
    family_size: int
    occupation: Optional[str] = None
    education_level: Optional[str] = None
    
    # Eligibility factors
    social_category: Optional[str] = None  # SC/ST/OBC/General
    is_disabled: bool = False
    disability_type: Optional[str] = None
    is_senior_citizen: bool = False
    is_single_woman: bool = False
    is_victim_of_crime: bool = False
    
    # Financial details
    assets_value: Optional[float] = None
    monthly_expenses: Optional[float] = None
    dependents: int = 0
    
    # Supporting documents
    income_certificate: bool = False
    caste_certificate: bool = False
    disability_certificate: bool = False
    identity_proof: bool = False
    address_proof: bool = False


@dataclass
class LegalAidCase:
    """Legal aid case details"""
    case_id: str
    applicant: LegalAidApplicant
    case_category: CaseCategory
    legal_aid_type: LegalAidType
    case_description: str
    urgency_level: str  # high/medium/low
    
    # Case specifics
    opposing_party: Optional[str] = None
    court_involved: Optional[str] = None
    case_number: Optional[str] = None
    case_stage: Optional[str] = None
    
    # Legal requirements
    lawyer_required: bool = False
    court_representation: bool = False
    document_assistance: bool = False
    legal_advice_only: bool = False
    
    # Timeline
    application_date: datetime = field(default_factory=datetime.now)
    required_by_date: Optional[datetime] = None
    
    # Previous legal aid
    previous_aid_received: bool = False
    previous_aid_details: Optional[str] = None
    
    # Special circumstances
    emergency_case: bool = False
    pro_bono_eligible: bool = False
    media_attention: bool = False


@dataclass
class LegalAidAssessment:
    """Legal aid assessment results"""
    case_id: str
    eligibility_score: float
    
    # Eligibility analysis
    income_eligibility: bool
    category_eligibility: bool
    case_merit: float
    urgency_assessment: float
    
    # Recommended aid
    recommended_aid_type: LegalAidType
    recommended_authority: LegalAidAuthority
    estimated_cost: float
    
    # Resource allocation
    lawyer_assignment: Dict[str, Any]
    priority_level: int  # 1-5, 1 being highest
    estimated_duration: int  # in days
    
    # Requirements
    additional_documents: List[str]
    verification_needed: List[str]
    conditions: List[str]
    
    # Financial analysis
    fee_waiver_eligible: bool
    court_fee_exemption: float
    legal_service_cost: float
    
    # Success factors
    case_strength: float
    likelihood_of_success: float
    social_impact: float
    
    # Quantum analysis
    quantum_confidence: float
    quantum_explanation: Dict[str, Any]
    
    # Recommendations
    next_steps: List[str]
    alternative_options: List[str]
    referral_suggestions: List[str]


class LegalAidManager:
    """Manager for legal aid assessment and distribution"""
    
    def __init__(self):
        self.quantum_model = QuantumLegalModel()
        self.preprocessor = LegalTextPreprocessor()
        self.knowledge_base = LegalKnowledgeBase()
        self.quantum_embedding = QuantumLegalEmbedding()
        self.quantum_reasoning = QuantumLegalReasoningCircuit()
        self.explainability = QuantumExplainabilityModule()
        
        # Legal aid specific data
        self.income_thresholds = self._load_income_thresholds()
        self.aid_schemes = self._load_aid_schemes()
        self.lawyer_database = self._load_lawyer_database()
    
    def assess_legal_aid_eligibility(self, case: LegalAidCase) -> LegalAidAssessment:
        """
        Assess legal aid eligibility and recommend appropriate aid
        
        Args:
            case: LegalAidCase object containing applicant and case details
            
        Returns:
            LegalAidAssessment with detailed analysis and recommendations
        """
        logger.info(f"Assessing legal aid eligibility for case: {case.case_id}")
        
        try:
            # Preprocess case data
            processed_data = self._preprocess_case_data(case)
            
            # Quantum analysis
            quantum_results = self._perform_quantum_analysis(processed_data)
            
            # Eligibility assessment
            eligibility = self._assess_eligibility(case)
            
            # Case merit analysis
            merit_analysis = self._analyze_case_merit(case)
            
            # Resource allocation
            resource_allocation = self._allocate_resources(case, eligibility)
            
            # Financial analysis
            financial_analysis = self._analyze_financial_aspects(case)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(case, quantum_results)
            
            # Explainability
            explanation = self.explainability.explain_quantum_decision(
                quantum_results, "legal_aid_assessment"
            )
            
            return LegalAidAssessment(
                case_id=case.case_id,
                eligibility_score=quantum_results.get('eligibility_score', 0.0),
                income_eligibility=eligibility['income'],
                category_eligibility=eligibility['category'],
                case_merit=merit_analysis['merit_score'],
                urgency_assessment=merit_analysis['urgency_score'],
                recommended_aid_type=resource_allocation['aid_type'],
                recommended_authority=resource_allocation['authority'],
                estimated_cost=resource_allocation['cost'],
                lawyer_assignment=resource_allocation['lawyer'],
                priority_level=resource_allocation['priority'],
                estimated_duration=resource_allocation['duration'],
                additional_documents=recommendations['documents'],
                verification_needed=recommendations['verification'],
                conditions=recommendations['conditions'],
                fee_waiver_eligible=financial_analysis['fee_waiver'],
                court_fee_exemption=financial_analysis['court_fee_exemption'],
                legal_service_cost=financial_analysis['service_cost'],
                case_strength=merit_analysis['case_strength'],
                likelihood_of_success=merit_analysis['success_likelihood'],
                social_impact=merit_analysis['social_impact'],
                quantum_confidence=quantum_results.get('confidence', 0.0),
                quantum_explanation=explanation,
                next_steps=recommendations['next_steps'],
                alternative_options=recommendations['alternatives'],
                referral_suggestions=recommendations['referrals']
            )
            
        except Exception as e:
            logger.error(f"Error assessing legal aid case {case.case_id}: {str(e)}")
            raise
    
    def _preprocess_case_data(self, case: LegalAidCase) -> Dict[str, Any]:
        """Preprocess legal aid case data for quantum analysis"""
        
        # Extract and clean text data
        case_text = self.preprocessor.preprocess_legal_text(case.case_description)
        
        # Calculate socio-economic score
        socio_economic_score = self._calculate_socio_economic_score(case.applicant)
        
        # Create feature vector
        features = {
            'case_category': case.case_category.value,
            'aid_type': case.legal_aid_type.value,
            'urgency': case.urgency_level,
            'income_ratio': case.applicant.annual_income / self._get_poverty_line(),
            'socio_economic_score': socio_economic_score,
            'case_complexity': self._assess_case_complexity(case),
            'text_features': case_text
        }
        
        return features
    
    def _perform_quantum_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform quantum-enhanced analysis of legal aid case"""
        
        # Create quantum embeddings
        text_embedding = self.quantum_embedding.encode_legal_text(
            processed_data['text_features']
        )
        
        # Quantum reasoning
        reasoning_result = self.quantum_reasoning.process_legal_reasoning(
            text_embedding,
            context_type="legal_aid",
            legal_domain="access_to_justice"
        )
        
        # Calculate eligibility score using quantum superposition
        eligibility_score = self._calculate_quantum_eligibility_score(
            reasoning_result, processed_data
        )
        
        return {
            'eligibility_score': eligibility_score,
            'confidence': reasoning_result.get('confidence', 0.0),
            'quantum_state': reasoning_result.get('final_state'),
            'coherence': reasoning_result.get('coherence', 0.0)
        }
    
    def _assess_eligibility(self, case: LegalAidCase) -> Dict[str, bool]:
        """Assess basic eligibility criteria"""
        
        applicant = case.applicant
        
        # Income eligibility
        income_threshold = self._get_income_threshold(applicant)
        income_eligible = applicant.annual_income <= income_threshold
        
        # Category eligibility
        category_eligible = (
            applicant.social_category in ['SC', 'ST', 'OBC'] or
            applicant.is_disabled or
            applicant.is_senior_citizen or
            applicant.is_single_woman or
            applicant.is_victim_of_crime or
            income_eligible
        )
        
        return {
            'income': income_eligible,
            'category': category_eligible
        }
    
    def _analyze_case_merit(self, case: LegalAidCase) -> Dict[str, float]:
        """Analyze the merit and urgency of the case"""
        
        # Merit score based on case type and circumstances
        merit_score = 0.5  # Base score
        
        if case.case_category in [CaseCategory.CRIMINAL, CaseCategory.CONSTITUTIONAL]:
            merit_score += 0.2
        elif case.case_category in [CaseCategory.WOMEN_RIGHTS, CaseCategory.CHILD_RIGHTS]:
            merit_score += 0.3
        
        if case.applicant.is_victim_of_crime:
            merit_score += 0.2
        
        # Urgency assessment
        urgency_score = 0.3  # Base urgency
        if case.urgency_level == "high":
            urgency_score = 0.9
        elif case.urgency_level == "medium":
            urgency_score = 0.6
        
        if case.emergency_case:
            urgency_score = min(urgency_score + 0.3, 1.0)
        
        # Case strength assessment
        case_strength = self._assess_case_strength(case)
        
        # Success likelihood
        success_likelihood = (merit_score + case_strength) / 2
        
        # Social impact
        social_impact = self._assess_social_impact(case)
        
        return {
            'merit_score': min(merit_score, 1.0),
            'urgency_score': urgency_score,
            'case_strength': case_strength,
            'success_likelihood': success_likelihood,
            'social_impact': social_impact
        }
    
    def _allocate_resources(self, case: LegalAidCase, eligibility: Dict[str, bool]) -> Dict[str, Any]:
        """Allocate appropriate resources for the case"""
        
        if not (eligibility['income'] or eligibility['category']):
            return self._create_rejection_allocation()
        
        # Determine appropriate aid type
        aid_type = case.legal_aid_type
        if case.court_representation:
            aid_type = LegalAidType.LEGAL_REPRESENTATION
        elif case.document_assistance:
            aid_type = LegalAidType.DOCUMENT_DRAFTING
        elif case.legal_advice_only:
            aid_type = LegalAidType.FREE_LEGAL_ADVICE
        
        # Determine authority
        authority = self._determine_appropriate_authority(case)
        
        # Estimate cost
        cost = self._estimate_aid_cost(case, aid_type)
        
        # Lawyer assignment
        lawyer = self._assign_lawyer(case, aid_type)
        
        # Priority level (1-5, 1 highest)
        priority = self._calculate_priority(case)
        
        # Duration estimate
        duration = self._estimate_duration(case, aid_type)
        
        return {
            'aid_type': aid_type,
            'authority': authority,
            'cost': cost,
            'lawyer': lawyer,
            'priority': priority,
            'duration': duration
        }
    
    def _analyze_financial_aspects(self, case: LegalAidCase) -> Dict[str, Any]:
        """Analyze financial aspects of legal aid"""
        
        applicant = case.applicant
        
        # Fee waiver eligibility
        fee_waiver = applicant.annual_income <= self._get_fee_waiver_threshold()
        
        # Court fee exemption amount
        court_fee_exemption = 0.0
        if fee_waiver:
            court_fee_exemption = self._calculate_court_fee_exemption(case)
        
        # Legal service cost
        service_cost = self._calculate_service_cost(case)
        
        return {
            'fee_waiver': fee_waiver,
            'court_fee_exemption': court_fee_exemption,
            'service_cost': service_cost
        }
    
    def _generate_recommendations(self, case: LegalAidCase, quantum_results: Dict[str, float]) -> Dict[str, List]:
        """Generate recommendations for the legal aid case"""
        
        eligibility_score = quantum_results.get('eligibility_score', 0.0)
        
        documents = []
        verification = []
        conditions = []
        next_steps = []
        alternatives = []
        referrals = []
        
        # Document requirements
        if not case.applicant.income_certificate:
            documents.append("Income Certificate from competent authority")
        if not case.applicant.identity_proof:
            documents.append("Valid identity proof (Aadhaar/Voter ID/Passport)")
        if not case.applicant.address_proof:
            documents.append("Address proof (Utility bill/Bank statement)")
        
        if case.applicant.social_category and not case.applicant.caste_certificate:
            documents.append("Caste certificate for reservation benefits")
        
        if case.applicant.is_disabled and not case.applicant.disability_certificate:
            documents.append("Disability certificate from medical authority")
        
        # Verification requirements
        verification.extend([
            "Income verification through field investigation",
            "Case details verification with court records",
            "Applicant interview by legal aid officer"
        ])
        
        # Conditions
        if eligibility_score > 0.7:
            conditions.append("Regular updates on case progress required")
            conditions.append("Cooperation with assigned legal aid lawyer")
        else:
            conditions.append("Additional documentation may be required")
            conditions.append("Case review after 30 days")
        
        # Next steps
        if eligibility_score > 0.6:
            next_steps.extend([
                "Submit complete application with required documents",
                "Attend verification interview",
                "Await lawyer assignment within 7 days"
            ])
        else:
            next_steps.extend([
                "Provide additional documentation",
                "Consider alternative legal aid options",
                "Seek assistance from paralegal services"
            ])
        
        # Alternative options
        alternatives.extend([
            "Lok Adalat for amicable settlement",
            "Mediation services for dispute resolution",
            "Legal literacy programs for self-help",
            "Pro bono services from private lawyers"
        ])
        
        # Referrals
        referrals.extend([
            "District Legal Services Authority",
            "State Legal Services Authority",
            "NGOs providing legal assistance",
            "Bar Association pro bono programs"
        ])
        
        return {
            'documents': documents,
            'verification': verification,
            'conditions': conditions,
            'next_steps': next_steps,
            'alternatives': alternatives,
            'referrals': referrals
        }
    
    # Helper methods
    def _load_income_thresholds(self) -> Dict[str, float]:
        """Load income thresholds for different categories"""
        return {
            'rural': 300000,  # 3 lakh per annum
            'urban': 500000,  # 5 lakh per annum
            'metro': 600000   # 6 lakh per annum
        }
    
    def _load_aid_schemes(self) -> Dict[str, Any]:
        """Load available legal aid schemes"""
        return {
            'nalsa_schemes': {
                'free_legal_aid': {
                    'description': 'Free legal services to eligible persons',
                    'eligibility': 'Income below threshold, SC/ST/OBC, women, children',
                    'coverage': 'Legal advice, representation, document drafting'
                },
                'lok_adalat': {
                    'description': 'Alternative dispute resolution mechanism',
                    'eligibility': 'All citizens',
                    'coverage': 'Mediation, conciliation, settlement'
                },
                'legal_literacy': {
                    'description': 'Legal awareness programs',
                    'eligibility': 'All citizens, especially marginalized',
                    'coverage': 'Rights awareness, legal procedures'
                }
            },
            'state_schemes': {
                'victim_compensation': {
                    'description': 'Compensation for crime victims',
                    'eligibility': 'Victims of violent crimes',
                    'coverage': 'Financial assistance, legal support'
                },
                'women_helpline': {
                    'description': '24x7 helpline for women in distress',
                    'eligibility': 'Women facing violence or harassment',
                    'coverage': 'Emergency assistance, legal guidance'
                }
            }
        }
    
    def _load_lawyer_database(self) -> List[Dict[str, Any]]:
        """Load database of available lawyers"""
        return [
            {
                'lawyer_id': 'LAW001',
                'name': 'Advocate Priya Sharma',
                'specialization': ['criminal', 'women_rights'],
                'experience': 8,
                'location': 'Delhi',
                'availability': True,
                'languages': ['Hindi', 'English'],
                'rating': 4.5
            },
            {
                'lawyer_id': 'LAW002',
                'name': 'Advocate Rajesh Kumar',
                'specialization': ['civil', 'property'],
                'experience': 12,
                'location': 'Mumbai',
                'availability': True,
                'languages': ['Hindi', 'English', 'Marathi'],
                'rating': 4.7
            },
            {
                'lawyer_id': 'LAW003',
                'name': 'Advocate Sunita Devi',
                'specialization': ['family', 'child_rights'],
                'experience': 6,
                'location': 'Bangalore',
                'availability': True,
                'languages': ['Hindi', 'English', 'Kannada'],
                'rating': 4.3
            },
            {
                'lawyer_id': 'LAW004',
                'name': 'Advocate Mohammed Ali',
                'specialization': ['labor', 'constitutional'],
                'experience': 15,
                'location': 'Kolkata',
                'availability': True,
                'languages': ['Hindi', 'English', 'Bengali', 'Urdu'],
                'rating': 4.8
            },
            {
                'lawyer_id': 'LAW005',
                'name': 'Advocate Lakshmi Menon',
                'specialization': ['consumer', 'environmental'],
                'experience': 10,
                'location': 'Chennai',
                'availability': True,
                'languages': ['Hindi', 'English', 'Tamil'],
                'rating': 4.6
            }
        ]
    
    def _calculate_socio_economic_score(self, applicant: LegalAidApplicant) -> float:
        """Calculate socio-economic vulnerability score"""
        score = 0.0
        
        # Income factor
        poverty_line = self._get_poverty_line()
        if applicant.annual_income <= poverty_line:
            score += 0.3
        elif applicant.annual_income <= poverty_line * 2:
            score += 0.2
        
        # Social category
        if applicant.social_category in ['SC', 'ST']:
            score += 0.2
        elif applicant.social_category == 'OBC':
            score += 0.1
        
        # Vulnerability factors
        if applicant.is_disabled:
            score += 0.2
        if applicant.is_senior_citizen:
            score += 0.1
        if applicant.is_single_woman:
            score += 0.15
        if applicant.is_victim_of_crime:
            score += 0.25
        
        return min(score, 1.0)
    
    def _get_poverty_line(self) -> float:
        """Get current poverty line income"""
        return 150000  # 1.5 lakh per annum (placeholder)
    
    def _assess_case_complexity(self, case: LegalAidCase) -> float:
        """Assess complexity of the legal case"""
        complexity = 0.5  # Base complexity
        
        if case.case_category in [CaseCategory.CONSTITUTIONAL, CaseCategory.ENVIRONMENTAL]:
            complexity += 0.3
        elif case.case_category in [CaseCategory.CRIMINAL, CaseCategory.CIVIL]:
            complexity += 0.2
        
        if case.court_representation:
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _calculate_quantum_eligibility_score(self, reasoning_result: Dict, processed_data: Dict) -> float:
        """Calculate eligibility score using quantum analysis"""
        base_score = reasoning_result.get('confidence', 0.5)
        
        # Adjust based on socio-economic factors
        socio_factor = processed_data.get('socio_economic_score', 0.5)
        income_factor = min(2.0 / processed_data.get('income_ratio', 1.0), 1.0)
        
        adjusted_score = base_score * (0.3 + socio_factor * 0.4 + income_factor * 0.3)
        
        return min(max(adjusted_score, 0.0), 1.0)
    
    def _get_income_threshold(self, applicant: LegalAidApplicant) -> float:
        """Get income threshold based on location and family size"""
        base_threshold = 300000  # Rural threshold
        
        # Adjust for family size
        threshold = base_threshold * (1 + (applicant.family_size - 1) * 0.1)
        
        return threshold
    
    def _assess_case_strength(self, case: LegalAidCase) -> float:
        """Assess the strength of the legal case"""
        strength = 0.5  # Base strength
        
        # Case category strength
        if case.case_category in [CaseCategory.CRIMINAL, CaseCategory.CONSTITUTIONAL]:
            strength += 0.2
        elif case.case_category in [CaseCategory.WOMEN_RIGHTS, CaseCategory.CHILD_RIGHTS]:
            strength += 0.3
        
        # Evidence and documentation
        if case.applicant.identity_proof and case.applicant.address_proof:
            strength += 0.1
        
        # Legal representation history
        if case.previous_aid_received:
            strength += 0.1
        
        # Urgency and emergency factors
        if case.emergency_case:
            strength += 0.2
        
        # Opposing party considerations
        if case.opposing_party:
            # If opposing party is government/large corporation, case might be weaker
            if any(keyword in case.opposing_party.lower() for keyword in ['government', 'state', 'corporation']):
                strength -= 0.1
        
        return min(max(strength, 0.1), 1.0)
    
    def _assess_social_impact(self, case: LegalAidCase) -> float:
        """Assess the social impact of the case"""
        impact = 0.5
        
        if case.case_category in [CaseCategory.WOMEN_RIGHTS, CaseCategory.CHILD_RIGHTS]:
            impact += 0.3
        elif case.case_category in [CaseCategory.HUMAN_RIGHTS, CaseCategory.CONSTITUTIONAL]:
            impact += 0.4
        
        if case.media_attention:
            impact += 0.2
        
        return min(impact, 1.0)
    
    def _create_rejection_allocation(self) -> Dict[str, Any]:
        """Create allocation for rejected cases"""
        return {
            'aid_type': None,
            'authority': None,
            'cost': 0.0,
            'lawyer': {'assigned': False, 'reason': 'Not eligible'},
            'priority': 5,
            'duration': 0
        }
    
    def _determine_appropriate_authority(self, case: LegalAidCase) -> LegalAidAuthority:
        """Determine appropriate legal aid authority"""
        if case.case_category == CaseCategory.CONSTITUTIONAL:
            return LegalAidAuthority.SUPREME_COURT_LEGAL_SERVICES
        elif case.court_involved and "High Court" in str(case.court_involved):
            return LegalAidAuthority.HIGH_COURT_LEGAL_SERVICES
        else:
            return LegalAidAuthority.DISTRICT_LEGAL_SERVICES_AUTHORITY
    
    def _estimate_aid_cost(self, case: LegalAidCase, aid_type: LegalAidType) -> float:
        """Estimate cost of providing legal aid"""
        base_costs = {
            LegalAidType.FREE_LEGAL_ADVICE: 500,
            LegalAidType.LEGAL_REPRESENTATION: 15000,
            LegalAidType.DOCUMENT_DRAFTING: 2000,
            LegalAidType.MEDIATION_SERVICES: 3000,
            LegalAidType.COURT_FEE_WAIVER: 5000
        }
        
        return base_costs.get(aid_type, 1000)
    
    def _assign_lawyer(self, case: LegalAidCase, aid_type: LegalAidType) -> Dict[str, Any]:
        """Assign appropriate lawyer for the case"""
        if aid_type == LegalAidType.LEGAL_REPRESENTATION:
            # Find best matching lawyer
            best_lawyer = None
            best_score = 0
            
            for lawyer in self.lawyer_database:
                if not lawyer['availability']:
                    continue
                
                score = 0
                # Specialization match
                if case.case_category.value in lawyer['specialization']:
                    score += 0.5
                
                # Experience factor
                score += min(lawyer['experience'] / 20, 0.3)
                
                # Rating factor
                score += lawyer['rating'] / 10
                
                if score > best_score:
                    best_score = score
                    best_lawyer = lawyer
            
            if best_lawyer:
                return {
                    'assigned': True,
                    'lawyer_id': best_lawyer['lawyer_id'],
                    'name': best_lawyer['name'],
                    'specialization': best_lawyer['specialization'],
                    'experience': f"{best_lawyer['experience']} years",
                    'rating': best_lawyer['rating'],
                    'languages': best_lawyer['languages'],
                    'match_score': best_score
                }
            else:
                return {
                    'assigned': False,
                    'reason': 'No suitable lawyer available currently'
                }
        else:
            return {
                'assigned': False,
                'reason': 'Paralegal assistance sufficient for this aid type'
            }
    
    def _calculate_priority(self, case: LegalAidCase) -> int:
        """Calculate priority level (1-5, 1 highest)"""
        if case.emergency_case:
            return 1
        elif case.urgency_level == "high":
            return 2
        elif case.case_category in [CaseCategory.CRIMINAL, CaseCategory.CONSTITUTIONAL]:
            return 2
        elif case.urgency_level == "medium":
            return 3
        else:
            return 4
    
    def _estimate_duration(self, case: LegalAidCase, aid_type: LegalAidType) -> int:
        """Estimate duration in days"""
        durations = {
            LegalAidType.FREE_LEGAL_ADVICE: 1,
            LegalAidType.LEGAL_REPRESENTATION: 180,
            LegalAidType.DOCUMENT_DRAFTING: 7,
            LegalAidType.MEDIATION_SERVICES: 30,
            LegalAidType.COURT_FEE_WAIVER: 3
        }
        
        return durations.get(aid_type, 30)
    
    def _get_fee_waiver_threshold(self) -> float:
        """Get threshold for fee waiver eligibility"""
        return 200000  # 2 lakh per annum
    
    def _calculate_court_fee_exemption(self, case: LegalAidCase) -> float:
        """Calculate court fee exemption amount"""
        # This would depend on the type of case and court
        return 5000  # Placeholder
    
    def _calculate_service_cost(self, case: LegalAidCase) -> float:
        """Calculate cost of legal services"""
        return self._estimate_aid_cost(case, case.legal_aid_type)