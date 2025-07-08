"""
Synthetic data generator for XQELM legal case datasets.
Generates realistic legal case data for training and testing purposes.
"""

import json
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    num_cases: int = 100
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    complexity_distribution: Dict[str, float] = None
    jurisdiction_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.complexity_distribution is None:
            self.complexity_distribution = {"low": 0.3, "medium": 0.5, "high": 0.2}
        if self.jurisdiction_weights is None:
            self.jurisdiction_weights = {
                "delhi": 0.15, "maharashtra": 0.15, "karnataka": 0.12,
                "uttar_pradesh": 0.12, "gujarat": 0.10, "tamil_nadu": 0.10,
                "west_bengal": 0.08, "rajasthan": 0.08, "haryana": 0.05, "punjab": 0.05
            }

class SyntheticDataGenerator:
    """
    Generates synthetic legal case data for training and testing.
    Supports multiple case types with realistic variations.
    """
    
    def __init__(self):
        """Initialize the synthetic data generator."""
        self.indian_names = {
            "male": ["Rajesh", "Amit", "Suresh", "Ravi", "Vikash", "Anil", "Deepak", "Manoj", "Sanjay", "Rahul"],
            "female": ["Priya", "Sunita", "Kavita", "Meera", "Pooja", "Anita", "Rekha", "Geeta", "Sita", "Radha"]
        }
        
        self.indian_cities = {
            "delhi": ["Connaught Place", "Karol Bagh", "Lajpat Nagar", "Dwarka", "Rohini"],
            "maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
            "karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum"],
            "uttar_pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Meerut"],
            "gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
            "tamil_nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli"],
            "west_bengal": ["Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri"],
            "rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner"],
            "haryana": ["Gurgaon", "Faridabad", "Panipat", "Ambala", "Karnal"],
            "punjab": ["Chandigarh", "Ludhiana", "Amritsar", "Jalandhar", "Patiala"]
        }
        
        self.company_names = [
            "Samsung Electronics India", "HDFC Bank Ltd", "ICICI Bank Ltd", "Reliance Industries",
            "Tata Motors", "Bajaj Auto", "Maruti Suzuki", "Infosys Ltd", "TCS Ltd", "Wipro Ltd"
        ]
        
    def generate_property_dispute_case(self, config: GenerationConfig) -> Dict[str, Any]:
        """Generate a synthetic property dispute case."""
        case_id = f"PD-{random.randint(1000, 9999)}-2024"
        
        # Random selections
        dispute_types = ["title_dispute", "partition_suit", "boundary_dispute", "possession_dispute", "easement_dispute"]
        property_types = ["residential", "commercial", "agricultural", "industrial", "vacant_land"]
        
        dispute_type = random.choice(dispute_types)
        property_type = random.choice(property_types)
        
        # Select jurisdiction
        state = self._select_weighted_random(config.jurisdiction_weights)
        city = random.choice(self.indian_cities[state])
        
        # Generate property details
        if property_type == "agricultural":
            area = random.uniform(1000, 50000)  # sq ft
            value = area * random.uniform(50, 200)  # per sq ft
        elif property_type == "commercial":
            area = random.uniform(200, 5000)
            value = area * random.uniform(5000, 25000)
        else:  # residential
            area = random.uniform(500, 3000)
            value = area * random.uniform(2000, 15000)
            
        # Generate parties
        plaintiff_name = self._generate_indian_name()
        defendant_name = self._generate_indian_name()
        
        # Generate case details
        damages_claimed = random.uniform(50000, 1000000)
        
        # Calculate success probability based on case characteristics
        base_probability = 0.6
        if dispute_type == "title_dispute":
            base_probability += random.uniform(-0.2, 0.3)
        elif dispute_type == "partition_suit":
            base_probability += random.uniform(0.1, 0.3)
        elif dispute_type == "boundary_dispute":
            base_probability += random.uniform(-0.1, 0.2)
            
        success_probability = max(0.1, min(0.95, base_probability))
        
        # Generate compensation estimate
        compensation = {
            "damages": damages_claimed * random.uniform(0.5, 1.2),
            "costs": damages_claimed * random.uniform(0.05, 0.15),
            "total": 0
        }
        compensation["total"] = compensation["damages"] + compensation["costs"]
        
        return {
            "case_id": case_id,
            "case_type": "property_dispute",
            "input_data": {
                "dispute_type": dispute_type,
                "property_type": property_type,
                "property_address": f"{random.randint(1, 999)} {random.choice(['Main Road', 'Park Street', 'MG Road'])}, {city}, {state.title()}",
                "property_area": round(area, 2),
                "property_value": round(value, 2),
                "survey_number": f"{random.randint(100, 999)}/{random.randint(1, 10)}",
                "plaintiff_name": plaintiff_name,
                "plaintiff_address": f"{random.randint(1, 999)} {random.choice(['Street', 'Colony', 'Nagar'])}, {city}",
                "defendant_name": defendant_name,
                "defendant_address": f"{random.randint(1, 999)} {random.choice(['Avenue', 'Road', 'Lane'])}, {city}",
                "title_documents": random.sample(["Sale Deed", "Registry", "Mutation Records", "Revenue Records", "Survey Settlement"], k=random.randint(2, 4)),
                "dispute_description": f"{dispute_type.replace('_', ' ').title()} regarding {property_type} property",
                "possession_status": random.choice(["Plaintiff in possession", "Defendant in possession", "Joint possession", "Disputed possession"]),
                "case_stage": random.choice(["pre_litigation", "suit_filed", "trial", "evidence", "arguments"]),
                "possession_sought": random.choice([True, False]),
                "damages_claimed": round(damages_claimed, 2),
                "declaration_sought": random.choice([True, False])
            },
            "expected_output": {
                "title_strength": {
                    "plaintiff": round(random.uniform(0.3, 0.9), 2),
                    "defendant": round(random.uniform(0.1, 0.7), 2)
                },
                "possession_rights": {
                    "current_possession": round(random.uniform(0.2, 0.9), 2),
                    "legal_possession": round(random.uniform(0.3, 0.9), 2)
                },
                "success_probability": round(success_probability, 2),
                "compensation_estimate": {
                    "damages": round(compensation["damages"], 2),
                    "costs": round(compensation["costs"], 2),
                    "total": round(compensation["total"], 2)
                },
                "recommendations": self._generate_property_recommendations(dispute_type),
                "case_duration_estimate": {
                    "min_months": random.randint(6, 18),
                    "max_months": random.randint(18, 48)
                },
                "litigation_cost_estimate": round(value * random.uniform(0.02, 0.08), 2)
            },
            "metadata": {
                "source": "synthetic",
                "jurisdiction": "india",
                "state": state,
                "date_created": self._generate_random_date(config.start_date, config.end_date),
                "complexity_level": self._select_weighted_random(config.complexity_distribution),
                "verified": True,
                "legal_principles": self._get_property_legal_principles(dispute_type)
            }
        }
        
    def generate_motor_vehicle_case(self, config: GenerationConfig) -> Dict[str, Any]:
        """Generate a synthetic motor vehicle claim case."""
        case_id = f"MVC-{random.randint(1000, 9999)}-2024"
        
        # Random selections
        accident_types = ["head_on_collision", "rear_end_collision", "side_impact", "hit_and_run", "rollover"]
        injury_types = ["simple_hurt", "grievous_hurt", "permanent_disability", "fatal", "minor_injury"]
        
        accident_type = random.choice(accident_types)
        injury_type = random.choice(injury_types)
        
        # Select jurisdiction
        state = self._select_weighted_random(config.jurisdiction_weights)
        city = random.choice(self.indian_cities[state])
        
        # Generate claimant details
        claimant_name = self._generate_indian_name()
        claimant_age = random.randint(18, 65)
        
        # Generate financial details based on injury severity
        if injury_type == "fatal":
            medical_expenses = random.uniform(100000, 500000)
            loss_of_income = random.uniform(500000, 2000000)
        elif injury_type == "permanent_disability":
            medical_expenses = random.uniform(200000, 1000000)
            loss_of_income = random.uniform(300000, 1500000)
        elif injury_type == "grievous_hurt":
            medical_expenses = random.uniform(50000, 400000)
            loss_of_income = random.uniform(50000, 300000)
        else:
            medical_expenses = random.uniform(5000, 50000)
            loss_of_income = random.uniform(5000, 50000)
            
        vehicle_damage = random.uniform(10000, 200000)
        
        # Calculate success probability
        base_probability = 0.7
        if accident_type == "hit_and_run":
            base_probability -= 0.2
        if injury_type in ["fatal", "permanent_disability"]:
            base_probability += 0.1
            
        success_probability = max(0.1, min(0.95, base_probability + random.uniform(-0.1, 0.1)))
        
        # Generate compensation
        pain_suffering = medical_expenses * random.uniform(0.3, 0.8)
        total_compensation = medical_expenses + loss_of_income + pain_suffering + vehicle_damage
        
        return {
            "case_id": case_id,
            "case_type": "motor_vehicle_claim",
            "input_data": {
                "accident_type": accident_type,
                "accident_date": self._generate_random_date(config.start_date, config.end_date),
                "accident_location": f"NH-{random.randint(1, 50)}, {city}, {state.title()}",
                "accident_time": f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
                "weather_conditions": random.choice(["clear", "rainy", "foggy", "cloudy"]),
                "road_conditions": random.choice(["good", "poor", "under_construction", "poor_visibility"]),
                "claimant_name": claimant_name,
                "claimant_age": claimant_age,
                "claimant_occupation": random.choice(["Software Engineer", "Teacher", "Business Owner", "Driver", "Student"]),
                "claimant_monthly_income": round(random.uniform(15000, 150000), 2),
                "injury_type": injury_type,
                "injury_description": self._generate_injury_description(injury_type),
                "medical_expenses": round(medical_expenses, 2),
                "loss_of_income": round(loss_of_income, 2),
                "vehicle_damage": round(vehicle_damage, 2),
                "at_fault_driver": random.choice(["Opposite vehicle driver", "Claimant", "Unknown (Hit and Run)", "Both parties"]),
                "insurance_company": random.choice(["ICICI Lombard", "HDFC ERGO", "New India Assurance", "Oriental Insurance"]),
                "policy_number": f"POL{random.randint(100000000, 999999999)}",
                "fir_number": f"FIR/2024/{random.randint(100000, 999999)}",
                "police_station": f"{city} Police Station",
                "witnesses": [f"{self._generate_indian_name()} - {random.randint(9000000000, 9999999999)}"]
            },
            "expected_output": {
                "liability_assessment": {
                    "claimant_fault": round(random.uniform(0.0, 0.3), 2),
                    "opposite_party_fault": round(random.uniform(0.7, 1.0), 2)
                },
                "compensation_estimate": {
                    "medical_expenses": round(medical_expenses, 2),
                    "loss_of_income": round(loss_of_income, 2),
                    "pain_and_suffering": round(pain_suffering, 2),
                    "vehicle_damage": round(vehicle_damage, 2),
                    "total": round(total_compensation, 2)
                },
                "insurance_coverage": {
                    "covered_amount": round(total_compensation * random.uniform(0.8, 1.2), 2),
                    "deductible": round(random.uniform(2000, 10000), 2),
                    "coverage_percentage": round(random.uniform(0.8, 1.0), 2)
                },
                "success_probability": round(success_probability, 2),
                "recommendations": self._generate_mvc_recommendations(accident_type, injury_type),
                "case_duration_estimate": {
                    "min_months": random.randint(3, 12),
                    "max_months": random.randint(12, 36)
                },
                "litigation_cost_estimate": round(total_compensation * random.uniform(0.05, 0.15), 2)
            },
            "metadata": {
                "source": "synthetic",
                "jurisdiction": "india",
                "state": state,
                "date_created": self._generate_random_date(config.start_date, config.end_date),
                "complexity_level": self._select_weighted_random(config.complexity_distribution),
                "verified": True,
                "legal_principles": ["motor_vehicles_act_1988", "negligence", "compensation"]
            }
        }
        
    def generate_consumer_dispute_case(self, config: GenerationConfig) -> Dict[str, Any]:
        """Generate a synthetic consumer dispute case."""
        case_id = f"CD-{random.randint(1000, 9999)}-2024"
        
        # Random selections
        complaint_types = ["defective_goods", "deficient_service", "unfair_trade_practice", "overcharging", "false_advertisement"]
        
        complaint_type = random.choice(complaint_types)
        
        # Select jurisdiction
        state = self._select_weighted_random(config.jurisdiction_weights)
        city = random.choice(self.indian_cities[state])
        
        # Generate consumer details
        consumer_name = self._generate_indian_name()
        
        # Generate product/service details
        if complaint_type == "defective_goods":
            products = ["Mobile Phone", "Laptop", "Television", "Refrigerator", "Washing Machine"]
            product = random.choice(products)
            purchase_amount = random.uniform(15000, 150000)
        elif complaint_type == "deficient_service":
            services = ["Banking Service", "Insurance Service", "Telecom Service", "Internet Service"]
            product = random.choice(services)
            purchase_amount = random.uniform(1000, 50000)
        else:
            products = ["Electronics", "Furniture", "Clothing", "Automobile"]
            product = random.choice(products)
            purchase_amount = random.uniform(5000, 200000)
            
        compensation_claimed = purchase_amount * random.uniform(0.2, 2.0)
        
        # Calculate success probability
        base_probability = 0.75
        if complaint_type == "defective_goods":
            base_probability += 0.1
        elif complaint_type == "unfair_trade_practice":
            base_probability += 0.05
            
        success_probability = max(0.1, min(0.95, base_probability + random.uniform(-0.15, 0.15)))
        
        return {
            "case_id": case_id,
            "case_type": "consumer_dispute",
            "input_data": {
                "complaint_type": complaint_type,
                "consumer_name": consumer_name,
                "consumer_address": f"{random.randint(1, 999)} {random.choice(['Street', 'Road', 'Colony'])}, {city}, {state.title()}",
                "consumer_phone": str(random.randint(9000000000, 9999999999)),
                "consumer_email": f"{consumer_name.lower().replace(' ', '.')}@email.com",
                "opposite_party": random.choice(self.company_names),
                "opposite_party_address": f"{random.choice(['Tower', 'Complex', 'Building'])}, {city}, {state.title()}",
                "product_service": product,
                "purchase_date": self._generate_random_date(config.start_date, config.end_date),
                "purchase_amount": round(purchase_amount, 2),
                "complaint_description": self._generate_complaint_description(complaint_type, product),
                "evidence_documents": random.sample(["Purchase Invoice", "Warranty Card", "Email Communications", "Photos", "Service Records"], k=random.randint(2, 4)),
                "forum_level": random.choice(["district", "state", "national"]),
                "relief_sought": f"Refund/Replacement of {product} and compensation",
                "compensation_claimed": round(compensation_claimed, 2),
                "previous_complaints": random.choice([True, False]),
                "legal_notice_sent": random.choice([True, False]),
                "legal_notice_date": self._generate_random_date(config.start_date, config.end_date) if random.choice([True, False]) else None
            },
            "expected_output": {
                "jurisdiction_validity": True,
                "forum_competence": random.choice(["district_forum", "state_commission", "national_commission"]),
                "evidence_strength": {
                    "documentary_evidence": round(random.uniform(0.6, 0.9), 2),
                    "witness_evidence": round(random.uniform(0.3, 0.7), 2),
                    "expert_evidence": round(random.uniform(0.5, 0.8), 2),
                    "overall_strength": round(random.uniform(0.6, 0.85), 2)
                },
                "success_probability": round(success_probability, 2),
                "relief_likelihood": {
                    "refund": round(random.uniform(0.6, 0.9), 2),
                    "replacement": round(random.uniform(0.5, 0.8), 2),
                    "compensation": round(random.uniform(0.4, 0.8), 2),
                    "punitive_damages": round(random.uniform(0.2, 0.6), 2)
                },
                "estimated_compensation": {
                    "product_value": round(purchase_amount, 2),
                    "mental_agony": round(purchase_amount * random.uniform(0.1, 0.3), 2),
                    "litigation_costs": round(purchase_amount * random.uniform(0.05, 0.1), 2),
                    "total": round(compensation_claimed, 2)
                },
                "recommendations": self._generate_consumer_recommendations(complaint_type),
                "case_duration_estimate": {
                    "min_months": random.randint(4, 12),
                    "max_months": random.randint(12, 30)
                },
                "litigation_cost_estimate": round(purchase_amount * random.uniform(0.1, 0.25), 2)
            },
            "metadata": {
                "source": "synthetic",
                "jurisdiction": "india",
                "state": state,
                "date_created": self._generate_random_date(config.start_date, config.end_date),
                "complexity_level": self._select_weighted_random(config.complexity_distribution),
                "verified": True,
                "legal_principles": ["consumer_protection_act_2019", complaint_type, "consumer_rights"]
            }
        }
        
    def generate_dataset(self, case_type: str, config: GenerationConfig) -> List[Dict[str, Any]]:
        """
        Generate a complete synthetic dataset for a case type.
        
        Args:
            case_type: Type of legal case to generate
            config: Generation configuration
            
        Returns:
            List of synthetic cases
        """
        cases = []
        
        for i in range(config.num_cases):
            try:
                if case_type == "property_disputes":
                    case = self.generate_property_dispute_case(config)
                elif case_type == "motor_vehicle_claims":
                    case = self.generate_motor_vehicle_case(config)
                elif case_type == "consumer_disputes":
                    case = self.generate_consumer_dispute_case(config)
                else:
                    raise ValueError(f"Unsupported case type: {case_type}")
                    
                cases.append(case)
                
            except Exception as e:
                logger.error(f"Error generating case {i} for {case_type}: {e}")
                continue
                
        logger.info(f"Generated {len(cases)} synthetic cases for {case_type}")
        return cases
        
    def save_dataset(self, cases: List[Dict[str, Any]], output_path: str):
        """Save generated dataset to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(cases)} cases to {output_path}")
        
    # Helper methods
    def _generate_indian_name(self) -> str:
        """Generate a random Indian name."""
        gender = random.choice(["male", "female"])
        first_name = random.choice(self.indian_names[gender])
        last_names = ["Kumar", "Singh", "Sharma", "Patel", "Gupta", "Agarwal", "Jain", "Shah", "Verma", "Yadav"]
        last_name = random.choice(last_names)
        return f"{first_name} {last_name}"
        
    def _select_weighted_random(self, weights: Dict[str, float]) -> str:
        """Select random item based on weights."""
        items = list(weights.keys())
        weights_list = list(weights.values())
        return random.choices(items, weights=weights_list)[0]
        
    def _generate_random_date(self, start_date: str, end_date: str) -> str:
        """Generate random date between start and end dates."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        time_between = end - start
        days_between = time_between.days
        random_days = random.randrange(days_between)
        
        random_date = start + timedelta(days=random_days)
        return random_date.strftime("%Y-%m-%d")
        
    def _generate_property_recommendations(self, dispute_type: str) -> List[str]:
        """Generate recommendations for property disputes."""
        base_recommendations = [
            "Collect all relevant title documents",
            "Conduct professional property survey",
            "File case within limitation period"
        ]
        
        if dispute_type == "title_dispute":
            base_recommendations.extend([
                "Verify chain of title thoroughly",
                "Check for any encumbrances or liens"
            ])
        elif dispute_type == "boundary_dispute":
            base_recommendations.extend([
                "Obtain detailed boundary survey",
                "Review original property plans"
            ])
        elif dispute_type == "partition_suit":
            base_recommendations.extend([
                "Prepare family genealogy chart",
                "Attempt family settlement first"
            ])
            
        return random.sample(base_recommendations, k=min(4, len(base_recommendations)))
        
    def _get_property_legal_principles(self, dispute_type: str) -> List[str]:
        """Get legal principles for property disputes."""
        principles_map = {
            "title_dispute": ["title_by_registration", "adverse_possession", "burden_of_proof"],
            "boundary_dispute": ["boundary_determination", "encroachment", "survey_evidence"],
            "partition_suit": ["partition_by_metes_and_bounds", "coparcenary_rights", "family_settlement"],
            "possession_dispute": ["possession_rights", "trespass", "injunctive_relief"],
            "easement_dispute": ["easement_rights", "dominant_servient_tenement", "prescription"]
        }
        return principles_map.get(dispute_type, ["property_law", "civil_procedure"])
        
    def _generate_injury_description(self, injury_type: str) -> str:
        """Generate injury description for motor vehicle cases."""
        descriptions = {
            "simple_hurt": "Minor cuts and bruises, soft tissue injury",
            "grievous_hurt": "Multiple fractures, significant soft tissue damage",
            "permanent_disability": "Spinal cord injury resulting in permanent disability",
            "fatal": "Fatal injuries resulting in death",
            "minor_injury": "Minor scratches and bruises"
        }
        return descriptions.get(injury_type, "Injury details to be assessed")
        
    def _generate_mvc_recommendations(self, accident_type: str, injury_type: str) -> List[str]:
        """Generate recommendations for motor vehicle claims."""
        recommendations = [
            "File claim within statutory limitation period",
            "Collect all medical records and bills",
            "Obtain police investigation report"
        ]
        
        if accident_type == "hit_and_run":
            recommendations.append("Apply to MACT Solatium Fund")
        if injury_type in ["permanent_disability", "fatal"]:
            recommendations.append("Obtain disability/death certificate")
            
        return recommendations
        
    def _generate_complaint_description(self, complaint_type: str, product: str) -> str:
        """Generate complaint description for consumer disputes."""
        descriptions = {
            "defective_goods": f"{product} developed defects within warranty period",
            "deficient_service": f"Poor quality service provided for {product}",
            "unfair_trade_practice": f"Misleading claims made about {product}",
            "overcharging": f"Excessive charges levied for {product}",
            "false_advertisement": f"False claims in advertisement for {product}"
        }
        return descriptions.get(complaint_type, f"Issue with {product}")
        
    def _generate_consumer_recommendations(self, complaint_type: str) -> List[str]:
        """Generate recommendations for consumer disputes."""
        base_recommendations = [
            "File complaint in appropriate consumer forum",
            "Collect all purchase and communication records",
            "Send legal notice before filing complaint"
        ]
        
        if complaint_type == "defective_goods":
            base_recommendations.append("Obtain technical expert opinion")
        elif complaint_type == "deficient_service":
            base_recommendations.append("Document all service interactions")
        elif complaint_type == "unfair_trade_practice":
            base_recommendations.append("Collect advertisement materials as evidence")
            
        return base_recommendations

# Convenience function
def generate_synthetic_dataset(case_type: str, num_cases: int = 100, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate synthetic dataset for a case type.
    
    Args:
        case_type: Type of legal case
        num_cases: Number of cases to generate
        output_path: Optional path to save the dataset
        
    Returns:
        List of synthetic cases
    """
    generator = SyntheticDataGenerator()
    config = GenerationConfig(num_cases=num_cases)
    
    cases = generator.generate_dataset(case_type, config)
    
    if output_path:
        generator.save_dataset(cases, output_path)
        
    return cases