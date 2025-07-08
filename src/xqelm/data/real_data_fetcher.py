"""
Real Legal Data Fetcher for XQELM

This module provides functionality to fetch, process, and integrate real legal data
from various Indian legal databases and public sources.
"""

import asyncio
import aiohttp
import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import re
import time

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a legal data source."""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 10  # requests per minute
    requires_auth: bool = False
    data_format: str = "json"  # json, xml, csv
    case_types: List[str] = None

@dataclass
class LegalCase:
    """Standardized legal case structure."""
    case_id: str
    case_type: str
    court_name: str
    case_number: str
    filing_date: str
    decision_date: Optional[str]
    case_status: str
    parties: Dict[str, str]
    case_facts: str
    legal_issues: List[str]
    judgment_summary: str
    citations: List[str]
    applicable_laws: List[str]
    case_outcome: str
    compensation_awarded: Optional[float]
    source: str
    jurisdiction: str
    complexity_level: str

class RealLegalDataFetcher:
    """
    Fetches real legal data from various Indian legal databases and sources.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the data fetcher with configured sources."""
        self.data_sources = self._load_data_sources(config_path)
        self.session = None
        self.rate_limiters = {}
        
    def _load_data_sources(self, config_path: Optional[str]) -> List[DataSource]:
        """Load data source configurations."""
        # Indian Legal Data Sources
        sources = [
            DataSource(
                name="Indian Kanoon",
                base_url="https://api.indiankanoon.org",
                rate_limit=60,
                case_types=["civil", "criminal", "constitutional", "consumer", "motor_vehicle"]
            ),
            DataSource(
                name="Supreme Court of India",
                base_url="https://main.sci.gov.in/api",
                rate_limit=30,
                case_types=["constitutional", "civil", "criminal"]
            ),
            DataSource(
                name="High Court Records",
                base_url="https://hcservices.ecourts.gov.in/api",
                rate_limit=20,
                case_types=["civil", "criminal", "consumer", "property"]
            ),
            DataSource(
                name="Consumer Forum Database",
                base_url="https://confonet.nic.in/api",
                rate_limit=15,
                case_types=["consumer_disputes"]
            ),
            DataSource(
                name="Motor Accident Claims Tribunal",
                base_url="https://mact.ecourts.gov.in/api",
                rate_limit=10,
                case_types=["motor_vehicle_claims"]
            ),
            DataSource(
                name="Property Dispute Records",
                base_url="https://landrecords.gov.in/api",
                rate_limit=20,
                case_types=["property_disputes"]
            )
        ]
        
        # Load custom configuration if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                # Merge with default sources
                for source_config in custom_config.get('data_sources', []):
                    sources.append(DataSource(**source_config))
        
        return sources
    
    async def fetch_cases_by_type(
        self, 
        case_type: str, 
        limit: int = 100,
        date_range: Optional[Tuple[str, str]] = None,
        jurisdiction: Optional[str] = None
    ) -> List[LegalCase]:
        """
        Fetch real legal cases by type from configured sources.
        
        Args:
            case_type: Type of legal case (e.g., 'consumer_disputes')
            limit: Maximum number of cases to fetch
            date_range: Optional date range (start_date, end_date)
            jurisdiction: Optional jurisdiction filter
            
        Returns:
            List of standardized legal cases
        """
        logger.info(f"Fetching {limit} cases of type '{case_type}'")
        
        all_cases = []
        
        # Filter sources that support this case type
        relevant_sources = [
            source for source in self.data_sources 
            if case_type in source.case_types
        ]
        
        if not relevant_sources:
            logger.warning(f"No data sources configured for case type: {case_type}")
            return []
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            for source in relevant_sources:
                try:
                    cases = await self._fetch_from_source(
                        source, case_type, limit // len(relevant_sources),
                        date_range, jurisdiction
                    )
                    all_cases.extend(cases)
                    
                    logger.info(f"Fetched {len(cases)} cases from {source.name}")
                    
                except Exception as e:
                    logger.error(f"Error fetching from {source.name}: {e}")
                    continue
        
        # Remove duplicates and standardize
        unique_cases = self._deduplicate_cases(all_cases)
        standardized_cases = [self._standardize_case(case) for case in unique_cases]
        
        logger.info(f"Total unique cases fetched: {len(standardized_cases)}")
        return standardized_cases[:limit]
    
    async def _fetch_from_source(
        self,
        source: DataSource,
        case_type: str,
        limit: int,
        date_range: Optional[Tuple[str, str]],
        jurisdiction: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Fetch cases from a specific data source."""
        
        # Rate limiting
        await self._apply_rate_limit(source.name, source.rate_limit)
        
        # Build query parameters
        params = {
            'case_type': case_type,
            'limit': limit,
            'format': 'json'
        }
        
        if date_range:
            params['start_date'] = date_range[0]
            params['end_date'] = date_range[1]
            
        if jurisdiction:
            params['jurisdiction'] = jurisdiction
        
        # Add API key if required
        if source.api_key:
            params['api_key'] = source.api_key
        
        # Construct URL based on source
        if source.name == "Indian Kanoon":
            url = f"{source.base_url}/search"
            params['doctype'] = 'judgments'
        elif source.name == "Consumer Forum Database":
            url = f"{source.base_url}/cases/search"
        elif source.name == "Motor Accident Claims Tribunal":
            url = f"{source.base_url}/claims/search"
        else:
            url = f"{source.base_url}/cases"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_response(data, source)
                else:
                    logger.error(f"HTTP {response.status} from {source.name}")
                    return []
                    
        except Exception as e:
            logger.error(f"Request failed for {source.name}: {e}")
            return []
    
    def _parse_response(self, data: Dict[str, Any], source: DataSource) -> List[Dict[str, Any]]:
        """Parse response data based on source format."""
        
        cases = []
        
        if source.name == "Indian Kanoon":
            # Parse Indian Kanoon response format
            for doc in data.get('docs', []):
                case = {
                    'case_id': doc.get('tid'),
                    'title': doc.get('title'),
                    'court': doc.get('court'),
                    'date': doc.get('date'),
                    'content': doc.get('content'),
                    'citations': doc.get('citations', []),
                    'source': source.name
                }
                cases.append(case)
                
        elif source.name == "Consumer Forum Database":
            # Parse consumer forum format
            for case_data in data.get('cases', []):
                case = {
                    'case_id': case_data.get('case_number'),
                    'consumer_name': case_data.get('complainant'),
                    'opposite_party': case_data.get('opposite_party'),
                    'complaint_type': case_data.get('nature_of_complaint'),
                    'forum_level': case_data.get('forum'),
                    'case_status': case_data.get('status'),
                    'relief_sought': case_data.get('relief'),
                    'order_details': case_data.get('order'),
                    'source': source.name
                }
                cases.append(case)
                
        elif source.name == "Motor Accident Claims Tribunal":
            # Parse MACT format
            for claim in data.get('claims', []):
                case = {
                    'case_id': claim.get('claim_number'),
                    'accident_date': claim.get('accident_date'),
                    'claimant_name': claim.get('claimant'),
                    'respondent': claim.get('respondent'),
                    'injury_type': claim.get('injury_nature'),
                    'compensation_claimed': claim.get('amount_claimed'),
                    'compensation_awarded': claim.get('amount_awarded'),
                    'tribunal_order': claim.get('order'),
                    'source': source.name
                }
                cases.append(case)
        
        else:
            # Generic parsing for other sources
            cases = data.get('cases', data.get('results', []))
            for case in cases:
                case['source'] = source.name
        
        return cases
    
    def _standardize_case(self, raw_case: Dict[str, Any]) -> LegalCase:
        """Convert raw case data to standardized LegalCase format."""
        
        source = raw_case.get('source', 'unknown')
        
        # Extract common fields
        case_id = raw_case.get('case_id', f"CASE-{int(time.time())}")
        
        # Determine case type from content
        case_type = self._infer_case_type(raw_case)
        
        # Extract parties
        parties = self._extract_parties(raw_case)
        
        # Extract legal issues and facts
        legal_issues = self._extract_legal_issues(raw_case)
        case_facts = self._extract_case_facts(raw_case)
        
        # Extract outcome and compensation
        case_outcome = self._extract_outcome(raw_case)
        compensation = self._extract_compensation(raw_case)
        
        # Determine complexity
        complexity = self._assess_complexity(raw_case)
        
        return LegalCase(
            case_id=case_id,
            case_type=case_type,
            court_name=raw_case.get('court', 'Unknown Court'),
            case_number=raw_case.get('case_number', case_id),
            filing_date=raw_case.get('filing_date', raw_case.get('date', '')),
            decision_date=raw_case.get('decision_date'),
            case_status=raw_case.get('case_status', 'unknown'),
            parties=parties,
            case_facts=case_facts,
            legal_issues=legal_issues,
            judgment_summary=raw_case.get('order_details', raw_case.get('content', '')),
            citations=raw_case.get('citations', []),
            applicable_laws=self._extract_applicable_laws(raw_case),
            case_outcome=case_outcome,
            compensation_awarded=compensation,
            source=source,
            jurisdiction=raw_case.get('jurisdiction', 'india'),
            complexity_level=complexity
        )
    
    def _infer_case_type(self, case_data: Dict[str, Any]) -> str:
        """Infer case type from case data."""
        
        # Check explicit case type
        if 'case_type' in case_data:
            return case_data['case_type']
        
        # Infer from content
        content = str(case_data.get('content', '') + 
                     case_data.get('title', '') + 
                     case_data.get('complaint_type', '')).lower()
        
        if any(term in content for term in ['consumer', 'defective', 'service', 'goods']):
            return 'consumer_dispute'
        elif any(term in content for term in ['motor', 'vehicle', 'accident', 'mact']):
            return 'motor_vehicle_claim'
        elif any(term in content for term in ['property', 'land', 'title', 'possession']):
            return 'property_dispute'
        elif any(term in content for term in ['bail', 'custody', 'arrest']):
            return 'bail_application'
        elif any(term in content for term in ['cheque', 'dishonour', 'bounce']):
            return 'cheque_bounce'
        else:
            return 'general_civil'
    
    def _extract_parties(self, case_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract party information from case data."""
        parties = {}
        
        # Common party fields
        if 'complainant' in case_data:
            parties['plaintiff'] = case_data['complainant']
        if 'claimant_name' in case_data:
            parties['plaintiff'] = case_data['claimant_name']
        if 'consumer_name' in case_data:
            parties['plaintiff'] = case_data['consumer_name']
            
        if 'opposite_party' in case_data:
            parties['defendant'] = case_data['opposite_party']
        if 'respondent' in case_data:
            parties['defendant'] = case_data['respondent']
        
        return parties
    
    def _extract_legal_issues(self, case_data: Dict[str, Any]) -> List[str]:
        """Extract legal issues from case content."""
        issues = []
        
        content = case_data.get('content', '') + case_data.get('order_details', '')
        
        # Common legal issue patterns
        issue_patterns = [
            r'issue.*?is.*?whether',
            r'question.*?law',
            r'legal.*?issue',
            r'point.*?determination'
        ]
        
        for pattern in issue_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            issues.extend(matches)
        
        return issues[:5]  # Limit to top 5 issues
    
    def _extract_case_facts(self, case_data: Dict[str, Any]) -> str:
        """Extract case facts summary."""
        facts = []
        
        if 'complaint_description' in case_data:
            facts.append(case_data['complaint_description'])
        if 'case_facts' in case_data:
            facts.append(case_data['case_facts'])
        if 'content' in case_data:
            # Extract first paragraph as facts
            content = case_data['content']
            first_para = content.split('\n')[0] if content else ''
            if len(first_para) > 50:
                facts.append(first_para[:500])
        
        return ' '.join(facts)
    
    def _extract_outcome(self, case_data: Dict[str, Any]) -> str:
        """Extract case outcome."""
        if 'case_outcome' in case_data:
            return case_data['case_outcome']
        
        status = case_data.get('case_status', '').lower()
        if 'allowed' in status or 'granted' in status:
            return 'favorable'
        elif 'dismissed' in status or 'rejected' in status:
            return 'unfavorable'
        else:
            return 'pending'
    
    def _extract_compensation(self, case_data: Dict[str, Any]) -> Optional[float]:
        """Extract compensation amount if available."""
        if 'compensation_awarded' in case_data:
            try:
                return float(case_data['compensation_awarded'])
            except (ValueError, TypeError):
                pass
        
        # Try to extract from order text
        order_text = case_data.get('order_details', case_data.get('content', ''))
        
        # Look for currency amounts
        amount_patterns = [
            r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'rupees\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, order_text, re.IGNORECASE)
            if matches:
                try:
                    amount_str = matches[0].replace(',', '')
                    return float(amount_str)
                except ValueError:
                    continue
        
        return None
    
    def _extract_applicable_laws(self, case_data: Dict[str, Any]) -> List[str]:
        """Extract applicable laws and statutes."""
        laws = []
        
        content = case_data.get('content', '') + case_data.get('order_details', '')
        
        # Common Indian laws
        law_patterns = [
            r'Consumer Protection Act.*?(\d{4})',
            r'Motor Vehicles Act.*?(\d{4})',
            r'Transfer of Property Act.*?(\d{4})',
            r'Indian Penal Code',
            r'Code of Civil Procedure',
            r'Indian Evidence Act'
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            laws.extend(matches)
        
        return list(set(laws))  # Remove duplicates
    
    def _assess_complexity(self, case_data: Dict[str, Any]) -> str:
        """Assess case complexity based on various factors."""
        complexity_score = 0
        
        # Factors that increase complexity
        if case_data.get('compensation_awarded', 0) > 1000000:
            complexity_score += 2
        elif case_data.get('compensation_awarded', 0) > 100000:
            complexity_score += 1
        
        if len(case_data.get('citations', [])) > 5:
            complexity_score += 1
        
        content_length = len(case_data.get('content', ''))
        if content_length > 5000:
            complexity_score += 2
        elif content_length > 2000:
            complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 4:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _deduplicate_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate cases based on case ID and content similarity."""
        seen_ids = set()
        unique_cases = []
        
        for case in cases:
            case_id = case.get('case_id')
            if case_id and case_id not in seen_ids:
                seen_ids.add(case_id)
                unique_cases.append(case)
        
        return unique_cases
    
    async def _apply_rate_limit(self, source_name: str, rate_limit: int):
        """Apply rate limiting for API requests."""
        if source_name not in self.rate_limiters:
            self.rate_limiters[source_name] = []
        
        now = time.time()
        # Remove requests older than 1 minute
        self.rate_limiters[source_name] = [
            req_time for req_time in self.rate_limiters[source_name]
            if now - req_time < 60
        ]
        
        # Check if we need to wait
        if len(self.rate_limiters[source_name]) >= rate_limit:
            sleep_time = 60 - (now - self.rate_limiters[source_name][0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.rate_limiters[source_name].append(now)
    
    async def save_cases_to_dataset(
        self, 
        cases: List[LegalCase], 
        case_type: str,
        output_dir: str = "data/datasets"
    ) -> str:
        """Save fetched cases to dataset format."""
        
        # Convert to XQELM dataset format
        dataset_cases = []
        
        for case in cases:
            dataset_case = self._convert_to_dataset_format(case, case_type)
            dataset_cases.append(dataset_case)
        
        # Save to file
        output_path = Path(output_dir) / case_type / "real_cases.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_cases, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(dataset_cases)} real cases to {output_path}")
        return str(output_path)
    
    def _convert_to_dataset_format(self, case: LegalCase, case_type: str) -> Dict[str, Any]:
        """Convert LegalCase to XQELM dataset format."""
        
        # Base structure
        dataset_case = {
            "case_id": case.case_id,
            "case_type": case_type,
            "input_data": {},
            "expected_output": {},
            "metadata": {
                "source": "real_data",
                "jurisdiction": case.jurisdiction,
                "date_created": datetime.now().isoformat(),
                "complexity_level": case.complexity_level,
                "verified": True,
                "legal_principles": case.applicable_laws,
                "original_source": case.source
            }
        }
        
        # Case-specific input data
        if case_type == "consumer_dispute":
            dataset_case["input_data"] = {
                "consumer_name": case.parties.get('plaintiff', ''),
                "opposite_party": case.parties.get('defendant', ''),
                "complaint_description": case.case_facts,
                "case_status": case.case_status,
                "court_name": case.court_name
            }
            
            dataset_case["expected_output"] = {
                "case_outcome": case.case_outcome,
                "compensation_awarded": case.compensation_awarded,
                "success_probability": 1.0 if case.case_outcome == 'favorable' else 0.0,
                "judgment_summary": case.judgment_summary
            }
            
        elif case_type == "motor_vehicle_claim":
            dataset_case["input_data"] = {
                "claimant_name": case.parties.get('plaintiff', ''),
                "respondent": case.parties.get('defendant', ''),
                "case_facts": case.case_facts,
                "compensation_claimed": case.compensation_awarded
            }
            
            dataset_case["expected_output"] = {
                "compensation_awarded": case.compensation_awarded,
                "case_outcome": case.case_outcome,
                "success_probability": 1.0 if case.case_outcome == 'favorable' else 0.0
            }
            
        elif case_type == "property_dispute":
            dataset_case["input_data"] = {
                "plaintiff_name": case.parties.get('plaintiff', ''),
                "defendant_name": case.parties.get('defendant', ''),
                "dispute_description": case.case_facts,
                "legal_issues": case.legal_issues
            }
            
            dataset_case["expected_output"] = {
                "case_outcome": case.case_outcome,
                "success_probability": 1.0 if case.case_outcome == 'favorable' else 0.0,
                "compensation_estimate": case.compensation_awarded
            }
        
        return dataset_case

# Convenience functions
async def fetch_real_legal_data(
    case_type: str, 
    num_cases: int = 100,
    output_dir: str = "data/datasets"
) -> str:
    """Fetch real legal data and save to dataset format."""
    
    fetcher = RealLegalDataFetcher()
    
    # Fetch cases
    cases = await fetcher.fetch_cases_by_type(case_type, num_cases)
    
    # Save to dataset
    output_path = await fetcher.save_cases_to_dataset(cases, case_type, output_dir)
    
    return output_path

async def update_all_datasets_with_real_data(num_cases_per_type: int = 50):
    """Update all case types with real data."""
    
    case_types = ["consumer_disputes", "motor_vehicle_claims", "property_disputes"]
    
    for case_type in case_types:
        try:
            output_path = await fetch_real_legal_data(case_type, num_cases_per_type)
            logger.info(f"Updated {case_type} with real data: {output_path}")
        except Exception as e:
            logger.error(f"Failed to update {case_type}: {e}")