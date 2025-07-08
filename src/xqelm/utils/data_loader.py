"""
Data loading utilities for XQELM training and testing datasets.
Provides functionality to load, validate, and process legal case datasets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DatasetMetrics:
    """Metrics for dataset validation and analysis."""
    total_cases: int
    case_types: Dict[str, int]
    complexity_distribution: Dict[str, int]
    jurisdictions: Dict[str, int]
    date_range: Tuple[str, str]
    average_case_value: float
    success_rate_distribution: Dict[str, float]

class LegalDatasetLoader:
    """
    Loads and validates legal datasets for XQELM training and testing.
    Supports multiple case types and provides data quality validation.
    """
    
    def __init__(self, base_path: str = "data/datasets"):
        """
        Initialize the dataset loader.
        
        Args:
            base_path: Base directory path for datasets
        """
        self.base_path = Path(base_path)
        self.supported_case_types = [
            "property_disputes",
            "motor_vehicle_claims", 
            "consumer_disputes",
            "bail_applications",
            "cheque_bounce"
        ]
        
    def load_case_dataset(self, case_type: str) -> List[Dict[str, Any]]:
        """
        Load dataset for a specific case type.
        
        Args:
            case_type: Type of legal case (e.g., 'property_disputes')
            
        Returns:
            List of case dictionaries
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If case type is not supported
        """
        if case_type not in self.supported_case_types:
            raise ValueError(f"Unsupported case type: {case_type}")
            
        dataset_path = self.base_path / case_type / "sample_cases.json"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                cases = json.load(f)
                
            logger.info(f"Loaded {len(cases)} cases for {case_type}")
            return cases
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset {dataset_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_path}: {e}")
            raise
            
    def load_all_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all available datasets.
        
        Returns:
            Dictionary mapping case types to their datasets
        """
        all_datasets = {}
        
        for case_type in self.supported_case_types:
            try:
                dataset = self.load_case_dataset(case_type)
                all_datasets[case_type] = dataset
            except FileNotFoundError:
                logger.warning(f"Dataset not found for {case_type}, skipping")
                continue
            except Exception as e:
                logger.error(f"Failed to load {case_type}: {e}")
                continue
                
        return all_datasets
        
    def validate_case_structure(self, case: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate the structure of a single case.
        
        Args:
            case: Case dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        required_fields = ["case_id", "case_type", "input_data", "expected_output", "metadata"]
        
        # Check required top-level fields
        for field in required_fields:
            if field not in case:
                errors.append(f"Missing required field: {field}")
                
        # Validate input_data structure
        if "input_data" in case:
            input_data = case["input_data"]
            if not isinstance(input_data, dict):
                errors.append("input_data must be a dictionary")
            elif len(input_data) == 0:
                errors.append("input_data cannot be empty")
                
        # Validate expected_output structure
        if "expected_output" in case:
            expected_output = case["expected_output"]
            if not isinstance(expected_output, dict):
                errors.append("expected_output must be a dictionary")
            elif "success_probability" not in expected_output:
                errors.append("expected_output must contain success_probability")
                
        # Validate metadata
        if "metadata" in case:
            metadata = case["metadata"]
            required_metadata = ["source", "jurisdiction", "date_created", "complexity_level"]
            for field in required_metadata:
                if field not in metadata:
                    errors.append(f"Missing metadata field: {field}")
                    
        return len(errors) == 0, errors
        
    def validate_dataset(self, case_type: str) -> Tuple[bool, List[str], DatasetMetrics]:
        """
        Validate entire dataset for a case type.
        
        Args:
            case_type: Type of legal case to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors, dataset_metrics)
        """
        try:
            cases = self.load_case_dataset(case_type)
        except Exception as e:
            return False, [f"Failed to load dataset: {e}"], None
            
        all_errors = []
        valid_cases = 0
        
        # Validate each case
        for i, case in enumerate(cases):
            is_valid, errors = self.validate_case_structure(case)
            if is_valid:
                valid_cases += 1
            else:
                all_errors.extend([f"Case {i}: {error}" for error in errors])
                
        # Calculate metrics
        metrics = self._calculate_metrics(cases)
        
        dataset_valid = len(all_errors) == 0
        if not dataset_valid:
            all_errors.insert(0, f"Dataset validation failed: {valid_cases}/{len(cases)} cases valid")
            
        return dataset_valid, all_errors, metrics
        
    def _calculate_metrics(self, cases: List[Dict[str, Any]]) -> DatasetMetrics:
        """Calculate dataset metrics for analysis."""
        if not cases:
            return DatasetMetrics(0, {}, {}, {}, ("", ""), 0.0, {})
            
        case_types = {}
        complexity_dist = {}
        jurisdictions = {}
        dates = []
        case_values = []
        success_rates = []
        
        for case in cases:
            # Case type distribution
            case_type = case.get("case_type", "unknown")
            case_types[case_type] = case_types.get(case_type, 0) + 1
            
            # Complexity distribution
            if "metadata" in case:
                complexity = case["metadata"].get("complexity_level", "unknown")
                complexity_dist[complexity] = complexity_dist.get(complexity, 0) + 1
                
                # Jurisdiction distribution
                jurisdiction = case["metadata"].get("state", "unknown")
                jurisdictions[jurisdiction] = jurisdictions.get(jurisdiction, 0) + 1
                
                # Date range
                date_created = case["metadata"].get("date_created")
                if date_created:
                    dates.append(date_created)
                    
            # Case values and success rates
            if "input_data" in case:
                if "property_value" in case["input_data"]:
                    case_values.append(case["input_data"]["property_value"])
                elif "purchase_amount" in case["input_data"]:
                    case_values.append(case["input_data"]["purchase_amount"])
                elif "damages_claimed" in case["input_data"]:
                    case_values.append(case["input_data"]["damages_claimed"])
                    
            if "expected_output" in case:
                success_prob = case["expected_output"].get("success_probability", 0.0)
                success_rates.append(success_prob)
                
        # Calculate date range
        date_range = ("", "")
        if dates:
            dates.sort()
            date_range = (dates[0], dates[-1])
            
        # Calculate average case value
        avg_case_value = sum(case_values) / len(case_values) if case_values else 0.0
        
        # Success rate distribution
        success_rate_dist = {}
        if success_rates:
            success_rate_dist = {
                "min": min(success_rates),
                "max": max(success_rates),
                "avg": sum(success_rates) / len(success_rates),
                "high_success": len([r for r in success_rates if r >= 0.8]) / len(success_rates)
            }
            
        return DatasetMetrics(
            total_cases=len(cases),
            case_types=case_types,
            complexity_distribution=complexity_dist,
            jurisdictions=jurisdictions,
            date_range=date_range,
            average_case_value=avg_case_value,
            success_rate_distribution=success_rate_dist
        )
        
    def export_training_data(self, case_type: str, output_format: str = "json") -> str:
        """
        Export dataset in format suitable for model training.
        
        Args:
            case_type: Type of legal case
            output_format: Export format ('json', 'csv', 'parquet')
            
        Returns:
            Path to exported file
        """
        cases = self.load_case_dataset(case_type)
        
        # Prepare training data
        training_data = []
        for case in cases:
            training_record = {
                "case_id": case["case_id"],
                "input_features": case["input_data"],
                "target_output": case["expected_output"],
                "metadata": case["metadata"]
            }
            training_data.append(training_record)
            
        # Export based on format
        output_dir = self.base_path / "processed"
        output_dir.mkdir(exist_ok=True)
        
        if output_format == "json":
            output_path = output_dir / f"{case_type}_training.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
                
        elif output_format == "csv":
            output_path = output_dir / f"{case_type}_training.csv"
            df = pd.json_normalize(training_data)
            df.to_csv(output_path, index=False)
            
        elif output_format == "parquet":
            output_path = output_dir / f"{case_type}_training.parquet"
            df = pd.json_normalize(training_data)
            df.to_parquet(output_path, index=False)
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        logger.info(f"Exported {len(training_data)} training records to {output_path}")
        return str(output_path)
        
    def generate_dataset_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report on all datasets.
        
        Returns:
            Dictionary containing dataset analysis report
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "datasets": {},
            "summary": {}
        }
        
        total_cases = 0
        all_case_types = set()
        all_jurisdictions = set()
        
        for case_type in self.supported_case_types:
            try:
                is_valid, errors, metrics = self.validate_dataset(case_type)
                
                report["datasets"][case_type] = {
                    "valid": is_valid,
                    "errors": errors,
                    "metrics": {
                        "total_cases": metrics.total_cases,
                        "case_types": metrics.case_types,
                        "complexity_distribution": metrics.complexity_distribution,
                        "jurisdictions": metrics.jurisdictions,
                        "date_range": metrics.date_range,
                        "average_case_value": metrics.average_case_value,
                        "success_rate_distribution": metrics.success_rate_distribution
                    } if metrics else None
                }
                
                if metrics:
                    total_cases += metrics.total_cases
                    all_case_types.update(metrics.case_types.keys())
                    all_jurisdictions.update(metrics.jurisdictions.keys())
                    
            except FileNotFoundError:
                report["datasets"][case_type] = {
                    "valid": False,
                    "errors": ["Dataset file not found"],
                    "metrics": None
                }
                
        report["summary"] = {
            "total_cases_across_all_datasets": total_cases,
            "available_case_types": list(all_case_types),
            "covered_jurisdictions": list(all_jurisdictions),
            "datasets_available": len([d for d in report["datasets"].values() if d["valid"]])
        }
        
        return report

# Convenience functions for common operations
def load_dataset(case_type: str) -> List[Dict[str, Any]]:
    """Load dataset for a specific case type."""
    loader = LegalDatasetLoader()
    return loader.load_case_dataset(case_type)

def validate_all_datasets() -> Dict[str, Any]:
    """Validate all available datasets and return report."""
    loader = LegalDatasetLoader()
    return loader.generate_dataset_report()

def export_for_training(case_type: str, format: str = "json") -> str:
    """Export dataset in training format."""
    loader = LegalDatasetLoader()
    return loader.export_training_data(case_type, format)