#!/usr/bin/env python3
"""
Test script for XQELM dataset functionality.
Demonstrates data loading, validation, and synthetic data generation.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xqelm.utils.data_loader import LegalDatasetLoader, validate_all_datasets
from xqelm.utils.synthetic_data_generator import SyntheticDataGenerator, GenerationConfig

def test_data_loading():
    """Test loading existing datasets."""
    print("=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    loader = LegalDatasetLoader()
    
    # Test loading individual datasets
    case_types = ["property_disputes", "motor_vehicle_claims", "consumer_disputes"]
    
    for case_type in case_types:
        try:
            cases = loader.load_case_dataset(case_type)
            print(f"✅ Loaded {len(cases)} cases for {case_type}")
            
            # Show sample case structure
            if cases:
                sample_case = cases[0]
                print(f"   Sample case ID: {sample_case.get('case_id', 'N/A')}")
                print(f"   Case type: {sample_case.get('case_type', 'N/A')}")
                print(f"   Metadata: {sample_case.get('metadata', {}).get('complexity_level', 'N/A')} complexity")
                
        except FileNotFoundError:
            print(f"❌ Dataset not found for {case_type}")
        except Exception as e:
            print(f"❌ Error loading {case_type}: {e}")
    
    print()

def test_data_validation():
    """Test dataset validation."""
    print("=" * 60)
    print("TESTING DATA VALIDATION")
    print("=" * 60)
    
    # Generate comprehensive validation report
    report = validate_all_datasets()
    
    print(f"Generated at: {report['generated_at']}")
    print(f"Total cases across all datasets: {report['summary']['total_cases_across_all_datasets']}")
    print(f"Available case types: {', '.join(report['summary']['available_case_types'])}")
    print(f"Covered jurisdictions: {', '.join(report['summary']['covered_jurisdictions'])}")
    print(f"Valid datasets: {report['summary']['datasets_available']}")
    
    print("\nDataset Details:")
    for dataset_name, dataset_info in report['datasets'].items():
        status = "✅ Valid" if dataset_info['valid'] else "❌ Invalid"
        print(f"  {dataset_name}: {status}")
        
        if dataset_info['metrics']:
            metrics = dataset_info['metrics']
            print(f"    - Cases: {metrics['total_cases']}")
            print(f"    - Complexity: {metrics['complexity_distribution']}")
            print(f"    - Success rate avg: {metrics['success_rate_distribution'].get('avg', 'N/A'):.2f}")
            
        if dataset_info['errors']:
            print(f"    - Errors: {len(dataset_info['errors'])}")
            for error in dataset_info['errors'][:3]:  # Show first 3 errors
                print(f"      • {error}")
    
    print()

def test_synthetic_generation():
    """Test synthetic data generation."""
    print("=" * 60)
    print("TESTING SYNTHETIC DATA GENERATION")
    print("=" * 60)
    
    generator = SyntheticDataGenerator()
    
    # Test generating small datasets for each case type
    case_types = ["property_disputes", "motor_vehicle_claims", "consumer_disputes"]
    
    for case_type in case_types:
        print(f"Generating synthetic data for {case_type}...")
        
        config = GenerationConfig(
            num_cases=5,  # Small number for testing
            complexity_distribution={"low": 0.2, "medium": 0.6, "high": 0.2}
        )
        
        try:
            cases = generator.generate_dataset(case_type, config)
            print(f"✅ Generated {len(cases)} synthetic cases for {case_type}")
            
            # Show sample generated case
            if cases:
                sample_case = cases[0]
                print(f"   Sample case ID: {sample_case['case_id']}")
                print(f"   Success probability: {sample_case['expected_output']['success_probability']}")
                print(f"   Complexity: {sample_case['metadata']['complexity_level']}")
                print(f"   State: {sample_case['metadata']['state']}")
                
        except Exception as e:
            print(f"❌ Error generating {case_type}: {e}")
    
    print()

def test_data_export():
    """Test data export functionality."""
    print("=" * 60)
    print("TESTING DATA EXPORT")
    print("=" * 60)
    
    loader = LegalDatasetLoader()
    
    # Test exporting existing datasets
    case_types = ["property_disputes", "motor_vehicle_claims", "consumer_disputes"]
    
    for case_type in case_types:
        try:
            # Test JSON export
            json_path = loader.export_training_data(case_type, "json")
            print(f"✅ Exported {case_type} to JSON: {json_path}")
            
            # Verify the exported file
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    exported_data = json.load(f)
                print(f"   Exported {len(exported_data)} training records")
            
        except FileNotFoundError:
            print(f"❌ Cannot export {case_type}: dataset not found")
        except Exception as e:
            print(f"❌ Error exporting {case_type}: {e}")
    
    print()

def test_dataset_metrics():
    """Test dataset metrics calculation."""
    print("=" * 60)
    print("TESTING DATASET METRICS")
    print("=" * 60)
    
    loader = LegalDatasetLoader()
    
    case_types = ["property_disputes", "motor_vehicle_claims", "consumer_disputes"]
    
    for case_type in case_types:
        try:
            is_valid, errors, metrics = loader.validate_dataset(case_type)
            
            if metrics:
                print(f"📊 Metrics for {case_type}:")
                print(f"   Total cases: {metrics.total_cases}")
                print(f"   Case types: {metrics.case_types}")
                print(f"   Complexity distribution: {metrics.complexity_distribution}")
                print(f"   Jurisdictions: {list(metrics.jurisdictions.keys())}")
                print(f"   Date range: {metrics.date_range[0]} to {metrics.date_range[1]}")
                print(f"   Average case value: ₹{metrics.average_case_value:,.2f}")
                
                if metrics.success_rate_distribution:
                    print(f"   Success rate - Min: {metrics.success_rate_distribution.get('min', 0):.2f}, "
                          f"Max: {metrics.success_rate_distribution.get('max', 0):.2f}, "
                          f"Avg: {metrics.success_rate_distribution.get('avg', 0):.2f}")
                print()
            else:
                print(f"❌ No metrics available for {case_type}")
                
        except FileNotFoundError:
            print(f"❌ Dataset not found for {case_type}")
        except Exception as e:
            print(f"❌ Error calculating metrics for {case_type}: {e}")

def main():
    """Run all dataset tests."""
    print("🚀 XQELM Dataset Testing Suite")
    print("Testing dataset loading, validation, and generation functionality\n")
    
    try:
        # Run all tests
        test_data_loading()
        test_data_validation()
        test_synthetic_generation()
        test_data_export()
        test_dataset_metrics()
        
        print("=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        print("Dataset functionality is working correctly!")
        print("You can now use the datasets for training and testing XQELM models.")
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()