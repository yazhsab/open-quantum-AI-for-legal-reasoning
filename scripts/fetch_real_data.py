#!/usr/bin/env python3
"""
Real Legal Data Fetching Script for XQELM

This script fetches real legal data from various Indian legal databases
and integrates it with existing synthetic datasets for comprehensive training.
"""

import sys
import os
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xqelm.data.real_data_fetcher import RealLegalDataFetcher, fetch_real_legal_data
from xqelm.utils.data_loader import LegalDatasetLoader
from xqelm.utils.synthetic_data_generator import SyntheticDataGenerator, GenerationConfig

async def fetch_and_integrate_data(
    case_type: str,
    real_cases: int = 50,
    synthetic_cases: int = 50,
    output_dir: str = "data/datasets"
):
    """
    Fetch real data and integrate with synthetic data for comprehensive datasets.
    
    Args:
        case_type: Type of legal case
        real_cases: Number of real cases to fetch
        synthetic_cases: Number of synthetic cases to generate
        output_dir: Output directory for datasets
    """
    print(f"🔍 Fetching and integrating data for {case_type}")
    print(f"Real cases: {real_cases}, Synthetic cases: {synthetic_cases}")
    print("=" * 60)
    
    # Step 1: Fetch real legal data
    print("📥 Fetching real legal data...")
    try:
        fetcher = RealLegalDataFetcher()
        real_legal_cases = await fetcher.fetch_cases_by_type(case_type, real_cases)
        
        if real_legal_cases:
            real_output_path = await fetcher.save_cases_to_dataset(
                real_legal_cases, case_type, output_dir
            )
            print(f"✅ Fetched {len(real_legal_cases)} real cases")
            print(f"📁 Saved to: {real_output_path}")
        else:
            print("⚠️  No real cases found, proceeding with synthetic data only")
            
    except Exception as e:
        print(f"❌ Error fetching real data: {e}")
        print("⚠️  Proceeding with synthetic data only")
        real_legal_cases = []
    
    # Step 2: Generate synthetic data
    print("\n🤖 Generating synthetic data...")
    try:
        generator = SyntheticDataGenerator()
        config = GenerationConfig(
            num_cases=synthetic_cases,
            complexity_distribution={"low": 0.2, "medium": 0.6, "high": 0.2}
        )
        
        synthetic_cases_data = generator.generate_dataset(case_type, config)
        
        # Save synthetic data
        synthetic_output_dir = Path(output_dir) / case_type
        synthetic_output_dir.mkdir(parents=True, exist_ok=True)
        synthetic_output_path = synthetic_output_dir / "synthetic_cases.json"
        
        generator.save_dataset(synthetic_cases_data, str(synthetic_output_path))
        
        print(f"✅ Generated {len(synthetic_cases_data)} synthetic cases")
        print(f"📁 Saved to: {synthetic_output_path}")
        
    except Exception as e:
        print(f"❌ Error generating synthetic data: {e}")
        synthetic_cases_data = []
    
    # Step 3: Combine datasets
    print("\n🔗 Combining real and synthetic datasets...")
    try:
        combined_cases = []
        
        # Load existing sample cases if they exist
        sample_cases_path = Path(output_dir) / case_type / "sample_cases.json"
        if sample_cases_path.exists():
            with open(sample_cases_path, 'r', encoding='utf-8') as f:
                sample_cases = json.load(f)
                combined_cases.extend(sample_cases)
                print(f"📋 Loaded {len(sample_cases)} existing sample cases")
        
        # Add real cases (converted to dataset format)
        if real_legal_cases:
            real_cases_path = Path(output_dir) / case_type / "real_cases.json"
            if real_cases_path.exists():
                with open(real_cases_path, 'r', encoding='utf-8') as f:
                    real_dataset_cases = json.load(f)
                    combined_cases.extend(real_dataset_cases)
        
        # Add synthetic cases
        combined_cases.extend(synthetic_cases_data)
        
        # Save combined dataset
        combined_output_path = Path(output_dir) / case_type / "combined_cases.json"
        with open(combined_output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_cases, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Combined dataset created with {len(combined_cases)} total cases")
        print(f"📁 Saved to: {combined_output_path}")
        
        # Create train/test splits
        await create_train_test_splits(combined_cases, case_type, output_dir)
        
    except Exception as e:
        print(f"❌ Error combining datasets: {e}")

async def create_train_test_splits(
    cases: List[Dict[str, Any]], 
    case_type: str, 
    output_dir: str,
    train_ratio: float = 0.8
):
    """Create training and testing splits from combined dataset."""
    
    print(f"\n📊 Creating train/test splits (ratio: {train_ratio:.1%})")
    
    import random
    random.shuffle(cases)
    
    split_index = int(len(cases) * train_ratio)
    train_cases = cases[:split_index]
    test_cases = cases[split_index:]
    
    # Save training set
    train_path = Path(output_dir) / case_type / "train_cases.json"
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_cases, f, indent=2, ensure_ascii=False)
    
    # Save test set
    test_path = Path(output_dir) / case_type / "test_cases.json"
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Training set: {len(train_cases)} cases → {train_path}")
    print(f"✅ Test set: {len(test_cases)} cases → {test_path}")

async def validate_integrated_datasets(output_dir: str = "data/datasets"):
    """Validate the integrated datasets."""
    
    print("\n🔍 Validating integrated datasets...")
    print("=" * 60)
    
    loader = LegalDatasetLoader(output_dir)
    
    case_types = ["property_disputes", "motor_vehicle_claims", "consumer_disputes"]
    
    total_cases = 0
    total_real_cases = 0
    
    for case_type in case_types:
        try:
            # Validate main dataset
            is_valid, errors, metrics = loader.validate_dataset(case_type)
            
            if is_valid:
                print(f"✅ {case_type}: Valid dataset")
                print(f"   📊 Total cases: {metrics.total_cases}")
                print(f"   🏛️  Jurisdictions: {len(metrics.jurisdictions)}")
                print(f"   📈 Avg success rate: {metrics.success_rate_distribution.get('avg', 0):.2f}")
                
                total_cases += metrics.total_cases
                
                # Count real vs synthetic cases
                combined_path = Path(output_dir) / case_type / "combined_cases.json"
                if combined_path.exists():
                    with open(combined_path, 'r') as f:
                        cases = json.load(f)
                        real_count = len([c for c in cases if c.get('metadata', {}).get('source') == 'real_data'])
                        total_real_cases += real_count
                        print(f"   🔍 Real cases: {real_count}")
                        print(f"   🤖 Synthetic cases: {len(cases) - real_count}")
                
            else:
                print(f"❌ {case_type}: Invalid dataset")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"   • {error}")
                    
        except Exception as e:
            print(f"❌ {case_type}: Error during validation - {e}")
    
    print("\n" + "=" * 60)
    print("📊 INTEGRATION SUMMARY")
    print("=" * 60)
    print(f"Total cases across all datasets: {total_cases}")
    print(f"Real cases: {total_real_cases}")
    print(f"Synthetic cases: {total_cases - total_real_cases}")
    print(f"Real data percentage: {(total_real_cases/total_cases*100):.1f}%" if total_cases > 0 else "No data")

async def setup_data_sources_config():
    """Setup configuration for data sources."""
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / "data_sources.json"
    
    # Default configuration with Indian legal data sources
    config = {
        "data_sources": [
            {
                "name": "Indian Kanoon",
                "base_url": "https://api.indiankanoon.org",
                "rate_limit": 60,
                "case_types": ["civil", "criminal", "constitutional", "consumer", "motor_vehicle"],
                "requires_auth": False
            },
            {
                "name": "eCourts Services",
                "base_url": "https://services.ecourts.gov.in/ecourtindia_v6",
                "rate_limit": 30,
                "case_types": ["civil", "criminal", "consumer", "property"],
                "requires_auth": True
            },
            {
                "name": "Consumer Forum Database",
                "base_url": "https://confonet.nic.in",
                "rate_limit": 15,
                "case_types": ["consumer_disputes"],
                "requires_auth": False
            }
        ],
        "api_keys": {
            "indian_kanoon": "YOUR_API_KEY_HERE",
            "ecourts": "YOUR_API_KEY_HERE"
        },
        "rate_limiting": {
            "default_requests_per_minute": 20,
            "retry_attempts": 3,
            "retry_delay": 5
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Data sources configuration created: {config_path}")
    print("⚠️  Please update API keys in the configuration file")
    
    return str(config_path)

async def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description="Fetch and integrate real legal data for XQELM")
    parser.add_argument("--case-type", type=str, 
                       choices=["property_disputes", "motor_vehicle_claims", "consumer_disputes", "all"],
                       default="all", help="Specific case type to process (default: all)")
    parser.add_argument("--real-cases", type=int, default=50,
                       help="Number of real cases to fetch per type (default: 50)")
    parser.add_argument("--synthetic-cases", type=int, default=50,
                       help="Number of synthetic cases to generate per type (default: 50)")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for datasets (default: data/datasets)")
    parser.add_argument("--setup-config", action="store_true",
                       help="Setup data sources configuration")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing datasets")
    
    args = parser.parse_args()
    
    print("🚀 XQELM Real Data Integration Tool")
    print("=" * 60)
    
    # Setup configuration if requested
    if args.setup_config:
        await setup_data_sources_config()
        return
    
    # Validate only if requested
    if args.validate_only:
        await validate_integrated_datasets(args.output_dir)
        return
    
    # Determine case types to process
    if args.case_type == "all":
        case_types = ["property_disputes", "motor_vehicle_claims", "consumer_disputes"]
    else:
        case_types = [args.case_type]
    
    # Process each case type
    for case_type in case_types:
        await fetch_and_integrate_data(
            case_type=case_type,
            real_cases=args.real_cases,
            synthetic_cases=args.synthetic_cases,
            output_dir=args.output_dir
        )
        print()  # Add spacing between case types
    
    # Final validation
    await validate_integrated_datasets(args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ DATA INTEGRATION COMPLETED!")
    print("=" * 60)
    print("Your datasets now include both real and synthetic legal data.")
    print("You can now train XQELM models with comprehensive, realistic data.")
    print(f"📁 Check datasets in: {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())