#!/usr/bin/env python3
"""
Dataset generation script for XQELM.
Generates comprehensive synthetic datasets for training and testing.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xqelm.utils.synthetic_data_generator import SyntheticDataGenerator, GenerationConfig

def generate_comprehensive_datasets(output_dir: str = "data/datasets", num_cases: int = 100):
    """
    Generate comprehensive datasets for all supported case types.
    
    Args:
        output_dir: Directory to save generated datasets
        num_cases: Number of cases to generate per case type
    """
    print(f"🚀 Generating comprehensive XQELM datasets")
    print(f"Output directory: {output_dir}")
    print(f"Cases per type: {num_cases}")
    print("=" * 60)
    
    generator = SyntheticDataGenerator()
    output_path = Path(output_dir)
    
    # Configuration for realistic data distribution
    config = GenerationConfig(
        num_cases=num_cases,
        start_date="2023-01-01",
        end_date="2024-12-31",
        complexity_distribution={
            "low": 0.25,      # 25% low complexity
            "medium": 0.55,   # 55% medium complexity  
            "high": 0.20      # 20% high complexity
        },
        jurisdiction_weights={
            "delhi": 0.15,
            "maharashtra": 0.15,
            "karnataka": 0.12,
            "uttar_pradesh": 0.12,
            "gujarat": 0.10,
            "tamil_nadu": 0.10,
            "west_bengal": 0.08,
            "rajasthan": 0.08,
            "haryana": 0.05,
            "punjab": 0.05
        }
    )
    
    case_types = [
        "property_disputes",
        "motor_vehicle_claims", 
        "consumer_disputes"
    ]
    
    total_generated = 0
    
    for case_type in case_types:
        print(f"📝 Generating {case_type}...")
        
        try:
            # Generate synthetic cases
            cases = generator.generate_dataset(case_type, config)
            
            # Create output directory
            case_output_dir = output_path / case_type
            case_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main dataset
            main_file = case_output_dir / "synthetic_cases.json"
            generator.save_dataset(cases, str(main_file))
            
            # Also append to existing sample_cases.json if it exists
            sample_file = case_output_dir / "sample_cases.json"
            if sample_file.exists():
                with open(sample_file, 'r', encoding='utf-8') as f:
                    existing_cases = json.load(f)
                
                # Combine existing and new cases
                combined_cases = existing_cases + cases
                
                # Save combined dataset
                combined_file = case_output_dir / "combined_cases.json"
                generator.save_dataset(combined_cases, str(combined_file))
                
                print(f"   ✅ Generated {len(cases)} new cases")
                print(f"   📁 Saved to: {main_file}")
                print(f"   📁 Combined with existing: {combined_file} ({len(combined_cases)} total cases)")
            else:
                print(f"   ✅ Generated {len(cases)} cases")
                print(f"   📁 Saved to: {main_file}")
            
            total_generated += len(cases)
            
        except Exception as e:
            print(f"   ❌ Error generating {case_type}: {e}")
            continue
    
    print("=" * 60)
    print(f"✅ Dataset generation completed!")
    print(f"📊 Total cases generated: {total_generated}")
    print(f"📁 Output directory: {output_dir}")
    
    return total_generated

def generate_training_test_split(case_type: str, train_ratio: float = 0.8, output_dir: str = "data/datasets"):
    """
    Split dataset into training and testing sets.
    
    Args:
        case_type: Type of legal case
        train_ratio: Ratio of training data (0.0 to 1.0)
        output_dir: Directory containing datasets
    """
    print(f"📊 Creating train/test split for {case_type}")
    print(f"Training ratio: {train_ratio:.1%}")
    
    # Load combined dataset
    dataset_path = Path(output_dir) / case_type / "combined_cases.json"
    if not dataset_path.exists():
        dataset_path = Path(output_dir) / case_type / "synthetic_cases.json"
    
    if not dataset_path.exists():
        dataset_path = Path(output_dir) / case_type / "sample_cases.json"
    
    if not dataset_path.exists():
        print(f"   ❌ No dataset found for {case_type}")
        return
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            all_cases = json.load(f)
        
        # Shuffle and split
        import random
        random.shuffle(all_cases)
        
        split_index = int(len(all_cases) * train_ratio)
        train_cases = all_cases[:split_index]
        test_cases = all_cases[split_index:]
        
        # Save splits
        output_path = Path(output_dir) / case_type
        
        train_file = output_path / "train_cases.json"
        test_file = output_path / "test_cases.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_cases, f, indent=2, ensure_ascii=False)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Training set: {len(train_cases)} cases → {train_file}")
        print(f"   ✅ Test set: {len(test_cases)} cases → {test_file}")
        
    except Exception as e:
        print(f"   ❌ Error creating split: {e}")

def generate_edge_cases(case_type: str, num_cases: int = 20, output_dir: str = "data/datasets"):
    """
    Generate edge cases for robust testing.
    
    Args:
        case_type: Type of legal case
        num_cases: Number of edge cases to generate
        output_dir: Directory to save edge cases
    """
    print(f"🔍 Generating edge cases for {case_type}")
    
    generator = SyntheticDataGenerator()
    
    # Configuration for edge cases - more extreme distributions
    edge_config = GenerationConfig(
        num_cases=num_cases,
        start_date="2020-01-01",
        end_date="2024-12-31",
        complexity_distribution={
            "low": 0.1,       # Few low complexity
            "medium": 0.2,    # Few medium complexity
            "high": 0.7       # Mostly high complexity
        },
        jurisdiction_weights={
            "delhi": 0.3,     # Concentrated in major jurisdictions
            "maharashtra": 0.3,
            "karnataka": 0.2,
            "uttar_pradesh": 0.2
        }
    )
    
    try:
        edge_cases = generator.generate_dataset(case_type, edge_config)
        
        # Modify cases to be more extreme
        for case in edge_cases:
            # Make success probabilities more extreme
            current_prob = case['expected_output']['success_probability']
            if current_prob > 0.5:
                case['expected_output']['success_probability'] = min(0.95, current_prob + 0.2)
            else:
                case['expected_output']['success_probability'] = max(0.05, current_prob - 0.2)
            
            # Increase case values for high complexity
            if case['metadata']['complexity_level'] == 'high':
                if 'property_value' in case['input_data']:
                    case['input_data']['property_value'] *= 2
                elif 'purchase_amount' in case['input_data']:
                    case['input_data']['purchase_amount'] *= 1.5
                elif 'damages_claimed' in case['input_data']:
                    case['input_data']['damages_claimed'] *= 2
        
        # Save edge cases
        output_path = Path(output_dir) / case_type
        output_path.mkdir(parents=True, exist_ok=True)
        
        edge_file = output_path / "edge_cases.json"
        generator.save_dataset(edge_cases, str(edge_file))
        
        print(f"   ✅ Generated {len(edge_cases)} edge cases → {edge_file}")
        
    except Exception as e:
        print(f"   ❌ Error generating edge cases: {e}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Generate XQELM datasets")
    parser.add_argument("--num-cases", type=int, default=100, 
                       help="Number of cases to generate per type (default: 100)")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for datasets (default: data/datasets)")
    parser.add_argument("--case-type", type=str, choices=["property_disputes", "motor_vehicle_claims", "consumer_disputes", "all"],
                       default="all", help="Specific case type to generate (default: all)")
    parser.add_argument("--train-test-split", action="store_true",
                       help="Create train/test splits")
    parser.add_argument("--edge-cases", action="store_true",
                       help="Generate edge cases for testing")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training data ratio for splits (default: 0.8)")
    
    args = parser.parse_args()
    
    print("🚀 XQELM Dataset Generation Tool")
    print("=" * 60)
    
    if args.case_type == "all":
        case_types = ["property_disputes", "motor_vehicle_claims", "consumer_disputes"]
    else:
        case_types = [args.case_type]
    
    # Generate main datasets
    if args.num_cases > 0:
        total_generated = 0
        for case_type in case_types:
            generator = SyntheticDataGenerator()
            config = GenerationConfig(num_cases=args.num_cases)
            
            try:
                cases = generator.generate_dataset(case_type, config)
                
                # Create output directory
                output_path = Path(args.output_dir) / case_type
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save dataset
                output_file = output_path / "synthetic_cases.json"
                generator.save_dataset(cases, str(output_file))
                
                print(f"✅ Generated {len(cases)} cases for {case_type}")
                total_generated += len(cases)
                
            except Exception as e:
                print(f"❌ Error generating {case_type}: {e}")
        
        print(f"\n📊 Total cases generated: {total_generated}")
    
    # Create train/test splits
    if args.train_test_split:
        print("\n📊 Creating train/test splits...")
        for case_type in case_types:
            generate_training_test_split(case_type, args.train_ratio, args.output_dir)
    
    # Generate edge cases
    if args.edge_cases:
        print("\n🔍 Generating edge cases...")
        for case_type in case_types:
            generate_edge_cases(case_type, max(10, args.num_cases // 5), args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Dataset generation completed!")
    print(f"📁 Check output directory: {args.output_dir}")

if __name__ == "__main__":
    main()