#!/usr/bin/env python3
"""
Comprehensive Training Script for XQELM with Real Legal Data

This script demonstrates the complete workflow:
1. Fetch real legal data from Indian legal databases
2. Generate synthetic data to supplement real data
3. Train quantum-enhanced legal models with mixed datasets
4. Evaluate model performance and generate reports
"""

import sys
import os
import asyncio
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xqelm.data.real_data_fetcher import RealLegalDataFetcher
from xqelm.utils.synthetic_data_generator import SyntheticDataGenerator, GenerationConfig
from xqelm.training.enhanced_trainer import EnhancedTrainer, TrainingConfig, train_xqelm_with_real_data
from xqelm.core.quantum_legal_model import QuantumLegalModel
from xqelm.utils.data_loader import LegalDatasetLoader
from xqelm.utils.config import XQELMConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def setup_comprehensive_datasets(
    case_types: list,
    real_cases_per_type: int = 50,
    synthetic_cases_per_type: int = 100,
    output_dir: str = "data/datasets"
):
    """
    Setup comprehensive datasets with both real and synthetic data.
    
    Args:
        case_types: List of case types to prepare
        real_cases_per_type: Number of real cases to fetch per type
        synthetic_cases_per_type: Number of synthetic cases to generate per type
        output_dir: Output directory for datasets
    """
    
    logger.info("🚀 Setting up comprehensive datasets")
    logger.info(f"Case types: {case_types}")
    logger.info(f"Real cases per type: {real_cases_per_type}")
    logger.info(f"Synthetic cases per type: {synthetic_cases_per_type}")
    logger.info("=" * 80)
    
    dataset_summary = {}
    
    for case_type in case_types:
        logger.info(f"\n📋 Processing {case_type}")
        logger.info("-" * 50)
        
        case_summary = {
            'real_cases': 0,
            'synthetic_cases': 0,
            'total_cases': 0,
            'status': 'pending'
        }
        
        try:
            # Step 1: Fetch real legal data
            logger.info("🔍 Fetching real legal data...")
            
            fetcher = RealLegalDataFetcher()
            real_cases = await fetcher.fetch_cases_by_type(
                case_type, 
                limit=real_cases_per_type
            )
            
            if real_cases:
                real_output_path = await fetcher.save_cases_to_dataset(
                    real_cases, case_type, output_dir
                )
                case_summary['real_cases'] = len(real_cases)
                logger.info(f"✅ Fetched {len(real_cases)} real cases")
                logger.info(f"📁 Saved to: {real_output_path}")
            else:
                logger.warning("⚠️  No real cases found for this type")
            
            # Step 2: Generate synthetic data
            logger.info("🤖 Generating synthetic data...")
            
            generator = SyntheticDataGenerator()
            config = GenerationConfig(
                num_cases=synthetic_cases_per_type,
                complexity_distribution={"low": 0.2, "medium": 0.6, "high": 0.2},
                jurisdiction_weights={
                    "delhi": 0.15, "maharashtra": 0.15, "karnataka": 0.12,
                    "uttar_pradesh": 0.12, "gujarat": 0.10, "tamil_nadu": 0.10,
                    "west_bengal": 0.08, "rajasthan": 0.08, "haryana": 0.05, "punjab": 0.05
                }
            )
            
            synthetic_cases = generator.generate_dataset(case_type, config)
            
            # Save synthetic data
            synthetic_output_dir = Path(output_dir) / case_type
            synthetic_output_dir.mkdir(parents=True, exist_ok=True)
            synthetic_output_path = synthetic_output_dir / "synthetic_cases.json"
            
            generator.save_dataset(synthetic_cases, str(synthetic_output_path))
            case_summary['synthetic_cases'] = len(synthetic_cases)
            
            logger.info(f"✅ Generated {len(synthetic_cases)} synthetic cases")
            logger.info(f"📁 Saved to: {synthetic_output_path}")
            
            # Step 3: Create combined dataset
            logger.info("🔗 Creating combined dataset...")
            
            combined_cases = []
            
            # Load existing sample cases
            sample_cases_path = synthetic_output_dir / "sample_cases.json"
            if sample_cases_path.exists():
                with open(sample_cases_path, 'r', encoding='utf-8') as f:
                    sample_cases = json.load(f)
                    combined_cases.extend(sample_cases)
                    logger.info(f"📋 Added {len(sample_cases)} existing sample cases")
            
            # Add real cases (already in dataset format)
            if real_cases:
                real_cases_path = synthetic_output_dir / "real_cases.json"
                if real_cases_path.exists():
                    with open(real_cases_path, 'r', encoding='utf-8') as f:
                        real_dataset_cases = json.load(f)
                        combined_cases.extend(real_dataset_cases)
            
            # Add synthetic cases
            combined_cases.extend(synthetic_cases)
            
            # Save combined dataset
            combined_output_path = synthetic_output_dir / "combined_cases.json"
            with open(combined_output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_cases, f, indent=2, ensure_ascii=False)
            
            case_summary['total_cases'] = len(combined_cases)
            case_summary['status'] = 'completed'
            
            logger.info(f"✅ Combined dataset created with {len(combined_cases)} total cases")
            logger.info(f"📁 Saved to: {combined_output_path}")
            
            # Step 4: Create train/test splits
            logger.info("📊 Creating train/test splits...")
            
            import random
            random.shuffle(combined_cases)
            
            train_ratio = 0.8
            split_index = int(len(combined_cases) * train_ratio)
            train_cases = combined_cases[:split_index]
            test_cases = combined_cases[split_index:]
            
            # Save splits
            train_path = synthetic_output_dir / "train_cases.json"
            test_path = synthetic_output_dir / "test_cases.json"
            
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(train_cases, f, indent=2, ensure_ascii=False)
            
            with open(test_path, 'w', encoding='utf-8') as f:
                json.dump(test_cases, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Training set: {len(train_cases)} cases → {train_path}")
            logger.info(f"✅ Test set: {len(test_cases)} cases → {test_path}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {case_type}: {e}")
            case_summary['status'] = 'failed'
            case_summary['error'] = str(e)
        
        dataset_summary[case_type] = case_summary
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 DATASET SETUP SUMMARY")
    logger.info("=" * 80)
    
    total_real = sum(summary['real_cases'] for summary in dataset_summary.values())
    total_synthetic = sum(summary['synthetic_cases'] for summary in dataset_summary.values())
    total_cases = sum(summary['total_cases'] for summary in dataset_summary.values())
    
    for case_type, summary in dataset_summary.items():
        status_emoji = "✅" if summary['status'] == 'completed' else "❌"
        logger.info(f"{status_emoji} {case_type}:")
        logger.info(f"   Real cases: {summary['real_cases']}")
        logger.info(f"   Synthetic cases: {summary['synthetic_cases']}")
        logger.info(f"   Total cases: {summary['total_cases']}")
        if summary['status'] == 'failed':
            logger.info(f"   Error: {summary.get('error', 'Unknown error')}")
    
    logger.info("-" * 80)
    logger.info(f"OVERALL TOTALS:")
    logger.info(f"Real cases: {total_real}")
    logger.info(f"Synthetic cases: {total_synthetic}")
    logger.info(f"Total cases: {total_cases}")
    logger.info(f"Real data percentage: {(total_real/total_cases*100):.1f}%" if total_cases > 0 else "No data")
    logger.info("=" * 80)
    
    return dataset_summary

async def train_comprehensive_model(
    case_types: list,
    training_config: dict,
    model_config: dict = None
):
    """
    Train a comprehensive XQELM model with real and synthetic data.
    
    Args:
        case_types: List of case types to train on
        training_config: Training configuration parameters
        model_config: Model configuration parameters
    """
    
    logger.info("🧠 Starting comprehensive model training")
    logger.info(f"Case types: {case_types}")
    logger.info(f"Training config: {training_config}")
    logger.info("=" * 80)
    
    try:
        # Initialize model
        if model_config:
            config = XQELMConfig(**model_config)
        else:
            config = XQELMConfig()
        
        model = QuantumLegalModel(
            config=config,
            n_qubits=training_config.get('n_qubits', 20),
            n_layers=training_config.get('n_layers', 4)
        )
        
        # Initialize data loader
        data_loader = LegalDatasetLoader()
        
        # Create training configuration
        train_config = TrainingConfig(
            epochs=training_config.get('epochs', 100),
            learning_rate=training_config.get('learning_rate', 0.01),
            batch_size=training_config.get('batch_size', 32),
            validation_split=training_config.get('validation_split', 0.2),
            early_stopping_patience=training_config.get('early_stopping_patience', 10),
            use_cross_validation=training_config.get('use_cross_validation', True),
            cv_folds=training_config.get('cv_folds', 5),
            real_data_weight=training_config.get('real_data_weight', 1.5),
            synthetic_data_weight=training_config.get('synthetic_data_weight', 1.0)
        )
        
        # Initialize trainer
        trainer = EnhancedTrainer(model, train_config, data_loader)
        
        # Train the model
        logger.info("🚀 Starting training process...")
        
        training_results = await trainer.train_comprehensive(
            case_types=case_types,
            use_real_data=True,
            use_synthetic_data=True
        )
        
        logger.info("✅ Training completed successfully!")
        
        # Save the trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/xqelm_trained_{timestamp}.pt"
        
        Path("models").mkdir(exist_ok=True)
        model.save_model(model_path)
        
        logger.info(f"💾 Model saved to: {model_path}")
        
        # Print training summary
        logger.info("\n" + "=" * 80)
        logger.info("🎯 TRAINING RESULTS SUMMARY")
        logger.info("=" * 80)
        
        if training_results.get('training_type') == 'cross_validation':
            cv_stats = training_results['cv_statistics']
            logger.info(f"Cross-validation results:")
            logger.info(f"  Mean validation loss: {cv_stats['mean_val_loss']:.4f} ± {cv_stats['std_val_loss']:.4f}")
            logger.info(f"  Mean validation accuracy: {cv_stats['mean_val_accuracy']:.4f} ± {cv_stats['std_val_accuracy']:.4f}")
            logger.info(f"  Best validation loss: {cv_stats['best_val_loss']:.4f}")
            logger.info(f"  Best validation accuracy: {cv_stats['best_val_accuracy']:.4f}")
        
        # Data statistics
        data_stats = training_results.get('data_statistics', {})
        total_cases = sum(stats['total_cases'] for stats in data_stats.values())
        total_real = sum(stats['real_cases'] for stats in data_stats.values())
        
        logger.info(f"\nData used for training:")
        logger.info(f"  Total cases: {total_cases}")
        logger.info(f"  Real cases: {total_real} ({total_real/total_cases*100:.1f}%)")
        logger.info(f"  Synthetic cases: {total_cases - total_real} ({(total_cases-total_real)/total_cases*100:.1f}%)")
        
        logger.info("=" * 80)
        
        return {
            'model_path': model_path,
            'training_results': training_results,
            'model': model
        }
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

async def evaluate_model_performance(
    model: QuantumLegalModel,
    case_types: list,
    test_data_dir: str = "data/datasets"
):
    """
    Evaluate the trained model on test datasets.
    
    Args:
        model: Trained QuantumLegalModel
        case_types: List of case types to evaluate
        test_data_dir: Directory containing test datasets
    """
    
    logger.info("📊 Evaluating model performance")
    logger.info("=" * 80)
    
    evaluation_results = {}
    
    for case_type in case_types:
        logger.info(f"\n🔍 Evaluating {case_type}")
        logger.info("-" * 50)
        
        try:
            # Load test data
            test_data_path = Path(test_data_dir) / case_type / "test_cases.json"
            
            if not test_data_path.exists():
                logger.warning(f"⚠️  No test data found for {case_type}")
                continue
            
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
            
            logger.info(f"📋 Loaded {len(test_cases)} test cases")
            
            # Evaluate on test cases
            correct_predictions = 0
            total_predictions = 0
            real_data_correct = 0
            real_data_total = 0
            synthetic_data_correct = 0
            synthetic_data_total = 0
            
            confidence_scores = []
            processing_times = []
            
            for i, test_case in enumerate(test_cases):
                try:
                    # Create query from test case
                    input_data = test_case.get('input_data', {})
                    
                    if case_type == 'consumer_dispute':
                        query = f"Consumer dispute: {input_data.get('complaint_description', '')}"
                    elif case_type == 'motor_vehicle_claim':
                        query = f"Motor vehicle claim: {input_data.get('case_facts', '')}"
                    elif case_type == 'property_dispute':
                        query = f"Property dispute: {input_data.get('dispute_description', '')}"
                    else:
                        query = str(input_data)
                    
                    # Process through model
                    result = await model.process_legal_query(
                        query,
                        input_data,
                        case_type
                    )
                    
                    # Calculate accuracy
                    expected_success_prob = test_case.get('expected_output', {}).get('success_probability', 0.5)
                    predicted_success = 1.0 if result.confidence > 0.5 else 0.0
                    actual_success = 1.0 if expected_success_prob > 0.5 else 0.0
                    
                    if predicted_success == actual_success:
                        correct_predictions += 1
                        
                        # Track by data source
                        data_source = test_case.get('metadata', {}).get('source', 'unknown')
                        if data_source == 'real_data':
                            real_data_correct += 1
                        else:
                            synthetic_data_correct += 1
                    
                    total_predictions += 1
                    
                    # Track by data source totals
                    data_source = test_case.get('metadata', {}).get('source', 'unknown')
                    if data_source == 'real_data':
                        real_data_total += 1
                    else:
                        synthetic_data_total += 1
                    
                    confidence_scores.append(result.confidence)
                    processing_times.append(result.processing_time)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"   Processed {i + 1}/{len(test_cases)} test cases")
                
                except Exception as e:
                    logger.warning(f"   Error processing test case {i}: {e}")
                    continue
            
            # Calculate metrics
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            real_accuracy = real_data_correct / real_data_total if real_data_total > 0 else 0.0
            synthetic_accuracy = synthetic_data_correct / synthetic_data_total if synthetic_data_total > 0 else 0.0
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
            
            case_results = {
                'total_test_cases': len(test_cases),
                'processed_cases': total_predictions,
                'overall_accuracy': accuracy,
                'real_data_accuracy': real_accuracy,
                'synthetic_data_accuracy': synthetic_accuracy,
                'real_data_cases': real_data_total,
                'synthetic_data_cases': synthetic_data_total,
                'average_confidence': avg_confidence,
                'average_processing_time': avg_processing_time
            }
            
            evaluation_results[case_type] = case_results
            
            logger.info(f"✅ Evaluation completed for {case_type}")
            logger.info(f"   Overall accuracy: {accuracy:.3f}")
            logger.info(f"   Real data accuracy: {real_accuracy:.3f} ({real_data_total} cases)")
            logger.info(f"   Synthetic data accuracy: {synthetic_accuracy:.3f} ({synthetic_data_total} cases)")
            logger.info(f"   Average confidence: {avg_confidence:.3f}")
            logger.info(f"   Average processing time: {avg_processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {case_type}: {e}")
            evaluation_results[case_type] = {'error': str(e)}
    
    # Print overall evaluation summary
    logger.info("\n" + "=" * 80)
    logger.info("🎯 EVALUATION SUMMARY")
    logger.info("=" * 80)
    
    overall_accuracy = []
    overall_real_accuracy = []
    overall_synthetic_accuracy = []
    
    for case_type, results in evaluation_results.items():
        if 'error' not in results:
            overall_accuracy.append(results['overall_accuracy'])
            if results['real_data_accuracy'] > 0:
                overall_real_accuracy.append(results['real_data_accuracy'])
            if results['synthetic_data_accuracy'] > 0:
                overall_synthetic_accuracy.append(results['synthetic_data_accuracy'])
    
    if overall_accuracy:
        logger.info(f"Average accuracy across all case types: {sum(overall_accuracy)/len(overall_accuracy):.3f}")
        
    if overall_real_accuracy:
        logger.info(f"Average accuracy on real data: {sum(overall_real_accuracy)/len(overall_real_accuracy):.3f}")
        
    if overall_synthetic_accuracy:
        logger.info(f"Average accuracy on synthetic data: {sum(overall_synthetic_accuracy)/len(overall_synthetic_accuracy):.3f}")
    
    logger.info("=" * 80)
    
    return evaluation_results

async def main():
    """Main function with comprehensive workflow."""
    
    parser = argparse.ArgumentParser(description="Comprehensive XQELM Training with Real Legal Data")
    parser.add_argument("--case-types", nargs='+', 
                       choices=["property_disputes", "motor_vehicle_claims", "consumer_disputes"],
                       default=["property_disputes", "motor_vehicle_claims", "consumer_disputes"],
                       help="Case types to process")
    parser.add_argument("--real-cases", type=int, default=50,
                       help="Number of real cases to fetch per type")
    parser.add_argument("--synthetic-cases", type=int, default=100,
                       help="Number of synthetic cases to generate per type")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate for training")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--n-qubits", type=int, default=12,
                       help="Number of qubits for quantum circuits")
    parser.add_argument("--n-layers", type=int, default=3,
                       help="Number of layers in quantum circuits")
    parser.add_argument("--skip-data-setup", action="store_true",
                       help="Skip data setup and use existing datasets")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and only setup data")
    parser.add_argument("--use-cross-validation", action="store_true", default=True,
                       help="Use cross-validation for training")
    
    args = parser.parse_args()
    
    logger.info("🚀 XQELM Comprehensive Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Case types: {args.case_types}")
    logger.info(f"Real cases per type: {args.real_cases}")
    logger.info(f"Synthetic cases per type: {args.synthetic_cases}")
    logger.info(f"Training epochs: {args.epochs}")
    logger.info(f"Quantum circuit: {args.n_qubits} qubits, {args.n_layers} layers")
    logger.info("=" * 80)
    
    try:
        # Step 1: Setup comprehensive datasets
        if not args.skip_data_setup:
            dataset_summary = await setup_comprehensive_datasets(
                case_types=args.case_types,
                real_cases_per_type=args.real_cases,
                synthetic_cases_per_type=args.synthetic_cases
            )
        else:
            logger.info("⏭️  Skipping data setup - using existing datasets")
        
        # Step 2: Train comprehensive model
        if not args.skip_training:
            training_config = {
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'n_qubits': args.n_qubits,
                'n_layers': args.n_layers,
                'use_cross_validation': args.use_cross_validation,
                'real_data_weight': 1.5,
                'synthetic_data_weight': 1.0
            }
            
            training_result = await train_comprehensive_model(
                case_types=args.case_types,
                training_config=training_config
            )
            
            # Step 3: Evaluate model performance
            evaluation_results = await evaluate_model_performance(
                model=training_result['model'],
                case_types=args.case_types
            )
            
            # Save evaluation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_path = f"training/evaluation_results_{timestamp}.json"
            
            Path("training").mkdir(exist_ok=True)
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Evaluation results saved to: {eval_path}")
        
        else:
            logger.info("⏭️  Skipping training - data setup only")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ COMPREHENSIVE TRAINING PIPELINE COMPLETED!")
        logger.info("=" * 80)
        logger.info("Your XQELM model has been trained with real legal data and is ready for use.")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())