"""
Enhanced Training Pipeline for XQELM with Real and Synthetic Data

This module provides an improved training pipeline that can handle mixed datasets
of real legal cases and synthetic data for more robust model training.
"""

import asyncio
import logging
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from ..core.quantum_legal_model import QuantumLegalModel
from ..utils.data_loader import LegalDatasetLoader
from ..utils.config import XQELMConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for enhanced training."""
    epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    use_cross_validation: bool = True
    cv_folds: int = 5
    real_data_weight: float = 1.5  # Weight real data higher than synthetic
    synthetic_data_weight: float = 1.0
    save_checkpoints: bool = True
    checkpoint_dir: str = "models/checkpoints"
    metrics_dir: str = "training/metrics"

@dataclass
class TrainingMetrics:
    """Training metrics and results."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    real_data_accuracy: float
    synthetic_data_accuracy: float
    learning_rate: float
    timestamp: str

class EnhancedTrainer:
    """
    Enhanced trainer for XQELM models with support for mixed real/synthetic data.
    """
    
    def __init__(
        self,
        model: QuantumLegalModel,
        config: TrainingConfig,
        data_loader: LegalDatasetLoader
    ):
        """
        Initialize the enhanced trainer.
        
        Args:
            model: QuantumLegalModel instance
            config: Training configuration
            data_loader: Data loader for legal datasets
        """
        self.model = model
        self.config = config
        self.data_loader = data_loader
        
        # Training state
        self.training_history = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        # Create output directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.metrics_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Enhanced trainer initialized")
    
    async def train_comprehensive(
        self,
        case_types: List[str],
        use_real_data: bool = True,
        use_synthetic_data: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model comprehensively on multiple case types with mixed data.
        
        Args:
            case_types: List of case types to train on
            use_real_data: Whether to include real legal data
            use_synthetic_data: Whether to include synthetic data
            
        Returns:
            Comprehensive training results
        """
        logger.info(f"Starting comprehensive training on {len(case_types)} case types")
        
        # Load and prepare datasets
        all_training_data = []
        all_validation_data = []
        data_statistics = {}
        
        for case_type in case_types:
            train_data, val_data, stats = await self._prepare_case_type_data(
                case_type, use_real_data, use_synthetic_data
            )
            
            all_training_data.extend(train_data)
            all_validation_data.extend(val_data)
            data_statistics[case_type] = stats
            
            logger.info(f"Loaded {len(train_data)} training and {len(val_data)} validation cases for {case_type}")
        
        # Log data statistics
        self._log_data_statistics(data_statistics)
        
        # Perform training
        if self.config.use_cross_validation:
            results = await self._train_with_cross_validation(all_training_data, case_types)
        else:
            results = await self._train_standard(all_training_data, all_validation_data)
        
        # Add data statistics to results
        results['data_statistics'] = data_statistics
        results['case_types'] = case_types
        
        # Save final results
        await self._save_training_results(results)
        
        logger.info("Comprehensive training completed")
        return results
    
    async def _prepare_case_type_data(
        self,
        case_type: str,
        use_real_data: bool,
        use_synthetic_data: bool
    ) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
        """Prepare training and validation data for a specific case type."""
        
        all_cases = []
        real_cases_count = 0
        synthetic_cases_count = 0
        
        # Load real data if available and requested
        if use_real_data:
            try:
                real_cases_path = Path("data/datasets") / case_type / "real_cases.json"
                if real_cases_path.exists():
                    with open(real_cases_path, 'r', encoding='utf-8') as f:
                        real_cases = json.load(f)
                        
                    # Add weights to real cases
                    for case in real_cases:
                        case['data_weight'] = self.config.real_data_weight
                        case['data_source'] = 'real'
                    
                    all_cases.extend(real_cases)
                    real_cases_count = len(real_cases)
                    
            except Exception as e:
                logger.warning(f"Could not load real data for {case_type}: {e}")
        
        # Load synthetic data if requested
        if use_synthetic_data:
            try:
                # Try combined cases first, then synthetic cases
                for filename in ["combined_cases.json", "synthetic_cases.json", "sample_cases.json"]:
                    synthetic_path = Path("data/datasets") / case_type / filename
                    if synthetic_path.exists():
                        with open(synthetic_path, 'r', encoding='utf-8') as f:
                            synthetic_cases = json.load(f)
                        
                        # Filter out real cases if we already loaded them
                        if use_real_data and filename == "combined_cases.json":
                            synthetic_cases = [
                                case for case in synthetic_cases 
                                if case.get('metadata', {}).get('source') != 'real_data'
                            ]
                        
                        # Add weights to synthetic cases
                        for case in synthetic_cases:
                            case['data_weight'] = self.config.synthetic_data_weight
                            case['data_source'] = 'synthetic'
                        
                        all_cases.extend(synthetic_cases)
                        synthetic_cases_count = len(synthetic_cases)
                        break
                        
            except Exception as e:
                logger.warning(f"Could not load synthetic data for {case_type}: {e}")
        
        if not all_cases:
            raise ValueError(f"No data available for case type: {case_type}")
        
        # Convert to training format
        training_cases = []
        for case in all_cases:
            training_case = self._convert_to_training_format(case)
            training_cases.append(training_case)
        
        # Split into train/validation
        split_idx = int(len(training_cases) * (1 - self.config.validation_split))
        
        # Shuffle while maintaining stratification by complexity
        import random
        random.shuffle(training_cases)
        
        train_data = training_cases[:split_idx]
        val_data = training_cases[split_idx:]
        
        # Calculate statistics
        stats = {
            'total_cases': len(all_cases),
            'real_cases': real_cases_count,
            'synthetic_cases': synthetic_cases_count,
            'train_cases': len(train_data),
            'val_cases': len(val_data),
            'complexity_distribution': self._calculate_complexity_distribution(all_cases),
            'success_rate_distribution': self._calculate_success_rate_distribution(all_cases)
        }
        
        return train_data, val_data, stats
    
    def _convert_to_training_format(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Convert case to training format expected by the model."""
        
        # Extract query from input data
        input_data = case.get('input_data', {})
        
        # Create a natural language query from input data
        if case.get('case_type') == 'consumer_dispute':
            query = f"Consumer dispute case: {input_data.get('complaint_description', '')}"
        elif case.get('case_type') == 'motor_vehicle_claim':
            query = f"Motor vehicle accident claim: {input_data.get('case_facts', '')}"
        elif case.get('case_type') == 'property_dispute':
            query = f"Property dispute case: {input_data.get('dispute_description', '')}"
        else:
            query = f"Legal case: {str(input_data)}"
        
        # Extract expected answer
        expected_output = case.get('expected_output', {})
        success_prob = expected_output.get('success_probability', 0.5)
        
        # Create expected answer
        if success_prob > 0.7:
            expected_answer = "The case has a high probability of success based on the evidence and legal precedents."
        elif success_prob > 0.4:
            expected_answer = "The case has a moderate probability of success with some challenges."
        else:
            expected_answer = "The case faces significant challenges and has a lower probability of success."
        
        return {
            'case_id': case.get('case_id'),
            'query': query,
            'expected_answer': expected_answer,
            'context': input_data,
            'use_case_type': case.get('case_type'),
            'data_weight': case.get('data_weight', 1.0),
            'data_source': case.get('data_source', 'unknown'),
            'success_probability': success_prob,
            'complexity_level': case.get('metadata', {}).get('complexity_level', 'medium')
        }
    
    async def _train_with_cross_validation(
        self,
        training_data: List[Dict],
        case_types: List[str]
    ) -> Dict[str, Any]:
        """Train with cross-validation for robust evaluation."""
        
        logger.info(f"Starting {self.config.cv_folds}-fold cross-validation")
        
        # Stratify by complexity level for balanced folds
        complexity_levels = [case['complexity_level'] for case in training_data]
        
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        cv_results = []
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(training_data, complexity_levels)):
            logger.info(f"Training fold {fold + 1}/{self.config.cv_folds}")
            
            # Split data for this fold
            fold_train_data = [training_data[i] for i in train_idx]
            fold_val_data = [training_data[i] for i in val_idx]
            
            # Create a fresh model for this fold
            fold_model = QuantumLegalModel(
                config=self.model.config,
                n_qubits=self.model.n_qubits,
                n_layers=self.model.n_layers
            )
            
            # Train on this fold
            fold_results = await self._train_single_fold(
                fold_model, fold_train_data, fold_val_data, fold
            )
            
            cv_results.append(fold_results)
            fold_models.append(fold_model)
            
            logger.info(f"Fold {fold + 1} completed - Val Loss: {fold_results['final_val_loss']:.4f}")
        
        # Calculate cross-validation statistics
        cv_stats = self._calculate_cv_statistics(cv_results)
        
        # Select best model
        best_fold = np.argmin([result['final_val_loss'] for result in cv_results])
        self.model = fold_models[best_fold]
        
        logger.info(f"Cross-validation completed. Best fold: {best_fold + 1}")
        
        return {
            'training_type': 'cross_validation',
            'cv_results': cv_results,
            'cv_statistics': cv_stats,
            'best_fold': best_fold,
            'final_model_metrics': cv_results[best_fold]
        }
    
    async def _train_single_fold(
        self,
        model: QuantumLegalModel,
        train_data: List[Dict],
        val_data: List[Dict],
        fold: int
    ) -> Dict[str, Any]:
        """Train a single fold."""
        
        # Reset training state for this fold
        fold_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training step
            train_loss, train_accuracy = await self._train_epoch(model, train_data)
            
            # Validation step
            val_loss, val_accuracy, detailed_metrics = await self._validate_epoch(model, val_data)
            
            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                real_data_accuracy=detailed_metrics['real_data_accuracy'],
                synthetic_data_accuracy=detailed_metrics['synthetic_data_accuracy'],
                learning_rate=self.config.learning_rate,
                timestamp=datetime.now().isoformat()
            )
            
            fold_history.append(metrics)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save checkpoint
                if self.config.save_checkpoints:
                    checkpoint_path = Path(self.config.checkpoint_dir) / f"fold_{fold}_epoch_{epoch}.pt"
                    model.save_model(str(checkpoint_path))
                    
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} for fold {fold}")
                break
        
        return {
            'fold': fold,
            'training_history': fold_history,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_train_accuracy': train_accuracy,
            'final_val_accuracy': val_accuracy,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(fold_history)
        }
    
    async def _train_epoch(self, model: QuantumLegalModel, train_data: List[Dict]) -> Tuple[float, float]:
        """Train for one epoch."""
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Create weighted batches
        batches = self._create_weighted_batches(train_data, self.config.batch_size)
        
        for batch in batches:
            batch_loss = 0.0
            
            for example in batch:
                try:
                    # Process through model
                    result = await model.process_legal_query(
                        example['query'],
                        example.get('context'),
                        example.get('use_case_type')
                    )
                    
                    # Calculate weighted loss
                    loss = model._compute_loss(result, example['expected_answer'])
                    weighted_loss = loss * example.get('data_weight', 1.0)
                    
                    batch_loss += weighted_loss
                    
                    # Calculate accuracy
                    predicted_success = 1.0 if result.confidence > 0.5 else 0.0
                    actual_success = 1.0 if example['success_probability'] > 0.5 else 0.0
                    
                    if predicted_success == actual_success:
                        correct_predictions += 1
                    total_predictions += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing training example: {e}")
                    continue
            
            total_loss += batch_loss / len(batch)
        
        avg_loss = total_loss / len(batches) if batches else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    async def _validate_epoch(
        self, 
        model: QuantumLegalModel, 
        val_data: List[Dict]
    ) -> Tuple[float, float, Dict[str, float]]:
        """Validate for one epoch with detailed metrics."""
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Separate metrics for real vs synthetic data
        real_correct = 0
        real_total = 0
        synthetic_correct = 0
        synthetic_total = 0
        
        for example in val_data:
            try:
                result = await model.process_legal_query(
                    example['query'],
                    example.get('context'),
                    example.get('use_case_type')
                )
                
                loss = model._compute_loss(result, example['expected_answer'])
                total_loss += loss
                
                # Calculate accuracy
                predicted_success = 1.0 if result.confidence > 0.5 else 0.0
                actual_success = 1.0 if example['success_probability'] > 0.5 else 0.0
                
                if predicted_success == actual_success:
                    correct_predictions += 1
                    
                    # Track by data source
                    if example.get('data_source') == 'real':
                        real_correct += 1
                    else:
                        synthetic_correct += 1
                
                total_predictions += 1
                
                # Track totals by data source
                if example.get('data_source') == 'real':
                    real_total += 1
                else:
                    synthetic_total += 1
                    
            except Exception as e:
                logger.warning(f"Error processing validation example: {e}")
                continue
        
        avg_loss = total_loss / len(val_data) if val_data else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        detailed_metrics = {
            'real_data_accuracy': real_correct / real_total if real_total > 0 else 0.0,
            'synthetic_data_accuracy': synthetic_correct / synthetic_total if synthetic_total > 0 else 0.0
        }
        
        return avg_loss, accuracy, detailed_metrics
    
    def _create_weighted_batches(self, data: List[Dict], batch_size: int) -> List[List[Dict]]:
        """Create batches with consideration for data weights."""
        
        # Sort by weight to ensure balanced batches
        sorted_data = sorted(data, key=lambda x: x.get('data_weight', 1.0), reverse=True)
        
        batches = []
        for i in range(0, len(sorted_data), batch_size):
            batch = sorted_data[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _calculate_complexity_distribution(self, cases: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of complexity levels."""
        distribution = {}
        for case in cases:
            complexity = case.get('metadata', {}).get('complexity_level', 'unknown')
            distribution[complexity] = distribution.get(complexity, 0) + 1
        return distribution
    
    def _calculate_success_rate_distribution(self, cases: List[Dict]) -> Dict[str, float]:
        """Calculate success rate statistics."""
        success_rates = []
        for case in cases:
            success_prob = case.get('expected_output', {}).get('success_probability', 0.5)
            success_rates.append(success_prob)
        
        if success_rates:
            return {
                'min': min(success_rates),
                'max': max(success_rates),
                'mean': np.mean(success_rates),
                'std': np.std(success_rates)
            }
        return {}
    
    def _calculate_cv_statistics(self, cv_results: List[Dict]) -> Dict[str, float]:
        """Calculate cross-validation statistics."""
        
        val_losses = [result['final_val_loss'] for result in cv_results]
        val_accuracies = [result['final_val_accuracy'] for result in cv_results]
        
        return {
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'mean_val_accuracy': np.mean(val_accuracies),
            'std_val_accuracy': np.std(val_accuracies),
            'best_val_loss': min(val_losses),
            'best_val_accuracy': max(val_accuracies)
        }
    
    def _log_data_statistics(self, data_statistics: Dict[str, Dict]):
        """Log comprehensive data statistics."""
        
        logger.info("=" * 60)
        logger.info("DATA STATISTICS")
        logger.info("=" * 60)
        
        total_cases = 0
        total_real = 0
        total_synthetic = 0
        
        for case_type, stats in data_statistics.items():
            logger.info(f"{case_type.upper()}:")
            logger.info(f"  Total cases: {stats['total_cases']}")
            logger.info(f"  Real cases: {stats['real_cases']}")
            logger.info(f"  Synthetic cases: {stats['synthetic_cases']}")
            logger.info(f"  Training cases: {stats['train_cases']}")
            logger.info(f"  Validation cases: {stats['val_cases']}")
            
            total_cases += stats['total_cases']
            total_real += stats['real_cases']
            total_synthetic += stats['synthetic_cases']
        
        logger.info("=" * 60)
        logger.info(f"OVERALL TOTALS:")
        logger.info(f"  Total cases: {total_cases}")
        logger.info(f"  Real cases: {total_real} ({total_real/total_cases*100:.1f}%)")
        logger.info(f"  Synthetic cases: {total_synthetic} ({total_synthetic/total_cases*100:.1f}%)")
        logger.info("=" * 60)
    
    async def _save_training_results(self, results: Dict[str, Any]):
        """Save comprehensive training results."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        results_path = Path(self.config.metrics_dir) / f"training_results_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Deep convert the results
        import copy
        json_results = copy.deepcopy(results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False, default=convert_numpy)
        
        logger.info(f"Training results saved to: {results_path}")
        
        # Generate and save training plots
        await self._generate_training_plots(results, timestamp)
    
    async def _generate_training_plots(self, results: Dict[str, Any], timestamp: str):
        """Generate training visualization plots."""
        
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('XQELM Training Results', fontsize=16)
            
            if results.get('training_type') == 'cross_validation':
                # Cross-validation results
                cv_results = results['cv_results']
                
                # Plot 1: Validation loss across folds
                fold_losses = [result['final_val_loss'] for result in cv_results]
                axes[0, 0].bar(range(1, len(fold_losses) + 1), fold_losses)
                axes[0, 0].set_title('Validation Loss by Fold')
                axes[0, 0].set_xlabel('Fold')
                axes[0, 0].set_ylabel('Loss')
                
                # Plot 2: Validation accuracy across folds
                fold_accuracies = [result['final_val_accuracy'] for result in cv_results]
                axes[0, 1].bar(range(1, len(fold_accuracies) + 1), fold_accuracies)
                axes[0, 1].set_title('Validation Accuracy by Fold')
                axes[0, 1].set_xlabel('Fold')
                axes[0, 1].set_ylabel('Accuracy')
                
                # Plot 3: Training history for best fold
                best_fold = results['best_fold']
                best_history = cv_results[best_fold]['training_history']
                
                epochs = [m.epoch for m in best_history]
                train_losses = [m.train_loss for m in best_history]
                val_losses = [m.val_loss for m in best_history]
                
                axes[1, 0].plot(epochs, train_losses, label='Training Loss')
                axes[1, 0].plot(epochs, val_losses, label='Validation Loss')
                axes[1, 0].set_title(f'Training History - Best Fold ({best_fold + 1})')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                
                # Plot 4: Data distribution
                data_stats = results['data_statistics']
                case_types = list(data_stats.keys())
                real_counts = [stats['real_cases'] for stats in data_stats.values()]
                synthetic_counts = [stats['synthetic_cases'] for stats in data_stats.values()]
                
                x = np.arange(len(case_types))
                width = 0.35
                
                axes[1, 1].bar(x - width/2, real_counts, width, label='Real Data')
                axes[1, 1].bar(x + width/2, synthetic_counts, width, label='Synthetic Data')
                axes[1, 1].set_title('Data Distribution by Case Type')
                axes[1, 1].set_xlabel('Case Type')
                axes[1, 1].set_ylabel('Number of Cases')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels([ct.replace('_', ' ').title() for ct in case_types])
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path(self.config.metrics_dir) / f"training_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate training plots: {e}")

# Convenience function for easy training
async def train_xqelm_with_real_data(
    case_types: List[str] = None,
    real_cases_per_type: int = 50,
    synthetic_cases_per_type: int = 50,
    epochs: int = 100,
    use_cross_validation: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to train XQELM with real and synthetic data.
    
    Args:
        case_types: List of case types to train on
        real_cases_per_type: Number of real cases per type
        synthetic_cases_per_type: Number of synthetic cases per type
        epochs: Number of training epochs
        use_cross_validation: Whether to use cross-validation
        
    Returns:
        Training results
    """
    
    if case_types is None:
        case_types = ["property_disputes", "motor_vehicle_claims", "consumer_disputes"]
    
    # Initialize components
    config = XQELMConfig()
    model = QuantumLegalModel(config=config)
    data_loader = LegalDatasetLoader()
    
    training_config = TrainingConfig(
        epochs=epochs,
        use_cross_validation=use_cross_validation,
        real_data_weight=1.5,
        synthetic_data_weight=1.0
    )
    
    trainer = EnhancedTrainer(model, training_config, data_loader)
    
    # Train the model
    results = await trainer.train_comprehensive(
        case_types=case_types,
        use_real_data=True,
        use_synthetic_data=True
    )
    
    return results