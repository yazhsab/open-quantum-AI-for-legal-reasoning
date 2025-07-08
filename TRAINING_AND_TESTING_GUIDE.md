# XQELM Training and Testing Guide with Real Legal Data

This guide explains how the XQELM (Explainable Quantum-Enhanced Language Models for Legal Reasoning) project trains and tests models using both real legal data and synthetic data.

## Overview

The XQELM project now supports a comprehensive training pipeline that combines:
- **Real legal data** from Indian legal databases and court records
- **Synthetic legal data** generated using AI to supplement real data
- **Quantum-enhanced training** using PennyLane quantum circuits
- **Cross-validation** and robust evaluation metrics

## Current Training Architecture

### 1. Data Sources

#### Real Legal Data Sources
- **Indian Kanoon**: Comprehensive legal database with court judgments
- **eCourts Services**: Government portal for court case information
- **Consumer Forum Database**: Consumer dispute cases and orders
- **Motor Accident Claims Tribunal**: Vehicle accident compensation cases
- **Property Dispute Records**: Land and property related legal cases

#### Synthetic Data Generation
- **AI-generated cases** based on real legal patterns
- **Configurable complexity** distributions (low/medium/high)
- **Jurisdiction-aware** generation across Indian states
- **Legal principle compliance** ensuring realistic scenarios

### 2. Training Pipeline Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Real Data     │    │  Synthetic Data  │    │  Sample Cases   │
│   Fetcher       │    │   Generator      │    │   (Manual)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Combined Dataset   │
                    │  (Train/Test Split) │
                    └─────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │ Enhanced Trainer    │
                    │ - Cross-validation  │
                    │ - Weighted training │
                    │ - Early stopping    │
                    └─────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │ Quantum Legal Model │
                    │ - Quantum circuits  │
                    │ - Classical NLP     │
                    │ - Explainability    │
                    └─────────────────────┘
```

## Getting Started

### Prerequisites

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Setup Data Sources Configuration**:
```bash
python scripts/fetch_real_data.py --setup-config
```

3. **Configure API Keys** (edit `config/data_sources.json`):
```json
{
  "api_keys": {
    "indian_kanoon": "YOUR_API_KEY_HERE",
    "ecourts": "YOUR_API_KEY_HERE"
  }
}
```

### Quick Start Training

#### Option 1: Comprehensive Training (Recommended)
```bash
# Complete pipeline: fetch real data + generate synthetic data + train model
python scripts/train_with_real_data.py \
    --case-types property_disputes motor_vehicle_claims consumer_disputes \
    --real-cases 50 \
    --synthetic-cases 100 \
    --epochs 50 \
    --n-qubits 12 \
    --use-cross-validation
```

#### Option 2: Step-by-Step Process

**Step 1: Fetch Real Legal Data**
```bash
python scripts/fetch_real_data.py \
    --case-type all \
    --real-cases 50 \
    --synthetic-cases 100
```

**Step 2: Train with Enhanced Pipeline**
```bash
python scripts/train_with_real_data.py \
    --skip-data-setup \
    --epochs 100 \
    --learning-rate 0.01
```

#### Option 3: Using Existing Synthetic Data Only
```bash
# Generate comprehensive synthetic datasets
python scripts/generate_datasets.py \
    --num-cases 200 \
    --case-type all \
    --train-test-split

# Train with synthetic data
python scripts/train_with_real_data.py \
    --skip-data-setup \
    --epochs 50
```

## Data Structure and Quality

### Dataset Format

Each legal case follows this standardized structure:

```json
{
  "case_id": "CD-001-2024",
  "case_type": "consumer_dispute",
  "input_data": {
    "complaint_type": "defective_goods",
    "consumer_name": "John Doe",
    "complaint_description": "Product defect details...",
    "evidence_documents": ["Invoice", "Photos"],
    "compensation_claimed": 50000.0
  },
  "expected_output": {
    "success_probability": 0.78,
    "evidence_strength": {
      "documentary_evidence": 0.8,
      "overall_strength": 0.75
    },
    "estimated_compensation": {
      "total": 95000.0
    },
    "recommendations": [
      "File in District Consumer Forum",
      "Obtain expert technical opinion"
    ]
  },
  "metadata": {
    "source": "real_data",  // or "synthetic"
    "jurisdiction": "india",
    "state": "karnataka",
    "complexity_level": "medium",
    "legal_principles": ["consumer_protection_act_2019"]
  }
}
```

### Data Quality Assurance

- **Validation Framework**: Structural integrity, legal accuracy, realistic values
- **Deduplication**: Automatic removal of duplicate cases
- **Standardization**: Consistent format across all data sources
- **Legal Compliance**: Adherence to Indian legal principles and procedures

## Training Configuration

### Enhanced Trainer Features

```python
TrainingConfig(
    epochs=100,
    learning_rate=0.01,
    batch_size=32,
    validation_split=0.2,
    early_stopping_patience=10,
    use_cross_validation=True,
    cv_folds=5,
    real_data_weight=1.5,      # Weight real data higher
    synthetic_data_weight=1.0,
    save_checkpoints=True
)
```

### Key Training Features

1. **Weighted Training**: Real legal data is weighted higher than synthetic data
2. **Cross-Validation**: 5-fold stratified cross-validation for robust evaluation
3. **Early Stopping**: Prevents overfitting with patience-based stopping
4. **Mixed Data Handling**: Seamless integration of real and synthetic cases
5. **Quantum-Classical Hybrid**: Combines quantum circuits with classical NLP

## Model Architecture

### Quantum Components
- **Quantum Embeddings**: Legal concept encoding in quantum states
- **Quantum Attention**: Attention mechanism using quantum circuits
- **Quantum Reasoning**: Legal reasoning through quantum state evolution

### Classical Components
- **Text Preprocessing**: Legal document processing and entity extraction
- **Knowledge Base**: Legal precedents and statute retrieval
- **Response Generation**: Natural language answer generation

### Explainability
- **Quantum State Analysis**: Understanding quantum decision processes
- **Legal Principle Mapping**: Connecting decisions to legal principles
- **Confidence Scoring**: Reliability assessment of predictions

## Testing and Evaluation

### Evaluation Metrics

1. **Accuracy**: Overall prediction accuracy
2. **Real vs Synthetic Performance**: Separate metrics for different data types
3. **Confidence Calibration**: Alignment between confidence and accuracy
4. **Processing Time**: Efficiency of quantum-classical pipeline
5. **Legal Principle Adherence**: Compliance with legal reasoning

### Test Data Structure

- **80/20 Train-Test Split**: Standard machine learning practice
- **Stratified Sampling**: Balanced complexity and jurisdiction distribution
- **Edge Cases**: Challenging scenarios for robust testing
- **Cross-Case-Type Evaluation**: Testing generalization across legal domains

### Running Evaluations

```bash
# Evaluate trained model
python scripts/train_with_real_data.py \
    --skip-data-setup \
    --skip-training \
    --evaluate-only

# Validate datasets
python scripts/test_datasets.py
```

## Current Data Statistics

### Available Datasets (as of implementation)

| Case Type | Sample Cases | Real Data Capability | Synthetic Generation |
|-----------|--------------|---------------------|---------------------|
| Consumer Disputes | 3 | ✅ Consumer Forums | ✅ 100-10,000+ cases |
| Motor Vehicle Claims | 3 | ✅ MACT Records | ✅ 100-10,000+ cases |
| Property Disputes | 3 | ✅ Land Records | ✅ 100-10,000+ cases |
| Bail Applications | Available | 🔄 In Development | ✅ Available |
| Cheque Bounce | Available | 🔄 In Development | ✅ Available |

### Data Quality Metrics

- **Completeness**: 100% of required fields populated
- **Consistency**: Standardized formats across all cases
- **Legal Accuracy**: Verified against Indian legal principles
- **Diversity**: Balanced across 10 Indian states and complexity levels

## Real Data Integration Benefits

### Before (Synthetic Only)
- ❌ Only 9 hand-crafted sample cases
- ❌ Limited real-world complexity
- ❌ Potential overfitting to synthetic patterns
- ❌ Uncertain generalization to actual legal scenarios

### After (Real + Synthetic)
- ✅ 50-100+ real cases per legal domain
- ✅ Authentic legal complexity and variations
- ✅ Robust training with diverse data sources
- ✅ Validated performance on actual legal cases
- ✅ Improved model reliability and trustworthiness

## Advanced Usage

### Custom Data Sources

Add new data sources by extending the configuration:

```json
{
  "data_sources": [
    {
      "name": "Custom Legal Database",
      "base_url": "https://api.customlegal.com",
      "rate_limit": 30,
      "case_types": ["custom_case_type"],
      "requires_auth": true
    }
  ]
}
```

### Custom Training Configurations

```python
# High-performance training
training_config = {
    'epochs': 200,
    'learning_rate': 0.005,
    'batch_size': 64,
    'n_qubits': 20,
    'n_layers': 6,
    'real_data_weight': 2.0,
    'use_cross_validation': True,
    'cv_folds': 10
}

# Quick prototyping
training_config = {
    'epochs': 20,
    'learning_rate': 0.02,
    'batch_size': 16,
    'n_qubits': 8,
    'n_layers': 2,
    'use_cross_validation': False
}
```

### Monitoring and Debugging

1. **Training Logs**: Comprehensive logging in `training.log`
2. **Metrics Visualization**: Automatic plot generation
3. **Checkpoint Saving**: Model state preservation during training
4. **Error Handling**: Graceful handling of data fetching failures

## Troubleshooting

### Common Issues

1. **API Rate Limiting**
   ```bash
   # Reduce request rate
   python scripts/fetch_real_data.py --real-cases 20
   ```

2. **Memory Issues with Large Datasets**
   ```bash
   # Reduce batch size
   python scripts/train_with_real_data.py --batch-size 8
   ```

3. **No Real Data Available**
   ```bash
   # Use synthetic data only
   python scripts/generate_datasets.py --num-cases 500
   ```

### Performance Optimization

1. **GPU Acceleration**: Use CUDA-enabled PyTorch for classical components
2. **Quantum Simulators**: Choose appropriate PennyLane backends
3. **Parallel Processing**: Utilize multiple cores for data processing
4. **Caching**: Cache processed embeddings and features

## Future Enhancements

### Planned Improvements

1. **Additional Data Sources**
   - Supreme Court judgments
   - High Court databases
   - Tribunal records
   - Legal news and updates

2. **Advanced Training Techniques**
   - Transfer learning from pre-trained legal models
   - Federated learning across institutions
   - Active learning for optimal data selection

3. **Enhanced Evaluation**
   - Legal expert validation
   - Comparative analysis with traditional models
   - Real-world deployment testing

## Contributing

### Adding New Case Types

1. **Define Data Model**: Create input/output structure
2. **Implement Use Case Manager**: Add business logic
3. **Create Sample Cases**: Hand-craft representative examples
4. **Add Synthetic Generation**: Extend generator with domain logic
5. **Update Training Pipeline**: Include in comprehensive training

### Data Source Integration

1. **API Integration**: Implement data fetcher for new source
2. **Data Standardization**: Convert to XQELM format
3. **Quality Validation**: Ensure data meets quality standards
4. **Documentation**: Update this guide with new capabilities

## Conclusion

The enhanced XQELM training and testing pipeline provides a robust foundation for developing quantum-enhanced legal AI systems. By combining real legal data with synthetic augmentation, the system achieves:

- **Higher Accuracy**: Training on authentic legal scenarios
- **Better Generalization**: Exposure to real-world complexity
- **Improved Reliability**: Cross-validation and comprehensive evaluation
- **Legal Compliance**: Adherence to Indian legal principles and procedures

This approach ensures that XQELM models are not only technically sophisticated but also practically applicable to real legal reasoning tasks.

---

*For technical support or questions about the training pipeline, please refer to the project documentation or contact the development team.*