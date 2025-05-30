# Modular Deep CNN Evaluator

This document describes the modular deep CNN evaluator implementation for the chess engine.

## Overview

The deep CNN evaluator provides a sophisticated neural network architecture for chess position evaluation, featuring:

- **Residual connections** for better gradient flow
- **Batch normalization** for training stability
- **Modular architecture** with configurable depth and width
- **Two variants**: Full deep architecture and lightweight version
- **Dropout regularization** to prevent overfitting

## Architecture Variants

### 1. Deep Architecture (`DeepChessCNN`)
- **Default**: 8 residual blocks, 256 base channels
- **Parameters**: ~1.76M (configurable)
- **Features**: 
  - Value head for position evaluation
  - Policy head (prepared for future move prediction)
  - Heavy dropout (0.3) for regularization

### 2. Lightweight Architecture (`LightweightDeepCNN`)
- **Default**: 4 residual blocks, 128 base channels  
- **Parameters**: ~173K (configurable)
- **Features**:
  - Streamlined value head only
  - Lighter dropout (0.2)
  - Faster inference

## Usage Examples

### Basic Usage
```python
from src.engine.evaluators.deep_cnn_eval import DeepCNN_Eval

# Lightweight version
evaluator = DeepCNN_Eval(
    board,
    architecture="lightweight",
    num_residual_blocks=2,
    base_channels=64
)

# Full deep version
evaluator = DeepCNN_Eval(
    board,
    architecture="deep",
    num_residual_blocks=8,
    base_channels=256
)

score = evaluator.evaluate()
```

### With Pre-trained Model
```python
evaluator = DeepCNN_Eval(
    board,
    model_path="data/models/deep_cnn.pth",
    architecture="deep"
)
```

### Custom Configuration
```python
evaluator = DeepCNN_Eval(
    board,
    architecture="deep",
    num_residual_blocks=12,  # Very deep
    base_channels=512        # Very wide
)
```

## Model Information

Get detailed information about the model:

```python
info = evaluator.get_model_info()
print(info)
# Output:
# {
#     'architecture': 'deep',
#     'total_parameters': 1761034,
#     'trainable_parameters': 1761034,
#     'device': 'cpu',
#     'model_class': 'DeepChessCNN'
# }
```

## Performance Comparison

From test runs on a sample position:

| Evaluator | Score | Parameters | Architecture |
|-----------|-------|------------|--------------|
| Simple Eval | -1.0 | N/A | Rule-based |
| Simple NN | 0.26 | ~50K | 3 conv layers |
| Deep CNN (Light) | -1.17 | 173K | 2 residual blocks |
| Deep CNN (Full) | 1.68 | 1.76M | 4 residual blocks |

## Key Features

### Residual Blocks
Each residual block contains:
- Two 3x3 convolutions
- Batch normalization after each convolution
- ReLU activations
- Skip connection for gradient flow

### Modular Design
- **Configurable depth**: Adjust `num_residual_blocks`
- **Configurable width**: Adjust `base_channels`
- **Architecture selection**: Choose between "deep" and "lightweight"
- **Device agnostic**: Automatically uses GPU if available

### Training Ready
- Batch normalization for stable training
- Dropout for regularization
- Proper weight initialization
- Compatible with existing training pipeline

## Testing

Run the test suite:
```bash
python -m pytest tests/deep_cnn_eval_test.py -v
```

Tests cover:
- Architecture creation
- Evaluation functionality
- Checkmate detection
- Model information retrieval
- Different configurations

## Integration

The deep CNN evaluator integrates seamlessly with the existing chess engine:

- Inherits from the base `Eval` class
- Compatible with minimax search
- Works with Zobrist hashing
- Follows the same interface as other evaluators

## Future Enhancements

Potential improvements:
- **Policy head utilization**: Use the policy head for move ordering
- **Multi-task learning**: Train on both position evaluation and move prediction
- **Attention mechanisms**: Add self-attention for long-range dependencies
- **Quantization**: Reduce model size for faster inference
- **Knowledge distillation**: Train smaller models from larger ones

## File Structure

```
src/engine/evaluators/
├── deep_cnn_eval.py      # Main implementation
├── simple_nn_eval.py     # Simple NN for comparison
└── eval.py               # Base evaluator class

tests/
└── deep_cnn_eval_test.py # Test suite

docs/
└── deep_cnn_eval_readme.md # This documentation
