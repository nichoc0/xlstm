# Model Scaling and Performance Notes

## Initial Model Performance
The initial smaller model (92,925 parameters) showed promising results:
- Fast convergence: Loss dropped significantly from epoch 1 to 3
- MAPE improved from 86.72% to 45.88%
- RMSE decreased from 1.1539 to 0.6900
- Low memory usage (0.02 GB)
- Balanced regime and signal distributions

## Scaled Model Configuration
The enlarged model (1,403,709 parameters) shows:
- Increased hidden dimensions from 64 to 256
- Sequence length increased from 48 to 64
- ~15x more parameters while maintaining low memory footprint

## Recommendations

### Architecture Improvements
1. **Focused scaling**: The performance gains from early epochs suggest the model might benefit from:
   - Deeper architecture (more layers) rather than just wider layers
   - Adding skip connections between distant layers for better gradient flow
   - Increasing the matrix_size parameter to 3 (from 2) for more expressivity

2. **Ensemble approach**: Consider training multiple models with:
   - Different sequence lengths (48, 64, 96)
   - Different hyperparameters (learning rates, regularization strengths)
   - Different architectures (varying depth vs. width)

### Training Optimizations
1. **Learning rate**: Try a lower initial learning rate with the larger model (1e-4)
2. **Regularization**: With more parameters, increase dropout to 0.3-0.4
3. **Early stopping**: Watch for signs of overfitting with the larger model

### Feature Engineering
The balanced regime distribution suggests your model is distinguishing market regimes well.
Consider creating specialized feature sets for each regime and conditionally activating them.

## Tracking Trade-offs
As you scale further, monitor:
- Inference speed vs. accuracy
- Memory usage vs. model capacity
- Training time vs. performance gains
