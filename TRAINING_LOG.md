# Training Log

## Initial Model (92,925 parameters)

### Epoch 1
- Train Loss: 0.2390  
- Val Loss: 0.4416  
- MAPE: 86.72%  
- RMSE: 1.1539  
- MAE: 1.1320  
- Directional Accuracy: 1.48%  
- Regime Distribution: [0.25636956 0.26752508 0.22670312 0.2494022 ]  
- Signal Distribution: [0.35561776 0.34708965 0.2972926 ]  
- GPU Memory: 0.02 GB (after epoch)

### Epoch 2
- Train Loss: 0.0991  
- Val Loss: 0.2034  
- MAPE: 48.15%  
- RMSE: 0.7071  
- MAE: 0.6426  
- Directional Accuracy: 1.82%  
- Regime Distribution: [0.3175423  0.2758296  0.17545111 0.23117688]
- Signal Distribution: [0.4005036  0.35343257 0.24606384]
- GPU Memory: 0.02 GB (after epoch)

### Epoch 3
- Train Loss: 0.0677  
- Val Loss: 0.1935  
- MAPE: 45.88%  
- RMSE: 0.6900  
- MAE: 0.6170  
- Directional Accuracy: 1.75%  
- Regime Distribution: [0.32954675 0.26953137 0.17400785 0.22691406]
- Signal Distribution: [0.38873833 0.3561366  0.25512502]
- GPU Memory: 0.02 GB (after epoch)

## Scaled Model (1,403,709 parameters)

### Epoch 1
- Train Loss: 0.4361  
- Val Loss: 0.2342  
- MAPE: 53.53%  
- RMSE: 0.7539  
- MAE: 0.7112  
- Directional Accuracy: 2.43%  
- Regime Distribution: [0.312012 0.34626883 0.20387585 0.13784333]
- Signal Distribution: [0.26784554 0.39057726 0.34157714]
- GPU Memory: 0.04 GB (after epoch)

### Epoch 2
- Train Loss: 0.3490  
- Val Loss: 0.2064  
- MAPE: 48.83%  
- RMSE: 0.6986  
- MAE: 0.6529  
- Directional Accuracy: 2.20%  
- Regime Distribution: [0.30800092 0.33756804 0.2134448 0.14098623]
- Signal Distribution: [0.28029755 0.38078547 0.33891696]
- GPU Memory: 0.04 GB (after epoch)

### Epoch 3
- Train Loss: 0.3261  
- Val Loss: 0.1720  
- MAPE: 43.32%  
- RMSE: 0.6239  
- MAE: 0.5789  
- Directional Accuracy: 3.15%  
- Regime Distribution: [0.3093886 0.33654404 0.21506515 0.13900223]
- Signal Distribution: [0.29258963 0.370693 0.3367174]
- GPU Memory: 0.04 GB (after epoch)

### Epoch 4
- Train Loss: 0.3074  
- Val Loss: 0.1752  
- MAPE: 44.10%  
- RMSE: 0.6277  
- MAE: 0.5869  
- Directional Accuracy: 2.43%  
- Regime Distribution: [0.30704 0.32354388 0.22169878 0.1477173]
- Signal Distribution: [0.30612984 0.35723737 0.33663282]
- GPU Memory: 0.04 GB (after epoch)

## Performance Analysis

### Comparison between Initial and Scaled Models
1. **Validation Loss**: 
   - Initial model: 0.1935 (epoch 3)
   - Scaled model: 0.1720 (epoch 3)
   - **Conclusion**: Scaled model shows improved validation loss

2. **MAPE (Mean Absolute Percentage Error)**:
   - Initial model: 45.88% (epoch 3)
   - Scaled model: 43.32% (epoch 3)
   - **Conclusion**: Scaled model shows modest improvement in MAPE

3. **Directional Accuracy**:
   - Initial model: ~1.75%
   - Scaled model: ~3.15% (peak at epoch 3)
   - **Conclusion**: Slight improvement but still very low, suggesting further work needed on directional prediction

4. **Regime Distribution**:
   - Initial model had more balanced regime distribution
   - Scaled model shows preference for regimes 0 and 1 with less emphasis on regime 3
   - **Conclusion**: Different model behavior in regime detection

5. **Signal Distribution**:
   - Initial model had "Buy" bias
   - Scaled model has more balanced "Buy"/"Sell" but with "Hold" preference
   - **Conclusion**: Potentially more conservative trading strategy

### Observations
1. The slight increase in validation loss at epoch 4 may indicate early signs of overfitting
2. Memory usage remains extremely efficient despite 15x parameter increase
3. Both models achieve similar performance levels by epoch 3, suggesting diminishing returns from parameter scaling alone

### Next Steps
1. Continue training to see if performance plateaus or improves further
2. Consider adding skip connections between LSTM layers to improve gradient flow
3. Experiment with lower learning rates (1e-4) to stabilize training
4. Increase dropout to 0.3 to combat potential overfitting
5. Focus on improving directional accuracy, which remains very low
