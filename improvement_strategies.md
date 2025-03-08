# Model Improvement Strategies

## Improving Directional Accuracy

The current models show very low directional accuracy (1-3%), which is critical for trading applications. Strategies to improve:

1. **Directional Loss Function**: Add a specialized loss component that penalizes direction errors
   ```python
   def directional_loss(y_pred, y_true):
       direction_pred = torch.sign(y_pred[:, 1:] - y_pred[:, :-1])
       direction_true = torch.sign(y_true[:, 1:] - y_true[:, :-1])
       return torch.mean((direction_pred != direction_true).float())
   
   # Combined loss
   loss = criterion(outputs, y_batch) + 0.2 * directional_loss(outputs, y_batch) + l1_lambda * l1_norm
   ```

2. **Multi-task Learning**: Add explicit direction prediction as a separate task
   ```python
   # Additional output head in model
   self.direction_head = nn.Linear(hidden_size, 4)  # Up, Down, Flat, Strong-trend
   
   # In forward pass
   direction_pred = self.direction_head(features)
   ```

3. **Training Data Augmentation**: Oversample trend reversal points
   ```python
   # Identify reversal points in preprocessing
   price_diff = df['Close'].diff()
   sign_changes = (np.signbit(price_diff[1:]) != np.signbit(price_diff[:-1]))
   reversal_indices = np.where(sign_changes)[0]
   
   # Create more training samples around these points
   ```

## Enhancing Model Architecture

1. **Residual LSTM Connections**:
   ```python
   class ResidualLSTMLayer(nn.Module):
       def __init__(self, input_dim, hidden_dim):
           super().__init__()
           self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
           self.projection = nn.Linear(hidden_dim, input_dim) if hidden_dim != input_dim else nn.Identity()
           
       def forward(self, x):
           out, _ = self.lstm(x)
           return x + self.projection(out)
   ```

2. **Attention Mechanism** to focus on relevant parts of the sequence:
   ```python
   class TimeAttention(nn.Module):
       def __init__(self, hidden_dim):
           super().__init__()
           self.query = nn.Linear(hidden_dim, hidden_dim)
           self.key = nn.Linear(hidden_dim, hidden_dim)
           self.value = nn.Linear(hidden_dim, hidden_dim)
           
       def forward(self, x):
           # x: [batch, seq, features]
           q = self.query(x)
           k = self.key(x)
           v = self.value(x)
           
           # Scaled dot-product attention
           scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(x.size(-1))
           attention = F.softmax(scores, dim=-1)
           
           return torch.bmm(attention, v)
   ```

3. **Feature Gating** based on detected market regime:
   ```python
   # In the ParallelExtendedSMLSTM forward pass
   regime_gates = F.softmax(self.regime_gates(regime), dim=-1)  # [batch, num_features]
   gated_features = x * regime_gates.unsqueeze(1)  # Apply gates to features
   ```

## Training Enhancements

1. **Cosine Annealing with Warm Restarts**:
   ```python
   scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
       optimizer, 
       T_0=5,  # Initial restart period
       T_mult=2,  # Period multiplier after each restart
       eta_min=1e-6  # Minimum learning rate
   )
   ```

2. **Gradient Accumulation** for effective larger batch sizes:
   ```python
   accumulation_steps = 4
   
   for X_batch, y_batch in train_loader:
       # Forward pass
       outputs, regime, signals = model(X_batch)
       loss = criterion(outputs, y_batch)
       
       # Scale the loss
       loss = loss / accumulation_steps
       
       # Backward pass
       loss.backward()
       
       # Update weights after accumulation_steps
       if (batch_idx + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Cyclical Learning Rates**:
   ```python
   scheduler = lr_scheduler.CyclicLR(
       optimizer,
       base_lr=1e-5,
       max_lr=5e-4,
       step_size_up=len(train_loader) * 2,
       mode='triangular2'
   )
   ```

## Feature Engineering

1. **Technical Indicator Transformations**:
   - Standardize using z-scores rather than simple scaling
   - Use log transformations for indicators with high skewness
   - Add rolling volatility-adjusted indicators

2. **Feature Crosses**:
   ```python
   # Example: RSI and volume interaction
   df['RSI_Volume'] = df['RSI'] * df['Volume_MA'] / df['Volume'].mean()
   
   # Price momentum relative to market (if you have market data)
   df['Relative_Momentum'] = df['Returns'] - market_df['Returns']
   ```

3. **Time-based Features**:
   ```python
   # Time since last trend change
   df['Days_Since_Trend_Change'] = df.groupby((df['Close'].shift() > df['Close']).diff().ne(0).cumsum()).cumcount()
   
   # Distance from recent high/low
   df['Pct_From_20d_High'] = df['Close'] / df['Close'].rolling(20).max() - 1
   df['Pct_From_20d_Low'] = df['Close'] / df['Close'].rolling(20).min() - 1
   ```
