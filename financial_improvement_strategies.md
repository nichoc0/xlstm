# Financial Time Series Prediction Optimization Strategies

## Directional Accuracy Enhancement (High Priority)

The current model shows low directional accuracy (1-3%), which is the most critical metric for profitable trading. Here are specialized techniques to improve directional forecasting:

### 1. Direction-Optimized Loss Functions

```python
def asymmetric_directional_loss(y_pred, y_true, up_penalty=2.0, down_penalty=1.0):
    """
    Penalizes direction errors with asymmetric weighting.
    Typically markets have different dynamics in up vs down moves.
    """
    diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
    diff_true = y_true[:, 1:] - y_true[:, :-1]
    
    # Direction matches
    direction_match = (torch.sign(diff_pred) == torch.sign(diff_true))
    
    # Calculate penalties based on true direction (up vs down)
    up_moves = diff_true > 0
    down_moves = diff_true < 0
    
    # Apply asymmetric penalties
    penalties = torch.ones_like(diff_true)
    penalties = torch.where(up_moves, penalties * up_penalty, penalties)
    penalties = torch.where(down_moves, penalties * down_penalty, penalties)
    
    # Apply penalties only to mismatched directions
    weighted_errors = (~direction_match).float() * penalties
    
    return weighted_errors.mean()
```

### 2. Multi-Head Prediction Architecture

Modify the `FunnyMachine` class to include explicit directional prediction:

```python
class EnhancedFunnyMachine(nn.Module):
    def __init__(self, input_size, hidden_size=16, matrix_size=4, dropout=0.2):
        super().__init__()
        self.extended_mlstm = ParallelExtendedSMLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            matrix_size=matrix_size,
            num_layers=2,
            dropout=dropout,
            expansion_factor=2,
        )
        # Price prediction head
        self.price_head = nn.Linear(hidden_size * matrix_size**2, 5)
        
        # Direction prediction head (binary: up/down for each of the 5 future points)
        self.direction_head = nn.Linear(hidden_size * matrix_size**2, 5)
        
        # Volatility prediction head (useful for confidence intervals)
        self.volatility_head = nn.Linear(hidden_size * matrix_size**2, 5)
        
        # Market regime conditioning
        self.regime_adapter = nn.Linear(4, 5)

    def forward(self, x):
        outputs, regime, signals = self.extended_mlstm(x)
        pooled_output = F.gelu(outputs).mean(1)
        
        # Price predictions
        base_price = self.price_head(pooled_output)
        regime_adjustment = self.regime_adapter(regime) * 0.1
        price_predictions = base_price + regime_adjustment
        
        # Direction predictions (sigmoid for binary classification)
        direction_logits = self.direction_head(pooled_output)
        direction_predictions = torch.sigmoid(direction_logits)
        
        # Volatility predictions (always positive via softplus)
        volatility_predictions = F.softplus(self.volatility_head(pooled_output))
        
        return {
            'price': price_predictions, 
            'direction': direction_predictions,
            'volatility': volatility_predictions,
            'regime': regime, 
            'signals': signals
        }
```

### 3. Label Engineering for Directional Training

```python
def create_enhanced_sequences(scaled_data, target_scaled, seq_length, augment=True):
    X, y_price, y_direction = [], [], []
    
    for i in range(seq_length, len(scaled_data) - 5):
        # Input sequence
        seq = scaled_data[i - seq_length:i]
        
        # Price targets
        price_targets = target_scaled[i:i+5].flatten()
        
        # Direction targets (1 for up, 0 for down or flat)
        price_seq = target_scaled[i-1:i+5]
        direction_targets = (price_seq[1:] > price_seq[:-1]).astype(np.float32)
        
        if augment:
            # Existing augmentation techniques
            if np.random.random() < 0.2:
                noise = np.random.normal(0, 0.002 * np.std(seq), seq.shape)
                seq = seq + noise
                
            # Direction-preserving augmentation:
            # Scale magnitude but preserve sign of price changes
            if np.random.random() < 0.15:
                scale_factor = np.random.uniform(0.9, 1.1)
                # Apply to price targets while preserving direction
                price_diff = price_targets[1:] - price_targets[:-1]
                scaled_diff = price_diff * scale_factor
                new_prices = [price_targets[0]]
                for diff in scaled_diff:
                    new_prices.append(new_prices[-1] + diff)
                price_targets = np.array(new_prices)

        X.append(seq)
        y_price.append(price_targets)
        y_direction.append(direction_targets)
        
    return np.array(X), np.array(y_price), np.array(y_direction)
```

## Matrix Memory Enhancements for Financial Data

The s-mLSTM's matrix memory can be further optimized for capturing financial patterns:

### 1. Trend Persistence Memory Component

```python
def enhance_parallel_memory_update(self, v, k, i, f, C_prev, seq_len, regime_weights, volatility_scale):
    """Enhanced memory update with trend persistence."""
    # Existing memory update logic
    C_t = self.parallel_memory_update(v, k, i, f, C_prev, seq_len, regime_weights, volatility_scale)
    
    # Extract trend information from the input sequence
    batch_size, seq_len, _ = v.shape
    
    # Calculate trend persistence based on autocorrelation
    # For financial series, autocorrelation indicates trend strength
    trend_strength = self.calculate_trend_strength(v)  # [batch, 1, 1, 1]
    
    # Scale memory persistence based on detected trend strength
    # Strong trends should have higher memory persistence
    trend_memory = torch.zeros_like(C_t)
    trend_memory[:, 1:] = C_t[:, :-1]  # Shift memory by 1 step
    
    # Blend based on trend strength (stronger trends rely more on past memory)
    C_t_enhanced = C_t * (1 - trend_strength) + trend_memory * trend_strength
    
    return C_t_enhanced
```

### 2. Regime-Specific Memory Processing

```python
def regime_specific_memory_mixing(self, C_t, regime_weights):
    """Apply different memory mixing strategies based on detected market regime."""
    batch_size, seq_len, d, _ = C_t.shape
    
    # Four regimes: trending up, trending down, mean-reversion, high volatility
    regime_processors = [
        self.trend_up_processor,
        self.trend_down_processor,
        self.mean_reversion_processor,
        self.volatility_processor
    ]
    
    # Process the memory with each regime-specific processor
    regime_memories = [processor(C_t) for processor in regime_processors]
    
    # Blend memory versions based on regime weights
    # [batch, seq, regime] -> [batch, seq, 1, 1]
    blend_weights = [rw.unsqueeze(-1).unsqueeze(-1) for rw in regime_weights.split(1, dim=2)]
    
    blended_memory = sum(m * w for m, w in zip(regime_memories, blend_weights))
    
    return blended_memory
```

## Financial Feature Engineering for Directional Accuracy

### 1. Technical Pattern Recognition Features

```python
def add_pattern_recognition_features(df):
    """Add features that detect common technical patterns."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Detect double tops/bottoms (common reversal patterns)
    df['Double_Top'] = 0.0
    df['Double_Bottom'] = 0.0
    
    # Rolling windows to detect patterns
    window = 20
    for i in range(window, len(df)):
        # Simple double top detection
        recent_highs = high[i-window:i].nlargest(2).values
        if abs(recent_highs[0] - recent_highs[1]) / recent_highs[0] < 0.03:  # Within 3%
            df.loc[df.index[i], 'Double_Top'] = 1.0
            
        # Simple double bottom detection
        recent_lows = low[i-window:i].nsmallest(2).values
        if abs(recent_lows[0] - recent_lows[1]) / recent_lows[0] < 0.03:  # Within 3%
            df.loc[df.index[i], 'Double_Bottom'] = 1.0
    
    # Support and resistance levels
    df['Support_Strength'] = 0.0
    df['Resistance_Strength'] = 0.0
    
    # Calculate price ranges that have high volume (liquidity)
    price_volume = pd.DataFrame({'price': close, 'volume': df['Volume']})
    
    # Group by price bins and sum volume
    bins = np.linspace(low.min() * 0.95, high.max() * 1.05, 100)
    liquidity_levels = price_volume.groupby(pd.cut(price_volume['price'], bins))['volume'].sum()
    
    # Find high liquidity areas
    high_liquidity = liquidity_levels[liquidity_levels > liquidity_levels.quantile(0.8)].index
    
    # Mark proximity to support/resistance
    for idx, row in df.iterrows():
        price = row['Close']
        for level in high_liquidity:
            if price > level.left and price < level.right:
                if price < close.iloc[df.index.get_loc(idx) - 1]:  # Price falling to support
                    df.at[idx, 'Support_Strength'] = 1.0
                elif price > close.iloc[df.index.get_loc(idx) - 1]:  # Price rising to resistance
                    df.at[idx, 'Resistance_Strength'] = 1.0
    
    return df
```

### 2. Multi-Timeframe Information Integration

```python
def add_multi_timeframe_features(df, timeframes=[5, 21, 63]):
    """Add features from multiple timeframes to capture different cycle components."""
    for tf in timeframes:
        # Price momentum at different timeframes
        df[f'Return_{tf}d'] = df['Close'].pct_change(tf)
        
        # Volatility at different timeframes
        df[f'Volatility_{tf}d'] = df['Returns'].rolling(window=tf).std()
        
        # RSI at different timeframes
        df[f'RSI_{tf}d'] = ta.momentum.RSIIndicator(close=df['Close'], window=tf).rsi() / 100
        
        # Trend strength at different timeframes
        df[f'ADX_{tf}d'] = ta.trend.ADXIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], window=tf
        ).adx() / 100
    
    # Create trend agreement features
    # When short, medium and long timeframes align, trends are stronger
    df['Trend_Agreement'] = (
        np.sign(df['Return_5d']) == np.sign(df['Return_21d'])
    ) & (np.sign(df['Return_21d']) == np.sign(df['Return_63d']))
    
    # Convert boolean to float
    df['Trend_Agreement'] = df['Trend_Agreement'].astype(float)
    
    return df
```

## Advanced Training Techniques for Directional Prediction

### 1. Specialized Direction-Aware Training Loop

```python
def train_direction_optimized_model(model, train_loader, val_loader, target_scaler, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    scaler = torch.amp.GradScaler('cuda')
    
    # Price prediction loss
    price_criterion = nn.HuberLoss(delta=0.5)
    
    # Direction prediction loss (with class weights for imbalance)
    # Markets typically have upward bias (more up days than down)
    pos_weight = torch.tensor([1.2, 1.2, 1.2, 1.2, 1.2]).to(device)  # Slight emphasis on down moves
    direction_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Adaptive loss weighting
    price_weight = 1.0
    direction_weight = 2.0  # Start with higher emphasis on direction
    
    optimizer = AdamW(
        model.parameters(), 
        lr=1e-4,  # Lower learning rate for stability
        weight_decay=1e-4,
        eps=1e-8
    )
    
    # Warmup + cosine decay schedule
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * 2,
        num_training_steps=len(train_loader) * epochs,
        num_cycles=epochs/10
    )
    
    for epoch in range(epochs):
        model.train()
        
        # Adjust loss weights over time:
        # Start with higher direction weight, gradually balance
        if epoch > epochs // 3:
            direction_weight = max(1.0, direction_weight * 0.9)
        
        # Training loop
        for X_batch, y_price_batch, y_dir_batch in train_loader:
            X_batch = X_batch.to(device)
            y_price_batch = y_price_batch.to(device)
            y_dir_batch = y_dir_batch.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(X_batch)
                
                # Price prediction loss
                price_loss = price_criterion(outputs['price'], y_price_batch)
                
                # Direction prediction loss
                dir_loss = direction_criterion(outputs['direction'], y_dir_batch)
                
                # Combined loss
                loss = price_weight * price_loss + direction_weight * dir_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        # Validation logic...
        # Calculate both price accuracy and direction accuracy
```

### 2. Ensembling with Direction Specialization

```python
class DirectionalEnsemble(nn.Module):
    """Ensemble model specialized for directional prediction."""
    def __init__(self, input_size, seq_lengths=[32, 48, 64, 96]):
        super().__init__()
        # Different sequence length models capture different horizons
        self.models = nn.ModuleList([
            EnhancedFunnyMachine(
                input_size=input_size,
                hidden_size=16,
                matrix_size=4,
                dropout=0.2
            ) 
            for _ in seq_lengths
        ])
        self.seq_lengths = seq_lengths
        
        # Meta-learner to weight predictions
        self.meta_learner = nn.Sequential(
            nn.Linear(len(seq_lengths) * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        # Direction-specific meta-learner
        self.direction_meta = nn.Sequential(
            nn.Linear(len(seq_lengths) * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Sigmoid()
        )
    
    def forward(self, x_dict):
        """Forward pass with multiple sequence lengths."""
        # Each model uses different historical context length
        price_preds = []
        dir_preds = []
        
        for i, model in enumerate(self.models):
            seq_len = self.seq_lengths[i]
            outputs = model(x_dict[f'seq_{seq_len}'])
            price_preds.append(outputs['price'])
            dir_preds.append(outputs['direction'])
        
        # Concatenate all predictions
        all_price_preds = torch.cat(price_preds, dim=1)
        all_dir_preds = torch.cat(dir_preds, dim=1)
        
        # Meta-learning for final predictions
        final_price = self.meta_learner(all_price_preds)
        final_direction = self.direction_meta(all_dir_preds)
        
        return {
            'price': final_price,
            'direction': final_direction,
            # Other outputs...
        }
```

## Implementation Priorities

For maximum impact on directional accuracy, implement these changes in order:

1. First priority: 
   - Direction-specialized loss function
   - Multi-head prediction with explicit direction output
   - Trend-specific memory processing

2. Second priority:
   - Technical pattern recognition features
   - Multi-timeframe feature integration
   - Direction-optimized training loop

3. Final enhancements:
   - Ensemble approach with direction specialization
   - Fine-tuning hyperparameters with focus on directional metrics

These strategies directly address the shortcomings in directional prediction while leveraging the strengths of the s-mLSTM architecture's matrix memory for financial patterns.
