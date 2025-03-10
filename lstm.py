# scalar matrix parallel lstm


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.checkpoint import checkpoint

class StructuredStateSpace(nn.Module):
    def __init__(self, d_state, d_model, discretization='bilinear', dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.discretization = discretization
        
        self.log_lambda_real = nn.Parameter(torch.linspace(math.log(dt_min), math.log(dt_max), d_state))
        self.log_b = nn.Parameter(torch.randn(d_state, 1).uniform_(-0.5, 0.5))  # Changed to d_state x 1
        self.c = nn.Parameter(torch.randn(1, d_state) / math.sqrt(d_state))     # Changed to 1 x d_state
        self.log_d = nn.Parameter(torch.zeros(1))                               # Changed to scalar
        
        # Fix: Use a single step size parameter now
        self.log_step = nn.Parameter(torch.tensor(math.log(dt_min)))
        
        self.volatility_gate = nn.Parameter(torch.ones(d_state))
        self.alpha = nn.Parameter(torch.tensor(0.7))
        self.layer_norm = nn.LayerNorm(d_state)

    def forward(self, x, volatility_scale=None):
        batch, seq_len, _ = x.shape
        lambda_real = -torch.exp(self.log_lambda_real)  # [d_state]
        b = torch.exp(self.log_b)                       # [d_state, 1]
        c = self.c                                      # [1, d_state]
        d = torch.exp(self.log_d)                       # [1]
        step = torch.exp(self.log_step)                 # Scalar
        
        # Simplify the discretization calculation
        if self.discretization == 'zoh':
            a_discrete = torch.exp(lambda_real * step)                      # [d_state]
            b_discrete = (a_discrete - 1.0) / lambda_real * b.squeeze(-1)   # [d_state]
        elif self.discretization == 'bilinear':
            a_discrete = (2.0 + step * lambda_real) / (2.0 - step * lambda_real)  # [d_state]
            b_discrete = step * (torch.ones_like(a_discrete) + a_discrete) * b.squeeze(-1) / 2.0  # [d_state]
        
        # Initialize the hidden state
        h = torch.zeros(batch, self.d_state, device=x.device)  # [batch, d_state]
        outputs = []
        
        # Simplify the recurrent loop
        for t in range(seq_len):
            # State update: h = A*h + B*x
            h = a_discrete.unsqueeze(0) * h + b_discrete.unsqueeze(0) * x[:, t, :].mean(dim=1, keepdim=True)
            
            # Apply layer normalization
            h = self.layer_norm(h)
            
            # Apply volatility scaling if provided
            if volatility_scale is not None:
                vol_influence = torch.sigmoid(self.volatility_gate) * volatility_scale[:, t]
                h = h * (1.0 / (1.0 + vol_influence))
            
            # Output calculation: y = C*h + D*x
            y = torch.matmul(h.unsqueeze(1), c.transpose(0, 1)).squeeze(1) + d * x[:, t, :]
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)
        alpha = torch.sigmoid(self.alpha)
        
        # Residual connection
        return alpha * x + (1 - alpha) * y


class ParallelSMLSTMCell(nn.Module):
    """
    Enhanced mLSTM with structured state-space memory (s_mLSTM) for financial time series.
    """
    def __init__(self, input_size, hidden_size, matrix_size=4, stabilizer_threshold=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.matrix_size = matrix_size
        self.d = hidden_size  # State dimension
        self.stabilizer_threshold = stabilizer_threshold
        
        # Initialize memory matrix and state vectors
        self.C_init = nn.Parameter(torch.zeros(1, hidden_size, hidden_size))
        self.n_init = nn.Parameter(torch.zeros(1, hidden_size))
        self.m_init = nn.Parameter(torch.zeros(1, 1))
        
        # Input projections
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.W_i = nn.Linear(input_size, 1)
        self.W_f = nn.Linear(input_size, 1)
        self.W_o = nn.Linear(input_size, hidden_size)
        
        # Structured state-space memory model for each matrix dimension
        self.ssm = StructuredStateSpace(
            d_state=hidden_size,       # Reduced from hidden_size*2 to match properly
            d_model=hidden_size,                 # Simplified to scalar model dimension
            discretization='bilinear',
            dt_min=0.001,
            dt_max=0.1
        )
        
        # Financial regime detection network
        # Detects market regimes like trending, mean-reverting, high-volatility, etc.
        self.regime_detector = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),  # 4 typical market regimes
            nn.Softmax(dim=-1)
        )
        
        # Layer normalization for numerical stability
        self.layer_norm_input = nn.LayerNorm(input_size)
        self.layer_norm_memory = nn.LayerNorm(hidden_size)
        
        # Initialize with financial-specific biases
        # Slightly bias forget gate toward remembering (financial time series have long-term dependencies)
        self.W_f.bias.data.fill_(1.0)
        # Initialize input gate slightly lower to be more selective about new information
        self.W_i.bias.data.fill_(0.0)

    def detect_volatility(self, x):
        """Extract volatility signal from input features for adaptive processing."""
        # Simple volatility estimation: std of last few time steps in sequence
        # Assumes input contains price information
        batch_size, seq_len, _ = x.shape
        
        # Use the last 20% of sequence for volatility estimation (rolling window)
        window_size = max(1, int(seq_len * 0.2))
        if seq_len <= 1:
            return torch.ones(batch_size, seq_len, 1, device=x.device)
            
        # Compute rolling volatility (std of price changes)
        # For simplicity, we use the standard deviation across feature dimensions
        # In practice, extract specific volatility features or financial indicators
        volatility = torch.stack([
            torch.std(x[:, i-window_size:i], dim=1) 
            for i in range(window_size, seq_len+1)
        ], dim=1)
        
        # Pad beginning
        padding = torch.ones(batch_size, window_size-1, x.size(2), device=x.device) * \
                  volatility[:, 0:1, :]
        volatility = torch.cat([padding, volatility], dim=1)
        
        # Extract scalar volatility (mean across features)
        scalar_vol = volatility.mean(dim=2, keepdim=True)
        
        # Normalize and add a small constant for stability
        scalar_vol = scalar_vol / (torch.mean(scalar_vol, dim=1, keepdim=True) + 1e-6)
        
        return scalar_vol
    
    def stabilize_gates(self, i_tilde, f_tilde, m_prev):
        batch_size, seq_len = i_tilde.shape[0], i_tilde.shape[1]
        
        # Ensure m_prev has the right shape for broadcasting
        if m_prev.dim() == 1:
            m_prev = m_prev.unsqueeze(1)  # [batch, 1]
        
        # Reshape m_prev for proper broadcasting
        m_prev_exp = m_prev.unsqueeze(1).expand(batch_size, seq_len, 1).clamp(-100, 100)
        
        # Compute max term for LogSumExp stability
        m_t = torch.maximum(f_tilde, m_prev_exp + f_tilde)
        
        # Stable computation with clipping for financial data extremes
        i = torch.exp(torch.clamp(i_tilde - m_t, -15.0, 15.0))
        f = torch.exp(torch.clamp(f_tilde + m_prev_exp - m_t, -15.0, 15.0))
        
        return i, f, m_t
    def store_key_value(self, key, value, i):
        """
        Compute key-value storage via outer product with memory-efficient implementation.
        """
        batch_size, seq_len, d = key.shape
        
        # Scale keys and values to prevent extremely large outer products
        key_norm = F.normalize(key, dim=2) * math.sqrt(self.d)
        value_norm = value / (torch.norm(value, dim=2, keepdim=True) + 1e-6) * math.sqrt(self.d)
        
        # Ensure same sequence length for all inputs
        min_seq = min(key_norm.shape[1], value_norm.shape[1], i.shape[1])
        key_norm = key_norm[:, :min_seq]
        value_norm = value_norm[:, :min_seq]
        i = i[:, :min_seq]
        
        # Memory-efficient implementation: process in chunks
        chunk_size = min(32, min_seq)  # Process 32 sequence steps at a time
        num_chunks = (min_seq + chunk_size - 1) // chunk_size
        
        result_chunks = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, min_seq)
            
            # Process this chunk
            chunk_key = key_norm[:, start_idx:end_idx]
            chunk_value = value_norm[:, start_idx:end_idx]
            chunk_i = i[:, start_idx:end_idx]
            
            # Compute outer product for this chunk only
            outer_product = torch.bmm(
                chunk_value.reshape(batch_size * (end_idx - start_idx), d, 1),
                chunk_key.reshape(batch_size * (end_idx - start_idx), 1, d)
            ).reshape(batch_size, end_idx - start_idx, d, d)
            
            # Apply input gate with gradient clipping for stability
            i_clamped = torch.clamp(chunk_i, 0.0, 1.0)
            result_chunks.append(i_clamped.unsqueeze(-1) * outer_product)
            
        # Concatenate chunks along sequence dimension
        return torch.cat(result_chunks, dim=1)

    def parallel_gate_computation(self, x_norm):
        """
        Compute all gates in parallel with market regime detection.
        """
        batch_size, seq_len, _ = x_norm.shape
        
        # Detect market regime
        regime_weights = self.regime_detector(x_norm)  # [batch, seq_len, num_regimes]
        
        # Standard gate computation
        q = self.W_q(x_norm)
        k = self.W_k(x_norm) / math.sqrt(self.d)  # Scale keys
        v = self.W_v(x_norm)
        i_tilde = self.W_i(x_norm)
        f_tilde = self.W_f(x_norm)
        
        # Regime-weighted output gate for adaptive behavior in different market conditions
        o_base = self.W_o(x_norm)
        
        # Adapt output gate based on regime - different activations for different regimes
        # This allows model to behave differently in bull/bear/volatile markets
        o = torch.sigmoid(o_base)
        
        # Shape handling
        i_tilde = i_tilde.view(batch_size, seq_len, 1)
        f_tilde = f_tilde.view(batch_size, seq_len, 1)
        
        return q, k, v, i_tilde, f_tilde, o, regime_weights

    def apply_structured_memory_mixing(self, C_t, regime_weights, volatility_scale):
        """
        Apply structured state-space memory mixing with regime awareness,
        using a memory-efficient approach.
        """
        batch_size, seq_len, d, _ = C_t.shape
        
        # Process the memory matrix in chunks to save memory
        chunk_size = 32  # Process 32 rows at a time
        num_chunks = (d + chunk_size - 1) // chunk_size
        
        C_mixed = torch.zeros_like(C_t)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, d)
            
            # Process this chunk of rows
            C_chunk = C_t[:, :, start_idx:end_idx, :]
            
            # Apply SSM processing to each row in the chunk
            for i in range(start_idx, end_idx):
                idx = i - start_idx
                C_row = C_t[:, :, i, :]
                C_row_mixed = self.ssm(C_row, volatility_scale)
                C_mixed[:, :, i, :] = C_row_mixed
        
        # Final memory is adaptively mixed using regime weights
        mixing_factor = regime_weights[:, :, 0:1].unsqueeze(-1)
        
        return C_mixed * mixing_factor + C_t * (1 - mixing_factor)

    def parallel_memory_update(self, v, k, i, f, C_prev, seq_len, regime_weights, volatility_scale):
        """
        Update memory with structured state-space mixing and regime awareness.
        Memory-efficient implementation with chunking.
        """
        batch_size = v.shape[0]
        d = self.d  # Get matrix dimension
        
        # Handle extra dimensions for f once
        if f.dim() > 3:
            f = f.squeeze(-1)
        
        # Ensure i, k, f, v all have the same sequence length
        min_seq_len = min(v.shape[1], k.shape[1], i.shape[1], f.shape[1])
        v = v[:, :min_seq_len]
        k = k[:, :min_seq_len]
        i = i[:, :min_seq_len]
        f = f[:, :min_seq_len]
        regime_weights = regime_weights[:, :min_seq_len]
        volatility_scale = volatility_scale[:, :min_seq_len]
        
        # Update actual sequence length to the trimmed length
        seq_len = min_seq_len
        
        # Calculate cumulative forget product
        f_cum = torch.cumprod(f, dim=1).unsqueeze(-1).unsqueeze(-1)  # [batch, seq, 1, 1]
        
        # Check shape of f_cum, should be [batch, seq_len, 1, 1]
        # If there's an extra dimension, squeeze it
        if f_cum.dim() > 4:
            f_cum = f_cum.squeeze(-1)  # Remove the extra dimension
        
        # Compute key-value outer products
        update_matrices = self.store_key_value(k, v, i)  # [batch, seq, d, d]
        
        # Ensure update_matrices has correct sequence length
        if update_matrices.shape[1] != min_seq_len:
            update_matrices = update_matrices[:, :min_seq_len]
        
        # Compute cumulative updates efficiently
        C_updates = torch.cumsum(update_matrices, dim=1)  # [batch, seq, d, d]
        
        # Expand C_prev for broadcasting - CRITICAL FIX: only expand to min_seq_len
        C_prev_expanded = C_prev.unsqueeze(1).expand(-1, min_seq_len, -1, -1)
        
        # Debug shapes
        # print(f"C_prev_expanded shape: {C_prev_expanded.shape}, f_cum shape: {f_cum.shape}")
        
        # Before expanding, ensure f_cum has compatible dimensions with C_prev_expanded
        f_cum_expanded = f_cum.expand(batch_size, min_seq_len, 1, 1)  # Explicitly define expansion
        
        # Apply forget gates to previous memory and add updates
        C_t = f_cum_expanded * C_prev_expanded + C_updates
        
        # Apply structured mixing and normalization
        C_t = self.apply_structured_memory_mixing(C_t, regime_weights, volatility_scale)
        
        # Final normalization for stability
        C_t = C_t / (torch.norm(C_t, dim=(-2, -1), keepdim=True) + 1e-6) * math.sqrt(self.d)
        
        return C_t

    def parallel_forward(self, x, state):
        """
        Forward pass with parallel sequence processing and financial enhancements.
        """
        batch_size, seq_len, _ = x.size()
        C_prev, n_prev, m_prev = state
        
        # Extract volatility signal for adaptive processing
        volatility_scale = self.detect_volatility(x)
        
        # Input normalization
        x_norm = self.layer_norm_input(x)
        
        # Compute all gates and market regime in parallel
        q, k, v, i_tilde, f_tilde, o, regime_weights = self.parallel_gate_computation(x_norm)
        
        # Stabilized gate computation
        i, f, m_t = self.stabilize_gates(i_tilde, f_tilde, m_prev)
        
        # Shape enforcement
        i = i.view(batch_size, seq_len, 1)
        f = f.view(batch_size, seq_len, 1)
        
        # Memory update with structured state-space mixing
        C_t = self.parallel_memory_update(v, k, i, f, C_prev, seq_len, regime_weights, volatility_scale)
        
        # Get actual sequence length from C_t for consistency
        actual_seq_len = C_t.shape[1]
        
        # Key tracking update (for normalizing query matching)
        n_prev_expanded = n_prev.unsqueeze(1).expand(-1, actual_seq_len, -1)
        
        # Ensure all tensors have the same sequence length as C_t
        q_actual = q[:, :actual_seq_len]
        k_actual = k[:, :actual_seq_len]
        i_actual = i[:, :actual_seq_len]
        f_actual = f[:, :actual_seq_len]
        o_actual = o[:, :actual_seq_len]
        
        i_k_product = i_actual * k_actual
        f_cum = torch.cumprod(f_actual, dim=1)
        n_t = torch.cumsum(i_k_product, dim=1) + f_cum * n_prev_expanded
        
        # Query matching with numerical stability
        h_tilde = torch.einsum('bsde,bse->bsd', C_t, q_actual)
        q_n_dot = torch.sum(n_t * q_actual, dim=-1)
        
        # Stable denominator with market-specific threshold
        denominator = torch.maximum(
            torch.abs(q_n_dot),
            torch.tensor(self.stabilizer_threshold, device=x.device)
        ).unsqueeze(-1)
        
        # Normalized query matching results
        h_tilde = h_tilde / denominator
        
        # Output computation with regime-aware gating
        h_t = o_actual * h_tilde
        
        # Final states for next step
        final_state = (C_t[:, -1], n_t[:, -1], m_t[:, -1])
        
        return h_t, final_state


class ParallelSMLSTM(nn.Module):
    """
    Multi-layer Parallel S-mLSTM optimized for financial time series.
    """
    def __init__(self, input_size, hidden_size, matrix_size=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.matrix_size = matrix_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Stack of S-mLSTM cells
        self.cells = nn.ModuleList([
            ParallelSMLSTMCell(
                input_size if i == 0 else hidden_size, 
                hidden_size,
                matrix_size=matrix_size
            )
            for i in range(num_layers)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Financial-specific output adaptation
        self.trend_detector = nn.Linear(hidden_size, 3)  # Bullish, bearish, sideways
        
    def init_state(self, batch_size, device):
        """Initialize hidden states with financial-specific priors."""
        states = []
        for cell in self.cells:
            # Initialize memory matrix with small random values (financial time series benefit from some memory bias)
            C = torch.randn(batch_size, cell.d, cell.d, device=device) * 0.01
            # No initial keys tracked
            n = torch.zeros(batch_size, cell.d, device=device)
            # Initial log-scale at zero
            m = torch.zeros(batch_size, 1, device=device)
            states.append((C, n, m))
        return states
    
    def forward(self, x):
        """Forward pass with checkpointing for memory efficiency."""
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # Initialize states
        states = self.init_state(batch_size, device)
        
        # Process through layers with residual connections
        layer_input = x
        outputs = []
        
        for layer_idx, cell in enumerate(self.cells):
            # Use checkpointing for memory efficiency with long sequences
            if seq_len > 100 and self.training:
                # Custom checkpointing function to handle state
                def run_cell(inp, state):
                    return cell.parallel_forward(inp, state)
                
                layer_output, states[layer_idx] = checkpoint(
                    run_cell, layer_input, states[layer_idx]
                )
            else:
                layer_output, states[layer_idx] = cell.parallel_forward(
                    layer_input, states[layer_idx]
                )
            
            outputs.append(layer_output)
            
            # Residual connection with dimension matching
            if layer_idx > 0 and layer_input.shape[-1] == layer_output.shape[-1]:
                layer_output = layer_output + layer_input
                
            # Apply dropout between layers
            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
                
            layer_input = layer_output
        
        # Detect overall market trend from final layer
        trend_logits = self.trend_detector(layer_output.mean(dim=1))
        
        return layer_output, trend_logits


class ParallelMLSTMBlock(nn.Module):
    def __init__(self, input_dim, expanded_dim=None, num_layers=2, dropout=0.2, 
                 expansion_factor=4, lstm_class=ParallelSMLSTM, matrix_size=4):
        super().__init__()
        expanded_dim = expanded_dim or input_dim * expansion_factor
        self.layer_norm = nn.LayerNorm(input_dim)
        self.up_proj = nn.Sequential(
            nn.Linear(input_dim, expanded_dim),
            nn.GELU()
        )
        self.mlstm = lstm_class(
            expanded_dim, 
            expanded_dim // 2,
            matrix_size=matrix_size,
            num_layers=num_layers, 
            dropout=dropout
        )
        self.down_proj = nn.Linear(expanded_dim // 2, input_dim)  # Corrected
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward pass with financial optimization."""
        # Apply normalization for numerical stability
        x_norm = self.layer_norm(x)
        
        # Dimension expansion
        x_up = self.up_proj(x_norm)
        
        # Apply structured mLSTM
        x_mlstm, trend_logits = self.mlstm(x_up)
        
        # Project back to input dimension
        x_down = self.down_proj(x_mlstm)
        
        # Ensure dimensions match for residual connection
        if x.shape != x_down.shape:
            x_down = x_down[:, :x.shape[1], :x.shape[2]]
        
        # Scaled residual connection
        skip_strength = torch.sigmoid(self.skip_scale)
        return x + skip_strength * x_down, trend_logits


class ParallelExtendedSMLSTM(nn.Module):
    """
    Complete financial time series model with structured memory mixing.
    Optimized for stock price prediction with multi-scale modeling.
    """
    def __init__(self, input_size, hidden_size=16, matrix_size=4, num_layers=2, 
                 dropout=0.2, expansion_factor=4):
        super().__init__()
        self.pre_fc_dim = hidden_size * matrix_size * matrix_size
        
        # Financial feature extraction
        self.input_projection = nn.Linear(input_size, self.pre_fc_dim)
        
        # Adaptive normalization (better for varying financial regimes)
        self.layer_norm = nn.LayerNorm(self.pre_fc_dim)
        
        # Multi-scale processing block with financial optimizations
        self.mlstm_block = ParallelMLSTMBlock(
            input_dim=self.pre_fc_dim,
            expanded_dim=self.pre_fc_dim * expansion_factor,
            num_layers=num_layers,
            dropout=dropout,
            expansion_factor=expansion_factor,
            lstm_class=ParallelSMLSTM,
            matrix_size=matrix_size
        )
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final projection with financial-specific initialization
        self.output_projection = nn.Linear(self.pre_fc_dim, self.pre_fc_dim)
        # Initialize with small weights (financial predictions should be conservative)
        self.output_projection.weight.data *= 0.1
        
        # Market regime detector for final output conditioning
        self.regime_detector = nn.Linear(self.pre_fc_dim, 4)
        
        # Trading signal generators (additional outputs beyond price prediction)
        self.signal_generator = nn.Linear(self.pre_fc_dim, 3)  # Buy, hold, sell
    
    def forward(self, x):
        """
        Forward pass with financial-specific processing.
        
        Args:
            x: Input tensor [batch, seq, features]
            
        Returns:
            outputs: Price predictions
            regime: Market regime probabilities
            signals: Trading signals
        """
        # Project input features
        x = self.input_projection(x)
        
        # Normalize (financial data can have extreme values)
        x = self.layer_norm(x)
        
        # Apply structured memory LSTM block
        x, trend_logits = self.mlstm_block(x)
        
        # Nonlinear activation
        x = F.gelu(x)
        
        # Regularization
        x = self.dropout(x)
        
        # Final projection
        outputs = self.output_projection(x)
        
        # Detect market regime
        regime = F.softmax(self.regime_detector(outputs.mean(dim=1)), dim=-1)
        
        # Generate trading signals
        signals = F.softmax(self.signal_generator(outputs.mean(dim=1)), dim=-1)
        
        return outputs, regime, signals