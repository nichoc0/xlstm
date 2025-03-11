# TPU-Accelerated Scalar-Matrix LSTM for Financial Time Series

## Introduction

This repository contains an advanced implementation of a memory-efficient Scalar-Matrix LSTM (SM-LSTM) model specifically designed for financial time series analysis. The model combines elements from modern architectural innovations including:

1. State Space Models (SSMs)
2. Matrix-based memory representations
3. Regime-aware processing
4. Volatility-adaptive computations
5. Trend detection and exploitation

This document provides an in-depth mathematical explanation of the model's components and operations.

## Mathematical Foundations

### 1. Scalar-Matrix LSTM Cell Structure

The core innovation of the SM-LSTM cell is its use of matrix-based memory instead of traditional vector memory. While standard LSTMs use hidden states $h_t \in \mathbb{R}^d$, our SM-LSTM uses $C_t \in \mathbb{R}^{d \times d}$, enabling richer state representation.

#### Core Memory Update Equation

$$C_t = f_t \odot C_{t-1} + i_t \cdot (v_t \otimes k_t)$$

Where:
- $C_t \in \mathbb{R}^{d \times d}$ is the cell state matrix
- $f_t \in \mathbb{R}$ is the forget gate scalar
- $i_t \in \mathbb{R}$ is the input gate scalar
- $v_t \in \mathbb{R}^d$ is the value vector
- $k_t \in \mathbb{R}^d$ is the key vector
- $\otimes$ denotes the outer product $(v_t \otimes k_t) \in \mathbb{R}^{d \times d}$

#### Hidden State Computation

The hidden state is computed as:

$$h_t = o_t \odot \frac{C_t q_t}{max(|q_t^T n_t|, \theta)}$$

Where:
- $o_t \in \mathbb{R}^d$ is the output gate
- $q_t \in \mathbb{R}^d$ is the query vector
- $n_t \in \mathbb{R}^d$ is the normalization vector updated as: $n_t = f_t \odot n_{t-1} + i_t \cdot k_t$
- $\theta$ is a stabilization threshold (typically 1.0)

### 2. Structured State Space Component

The model incorporates a structured state space module defined by:

$$\dot{x}(t) = Ax(t) + Bu(t)$$
$$y(t) = Cx(t) + Du(t)$$

In discrete form with step size $\Delta t$:

#### Bilinear Discretization:

$$A_d = \frac{2 + \Delta t \cdot \lambda}{2 - \Delta t \cdot \lambda}$$
$$B_d = \frac{\Delta t \cdot (1 + A_d)}{2} \cdot b$$

#### Zero-Order Hold (ZOH) Discretization:

$$A_d = e^{\lambda \Delta t}$$
$$B_d = \frac{A_d - 1}{\lambda} \cdot b$$

The SSM implementation uses reduced parameters:
- $\lambda \in \mathbb{R}^{d_{state}}$ (eigenvalues)
- $b \in \mathbb{R}^{d_{state} \times 1}$ (input projection)
- $c \in \mathbb{R}^{1 \times d_{state}}$ (output projection)

### 3. Parallel Memory Update

The efficient parallel implementation for sequence processing uses cumulative products and sums:

$$f_{cum}[t] = \prod_{i=1}^{t} f[i]$$
$$C_{updates}[t] = \sum_{i=1}^{t} i[i] \cdot (v[i] \otimes k[i])$$
$$C_t[t] = f_{cum}[t] \cdot C_{prev} + C_{updates}[t]$$

### 4. Financial-Specific Loss Functions

#### Asymmetric Directional Loss

$$L_{dir}(y_{pred}, y_{true}) = \frac{1}{T-1}\sum_{t=1}^{T-1} \mathbf{1}_{sign(y_{pred,t+1} - y_{pred,t}) \neq sign(y_{true,t+1} - y_{true,t})} \cdot w_t$$

Where $w_t$ is:
- $w_t = w_{up}$ when $y_{true,t+1} - y_{true,t} > 0$
- $w_t = w_{down}$ when $y_{true,t+1} - y_{true,t} < 0$

This penalizes direction prediction errors asymmetrically, with typically $w_{up} > w_{down}$.

## Key Components and Their Mathematical Details

### 1. Gate Stabilization

To prevent numerical instability from exponential operations, we use the following stabilization technique:

$$m_t = \max(f_t, m_{t-1} + f_t)$$
$$i_t = \exp(i_t - m_t)$$
$$f_t = \exp(f_t + m_{t-1} - m_t)$$

### 2. Volatility Detection

The model detects volatility using a rolling standard deviation approach:

$$\sigma_t = \sqrt{\frac{1}{W} \sum_{i=t-W}^{t} (x_i - \bar{x}_{t-W:t})^2}$$

Where $W$ is the window size and $\bar{x}_{t-W:t}$ is the mean over the window. The normalized volatility is:

$$\hat{\sigma}_t = \frac{\sigma_t}{\bar{\sigma} + \epsilon}$$

### 3. Trend Strength Calculation

The model calculates trend strength through autocorrelation:

$$r = \frac{\sum_{t=1}^{T-1} (x_t - \bar{x})(x_{t+1} - \bar{x})}{\sqrt{\sum_{t=1}^{T} (x_t - \bar{x})^2 \sum_{t=1}^{T} (x_{t+1} - \bar{x})^2}}$$

$$\text{trend strength} = \frac{|r| + 1}{2}$$

For greater efficiency, the implementation also uses fast and slow exponential moving averages (EMAs):

$$\text{fast EMA}_t = \alpha_{\text{fast}} \cdot x_t + (1-\alpha_{\text{fast}}) \cdot \text{fast EMA}_{t-1}$$
$$\text{slow EMA}_t = \alpha_{\text{slow}} \cdot x_t + (1-\alpha_{\text{slow}}) \cdot \text{slow EMA}_{t-1}$$

Agreement between EMAs produces a trend signal:
$$\text{agreement}_t = \text{sign}(\Delta \text{fast EMA}_t) \cdot \text{sign}(\Delta \text{slow EMA}_t)$$

### 4. Memory Mixing

The model employs a sophisticated memory mixing approach:

1. **Regime-based Processing**: The memory matrix $C_t$ is processed differently based on detected regime probabilities:
   
   $$C_t^{\text{processed}} = \sum_{r=1}^{4} w_r \cdot \text{Processor}_r(C_t)$$
   
   Where $w_r$ are regime weights and $\text{Processor}_r$ are regime-specific neural networks.

2. **Structured Memory Mixing**:
   
   $$C_t^{\text{mixed}} = \text{SSM}(C_t, \sigma_t)$$
   
   The SSM applies row-wise state space dynamics to the memory matrix, conditioned by volatility.

3. **Trend-based Memory Persistence**:
   
   $$C_t^{\text{final}} = (1 - \text{trend strength}) \cdot C_t + \text{trend strength} \cdot C_{t-1}$$
   
   This allows the model to maintain persistent memory during strong trends.

### 5. Ensemble Model Architecture

The ensemble model combines multiple SM-LSTM models using different configurations:

$$y_{\text{ensemble}} = \text{MetaNetwork}([y_1, y_2, ..., y_n])$$

Where each $y_i$ is the output of an individual model operating on a different time scale or with different hyperparameters.

## Implementation Optimizations

### Memory Efficiency Techniques

1. **Chunked Processing**: Operations are performed in small chunks to reduce peak memory usage:

   ```python
   chunk_size = 8  # Reduced from 32 for memory efficiency
   num_chunks = (min_seq + chunk_size - 1) // chunk_size
   
   for chunk_idx in range(num_chunks):
       # Process chunk
       # Free memory after chunk processing
   ```

2. **Gradient Checkpointing**: For long sequences, we use PyTorch's gradient checkpointing:

   ```python
   if seq_len > 100 and self.training:
       def run_cell(inp, st):
           return cell.parallel_forward(inp, st)
       layer_output, states[layer_idx] = checkpoint(run_cell, layer_input, states[layer_idx])
   ```

3. **Dimension-wise Processing**: For large matrices, row-by-row processing is used:

   ```python
   for row in range(d):
       row_data = C_t[:, :, row, :]  # [batch, seq, d]
       # Process row
   ```

## Understanding Key Functions

### 1. ParallelSMLSTMCell.parallel_forward

This is the main entry point for processing a sequence through the cell. It:
1. Normalizes the input
2. Computes gates (i, f, o)
3. Transforms the input into queries, keys, and values
4. Updates the memory matrix
5. Computes the output

### 2. ParallelSMLSTMCell.parallel_memory_update

This function efficiently updates the matrix memory in parallel across the sequence dimension using:
1. Cumulative products of forget gates
2. Cumulative sums of memory updates
3. Regime-based memory processing
4. Structured state space mixing
5. Trend-based memory mixing

### 3. ParallelSMLSTMCell.stabilize_gates

This function implements the numerical stabilization technique described above, ensuring that the exponential operations in the gate computations don't cause overflow or underflow.

### 4. StructuredStateSpace.forward

This implements the structured state space model, which processes temporal data using:
1. Discretized state space dynamics
2. Volatility-conditioned state transitions
3. Trend-aware update steps

## Conclusion

The TPU-accelerated Scalar-Matrix LSTM represents a significant advancement in memory-based models for financial time series. By combining matrix memory, structured state spaces, and financial domain-specific optimizations, it achieves both expressiveness and efficiency. The implementation is carefully optimized for memory usage and computational efficiency, making it suitable for deployment in resource-constrained environments.
