# Matrix LSTM (xLSTM & S-mLSTM) Implementation

## Overview
This repository implements advanced **Matrix LSTM variants** including:
1. **xLSTM** - Extended LSTM with matrix-based memory
2. **S-mLSTM** - Structured Memory LSTM optimized for financial time series

These models enhance traditional LSTMs by storing information in **matrices rather than vectors**, enabling more powerful pattern recognition, associative memory, and long-term dependencies tracking.

## Mathematical Foundations

### Matrix Memory Mechanism

The key innovation in xLSTM/S-mLSTM is replacing the scalar cell state with a **matrix memory cell** $\mathbf{C}_t \in \mathbb{R}^{d \times d}$:

$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot (\mathbf{v}_t \mathbf{k}_t^T)$$

Where:
- $\mathbf{C}_t$ is the memory matrix at time $t$
- $\mathbf{f}_t$ is the forget gate (controls memory persistence)
- $\mathbf{i}_t$ is the input gate (controls new information flow)
- $\mathbf{v}_t$ is the value vector (information to store)
- $\mathbf{k}_t$ is the key vector (addressing/indexing information)
- $\mathbf{v}_t \mathbf{k}_t^T$ creates an **outer product** matrix representing associative memory

### Key Normalization
To ensure stable memory retrieval, a normalizer state $\mathbf{n}_t$ tracks key strengths:

$$\mathbf{n}_t = \mathbf{f}_t \odot \mathbf{n}_{t-1} + \mathbf{i}_t \odot \mathbf{k}_t$$

### Information Retrieval
Information is retrieved using query vectors $\mathbf{q}_t$ via matrix-vector product and normalization:

$$\mathbf{h}_t = \mathbf{o}_t \odot \frac{\mathbf{C}_t \mathbf{q}_t}{\max(|\mathbf{n}_t^T \mathbf{q}_t|, \lambda)}$$

Where:
- $\mathbf{o}_t$ is the output gate
- $\lambda$ is a stabilization threshold (typically 1.0)

### Exponential Gating
For improved gradient flow and stability, we use exponential parametrization of gates:

$$\mathbf{i}_t = \exp(\tilde{\mathbf{i}}_t - \mathbf{m}_t)$$
$$\mathbf{f}_t = \exp(\tilde{\mathbf{f}}_t + \mathbf{m}_{t-1} - \mathbf{m}_t)$$

Where $\mathbf{m}_t = \max(\tilde{\mathbf{f}}_t + \mathbf{m}_{t-1}, \tilde{\mathbf{i}}_t)$ ensures numerical stability.

## Model Variants

### xLSTM (Extended LSTM)
The base matrix-memory LSTM implementation with:
- Matrix cell state for storing associative memory
- Key-value storage mechanism
- Query-based information retrieval
- Parallel sequence processing

### S-mLSTM (Structured Memory LSTM)
Enhances xLSTM with:
- **State-Space Model (SSM)** integration for multi-scale temporal dynamics
- **Market regime detection** for financial time series
- **Volatility-aware** memory scaling
- **Multi-timescale modeling** optimized for financial patterns

## Financial Applications
The S-mLSTM is specifically designed for financial time series analysis with:

1. **Market Regime Detection**:
   - Automatically identifies bull/bear/sideways/volatile market conditions
   - Adapts memory dynamics based on detected regimes

2. **Volatility Awareness**:
   - Scales memory updates inversely with market volatility
   - More cautious predictions during high volatility periods

3. **Multi-scale Processing**:
   - Captures patterns from hourly to quarterly timeframes
   - Integrates different market cycle dynamics

## Parallel Processing Architecture
The implementation features:
- **Fully parallelized sequence processing** for training efficiency
- **Cumulative computations** for parallel gate and memory updates
- **Checkpointed gradient computation** for memory efficiency with long sequences

## Implementation Details

### Core Components:

### 1. Matrix Memory Update
**Equation:**
\[
C_t = f_t C_{t-1} + i_t v_t k_t^T
\]

**Explanation:**
- The memory cell \(C_t\) is updated using:
  - A **forget gate** \(f_t\) to **control decay** of past memory.
  - An **input gate** \(i_t\) to **regulate new memory updates**.
  - The **key-value pair** \((v_t, k_t)\) is stored via an **outer product**.

**Code (Parallel Memory Update):**
```python
update = torch.einsum('bsd,bse->bsde', value, key)  # Outer product v_t k_t^T
C_updates = torch.cumsum(update, dim=1)             # Cumulative memory updates
C_t = f_cum_expanded * C_prev_expanded + C_updates  # Weighted sum
```

---

### 2. Normalizer State Update
**Equation:**
\[
n_t = f_t n_{t-1} + i_t k_t
\]

**Explanation:**
- This tracks how strong each key vector \(k_t\) is over time.
- Helps stabilize **query-based memory retrieval**.

**Code:**
```python
n_t = torch.cumsum(i * k, dim=1) + f_cum * n_prev_expanded
```

---

### 3. Memory Retrieval (Hidden State Computation)
**Equation:**
\[
h_t = o_t \cdot \frac{C_t q_t}{\max(|n_t^T q_t|, 1)}
\]

**Explanation:**
- A **query vector \(q_t\)** retrieves stored information from memory.
- The denominator **normalizes the retrieval process** to prevent exploding values.
- The **output gate \(o_t\)** controls the final output.

**Code:**
```python
h_tilde = torch.einsum('bsde,bse->bsd', C_t, q)  # Memory retrieval
q_n_dot = torch.sum(n_t * q, dim=-1)  # Normalization factor
h_tilde = h_tilde / torch.maximum(torch.abs(q_n_dot), torch.tensor(1.0, device=x.device))
h_t = o * h_tilde  # Apply output gate
```

---

### 4. Exponential Gating Mechanism
**Equation:**
\[
i_t = \exp(i_t')\quad ,\quad f_t = \exp(f_t')
\]

**Explanation:**
- Instead of sigmoid activation, **exponential gates** are used.
- Ensures stronger gating behavior.

**Code:**
```python
i = torch.exp(i_tilde - m_t)
f = torch.exp(f_tilde + m_prev - m_t)
```

---

## Model Architecture
The core model consists of:
1. **ParallelxLSTMCell** – Implements matrix memory updates and retrieval.
2. **ParallelxLSTM** – Stacks multiple `ParallelxLSTMCell` layers.
3. **ParallelMLSTMBlock** – Implements residual xLSTM blocks.
4. **ParallelExtendedMLSTM** – Encapsulates the full model with input projection and normalization.
5. **FunnyMachine** – The final model combining xLSTM with fully connected output layers.

---

## Training Setup
The model is trained on **hourly & daily stock market data** with:
- **Huber Loss** (for stability against outliers).
- **AdamW Optimizer** (with L1 regularization for weight sparsity).
- **OneCycleLR Scheduler** (adaptive learning rate control).
- **Gradient Clipping** to prevent exploding gradients.

**Training Loop (Simplified):**
```python
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch) + l1_lambda * sum(p.abs().sum() for p in model.parameters())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
```

---

For more details pls ask me!

