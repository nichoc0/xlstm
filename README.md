# xLSTM Implementation

## Overview
This repository implements an **xLSTM-based architecture** using **matrix LSTMs (mLSTM)** for efficient long-term memory storage and retrieval. The model is designed to enhance traditional LSTMs by incorporating **matrix memory updates** and **parallel computation** for improved scalability and efficiency.

## Theoretical Foundation
The implementation is based on the xLSTM and mLSTM architectures described in various research papers. The core idea revolves around **storing information in a d×d matrix (C_t)** rather than a scalar, enabling better **associative memory retrieval**.

## Core Equations & Implementation

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

