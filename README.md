# Matrix LSTM (mLSTM) for Financial Time Series

## Project Overview
This repository implements advanced Matrix LSTM variants for financial time series prediction:
- **S-mLSTM**: Structured Memory LSTM with state-space models for multi-scale financial modeling
- **xLSTM**: Regular matrix-based LSTM implementation for comparison

The implementation is designed for high-performance processing of financial data with special attention to market regimes, volatility dynamics, and long-term dependencies in price movements.

## Mathematical Foundation

### Core Innovation: Matrix Memory Cells
Traditional LSTMs store information in vector-based memory cells, limiting their ability to capture complex interdependencies. Matrix LSTMs replace this with matrix-valued memory cells, enabling more sophisticated pattern recognition.

The primary mathematical innovation is:

$$\mathbf{C}_t \in \mathbb{R}^{d \times d}$$

Instead of:

$$c_t \in \mathbb{R}^d$$

Where:
- $\mathbf{C}_t$ is the matrix memory cell at time $t$
- $d$ is the hidden dimension

### Memory Update Equations

The key equations of matrix memory update are:

**1. Input & Forget Gates:**
$$\mathbf{i}_t = \exp(\tilde{\mathbf{i}}_t - \mathbf{m}_t)$$
$$\mathbf{f}_t = \exp(\tilde{\mathbf{f}}_t + \mathbf{m}_{t-1} - \mathbf{m}_t)$$

Where $\mathbf{m}_t = \max(\tilde{\mathbf{f}}_t + \mathbf{m}_{t-1}, \tilde{\mathbf{i}}_t)$ for numerical stability.

**2. Matrix Memory Update:**
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot (\mathbf{v}_t \mathbf{k}_t^T)$$

Where:
- $\mathbf{v}_t \in \mathbb{R}^d$ is the value vector to be stored
- $\mathbf{k}_t \in \mathbb{R}^d$ is the key vector for addressing
- $\mathbf{v}_t \mathbf{k}_t^T \in \mathbb{R}^{d \times d}$ is the outer product creating associative memory

**3. Key Tracking State:**
$$\mathbf{n}_t = \mathbf{f}_t \odot \mathbf{n}_{t-1} + \mathbf{i}_t \odot \mathbf{k}_t$$

**4. Query-Based Memory Retrieval:**
$$\mathbf{h}_t = \mathbf{o}_t \odot \frac{\mathbf{C}_t \mathbf{q}_t}{\max(|\mathbf{n}_t^T \mathbf{q}_t|, \lambda)}$$

Where:
- $\mathbf{q}_t \in \mathbb{R}^d$ is the query vector
- $\mathbf{o}_t \in \mathbb{R}^d$ is the output gate
- $\lambda$ is a stability threshold (typically 1.0)

### Parallel Processing Implementation

We implement parallelized computation across the sequence dimension using cumulative operations:

**1. Cumulative Gates:**
$$\mathbf{F}_{t,j} = \prod_{i=1}^j \mathbf{f}_{t,i}$$

**2. Cumulative Memory Updates:**
$$\Delta\mathbf{C}_{t,j} = \sum_{i=1}^j \mathbf{i}_{t,i} \odot (\mathbf{v}_{t,i} \mathbf{k}_{t,i}^T)$$

**3. Parallel Memory Computation:**
$$\mathbf{C}_{t,j} = \mathbf{F}_{t,j} \odot \mathbf{C}_{t-1} + \Delta\mathbf{C}_{t,j}$$

This enables processing the entire sequence at once rather than step-by-step.

### S-mLSTM: Structured State-Space Enhancement

The Structured Memory LSTM adds a state-space model layer that processes the memory matrices:

**1. Continuous Dynamics:**
$$\frac{ds(t)}{dt} = \mathbf{A}s(t) + \mathbf{B}u(t)$$
$$y(t) = \mathbf{C}s(t) + \mathbf{D}u(t)$$

**2. Discretized Implementation:**
For the bilinear discretization method:
$$\mathbf{A}_d = \frac{2 + \Delta t \mathbf{A}}{2 - \Delta t \mathbf{A}}$$
$$\mathbf{B}_d = \frac{\Delta t (I + \mathbf{A}_d)}{2}\mathbf{B}$$

**3. Structured Memory Mixing:**
$$\mathbf{C}_t^{\text{mixed}} = \text{SSM}(\mathbf{C}_t) \cdot \alpha + \mathbf{C}_t \cdot (1 - \alpha)$$

Where $\alpha$ is determined by market regime detection.

## Implementation Details

### Core Components

1. **StructuredStateSpace**: 
   - Implements discretized state-space models for multi-scale temporal dynamics
   - Uses bilinear discretization for stability
   - Incorporates volatility awareness

2. **ParallelSMLSTMCell**:
   - Implements matrix memory updates with parallelized sequence processing
   - Uses exponential gating for improved gradient flow
   - Integrates state-space memory mixing

3. **ParallelExtendedSMLSTM**:
   - Full financial prediction model with regime detection
   - Multi-scale memory processing
   - Generates price predictions, regime probabilities, and trading signals

### Financial Adaptations

1. **Market Regime Detection**:
   - Classifies market states into 4 regimes: Bull, Bear, Sideways, Volatile
   - Adapts memory dynamics based on detected regime

2. **Volatility-Aware Processing**:
   - Scales memory update intensity based on detected market volatility
   - Provides more stable predictions during high-volatility periods

3. **Multi-Scale Modeling**:
   - Captures patterns from hourly to quarterly timeframes
   - Structured state-space models with varying time constants

## Training Setup

The model is trained with:
- **Huber Loss**: For robustness against financial outliers
- **AdamW Optimizer**: With L1 regularization for weight sparsity
- **OneCycleLR Scheduler**: For adaptive learning rate control
- **Checkpointing**: For memory-efficient training with long sequences

## Progress Log

### Development Milestones:

1. **Initial Implementation**:
   - Created core matrix LSTM architecture
   - Implemented numerical stability optimizations

2. **Debugging Phase**:
   - Fixed dimension mismatches in parallel processing
   - Resolved shape errors in matrix operations
   - Improved memory update operations

3. **Performance Optimization**:
   - Enhanced structured state-space model
   - Implemented efficient memory mixing for multi-scale analysis

4. **Current Status**:
   - Model can successfully train on financial data
   - Architecture correctly detects different market regimes
   - First training run completed with minimal parameters

## Usage

### What You Need
- Python 3.8+, PyTorch 1.10+,
- Extras: `numpy`, `pandas`, `yfinance`, `scikit-learn`, `ta`, `torch_optimizer`.

### Data Setup
Defaults to MSFT stock via `yfinance`. For your own:
1. CSV with `['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']`,
2. Edit `get_stock_data` and `get_daily_data` in `paralstm.py`.

### Fire It Up
Run:
```bash
python paralstm.py
```
It grabs data, adds indicators (RSI, MACD), and trains. Tweak `SEQ_LENGTH`, `BATCH_SIZE`, or `EPOCHS` if you're feeling experimental.