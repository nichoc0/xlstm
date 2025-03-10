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

```math
\mathbf{C}_t \in \mathbb{R}^{d \times d}
```

Instead of:

```math
c_t \in \mathbb{R}^d
```

Where:
- \( \mathbf{C}_t \) is the matrix memory cell at time \( t \)
- \( d \) is the hidden dimension

### Memory Update Equations

#### 1. Input & Forget Gates:
```math
\mathbf{i}_t = \exp(\tilde{\mathbf{i}}_t - \mathbf{m}_t)
```
```math
\mathbf{f}_t = \exp(\tilde{\mathbf{f}}_t + \mathbf{m}_{t-1} - \mathbf{m}_t)
```

Where:
```math
\mathbf{m}_t = \max(\tilde{\mathbf{f}}_t + \mathbf{m}_{t-1}, \tilde{\mathbf{i}}_t)
```
For numerical stability.

#### 2. Matrix Memory Update:
```math
\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot (\mathbf{v}_t \mathbf{k}_t^T)
```

Where:
- \( \mathbf{v}_t \in \mathbb{R}^d \) is the value vector to be stored
- \( \mathbf{k}_t \in \mathbb{R}^d \) is the key vector for addressing
- \( \mathbf{v}_t \mathbf{k}_t^T \in \mathbb{R}^{d \times d} \) is the outer product creating associative memory

#### 3. Key Tracking State:
```math
\mathbf{n}_t = \mathbf{f}_t \odot \mathbf{n}_{t-1} + \mathbf{i}_t \odot \mathbf{k}_t
```

#### 4. Query-Based Memory Retrieval:
```math
\mathbf{h}_t = \mathbf{o}_t \odot \frac{\mathbf{C}_t \mathbf{q}_t}{\max(|\mathbf{n}_t^T \mathbf{q}_t|, \lambda)}
```

Where:
- \( \mathbf{q}_t \in \mathbb{R}^d \) is the query vector
- \( \mathbf{o}_t \in \mathbb{R}^d \) is the output gate
- \( \lambda \) is a stability threshold (typically 1.0)

### Parallel Processing Implementation

#### 1. Cumulative Gates:
```math
\mathbf{F}_{t,j} = \prod_{i=1}^j \mathbf{f}_{t,i}
```

#### 2. Cumulative Memory Updates:
```math
\Delta\mathbf{C}_{t,j} = \sum_{i=1}^j \mathbf{i}_{t,i} \odot (\mathbf{v}_{t,i} \mathbf{k}_{t,i}^T)
```

#### 3. Parallel Memory Computation:
```math
\mathbf{C}_{t,j} = \mathbf{F}_{t,j} \odot \mathbf{C}_{t-1} + \Delta\mathbf{C}_{t,j}
```

This enables processing the entire sequence at once rather than step-by-step.

### S-mLSTM: Structured State-Space Enhancement

The Structured Memory LSTM adds a state-space model layer that processes the memory matrices:

#### 1. Continuous Dynamics:
```math
\frac{ds(t)}{dt} = \mathbf{A}s(t) + \mathbf{B}u(t)
```
```math
y(t) = \mathbf{C}s(t) + \mathbf{D}u(t)
```

#### 2. Discretized Implementation:
For the bilinear discretization method:
```math
\mathbf{A}_d = \frac{2 + \Delta t \mathbf{A}}{2 - \Delta t \mathbf{A}}
```
```math
\mathbf{B}_d = \frac{\Delta t (I + \mathbf{A}_d)}{2}\mathbf{B}
```

#### 3. Structured Memory Mixing:
```math
\mathbf{C}_t^{\text{mixed}} = \text{SSM}(\mathbf{C}_t) \cdot \alpha + \mathbf{C}_t \cdot (1 - \alpha)
```

Where \( \alpha \) is determined by market regime detection.

## Training Setup

The model is trained with:
- **Huber Loss**: For robustness against financial outliers
- **AdamW Optimizer**: With L1 regularization for weight sparsity
- **OneCycleLR Scheduler**: For adaptive learning rate control
- **Checkpointing**: For memory-efficient training with long sequences

## Usage

### Requirements
- Python 3.8+
- PyTorch 1.10+
- Extras: `numpy`, `pandas`, `yfinance`, `scikit-learn`, `ta`, `torch_optimizer`

### Data Setup
Defaults to MSFT stock via `yfinance`. For custom data:
1. Prepare a CSV with columns: `['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']`
2. Modify `get_stock_data` and `get_daily_data` in `paralstm.py`

### Running the Model
Execute:
```bash
python paralstm.py
```
This script fetches data, applies indicators (RSI, MACD), and trains the model. Adjust `SEQ_LENGTH`, `BATCH_SIZE`, or `EPOCHS` for different experiments.

