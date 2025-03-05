# Matrix LSTM: Cracking Financial Time Series with Math

## Intro: What's This All About?
Matrix LSTM is a neural network trick for predicting financial time series—like stock prices—without losing your mind over the market's chaos. It's got roots in the CS Games competition (March 2024, Québec, loud techno, good times), where I tackled some wild coding challenges and thought, "Hey, let's math up those financial patterns!" Unlike boring old LSTMs with their single-number memory, this one uses **matrix memory cells** to juggle all the messy correlations in market data. Let's dive into the equations and see what makes it tick.

## The Math: Where the Magic Happens

### Old-School LSTMs: Meh
Regular LSTMs are like that friend who can only remember one thing at a time. They use a **scalar cell state** to track sequences, which is fine for simple stuff but flops with financial data's multi-dimensional madness (prices, volumes, volatility—all tangled up). We need more firepower.

### Matrix Memory: The Big Upgrade
Matrix LSTM swaps that scalar for a **matrix memory cell**, C_t ∈ ℝ^(d×d), turning it into a powerhouse for storing connections. Here's how it updates:

```
C_t = f_t ⊙ C_(t-1) + i_t ⊙ (v_t k_t^T)
```

Breaking it down:
- f_t: **Forget gate**—what to toss from last time,
- i_t: **Input gate**—what new stuff to add,
- v_t: **Value vector**—the info we're keeping,
- k_t: **Key vector**—where it goes in the matrix,
- v_t k_t^T: **Outer product**—a matrix that ties features together.

This isn't just memory—it's a web of associations, perfect for spotting patterns like a math wizard.

### Keeping It Stable: Normalizer State
To avoid blowing up with crazy market swings, we've got a **normalizer state**, n_t:

```
n_t = f_t ⊙ n_(t-1) + i_t ⊙ k_t
```

It tracks the keys over time, adjusted by the gates, so we don't get nonsense when pulling data out. Think of it as the math glue holding everything together.

### Grabbing the Goods: Query Time
The **hidden state**, h_t, is what we actually use, and it's pulled from the matrix with a **query vector**, q_t:

```
h_t = o_t ⊙ (C_t q_t) / max(|n_t^T q_t|, λ)
```

- o_t: **Output gate**—how much to show,
- λ: A tiny safety net (say, 1.0) to avoid dividing by zero.

This is like asking the matrix, "What's the scoop?" and getting a clean, stable answer.

### Exponential Gating: No Gradient Drama
Regular LSTMs use sigmoid gates that can choke the gradients. We go exponential instead:

```
i_t = exp(ĩ_t - m_t),   f_t = exp(f̃_t + m_(t-1) - m_t)
```

with m_t = max(f̃_t + m_(t-1), ĩ_t). This keeps gates positive and gradients flowing—crucial for nailing those long-term market trends.

## The Variants: Two Flavors of Awesome

### xLSTM: The Core Beast
**xLSTM** is the base model, rocking:
- Matrix memory for pattern-hunting,
- Parallel processing to chew through data fast.

It's a solid all-rounder, but we're just getting started.

### S-mLSTM: Financial Superpowers
**S-mLSTM** levels up for market mayhem:
- **Structured State-Space Model (SSM)**: Uses discretized differential equations to catch patterns from minutes to months,
- **Regime Detection**: Spots if the market's chill (bull), grumpy (bear), or wild (volatile),
- **Volatility Scaling**: Tweaks memory updates when the market freaks out.

The SSM bit? It's like:

```
ds(t)/dt = As(t) + Bu(t),   y(t) = Cs(t)
```

Discretized and learned with the LSTM weights—fancy, right? This thing adapts like a champ.

## The Code: How It Runs
Built in Python with PyTorch, here's the lineup:
- **`psmlstm.py`**:
  - `StructuredStateSpace`: SSM math in action,
  - `ParallelSMLSTMCell`: Mixes matrix memory and SSM,
  - `ParallelSMLSTM`: Stacks it all up with residuals.
- **`paralstm.py`**:
  - `FunnyMachine`: Runs the show—inputs, S-mLSTM, outputs (prices, regimes, signals),
  - Training with **Huber Loss**, **AdamW**, and **OneCycleLR**.

## Using It: Quick Start

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