# Options Trading Bot

This project currently collects option-chain snapshots for the 50 largest
U.S. companies by market capitalization, calculates Black-Scholes-Merton
Greeks, exposes the saved data through a browser interface, and provides a
deterministic `research_demo` environment for future reinforcement-learning
agents.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Collect data

Run one collection cycle:

```bash
collect-options --once
```

Run continuously every 15 minutes:

```bash
collect-options
```

Each ticker has an append-only file under `data/`, such as `data/AAPL.csv`.
Every row records the collection time, expiration, option type, market fields,
model inputs, and Delta, Gamma, Theta, and Vega. Collection defaults to the
nearest three expirations. Use `--expirations 1` for the lowest-latency mode or
`--expirations 0` to collect every listed expiration:

```bash
collect-options --once --expirations 1
```

## Explore data

```bash
streamlit run src/trading_bot/interface/app.py
```

The interface lets you choose a ticker, inspect its latest call or put
snapshot, and submit fake option orders. Paper buys fill at the saved ask and
paper sells fill at the saved bid.

## Paper portfolio sandbox

The sandbox starts with $100,000 in fake cash and persists to
`data/paper_portfolio.db`. It is long-only: contracts must be bought before they
can be sold. Each option uses the standard 100-share multiplier.

Future agents can use the same tested Python boundary as the interface:

```python
from pathlib import Path
from trading_bot.execution import PaperBroker

broker = PaperBroker(Path("data/paper_portfolio.db"))
account = broker.account()
positions = broker.positions()
trades = broker.trades()
```

`broker.buy(...)` and `broker.sell(...)` require explicit contract metadata,
quantity, and fill price. The paper broker never fetches quotes and has no live
broker adapter, which keeps execution decisions separate from market data.

## Project structure

```text
src/trading_bot/
├── analytics/       # Greeks now; portfolio statistics later
├── execution/       # Paper broker, ledger, and portfolio valuation
├── interface/       # User-facing data explorer
├── market_data/     # Option retrieval, rates, universe, and collection
└── training/        # Features, environment, recurrent/GNN models, and trainer
tests/               # Deterministic unit tests
data/                # Generated per-ticker CSVs (git-ignored)
```

Future deep-learning and live-order modules will be added only when their data
contracts, risk limits, and broker requirements are defined.

## RL research demo

The first RL surface is intentionally a smoke-test environment over one
ticker's existing CSV snapshots:

```python
from pathlib import Path
import numpy as np
from trading_bot.training import OptionsEnv

env = OptionsEnv.from_directory(Path("data"), "AAPL", slot_count=32)
observation, info = env.reset(seed=7)
env_action = np.zeros(env.action_shape[0], dtype=int)
observation, reward, terminated, truncated, info = env.step(
    env_action  # integer array: 0 hold, 1..Q buy, Q+1..2Q sell
)
```

It uses fixed padded contract slots, validity/action masks, bid/ask paper fills,
cash constraints, deterministic seeds, and decomposed rewards. Its manifest is
marked `research_demo`; it is not a historical backtest or a claim of trading
performance. A licensed point-in-time all-expiry dataset is required before a
historical RL environment is enabled.

Snapshot loading adds causal engineered features such as relative spread,
forward log-moneyness, DTE, extrinsic value, quote age, liquidity logs,
underlying return, IV change/skew, ATM term slope, put-call IV spread, and
put-call parity residual. Fixed policy slots are stratified across expiration
and option type before taking deeper strikes. Chronological windows are
available through `training.sequence`.

Optional recurrent actor-critic models support GRU, LSTM, and parallel hybrid
GRU+LSTM encoders:

```bash
python -m pip install -e '.[ml]'
```

```python
from trading_bot.training.recurrent import RecurrentConfig, build_recurrent_actor_critic

model = build_recurrent_actor_critic(
    RecurrentConfig(input_size=709, slot_count=32, action_count=7, kind="gru")
)
```

Run the research-demo PPO trainer against collected snapshots:

```bash
train-demo --symbol AAPL --kind hybrid --episodes 25 --sequence-length 8
```

Add `--encoder graph` to run masked message passing over the option surface
before temporal encoding:

```bash
train-demo --symbol AAPL --encoder graph --kind hybrid --episodes 25
```

The graph connects each valid contract to neighbors using cross-sectionally
standardized IV, delta, log-moneyness, and DTE, symmetrizes those relationships,
adds self edges, and applies two message-passing layers. Invalid/padded slots
cannot send or receive messages. This dense implementation is deliberate: with
the default 32 slots it avoids a separate graph-framework dependency and its
conversion overhead. Use the default `--encoder flat` when inference latency
matters more than relational capacity.

It writes a safely loadable PyTorch checkpoint and a readable `.pt.json`
provenance sidecar containing the environment fingerprint, model/training
configuration, selection decision, and episode metrics. The trainer uses
factorized per-contract PPO ratios, generalized advantage estimation, clipped
policy and value updates, shuffled minibatches, target-KL early stopping,
entropy regularization, and gradient clipping. It evaluates deterministic
actions after each rollout and restores the best checkpoint. Selection is
explicitly labeled `in_sample_research_demo`; it is integration evidence, not a
backtest or an alpha claim.

Restore weights without enabling arbitrary pickle execution:

```python
from pathlib import Path
from trading_bot.training import load_checkpoint

model, manifest = load_checkpoint(Path("data/models/AAPL-graph-hybrid.pt"))
```

The ML extra is optional so collector startup latency and ordinary paper use do
not import PyTorch.

## Greek conventions

- Model: Black-Scholes-Merton using Yahoo's implied volatility.
- Expiration: 4:00 PM America/New_York on the listed date, ACT/365.
- Risk-free rate: latest Yahoo `^IRX` 13-week Treasury-bill yield divided by 100.
- Dividend yield: Yahoo underlying value when available, otherwise zero.
- Theta: option-price change per calendar day.
- Vega: option-price change for a one-percentage-point IV increase.

These are European-model estimates for American-listed equity options. Early
exercise, discrete dividends, delayed quotes, and the short-rate proxy can make
them differ from broker-provided Greeks.

## Test

```bash
python -m unittest discover -s tests -v
```

Yahoo Finance data is suitable for research and prototyping, not live order
execution. A broker-supported market-data feed is required before live trading.
