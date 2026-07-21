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
cash constraints, portfolio Greek exposures, optional Greek risk budgets,
deterministic seeds, and decomposed rewards. Its manifest is marked
`research_demo`; it is not a historical backtest or a claim of trading
performance. A licensed point-in-time all-expiry dataset is required before a
historical RL environment is enabled.

The portfolio state includes cash, invested cost, NAV, Delta, Gamma, Theta, and
Vega. Exposure units are share-equivalent Delta, Delta change per $1 Gamma,
dollars per calendar day Theta, and dollars per one-percentage-point IV Vega.
Limits are optional; risk-reducing trades remain available after Greek drift:

```python
env = OptionsEnv.from_directory(
    Path("data"),
    "AAPL",
    max_abs_delta=500,
    max_abs_gamma=100,
    max_abs_theta=250,
    max_abs_vega=500,
)
```

Snapshot loading adds causal engineered features such as relative spread,
forward log-moneyness, DTE, extrinsic value, quote age, liquidity logs, IV
change/skew, ATM term slope, put-call IV spread, and put-call parity residual.
Underlying return and annualized realized volatility over 4- and 16-snapshot
windows live once in the market vector rather than being repeated for every
contract. Each volatility estimate has an explicit history-coverage feature.
Fixed policy slots are stratified across expiration and option type before
taking deeper strikes. Chronological windows are available through
`training.sequence`.

Before entering a policy, production-layout observations use the versioned
`dimensionless.v2` transform. Prices and strikes are divided by spot, contract
Gamma represents a 10% spot move, Greek exposures are scaled by spot and NAV,
portfolio values become ratios, DTE is in years, and heavy-tailed age/liquidity
fields are compressed. Raw volume and
open interest are omitted because their causal log features contain the useful
ordering at a much better numerical scale. The transform is fitted on no data,
so it cannot leak future distribution statistics into a training window.

Optional recurrent actor-critic models support GRU, LSTM, and parallel hybrid
GRU+LSTM encoders:

```bash
python -m pip install -e '.[ml]'
```

```python
from trading_bot.training.recurrent import RecurrentConfig, build_recurrent_actor_critic
from trading_bot.training.sequence import observation_vector

model = build_recurrent_actor_critic(
    RecurrentConfig(
        input_size=observation_vector(observation).size,
        slot_count=env.action_shape[0],
        action_count=env.action_shape[1],
        kind="gru",
        market_feature_count=observation.market.size,
        portfolio_feature_count=observation.portfolio.size,
    )
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

Training can enforce the same portfolio budgets:

```bash
train-demo --symbol AAPL --max-abs-delta 500 --max-abs-vega 500
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

Evaluation reports include return, drawdown, step volatility/downside
deviation, Sharpe/Sortino diagnostics, turnover, fees, invalid actions, fills,
and peak absolute Greek exposures. Cost stress uses the same policy and quotes
while widening executable spreads and commissions:

```python
from trading_bot.training import evaluate_cost_stress, walk_forward_splits
from trading_bot.training.baselines import no_op

cost_reports = evaluate_cost_stress(env, no_op, seeds=(7, 8, 9))
folds = walk_forward_splits(
    len(env.dataset),
    min_train_size=500,
    validation_size=100,
    test_size=100,
    embargo=8,
)
```

Walk-forward folds are strictly chronological, support expanding or bounded
rolling training windows, and place an explicit embargo between train,
validation, and test partitions. The current small Yahoo sample will return no
folds for production-sized thresholds, which is preferable to silently
pretending that it supports out-of-sample evidence.

The evidence and sequencing behind future alpha research—including
walk-forward validation, benchmark hedges, realized-volatility state, GNNs,
and volatility-surface compression—are tracked in
[`docs/research-roadmap.md`](docs/research-roadmap.md).

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
uv run --extra dev python -m pytest -q
```

Yahoo Finance data is suitable for research and prototyping, not live order
execution. A broker-supported market-data feed is required before live trading.
