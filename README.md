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

On macOS, install that loop as a login service so it is restarted after a
failure and does not depend on an open terminal:

```bash
collector-service install
collector-status
```

`collector-service uninstall` stops and removes the LaunchAgent. The service
writes its stdout and stderr logs under `data/`. `collector-status --json`
exposes the current PID, heartbeat age, cycle progress, success/failure counts,
appended/unchanged ticker counts, and errors. Its exit status is nonzero when
the heartbeat is stale, the last cycle failed, or a continuous process died.
Only one collector may hold an output directory at a time.

Each ticker has an append-only file under `data/`, such as `data/AAPL.csv`.
Every row records the collection time, expiration, option type, market fields,
model inputs, and Delta, Gamma, Theta, and Vega. Collection defaults to the
nearest three expirations. Use `--expirations 1` for the lowest-latency mode or
`--expirations 0` to collect every listed expiration:

```bash
collect-options --once --expirations 1
```

The collector fingerprints the raw quote surface, spot, dividend yield, and
risk-free rate before appending. If those inputs are unchanged, it records the
successful observation in the heartbeat but does not append rows whose only
differences would be elapsed time and recomputed Greeks. The training loader
applies the same consecutive-deduplication rule to older CSVs, preventing stale
closed-market quotes from becoming synthetic RL transitions. A changed rate,
spot, contract set, or quote remains a new snapshot.

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
    env_action
)
```

It uses identity-stable padded contract slots, validity/action masks, bid/ask paper fills,
cash constraints, portfolio Greek exposures, optional Greek risk budgets,
deterministic seeds, and decomposed rewards. Its manifest is marked
`research_demo`; it is not a historical backtest or a claim of trading
performance. A licensed point-in-time all-expiry dataset is required before a
historical RL environment is enabled.

The action matrix has one row per option contract plus a final underlying-share
row. In every row action `0` holds, `1..Q` buys, and `Q+1..2Q` sells. Option
buckets contain 1..Q contracts. The underlying buckets contain multiples of the
configured lot size—25, 50, and 75 shares by default—and may create bounded
short positions. Passing the older option-only vector remains valid and implies
an underlying hold.

Options remain long-only by default. Passing
`--allow-collateralized-option-shorts` to a training command, or
`allow_collateralized_option_shorts=True` to `OptionsEnv`, enables only covered
calls and cash-secured puts. Each covered call locks 100 owned shares; each
short put locks its full strike times 100 in cash. Locked collateral cannot
support another order, and every leg of a multi-order action is revalidated
against the running account. Naked option shorts remain unavailable.

The portfolio state includes cash, invested cost, NAV, Delta, Gamma, Theta,
Vega, underlying shares, reserved cash, and reserved covered shares. Total
Delta includes the share position. Exposure units are share-equivalent Delta,
Delta change per $1 Gamma, dollars per calendar day Theta, and dollars per
one-percentage-point IV Vega. Limits are optional; risk-reducing trades remain
available after Greek drift:

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
Underlying return, cumulative log return over 4- and 16-snapshot windows,
elapsed seconds from the causal prior snapshot, and annualized realized
volatility over the same windows live once in the market vector rather than
being repeated for every contract. The trend and volatility summaries share
the exact causal history-coverage masks, while explicit gap coverage
distinguishes a missing or invalid timestamp from a genuine interval.
Shared price-history coverage remains visible in both `price_trend` and
`volatility_regime` ablations so a removed signal never removes the policy's
knowledge that history is sparse.
Front-expiry ATM IV and its difference from both realized-volatility
horizons provide a compact volatility-risk-premium regime signal. The same
snapshot-level vector carries executable front-expiry 25-delta risk reversal
and butterfly factors, exposing smirk and wing convexity without repeating them
across graph nodes. Executable ATM points across expirations now produce a
market-level term-structure slope and discrete curvature. One-snapshot changes
in front ATM IV, 25-delta risk reversal/butterfly, and term slope expose surface
dynamics without asking the recurrent model to reconstruct sparse factors from
changing contract slots. ATM, wing, term, change, executable-quote, Greek, and
realized-volatility coverage features distinguish missing surfaces from genuine
zero signals. Wing and term selection ignore zero-bid or otherwise unexecutable
quotes. The IV-minus-
realized signal stays neutral until some causal return history exists. Nearest
wing contracts must also lie within 0.15 Delta of the 25-delta target, so an
ATM-only chain cannot masquerade as a complete smile.
The first snapshot stratifies policy slots across expiration and option type
before taking deeper strikes. Later snapshots retain each surviving contract
at the same index and rank only replacements into vacated slots. A held option
that reappears after a quote gap is prioritized so it remains sellable; a
currently visible held option is never displaced by that recovery. Every contract
row carries `slotContinuity`, while `info` and training artifacts retain
identity changes and churn. This prevents GRU/LSTM state from silently following
a different option after a cross-sectional rank reversal. Use
`--slot-assignment ranked` only as a declared legacy comparison; the
`slot_identity` ablation masks the continuity input without changing assignment.
Every visible contract row also carries the agent's held quantity, average
entry price, and executable unrealized return. These are current portfolio
state—not market-data columns—and are zero for an unheld contract. Without them,
different holdings can collapse to the same aggregate-Greek observation and the
value function cannot tell which slot may create future P&L. Use the
`position_state` ablation to test their marginal value without changing fills or
sell feasibility.
Each visible contract also carries a compact prior-snapshot dynamics group:
mid-quote log return, relative-spread change, IV change, and separate quote/IV
coverage bits. Changes are available only when the same contract exists in both
snapshots with positive, non-crossed bid/ask quotes; IV change additionally
requires positive IV at both endpoints. Missing history is represented as zero
change with zero coverage, not as an observed flat market. The calculation does
not use last-trade price, Delta changes, or unsigned volume as a proxy for order
flow. Use the `contract_dynamics` ablation to test whether this state earns its
per-slot cost.
Chronological windows are available through `training.sequence`.

Before entering a policy, production-layout observations use the versioned
`dimensionless.v13` transform. Prices, strikes, and average entry price are
divided by spot, contract Gamma represents a 10% spot move, Greek exposures are
scaled by spot and NAV, share positions and covered-share reserves are scaled
by their NAV weights, and cash collateral is divided by NAV. Portfolio values
become ratios, DTE is in years, and heavy-tailed age/liquidity/gap fields and
position quantity are log-compressed. Unrealized return uses a signed log transform.
Cumulative log returns use the same signed bounded transform as one-step return.
Signed contract changes are log-compressed at fixed scales. The `time_context`,
`price_trend`, `position_state`, and `contract_dynamics` walk-forward ablations
can mask their inputs without changing model shape.
Raw volume and
open interest are omitted because their causal log features contain the useful
ordering at a much better numerical scale. The transform is fitted on no data,
so it cannot leak future distribution statistics into a training window.
On the included 84-contract AAPL snapshot, the expanded full feature-engineering
pass took about 20.79 ms median and 22.60 ms p95 per snapshot in a local
100-iteration CPU microbenchmark. Collapsing redundant surface-dynamics column
writes reduced its median from 23.43 ms. Engineering runs once when loading a
dataset, not on every policy step; treat these numbers as machine-specific
evidence, not a production SLA.

Optional recurrent actor-critic models support GRU, LSTM, parallel concatenated
GRU+LSTM hybrids, and adaptively gated GRU-LSTM mixtures:

```bash
python -m pip install -e '.[ml]'
```

```python
from trading_bot.training.recurrent import RecurrentConfig, build_recurrent_actor_critic
from trading_bot.training.sequence import observation_vector

model = build_recurrent_actor_critic(
    RecurrentConfig(
        input_size=observation_vector(observation).size,
        slot_count=env.slot_count,
        action_slot_count=env.action_shape[0],
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

Use the same recurrent policy with a Monte-Carlo policy-gradient comparator:

```bash
train-demo --symbol AAPL --kind gru --algorithm reinforce --episodes 25
```

`reinforce` computes discounted trajectory returns, bootstraps only a bounded
nonterminal rollout, subtracts the learned value baseline, and makes one
on-policy optimizer pass through contiguous recurrent chunks. PPO remains the
default and retains clipped multi-epoch updates. Metrics distinguish PPO,
REINFORCE, and total optimizer updates so their compute is auditable.

Both algorithms use duration-adjusted continuation and eligibility factors by
default. With the default 900-second reference interval, a transition lasting
`dt` seconds uses `gamma ** (dt / 900)`; PPO applies the same composition to
GAE lambda. This preserves one physical-time objective when collection gaps
vary instead of treating a one-minute and one-hour interval as equivalent.
Configure the reference with `--discount-reference-seconds`, set the base
factors with `--gamma` and `--gae-lambda`, or request legacy fixed-transition
semantics with `--no-time-aware-discounting`. Episode metrics and checkpoints
record observed durations and effective gamma/lambda ranges. These operations
run only while constructing training targets and do not enter agent inference.
A local 5,000-episode microbenchmark over 128 transitions measured about 62
microseconds per episode, or 0.48 microseconds per transition, for timestamp
parsing plus both factor vectors. Treat that as machine-specific engineering
evidence, not a portable guarantee.

Sparse portfolio reward does not have to be the recurrent encoder's only
learning signal. Enable train-only multi-horizon dynamics prediction:

```bash
train-demo \
  --symbol AAPL \
  --kind hybrid \
  --auxiliary-coefficient 0.05 \
  --auxiliary-horizon 1 \
  --auxiliary-horizon 4
```

For every policy state, the shared GRU/LSTM representation predicts bounded
cumulative spot, front-ATM-IV, 25-delta risk-reversal/butterfly, and ATM
term-slope changes at each declared snapshot horizon. It also predicts the
cross-sectional median matched-contract mid-quote return, relative-spread
change, and IV change. Contract targets match identifiers at both endpoints,
require executable bid/ask quotes, cover at least half of the current valid
cross-section, and are independent of slot order. Both
endpoints require point-in-time coverage; incomplete tail horizons are
explicitly masked. Targets are constructed only from observations already
collected inside that training rollout and partition, so neither validation nor
test values enter the loss or policy input. Horizons count snapshots rather
than fixed wall-clock time, which must be considered when collection cadence
varies.

The linear head is excluded from action inference and therefore adds parameters
and training work but no policy-path matrix multiply. Episode metrics retain
Smooth-L1 loss, masked MAE, weighted loss, and nested per-horizon/per-target
coverage. The coefficient defaults to zero because this is a
representation-learning hypothesis, not established alpha. In the current
hidden-size-eight layout, horizons 1+4 add 72 parameters over the
one-step head; policy-only medians overlapped at roughly 108-118 microseconds.
All validation scores were zero and the one-step candidate won only through the
smaller-parameter tie-break.

The v0.41 target extension adds 771 parameters to a width-128 hybrid head and
387 to a width-128 mixture head relative to the prior five-target head. It adds
no operation to `forward` or action inference. Generating both one- and
four-snapshot targets for all nine transitions in the current AAPL sample took
1.45 ms total. The three contract targets were available on every one-step
transition and six of nine four-step rows; the mask, rather than a fabricated
zero, covers the remaining tails. A two-fold, width-eight enabled/disabled
smoke tied at zero validation score and selected the zero-coefficient candidate
through the declared tie rule. This verifies training, masking, checkpoint, and
ablation plumbing but is not evidence of alpha.

v0.42 adds an explicitly opt-in option-liability surface without introducing
naked margin. Positions are signed and may close or cross through zero at the
current bid/ask. Short equity options that are in the money are physically
assigned using 100 shares per contract at the first observed date after
expiration; out-of-the-money contracts expire worthless, while long intrinsic
value remains cash-settled in this research approximation. There is no early
assignment model. The action mask requires valid option type, positive strike,
and a parseable expiration before opening a short.

The `dimensionless.v12` layout has 1,129 inputs at 32 slots. On the included
AAPL observation, adding the two normalized collateral fields left the local
10,000-call preprocessing median effectively unchanged (30.25 versus 30.17
microseconds). Opt-in collateral checks increased observation construction
from 3.41 to 4.29 ms median; default mode retains its direct long-only mask
path. Width-128 parameter counts are 1,192,002 for flat hybrid, 1,161,539 for
flat mixture, 220,013 for zero-neighbor graph-set hybrid, and 217,326 for
zero-neighbor graph-set mixture. A two-fold, width-eight AAPL walk-forward smoke
loaded `research-demo.v15`/`dimensionless.v12` checkpoints with short mode
enabled. These are machine-specific integration and latency checks, not alpha.

v0.43 makes that liability surface answer to a deterministic comparator. The
`cash_secured_short_put_delta_hedge` baseline waits until covered front-ATM IV
exceeds backward-only realized volatility by a declared edge, sells one
feasible front-expiry ATM put through the normal action mask, and reduces its
signed Delta with the same underlying-share actions available to the agent. It
is a no-op unless collateralized option shorts are explicitly enabled. Every
single-ticker and universe held-out artifact now includes its configuration,
timestamp-paired comparison, and separate normal/doubled-cost paths.

A synthetic chronological fold proved one valid secured-put execution with no
invalid actions: final NAV was 19,984.35 under base costs and 19,973.70 when
spread and commission were doubled; fees rose from 0.65 to 1.30. This is an
integration/stress result, not a profitable result. Replacing a per-call
31-field market dictionary with fixed feature indices reduced the signal-read
microbenchmark from 3.041 to 1.834 microseconds median over 100,000 calls (40%).
The shared CLI configuration constructor also prevents single-ticker and
universe baseline settings from silently diverging.

v0.44 makes risk part of the training objective instead of only a checkpoint
selection diagnostic. Both PPO and REINFORCE can opt into path-causal shaping:

```bash
train-walk-forward \
  --symbol AAPL \
  --reward-drawdown-penalty 1.0 \
  --reward-downside-penalty 1.0
```

At each transition, the downside component charges the negative part of the
net P&L return. The drawdown component charges only the increase in the running
maximum NAV drawdown. Consequently, its episode sum is exactly the coefficient
times negative maximum drawdown; an unchanged underwater state is not charged
again. Both components use only current and prior state, reset at every episode
boundary, and are retained separately in rollout metrics and checkpoint
manifests. Their coefficients default to zero, leave observations and inference
unchanged, and preserve the previous raw-return objective. Validation selection
penalties below are a separate layer; enabling both is an explicit choice to
optimize path risk during training and rank checkpoints by validation risk.

v0.45 removes the cold recurrent boundary from sampled training windows. A
window now starts up to eight snapshots earlier, advances the environment with
hold actions, and warms the GRU/LSTM state on that causal prefix without
gradients or reward credit. The optimized rollout still begins at the sampled
index with zero positions; stable slot history and market memory no longer
appear from an artificial zero state. The prefix is capped at the available
history and never leaves the training partition. Configure it with
`--burn-in-steps`; use `--burn-in-ablation` to add a matched zero-burn-in
candidate whose validation-only lift is retained in single-ticker and universe
artifacts. Ties prefer the recurrent-context candidate.

Burn-in observations are evaluated in one recurrent batch and do not construct
action masks because the discarded actor outputs cannot affect hidden state.
On the included AAPL data, an eight-step width-128 flat GRU benchmark measured
194.77 microseconds median for the batched prefix versus 530.21 microseconds for
eight streaming calls (2.72x faster, one CPU thread, 1,000 paired iterations).
This changes training only; deployment inference and the feature schema remain
unchanged. The benchmark is machine-specific and is not evidence of alpha.

v0.46 adds two compact, ticker-relative volatility-regime signals without
expanding the contract nodes. `frontAtmIvZScore16` and
`volatilityRiskPremiumZScore16` compare the current front ATM IV and current
front-ATM-IV-minus-four-snapshot-realized-volatility against valid values from
the 16 strictly prior snapshots. Four prior values are required. The
standardizer never sees the current or future value, extreme scores are clipped to ±8, and
each signal has separate history coverage. Missing or constant history yields a
neutral zero rather than a fabricated direction.

Use `--ablation volatility_normalization` to add a matched candidate with only
the two z-scores masked; their coverage remains visible so missing history is
not conflated with a zero signal. The `dimensionless.v13` layout has 1,133
inputs at 32 slots. On the current ten-snapshot AAPL sample, the final ATM-IV
z-score is 0.894 at 56.25% history coverage; the normalized volatility premium
is zero at 31.25% coverage because its prior values have no usable dispersion.
The 10,000-call preprocessing median was 30.38 microseconds versus 30.92 before
the four fields. Repeated width-128 inference medians were about 119 microseconds
for flat GRU and 215–218 microseconds for zero-neighbor graph-set GRU, consistent
with the prior run. These are coverage/integration and machine-latency results,
not evidence that the features produce alpha. A one-episode flat-mixture smoke
tied at zero validation score and selected the normalization-masked candidate
through the active-input tie-break, so the current sample provides no reason to
promote the new signals.

v0.47 removes pandas row materialization from the environment's repeated
observation path. Contract slots now use immutable scalar views over snapshot
column arrays, and portfolio marking reuses the same first-occurrence quote
lookup. Ranking, duplicate handling, fill prices, action masks, Greek exposure,
and accounting are unchanged; an explicit regression test preserves the
first-quote rule for duplicate contract symbols. Invalid zero-midpoint quotes
also remain uncovered without emitting NumPy division warnings.

On the current 11-snapshot, 84-contract AAPL sample at 32 slots, 50 complete
no-op episodes reduced median transition time from 5.27 ms to 1.63 ms and p95
from 8.34 ms to 1.81 ms. Median reset time fell from 9.15 ms to 6.93 ms. These
are same-machine measurements and do not alter model inference latency or
establish alpha. Profiling showed Python/pandas object construction—not a
numerical kernel—as the bottleneck, so a Rust or C++ extension would add a
boundary without addressing the measured cause.

v0.48 adds `attention_set`, an opt-in learned cross-contract relation encoder.
Within each snapshot it projects every valid contract into a shared latent
space, applies masked multi-head self-attention and pointwise residual blocks,
then uses the same invariant mean/max pooling and shared option-action scorer as
`graph_set`. GRU, LSTM, hybrid, and mixture units still own temporal state.
There are no positional embeddings: permuting contracts permutes their option
logits while leaving value estimates, auxiliary predictions, and the underlying
row unchanged. Invalid slots cannot contribute keys or values, and an entirely
empty surface remains finite. Configure heads with `--attention-heads`; the
graph width must be divisible by that count.

The encoder works with PPO or REINFORCE, factorized or exact single-leg actions,
feature ablations, parameter caps, checkpoints, and both walk-forward runners.
It is a candidate rather than a default. On the current 1,133-input AAPL layout
with width-128 GRU, graph width 32, two graph layers, and one CPU thread, 500
streaming calls measured 384.50 microseconds median and 398.08 p95 with 112,781
parameters. Zero-neighbor `graph_set` measured 182.25/188.21 microseconds and
96,749 parameters; flat GRU measured 121.81/126.46 microseconds and 517,186
parameters. A tiny matched AAPL validation smoke gave both set encoders zero
reward and selected Deep Sets through the smaller-parameter tie-break. This is
integration and latency evidence, not evidence that learned attention adds
alpha.

v0.49 removes two semantic no-ops from rollout and validation. Stable slot
assignment now returns immediately when every currently visible contract is
already assigned, even when the configured surface has unused padded slots. It
still invokes the full ranker as soon as a new quote appears. Option-expiry
handling now returns before timestamp parsing only when the option portfolio is
empty; every held long or short position follows the unchanged settlement path.

On a 260-snapshot, two-contract synthetic surface padded to 32 slots, a complete
128-step PPO rollout plus full deterministic selection fell from 2.62 seconds
median to 0.48 seconds (about 82%). Isolated no-op transitions measured 0.282 ms
with stable assignment versus 3.858 ms when forced through full ranking. On the
current 84-contract AAPL sample, median no-op transition time improved again
from 1.63 ms to 1.41 ms. These are same-machine throughput measurements; policy
inputs, actions, rewards, checkpoints, and evidence for alpha are unchanged.

Collection intervals are not assumed to be regular. The market vector includes
the positive elapsed seconds from the immediately prior snapshot and a separate
coverage bit; `dimensionless.v13` log-compresses the interval before it reaches
the recurrent layer. On the current 22-snapshot AAPL integration sample, 21
intervals were covered, ranging from 53.37 to 967.26 seconds with a 963.26-second
median. A hidden-size-128 flat hybrid grew from 983,406 to 985,202 parameters
when the two inputs were added. Its local 1,000-iteration streaming benchmark
measured 184.83 microseconds median and 188.83 microseconds p95, within ordinary
run-to-run variation of the earlier layout. A one-episode matched
`time_context` walk-forward smoke produced zero validation reward for both
variants and selected the masked candidate through the active-input tie-break;
that verifies the comparison path but provides no evidence of alpha.

The 4/16-snapshot cumulative-return extension reuses the realized-volatility
history pass and adds two market inputs, not contract-node fields. On the same
AAPL sample, 18 snapshots had complete four-step history and six had complete
sixteen-step history, but every retained quote was after the close and spot
remained 327.74, so every cumulative return was zero. A hidden-size-128 flat
hybrid added 1,796 parameters (985,202 to 986,998) and measured 188.33
microseconds median and 203.17 microseconds p95 over 1,000 local streaming
iterations. In the final one-episode smoke, full and `price_trend`-masked GRUs
both scored zero on validation; the masked candidate won the active-input
tie-break, and the trend comparator correctly made no held-out trades. This is
negative integration evidence, not support for retaining the signal in a paper
strategy.

Train one ticker-invariant shared policy across the collected top-50 universe:

```bash
train-demo \
  --universe top50 \
  --kind hybrid \
  --episodes 100 \
  --selection-cross-ticker-std-penalty 0.25 \
  --selection-worst-ticker-weight 0.25
```

Multi-ticker training uses a seeded shuffled order within balanced cycles and
requires at least one episode per ticker. Every episode owns a separate
environment, so recurrent state, cash, positions, returns, and rollout windows
reset at symbol boundaries; trajectories are never concatenated across
tickers. Checkpoint evaluation runs every ticker independently and preserves
each report and fingerprint. The shared observation remains dimensionless and
does not add a symbol ID, encouraging transfer rather than memorizing the 50
training names. The executable `train-demo` selection is still explicitly
in-sample; use it for integration and representation research, not performance
claims.

Use the shared chronological research boundary for model evidence:

```bash
train-universe-walk-forward \
  --min-train-size 500 \
  --validation-size 100 \
  --test-size 100 \
  --embargo 8 \
  --candidate flat:gru:ppo \
  --candidate graph:hybrid:ppo \
  --episodes 100 \
  --selection-cross-ticker-std-penalty 0.25 \
  --selection-worst-ticker-weight 0.25
```

The universe runner defaults to the collected top 50; repeat
`--universe-symbol TICKER` to declare a smaller research subset. It applies the
same ordinal split to every ticker, then enforces a stronger global boundary:
the latest training arrival across all symbols must precede the earliest
validation arrival, and the latest validation arrival must precede the earliest
test arrival. Dataset lengths, unused tails, exact global timestamps, split
indices, and train/validation/test fingerprints remain in the artifact.

Every architecture is trained over the isolated training pool and restores its
best aggregate validation checkpoint. Parameter matching, feature ablations,
PPO/REINFORCE comparisons, and a worst-ticker latency ceiling remain available.
Only the validation winner causes test environments to be instantiated. The
winner is then evaluated separately per ticker against every baseline and cost
scenario; moving-block comparisons remain within ticker and seed rather than
pooling symbols as independent observations. The cross-ticker held-out summary
is explicitly descriptive. One winning shared checkpoint is saved per fold.

Checkpoint and architecture selection can penalize validation-path risk:

```bash
train-walk-forward \
  --symbol AAPL \
  --selection-drawdown-penalty 1.0 \
  --selection-downside-penalty 1.0 \
  --selection-turnover-penalty 0.01
```

The declared score is validation reward minus each coefficient times maximum
drawdown, downside deviation, or turnover. All quantities are dimensionless.
Zero remains the default for every coefficient, preserving raw-reward behavior
until an experiment declares its risk tradeoff. The same score controls
checkpoint restoration, patience, ablation lift, and tournament ranking; raw
reward and every component remain in the artifact. Test metrics never enter it.
For shared policies, per-ticker scores are aggregated as
`(1-w) * mean + w * worst - d * standard_deviation`, where `w` is
`--selection-worst-ticker-weight` in `[0, 1]` and `d` is
`--selection-cross-ticker-std-penalty`. Both default to zero, preserving the
single-ticker and mean-score behavior. Declare them before validation; they are
robustness controls, not evidence of alpha.

Training episodes default to reproducible random windows of at most 128
transitions inside the supplied training partition. This exposes PPO to more
market regimes and bounds update cost without touching validation or test data.
Each random window uses up to eight preceding causal hold observations to warm
the recurrent state by default. Burn-in transitions update neither the optimizer
trajectory nor its reward totals; actual prefix start/length are persisted.
Use `--max-steps` to change the window or `--no-random-start --max-steps ...`
for a fixed prefix; passing the Python API `max_steps=None` trains on the entire
partition from its first snapshot. In a local 500-snapshot, flat-GRU synthetic
benchmark, the bounded default processed 128 rather than 499 transitions and
reduced an otherwise default one-episode train-and-selection run from about
5.67 seconds to 3.49 seconds (1.63x). Validation selection defaults to every
five episodes rather than every rollout; use `--evaluation-interval` to change
that cadence. Training stops after three evaluated checkpoints fail to improve
the selection reward, which avoids spending the full episode budget on stalled
candidates. `--selection-patience 0` disables this behavior, while
`--selection-min-delta` requires a meaningful reward increase before resetting
patience. Each metric row and checkpoint manifest records the stopping state.
These are machine-specific throughput choices, not alpha results.

Add `--encoder graph` to run masked message passing over the option surface,
`--encoder graph_set` to use a permutation-equivariant fixed-graph set policy,
or `--encoder attention_set` to learn masked cross-contract relations before
temporal encoding:

```bash
train-demo --symbol AAPL --encoder graph --kind hybrid --episodes 25
train-demo --symbol AAPL --encoder graph_set --kind hybrid --episodes 25
train-demo --symbol AAPL --encoder attention_set --attention-heads 4 --kind hybrid --episodes 25
```

Training can enforce the same portfolio budgets:

```bash
train-demo --symbol AAPL --max-abs-delta 500 --max-abs-vega 500
```

The graph connects each valid contract to neighbors using cross-sectionally
standardized IV, delta, log-moneyness, and DTE, symmetrizes those relationships,
adds self edges, and applies two message-passing layers. Invalid/padded slots
cannot send or receive messages. The original `graph` encoder flattens the node
states into a dense policy head. `graph_set` instead applies masked mean/max
pooling for the temporal state and scores every option through one shared head,
with a separate underlying-share head. It therefore preserves option-logit
equivariance and global-output invariance when contract slots are permuted, and
shares policy parameters across slots and tickers. This dense implementation
avoids a separate graph-framework dependency at the default 32 slots. Keep
`flat` and flattened `graph` as measured baselines rather than assuming more
relational structure is automatically better.

`attention_set` retains those set-policy symmetries but replaces hand-built
nearest-neighbor edges with global learned multi-head attention among valid
contracts. It uses no slot or positional embedding, so surface coordinates must
come from causal contract features such as log-moneyness and DTE. The additional
quadratic cross-contract work is bounded at 32 slots and is measured by the
existing inference-latency gate. Compare it against both zero-neighbor Deep Sets
and fixed-neighbor `graph_set` under the same parameter and validation budgets.

Set `--graph-neighbors 0` to turn `graph_set` into a self-only Deep Sets path.
It retains pointwise contract encoding, validity-masked pooling, the recurrent
global state, and the shared contract scorer, but creates no adjacency matrix,
graph multiply, or unused neighbor weights. This is a first-class candidate,
not an inference-only shortcut, so its exact likelihood and trainable parameter
count remain valid during PPO or REINFORCE training.

Recurrent state is carried causally from one snapshot to the next. PPO updates
shuffle contiguous truncated-backpropagation chunks instead of independently
shuffling zero-padded windows; `--sequence-length` is the maximum gradient
chunk length, not a claim that earlier state is erased. Deterministic inference
feeds one observation at a time and caches the GRU/LSTM hidden state. On the
pre-position-state layout (899 inputs, 32 option slots, 33 action rows, and seven
actions per row), a matched local 1,000-iteration CPU benchmark with hybrid
width 128 and graph width 32 measured 187.50 microseconds median and 983,406
parameters for `flat`, 360.67 microseconds and 1,159,650 parameters for
flattened `graph`, 333.42 microseconds and 214,187 parameters for three-neighbor
`graph_set`, and 240.50 microseconds and 212,331 parameters for zero-neighbor
`graph_set`. The self-only path was about 28% faster than full `graph_set` while
remaining about 28% slower than `flat`. In a three-fold, tiny-width-eight AAPL
integration smoke, it measured about 213-215 microseconds versus 316-324 for
the full graph set and 112-115 for flat. All validation scores tied at zero, so
the smoke establishes integration and latency only—not alpha. Stable
steady-state slot assignment measured
1.57 ms median versus 4.95 ms for full reranking; across the 17-transition AAPL
no-op episode, the median environment loop fell from 144.38 ms to 84.53 ms.
Treat these as machine-specific engineering measurements, not trading results.

The v0.37 position-aware layout has 29 contract fields and 999 flattened inputs
at 32 slots. A width-128 hybrid `flat` model grows from 986,998 to 1,073,206
parameters, while zero-neighbor `graph_set` grows by only 102 parameters—from
215,923 to 216,025—because its pointwise contract encoder shares weights across
slots. Precomputed feature-index maps and in-place finite/clipping transforms
reduced policy-vector preprocessing from 47.36 to 41.72 microseconds mean in a
local 5,000-call benchmark. Final 2,000-iteration streaming medians were 191.13
microseconds for `flat` and 273.10 for zero-neighbor `graph_set`, effectively
recovering the earlier measured end-to-end latency despite the added state.
These are machine-specific latency results; the position features remain an
ablation hypothesis, not evidence of alpha.

The `mixture` recurrent candidate runs causal GRU and LSTM experts in parallel,
then uses one learned sigmoid gate per timestamp to form a convex combination
before the policy, value, and auxiliary heads. Its gate starts at exactly 0.5,
so neither expert receives an arbitrary initialization advantage. Unlike
`hybrid`, it keeps downstream width at `hidden_size` instead of concatenating
to twice that width. On the v0.39 width-128 layout, this reduced flat parameters
from 1,073,206 to 1,043,767 and zero-neighbor `graph_set` parameters from
216,025 to 214,362. Local 2,000-iteration streaming medians were 198.46 versus
188.71 microseconds for flat mixture versus hybrid and 284.90 versus 266.94 for
the graph-set pair. The extra adaptive gate was slower despite the smaller
heads, so `mixture` is an explicit validation/latency candidate—not the default
and not a latency optimization. A two-fold, width-eight AAPL integration smoke
tied at zero validation score and selected the mixture only through the declared
smaller-parameter tie-break; it establishes training/checkpoint plumbing, not
predictive benefit.

The v0.40 causal contract-dynamics layout has 33 contract fields and 1,127
flattened inputs at 32 slots. Relative to v0.39, a width-128 flat hybrid grows
from 1,073,206 to 1,188,150 parameters, while zero-neighbor `graph_set` grows
from 216,025 to 216,161 because its contract encoder is shared across slots.
Local 2,000-iteration streaming medians were 203.00 microseconds for the flat
hybrid, 209.75 for flat mixture, 241.67 for graph-set hybrid, and 259.88 for
graph-set mixture. Policy-vector preprocessing measured 47.95 microseconds mean
over 5,000 calls. Consolidating prior-contract alignment into one lookup cut the
new dynamics pass from 2.32 to 1.91 ms median on the same 288-row AAPL surface.
Only 34.1% of rows in the current ten-snapshot AAPL sample had matched executable
quotes and IV, which is why change values and coverage remain separate. A
two-fold, tiny-width-eight removal smoke tied at zero validation score and masked
exactly 20 inputs across four slots; the masked candidate won the declared tie.
These are machine-specific integration and latency measurements, not evidence
of predictive benefit or alpha.

v0.40.1 preserves the `dimensionless.v11` values while reducing transform work:
the signed contract columns are processed as one matrix, clipping handles
infinities directly, one explicit pass replaces NaNs, and the float32 policy
vector is filled without a float64 concatenation. On the same 32-slot AAPL
observation, preprocessing fell from 47.79 to 30.13 microseconds median over
10,000 calls (37%). Matched 3,000-iteration streaming medians fell from 203.00
to 181.75 microseconds for flat hybrid, 209.75 to 191.63 for flat mixture,
241.67 to 224.29 for graph-set hybrid, and 259.88 to 239.08 for graph-set
mixture. Tests explicitly retain NaN-to-zero, positive-infinity-to-10, and
negative-infinity-to-minus-10 boundary behavior; 500 randomized finite and
nonfinite 32-slot observations were bitwise identical to the committed v0.40
transform. These measurements are local engineering evidence, not a production
SLA or alpha result.

The categorical policy head starts with a configurable `5.0` logit bias toward
hold, while every feasible action remains sampleable and the bias remains fully
trainable. On the real AAPL 33-row mask, 1,024 untrained graph-hybrid samples
fell from 24.06 requested orders per snapshot with zero bias to 0.74 with the
sparse prior; the 95th percentile was two orders. PPO entropy regularization
defaults to `1e-4`, matching basis-point reward scale more closely than the old
`0.01`. Override these with `--initial-hold-bias` and
`--entropy-coefficient`. Episode metrics retain requested option orders,
underlying orders, mean orders per step, and action density so sparsity is
measured rather than assumed.

The default `factorized` action decoder can place several option/underlying
orders at one snapshot. An optional `single_leg` decoder instead learns one
exact categorical distribution over global hold plus every feasible
row/non-hold-action pair—199 choices for the default 33×7 action surface. It
therefore emits at most one order by construction while preserving exact PPO or
REINFORCE likelihoods and the environment's masks; no sampled action is capped
or rewritten. Enable it with `--action-decoder single_leg`. This simpler action
space cannot open a same-snapshot spread or option-plus-hedge combination, so it
is a validation candidate rather than a new default.

On the current AAPL layout, a hidden-size-128 flat hybrid fell from 986,998 to
978,774 parameters, while zero-neighbor graph-set fell from 215,923 to 215,634.
The joint decoder was slower because decoding its selected category back into
the environment vector costs more than row-wise argmax: local 1,000-iteration
medians were 185.25 versus 205.50 microseconds for flat and 272.06 versus 300.65
microseconds for graph-set. A tiny hidden-size-eight AAPL tournament tied at
zero validation reward; the joint decoder won only through the smaller-parameter
tie-break (25,563 versus 25,851) and made no held-out trades. These measurements
justify keeping both decoders and provide no evidence of alpha.

It writes a safely loadable PyTorch checkpoint and a readable `.pt.json`
provenance sidecar containing the environment fingerprint, model/training
configuration, selection decision, and episode metrics. The stateful trainer
uses decoder-consistent exact PPO ratios, generalized advantage estimation,
clipped policy and value updates, contiguous recurrent minibatches, target-KL
early stopping, entropy regularization, and gradient clipping. It evaluates
deterministic actions after each rollout and restores the best checkpoint.
Selection is explicitly labeled `in_sample_research_demo` for `train-demo`; it
is integration evidence, not a backtest or an alpha claim.

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
and final/peak absolute Greek exposures. Cost stress uses the same policy and quotes
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

Run the complete recurrent PPO workflow once enough snapshots exist:

```bash
train-walk-forward \
  --symbol AAPL \
  --min-train-size 500 \
  --validation-size 100 \
  --test-size 100 \
  --embargo 8 \
  --candidate flat:gru:ppo \
  --candidate flat:lstm:reinforce \
  --candidate graph:hybrid:ppo \
  --candidate graph_set:hybrid:ppo \
  --candidate graph_set:hybrid:ppo:0 \
  --candidate attention_set:hybrid:ppo \
  --candidate flat:gru:ppo:single_leg \
  --candidate graph_set:hybrid:ppo:0:single_leg \
  --hidden-size 256 \
  --parameter-budget 20000 \
  --latency-warmup-iterations 10 \
  --latency-measured-iterations 100 \
  --max-median-inference-latency-us 500 \
  --ablation surface_wings \
  --ablation volatility_regime
```

Repeat
`--candidate ENCODER:KIND[:ALGORITHM[:GRAPH_NEIGHBORS][:ACTION_DECODER]]` to run a
leak-safe architecture and learning-algorithm tournament. The optional
algorithm is `ppo` or `reinforce` and defaults to `--algorithm` when omitted.
The optional integer neighbor override permits full and zero-neighbor graph
candidates in one predeclared tournament; otherwise `--graph-neighbors` applies.
The decoder is `factorized` or `single_leg`; for a non-graph candidate it may
occupy the fourth field directly, as in `flat:gru:ppo:single_leg`.
Every GRU, LSTM, hybrid, mixture, flat, flattened-graph, graph-set, or
attention-set candidate receives
the same fold and training seed. Each candidate restores its best validation
checkpoint; the
highest declared validation selection score wins, with fewer trainable
parameters and then a smaller active input set, fewer optimizer updates, and
stable model ID breaking ties. Only that winner is instantiated against the
held-out test range and only its checkpoint is saved. The summary retains every
candidate's configuration, parameter count, and validation score, but never a
losing-candidate test result. It also records episodes completed and whether
validation patience stopped each candidate before its requested budget. Omit
`--candidate` to preserve the single-model `--encoder`/`--kind` workflow.

Use `--parameter-budget N` to compare recurrent and graph architectures under
the same trainable-parameter ceiling. In this mode `--hidden-size` is a search
cap: each candidate deterministically receives the widest recurrent state that
fits `N`, while its encoder, recurrent family, graph shape, and input ablation
remain fixed. Resolution reads only the training environment's observation and
action layout, is cached across folds, and never inspects validation or test
values. An impossible budget that cannot fit hidden size one fails rather than
silently changing architecture. Artifacts retain both the requested model and
resolved `RecurrentConfig`, exact parameter count, and unused budget headroom.
This controls a major capacity confound.

Every trained candidate also receives a standardized streaming, batch-one
inference benchmark using one training-partition observation. The artifact
records median, p95, and mean microseconds plus device, PyTorch version, thread
count, warm-up count, and measured count. Configure the run length with
`--latency-warmup-iterations` and `--latency-measured-iterations`. Timing is
diagnostic only: it never changes validation ranking, and it is not portable
across hardware or runtime configurations. Use it to expose graph construction
or recurrent execution cost. When a deployment SLA is known in advance,
`--max-median-inference-latency-us` makes that predeclared ceiling an
eligibility constraint: candidates above it retain their configuration,
validation evidence, measured latency, and exclusion reason, but cannot win or
reach test. The run fails if every candidate exceeds the ceiling. The default
is no ceiling, so timing does not affect selection unless explicitly requested.

Repeat `--ablation GROUP` to add one matched feature-removal candidate per
architecture while retaining each full-feature candidate. Available groups are
`slot_identity`, `position_state`, `time_context`, `price_trend`, `surface_wings`,
`term_structure`, `surface_dynamics`, `volatility_regime`, `data_quality`, and
`derived_contract_surface`. Masking happens inside the
checkpointed model after
the versioned transform, so training, restored inference, and all encoders use
the same ablation. Artifacts report each ablated candidate's validation
score and raw-reward lift versus its full-feature counterpart plus active and
masked input counts.
Only the validation winner reaches test, preventing feature research from
becoming repeated holdout peeking.

When auxiliary training is enabled, add `--auxiliary-ablation` to create an
otherwise matched candidate with coefficient zero. Both candidates start from
the same seeded initialization and train on the same fold; artifacts report the
disabled candidate's validation reward and selection-score lift relative to the
enabled version. This comparison is also available in the universe runner.
For multi-horizon experiments, add `--auxiliary-horizon-ablation` to include a
matched one-step candidate in the same validation tournament. Artifacts retain
effective horizons and the one-step candidate's lift relative to the configured
multi-horizon model.
Add `--fixed-step-discount-ablation` to include otherwise matched candidates
whose gamma and GAE lambda apply once per snapshot rather than per elapsed
wall-clock interval. The artifact records effective semantics and the fixed-step
candidate's validation lift versus time-aware discounting. Because it does not
reduce inference parameters or latency, an exact tie retains the time-aware
reference. This is a training objective comparison, separate from the
`time_context` input ablation.
Only the validation winner reaches test, so neither an auxiliary task nor a
discounting convention can be kept because it happened to look good on the
held-out range.

Each fold trains only on its training range and touches the test range only
after both architecture and checkpoint selection. It writes a safe checkpoint
per fold plus a JSON summary with exact split boundaries, distinct
train/validation/test fingerprints, held-out recurrent results, no-op and
first-feasible baselines, a buy-first-then-Delta-hedge comparator, and
normal/doubled-cost reports. A feature-aware long-volatility comparator waits
for sufficient realized-volatility coverage, buys a front-ATM call/put pair
only when realized volatility exceeds ATM IV by a configured edge, and then
hedges residual Delta with shares. Its defaults are a 16-snapshot horizon, 75%
coverage, a 0.02 volatility edge, and one contract per leg; tune them with the
`--long-volatility-*` flags. The rule is long-only and holds the pair for the
episode, so it is a benchmark rather than a complete volatility strategy.

The cash-secured short-put comparator uses the opposite IV edge with the same
defaults and is configured through `--short-volatility-*`. It selects only a
feasible front-expiry negative-Delta contract, then Delta-hedges on later
snapshots. It holds through expiry, so assignment, spread, commission, and
collateral are part of its realized path. This is a conservative carry hurdle,
not evidence that selling volatility is profitable or safe.

These are joined by a causal underlying-trend comparator that targets a small
long, flat, or short share position from the covered 4/16-snapshot cumulative
return.
It rebalances toward the target rather than buying repeatedly, obeys the same
action masks and synthetic underlying costs as the agent, and is configured by
`--trend-window`, `--trend-min-coverage`,
`--trend-min-abs-log-return`, and `--trend-quantity`. The current execution
model does not include borrow, margin, funding, or dividends, so the short leg
remains research-only. These comparators improve the evaluation boundary;
Yahoo snapshots and the baselines still do not establish alpha.

Held-out agent and baseline paths are also compared with a paired circular
moving-block bootstrap over cumulative log-return difference. Pairing uses the
same arrival timestamps and preserves short-range serial dependence inside each
resampled block. The default uses 2,000 samples, square-root block length, a 95%
interval, and requires at least 20 test transitions. Shorter folds explicitly
report `insufficient_history` with no confidence bounds. Configure the method
with `--bootstrap-samples`, `--bootstrap-block-length`,
`--bootstrap-confidence`, and `--bootstrap-min-observations`. These diagnostics
are computed only after checkpoint selection and must not become a test-set
hyperparameter loop.

The NumPy implementation is vectorized; a local 1,000-observation/2,000-sample
benchmark took about 7.5 ms median, keeping statistical QA outside policy
latency.

Underlying fills use the saved underlying price plus/minus configurable
synthetic slippage (one basis point by default) and a per-share commission.
This enables a reproducible Delta hedge but is not a substitute for historical
underlying bid/ask data, borrow availability, margin, dividends, or funding.

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
