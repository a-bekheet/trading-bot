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

The collector fingerprints the raw quote surface, spot, dividend yield,
risk-free rate, and provider-timestamped benchmark observation before
appending. If those inputs are unchanged, it records the
successful observation in the heartbeat but does not append rows whose only
differences would be elapsed time and recomputed Greeks. SPY is fetched once per
cycle by default and copied into each ticker snapshot; use
`--benchmark-symbol` to change the declared proxy. Benchmark failure is
fail-soft and leaves explicit zero coverage while ticker collection continues.
The training loader
applies the same consecutive-deduplication rule to older CSVs, preventing stale
closed-market quotes from becoming synthetic RL transitions. A changed rate,
spot, contract set, quote, or timestamped benchmark observation remains a new
snapshot.

## Explore data

```bash
streamlit run src/trading_bot/interface/app.py
```

The first tab is an agent-results workspace. It discovers saved walk-forward
runs under `data/agent_runs/` and presents the newest selected policy for every
ticker as a persisted agent with a stable ID, checkpoint, recurrent core,
flat/GNN topology, activation state, latency, held-out return, and latest action.
Its decision tape retains HOLD decisions as well as fills and shows the learned
research action beside the action actually permitted by the sandbox guard. The
complete flat and surface-GNN GRU/LSTM/mixture challenger fleet remains
inspectable below the roster. Candidates are ranked on validation, and only the
fixed winner is opened on the held-out slice. A conservative promotion gate
requires positive held-out and doubled-cost returns, improvement over no-op with
statistical support, adequate history, regular-session provenance, and no
invalid actions.

The same tab also shows a persistent paper-agent loop. Each selected checkpoint
gets an isolated account, exact checkpoint hash, recurrent hidden-state cursor,
and idempotent decision ledger in `data/agent_paper.db`. The loop records the
model's proposed orders on each new regular, fresh, executable post-evaluation
snapshot. A winner that did not clear the validation activation gate is still
visible, but its actual sandbox order is forced to HOLD.

The remaining tabs let you choose a ticker, inspect its latest call or put
snapshot, and submit fake option orders. Paper buys fill at the saved ask and
paper sells fill at the saved bid.

Run the reproducible five-ticker recurrent-agent arena used by the interface:

```bash
agent-arena
```

By default it independently compares PPO GRU, LSTM, and gated-mixture agents on
AAPL, NVDA, MSFT, AMZN, and GOOG. Every flat recurrent family gets factorized
multi-leg and exact sparse single-leg policies. Three additional
`surface_graph_set` agents pair the sparse decoder with local strike/expiry and
opposite-side message passing. Six matched sparse agents remove only the causal
contract-level smile residual across flat/surface-GNN and GRU/LSTM/mixture
families. Six more remove only cadence-normalized surface velocity, for 21
candidates per ticker. Repeat
`--symbol` to choose another set. The command keeps identical budgets and split
rules across tickers and uses the latest possible chronological fold, assigning
all earlier eligible history to training. Each invocation gets a timestamped
directory under `data/agent_runs/recurrent-arena`, containing one walk-forward
artifact per ticker plus `agent-arena.json`, so later runs cannot overwrite
earlier evidence. A per-ticker failure does not discard completed runs. It
trains each candidate with three deterministic seed
replicates. Selection retains every policy within one standard error of the raw
leader, with a one-basis-point materiality floor, then prefers the existing
ablation, actor-latency, and complexity ordering. A separate validation-only
gate requires the winner to beat no-op by one basis point before the sandbox
will invoke it; otherwise the research result stays visible while the active
policy abstains. Before any model is trained, the default arena filters to
provider-confirmed regular states with a fresh underlying timestamp and an
executable option quote, then requires enough eligible states for all three
partitions: six training, three validation, and four held-out test. An unready
run writes a readiness manifest and exits successfully without training; use
`--allow-unready-tail` only for an explicit plumbing experiment.

On macOS, the readiness-aware training loop can run without an open terminal:

```bash
arena-service install
```

It checks the same strict five-ticker gate every 60 seconds, trains the full
315-replica arena once for each New York market session whose data becomes
ready, and records its heartbeat in `data/_arena_watch_status.json`. It will not
retrain on every collector cycle. The Agent Results tab displays whether it is
waiting, running, complete, or already current. `arena-watch --once` performs a
single check; `arena-service uninstall` stops and removes the LaunchAgent.

After a compatible arena finishes, advance the selected policies once:

```bash
paper-agents --data-dir data
```

On macOS, install the change-aware paper loop alongside the collector and arena
watcher:

```bash
paper-agent-service install
```

It checks inputs every 30 seconds but reloads checkpoints only when a CSV or
walk-forward summary changed. Its atomic heartbeat is
`data/_paper_agent_watch_status.json`; stdout/stderr remain under `data/`.
`paper-agent-watch --once` is the auditable one-cycle form, and
`paper-agent-service uninstall` removes the service. Old checkpoints whose
feature/checkpoint schema predates the runtime fail closed and remain visible as
errors until a current arena produces compatible weights.

For a one-ticker drill-down with explicit settings:

```bash
train-walk-forward \
  --symbol AAPL \
  --output-dir data/agent_runs/aapl-recurrent-tournament \
  --min-train-size 6 --validation-size 2 --test-size 3 --step-size 100 \
  --candidate flat:gru --candidate flat:lstm --candidate flat:mixture \
  --hidden-size 8 --episodes 3 --sequence-length 2 --burn-in-steps 0 \
  --max-steps 4 --initial-hold-bias 0 --slot-count 8 --max-quantity 1
```

These commands are visible integration demos, not recommended experiment sizes.
In the first five-ticker arena, GRU won validation on AMZN, MSFT, and NVDA;
LSTM won GOOG; and the gated mixture won AAPL. Every selected held-out path lost
money after costs: returns ranged from -0.046% on NVDA to -0.443% on GOOG.
The five paths made 42 fills over 15 transitions. Test slices had legacy or
non-regular-session provenance, so the result demonstrates end-to-end model
selection, behavior, accounting, and honest failure reporting—not alpha or live
fillability.

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

Automated policies do not share this manual account. Their separate
`agent_paper.db` store atomically commits the portfolio state, model-bound
recurrent cursor, and unique per-snapshot decision. This avoids double fills
after a restart and prevents one ticker or checkpoint from mutating another
agent's capital. The newest decision labels its reward as same-snapshot
execution-only until a later eligible mark exists; current paper equity is the
account result. It remains simulated execution only; no code path connects the
agent store to a live broker.

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
Each executable contract also receives a same-expiry, same-side quadratic-smile
leave-one-out IV residual plus coverage. The residual measures rich/cheap
deviation from the currently observed smile without reconstructing a quote or
changing execution prices and is removable through `contract_smile_residual`.
Underlying return, cumulative log return over 4- and 16-snapshot windows,
elapsed seconds from the causal prior snapshot, and annualized realized
volatility over the same windows live once in the market vector rather than
being repeated for every contract. The trend and volatility summaries share
the exact causal history-coverage masks, while explicit gap coverage
distinguishes a missing or invalid timestamp from a genuine interval.
Shared price-history coverage remains visible in both `price_trend` and
`volatility_regime` ablations so a removed signal never removes the policy's
knowledge that history is sparse. A compact `systematic_context` block adds the
provider-timestamped benchmark return, 4/16-observation benchmark return and
realized volatility, ticker-minus-benchmark return, quote age, and explicit
coverage. Repeated or backward benchmark timestamps never masquerade as zero
market returns, and legacy CSVs remain neutral with zero coverage.
Front-expiry ATM IV and its difference from both realized-volatility
horizons provide a compact volatility-risk-premium regime signal. The same
snapshot-level vector carries executable front-expiry 25-delta risk reversal
and butterfly factors, exposing smirk and wing convexity without repeating them
across graph nodes. Executable ATM points across expirations now produce a
market-level term-structure slope and discrete curvature. One-snapshot changes
in front ATM IV, 25-delta risk reversal/butterfly, and term slope expose surface
dynamics without asking the recurrent model to reconstruct sparse factors from
changing contract slots. Four additional velocities divide those covered
changes by the causal snapshot gap in hours and clip them to plus or minus two
volatility units per hour. This lets a compact recurrent policy distinguish a
fast surface shock from the same change over a long collection interval without
inferring a division from two inputs. Missing or nonpositive gaps remain neutral,
and the four values are removable through `surface_velocity`. ATM, wing, term,
change, executable-quote, Greek, and
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
Two additional causal lifecycle clocks report snapshots since the position
opened and since its most recent trade. Same-sign adds retain the opening clock,
every adjustment resets the last-trade clock, and crossing through zero starts a
new lifecycle. Both are zero for unheld contracts, log-compressed before policy
inference, and removable through the separate `position_lifecycle` ablation.
The recurrent state and value head also receive compact action-capacity context:
each option slot reports the fraction of configured buy and sell quantity
buckets currently feasible, and the portfolio vector reports the same two
fractions for underlying-share actions. These values are computed from the
exact current per-row action mask after cash, collateral, position, quote, and
Greek checks; the exact mask remains authoritative at sampling and execution,
and multi-order actions are still revalidated sequentially for aggregate risk.
Use the
`action_feasibility` ablation to test whether this compact summary improves
learning without feeding the full combinatorial mask to the recurrent layer.
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
`dimensionless.v24` transform. Prices, strikes, and average entry price are
divided by spot, contract Gamma represents a 10% spot move, Greek exposures are
scaled by spot and NAV, share positions and covered-share reserves are scaled
by their NAV weights, and cash collateral is divided by NAV. Portfolio values
become ratios, DTE is in years, and heavy-tailed age/liquidity/gap fields and
position quantity are log-compressed. Provider underlying-quote age uses the
same fixed gap transform. Unrealized return uses a signed log transform.
Cumulative log returns, benchmark returns, and ticker-relative returns use the
same signed bounded transform as one-step return. Benchmark volatility and age
reuse the existing volatility and provider-age transforms. Fitted-smile
curvature and ATM residual use the signed surface transform, while relative
fit RMSE uses `log1p`.
Signed contract changes are log-compressed at fixed scales. The `time_context`,
`price_trend`, `position_state`, `position_lifecycle`, `action_feasibility`, and `contract_dynamics`
walk-forward ablations preserve the external observation/action contract. Flat
models physically compact masked inputs; graph encoders retain shape and zero
masked relations before graph construction.
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

Net liquidation value is the default reward/NAV contract. Legacy experiments
can be reproduced explicitly with `--portfolio-valuation midpoint`; do not mix
the two modes inside one comparison.

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
change, IV change, and current-Delta-hedged option P&L normalized by current
spot. The last target additionally requires explicit regular-session and fresh
underlying provenance at both endpoints. Contract targets match identifiers at both endpoints,
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

v0.50 adds four current-snapshot, bid/ask-aware static-arbitrage diagnostics to
each contract node. Adjacent same-expiry calls and puts expose the larger of an
executable monotonicity violation and a vertical-spread payoff-bound violation.
Adjacent strike triples expose an executable convexity violation using the
correct weights for uneven strike spacing. Both scores are divided by spot and
paired with separate coverage bits; only positive, non-crossed bid/ask quotes
participate, duplicate strikes use the deterministic first quote, and missing
coverage stays distinct from an observed zero. The fractional uneven-strike
butterfly is a surface-consistency diagnostic, not a claim that an integer-lot
arbitrage can always be filled. Calendar constraints are intentionally deferred
because these American equity options and the current data do not provide a
clean European forward surface.

The `static_arbitrage` ablation masks the two scores while retaining coverage.
The resulting `dimensionless.v14` layout has 37 contract fields and 1,261
flattened inputs at 32 slots. Across the current 12-snapshot AAPL sample, 50.35%
of 2,028 contract rows had adjacent executable-quote coverage and none showed a
positive vertical or butterfly violation. Feature engineering on one 84-row
snapshot measured 24.54 ms median and 26.75 ms p95; 10,000 policy-vector calls
measured 30.71 microseconds median. Width-128 GRU medians were 125.25
microseconds for `flat`, 185.15 for zero-neighbor `graph_set`, and 389.08 for
`attention_set`, with 566,594, 96,885, and 112,917 parameters respectively.
A one-episode matched smoke tied both feature candidates at zero validation
score and selected the masked candidate through the 1,197-versus-1,261 active
input tie-break. These are integration, coverage, and machine-latency results;
they do not establish mispricing or alpha.

v0.51 changes the default training wealth definition from midpoint accounting
to net liquidation value. Open option longs are marked at the executable bid,
shorts at the executable ask, and both reserve the commission required to close.
Underlying shares use the configured exit-side synthetic slippage and closing
commission. If a held option temporarily disappears, valuation carries its
last executable liquidation mark instead of resetting it to entry cost. As a
result, the complete round-trip spread and commissions are recognized when a
position is opened, and closing later at unchanged quotes creates zero
artificial reward. Use `--portfolio-valuation midpoint` only to reproduce the
legacy optimistic accounting contract.

The environment manifest, checkpoint, single-ticker walk-forward summary, and
universe summary all persist the valuation contract. The environment,
checkpoint, walk-forward, and universe schemas advance to v19, v36, v39, and
v23 respectively; the unchanged policy input remains `dimensionless.v14`.
With one held AAPL option, 20,000 direct portfolio valuations measured 5.92
microseconds median for liquidation versus 4.54 for midpoint. An identical
deterministic 12-snapshot AAPL run ended at $99,896.00 under liquidation versus
$99,912.50 under midpoint, exposing $16.50 of legacy terminal optimism. The
one-episode liquidation walk-forward smoke completed with zero trades and zero
held-out return. These are accounting and latency checks, not evidence of alpha.

v0.52 makes executable-quote eligibility consistent across the action mask,
fills, contract state, and surface coverage. A finite, positive, non-crossed
bid/ask book is executable even when `lastPrice` is absent; last trade remains
available as raw context and as a held-position fallback, but it cannot suppress
a currently fillable contract. The semantic feature schema advances to
`dimensionless.v15`; environment, checkpoint, single-ticker walk-forward, and
universe walk-forward schemas advance to v20, v37, v40, and v24.

Deterministic held-out evaluation now accepts exactly one seed per CSV path.
Changing a seed label while policy inference and the market path are deterministic
does not create an independent observation. Every fold records a
`heldout_evaluation_contract` with its actual path count, seed repetitions, and
moving-block independence unit; cross-ticker summaries remain descriptive because
the paths share market conditions. Multiple training seeds remain valuable for
measuring learned-policy variability, but each must train an independent policy
and must not be substituted with repeated evaluation of one checkpoint.

Across the current 50 CSVs, 214,499 of 326,608 rows had executable bid/ask books
and none were newly recovered by removing the stale last-trade gate, so this is a
correctness and future-data fix rather than measured alpha. On the 84-row AAPL
snapshot, feature engineering measured 23.55 ms median and 26.51 ms p95, roughly
1 ms lower at the median than the prior same-machine run but within ordinary run
variation. The v0.52 one-episode walk-forward smoke completed with zero trades,
zero held-out return, and one declared deterministic held-out path.

v0.53 adds genuine training-seed replication to both single-ticker and shared
universe walk-forward selection. Repeat `--training-seed-offset` to train
independently initialized PPO or REINFORCE GRU/LSTM/hybrid/mixture candidates.
Each architecture is ranked on validation by a predeclared blend of mean and worst
seed score minus a seed-dispersion penalty. The deployed checkpoint is the run
closest to the median validation score, so adding seed robustness does not
multiply live inference latency or cherry-pick the best seed.

Every candidate now records its seed-level validation score, reward, optimizer
updates, latency, aggregate score, and representative rule. Every replicate must
pass the deployment latency ceiling. The held-out contract records all training
seeds and the selected training seed while still evaluating exactly one
deterministic policy/path pair. Checkpoint, single-ticker walk-forward, and
universe walk-forward schemas advance to v38, v41, and v25; environment v20 and
feature schema `dimensionless.v15` are unchanged.

In a two-seed AAPL integration smoke, seeds 7 and 1007 both scored zero on the
tiny validation path; the median-representative tie-break selected seed 7. Their
streaming inference medians were 124.81 and 120.27 microseconds, and the selected
held-out policy made zero trades with zero return. This verifies independent
training, aggregation, checkpoint selection, and constant deployment model count;
it is not evidence of alpha.

v0.72 gives recurrent agents a causal intraday clock without turning timestamps
into an execution rule. The market vector now contains the current capture
timestamp's fraction of the day in `America/New_York` plus explicit timestamp
coverage. This lets GRU, LSTM, hybrid, and mixture candidates learn regular-hour
time-of-day effects while Yahoo's provider `marketState` remains the only session
label and the independent execution mask remains authoritative. The feature does
not infer holidays, early closes, exchange quote time, or fill contemporaneity.

```bash
train-walk-forward --symbol AAPL --ablation intraday_clock
```

The design is motivated by the reported concentration of smile-geometry return
predictability in the final half-hour in
[Intraday Volatility-Smile Geometry and Option Returns (revised 2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5893362).
It does not reproduce the paper's 30-minute holding period or high-frequency
data. An initial sine/cosine/coverage design was simplified to day fraction plus
coverage because the intended U.S. regular-hours interval does not cross
midnight. That cut the width-128 flat-GRU increment from 1,158 to 772 parameters.

Across the latest AAPL, NVDA, MSFT, AMZN, and GOOG samples, every retained
snapshot had valid clock coverage, but none had provider-confirmed regular-session
coverage. A two-fold, one-episode AAPL smoke therefore tied at zero validation
reward in both folds and selected one full and one removed model through later
latency tie-breaks. This verifies plumbing only and provides no alpha evidence.
An alternating nine-repeat actor benchmark measured 125.125 microseconds for the
full model and 126.584 for the compact ablation; the wider model is not
intrinsically faster, so the 1.15% ordering is treated as machine noise.
Environment, feature-vector, checkpoint, single-ticker walk-forward, and universe
walk-forward schemas advance to v27, `dimensionless.v22`, v54, v58, and v42.
Package version is 0.72.0.

v0.73 replaces the interface's model-free landing view with a tangible agent
results workspace. Walk-forward v59 artifacts retain the selected policy's
held-out NAV path, per-step decisions, and execution ledger plus baseline paths
and test-slice execution provenance. The UI presents the candidate leaderboard
as validation evidence, exposes only the fixed winner as held-out evidence, and
refuses to turn a tiny or legacy-session demo into an alpha claim. The current
reproducible AAPL GRU/LSTM/gated-mixture tournament selected the mixture on a
-0.000424 validation score; it returned -0.121% on two held-out transitions
with 13 fills and $7.40 in fees. Package version is 0.73.0.

v0.74 adds `agent-arena`, a single reproducible command for applying the same
GRU/LSTM/gated-mixture PPO tournament to multiple tickers with isolated failure
records. The Agent Results tab now opens with a cross-ticker scorecard, return
chart, validation-winner counts, fills, costs, evidence grade, and execution
provenance before the existing per-run drill-down. The initial five-ticker run
completed without orchestration failures, selected GRU three times, LSTM once,
and the gated mixture once, and produced five negative held-out paths. Package
version is 0.74.0.

v0.75 turns action sparsity into a selected agent property instead of a fixed
assumption. Each GRU, LSTM, and gated-mixture arena family now competes with
matched factorized multi-leg and exact single-leg decoders. The sparse decoder
won validation on all five initial tickers. Relative to the earlier factorized
arena, held-out fills fell from 42 to 7, fees from $20.48 to $2.98, and mean
return improved from -0.154% to -0.025%; GOOG moved from -0.443% to flat. These
paths remain too short and poorly session-covered for an alpha claim.

The deterministic single-leg actor now bypasses training-only safe-row cloning
and full joint-mask materialization, masks only non-hold logits in place, and
decodes the winning joint index directly. A nine-repeat alternating AAPL GRU
benchmark reduced its median from 113.6 to 108.2 microseconds (4.8%) and narrowed
the overhead versus factorized from 15.9% to 9.6%. Arena schema advances to v2;
package version is 0.75.0.

v0.76 restores topology-aware GNNs to the tangible default arena after v0.75's
action-decoder isolation. Surface-GNN GRU, LSTM, and gated-mixture PPO policies
now compete against all six flat controls using the sparse decoder that won the
earlier action-surface test. In the first nine-policy five-ticker run, a surface
GNN won validation on every ticker: the gated mixture on AAPL, AMZN, MSFT, and
NVDA, and GRU on GOOG. Actor medians on AAPL were roughly 343-390 microseconds
for the GNNs versus 102-156 for flat candidates, keeping the latency tradeoff
visible.

That selection result did not translate into profit. All five held-out paths
were negative, mean return was -0.027%, and 14 fills cost $2.28. The interface
therefore reports zero promotion-ready paths and shows the exact failed gates
instead of treating a validation win as alpha. Arena schema advances to v3;
package version is 0.76.0.

v0.77 makes that larger arena more resistant to marginal validation wins. Every
default candidate now trains on three seeds. A one-standard-error rule retains
policies statistically competitive with the raw leader; a declared one-basis-
point floor prevents a zero-variance result on a tiny validation slice from
claiming false precision. Existing ablation preferences, actor latency,
parameter count, and active input count select only within that competitive
pool. Artifacts store the raw leader, effective tolerance, competitive model
IDs, selected-score sacrifice, and seed replicates.

On the same five-ticker integration data, this rule selected flat LSTMs for
AAPL and AMZN, a flat GRU for NVDA, and retained surface-GNN winners for GOOG
and MSFT. Median selected actor latency fell from about 382 to 133 microseconds,
a 65% reduction. Mean held-out return improved slightly from -0.027% to -0.025%,
but every path remained negative and zero passed promotion. Because the same
tiny held-out data has already been inspected across development iterations,
this comparison is engineering feedback, not fresh alpha evidence. Walk-forward,
universe walk-forward, and arena schemas advance to v60, v43, and v4; package
version is 0.77.0.

v0.78 separates the validation-selected research winner from the policy allowed
to act in the sandbox. A validation-only activation gate runs the deterministic
no-op baseline on the same validation environment and requires the selected
agent's seed-robust score to beat it by a predeclared margin. The default arena
uses one basis point and fails closed on equality. Research checkpoints,
held-out paths, and diagnostics remain available even when the operational
surface abstains.

All five current research winners had negative validation advantages, from
-0.44 to -1.51 basis points, so every sandbox policy chose no-op. On the already
inspected held-out paths this changed the operational view from a -0.025% mean
return, 14 fills, $3.32 fees, and 133-microsecond median selected actor latency
to 0%, zero fills, zero fees, and no actor invocation. This is loss avoidance,
not alpha: the gate has not yet activated a profitable agent, and the same tiny
test paths are not fresh evidence. Walk-forward, universe walk-forward, and
arena schemas advance to v61, v44, and v5; package version is 0.78.0.

v0.79 adds a causal per-contract volatility-smile residual to both flat
recurrent and surface-GNN observations. It fits only positive, non-crossed
current quotes within each expiration and option side, requires five points and
three unique forward-log-moneyness coordinates, and reports a leverage-adjusted
leave-one-out residual with explicit coverage. The default arena now evaluates
15 policies per ticker: nine full-feature agents plus six exact sparse
`contract_smile_residual` ablations across flat/GNN and GRU/LSTM/mixture.

In the five-ticker integration run, the signal improved 13 of 30 matched
validation pairs, hurt 14, and tied 3. Mean feature lift was +1.02 bp for the 15
surface-GNN pairs and -1.04 bp for the 15 flat pairs, so it remains selectable
rather than mandatory. Winners retained the signal for AMZN and GOOG and
ablated it for AAPL, MSFT, and NVDA; GOOG and MSFT selected surface-GNN LSTMs.
Every winner still had a negative validation advantage, from -0.29 to -1.21 bp,
so all sandbox policies abstained. The already inspected held-out research paths
averaged -0.023%, with 13 fills and $3.73 in fees; operational return remained
0%. Median selected actor latency was 118.5 microseconds. These small repeated
samples are engineering and hypothesis-ranking evidence, not fresh alpha.
Environment, feature-vector, checkpoint, walk-forward, universe, and arena
schemas advance to v28, `dimensionless.v23`, v55, v62, v45, and v6; package
version is 0.79.0.

v0.80 fixes a stale-evidence flaw in the fast default arena. Its previous
100-snapshot step produced only the earliest possible fold, so a growing live
collector never moved training, validation, or test forward until 100 additional
deduplicated states existed. The arena now requests one latest chronological
fold directly: the validation/test tail remains untouched, embargoes are
preserved, and every earlier eligible state expands training. General single-
ticker and shared-universe commands expose the same behavior through
`--latest-fold-only`; ordinary multi-fold walk-forward remains the default.

On the current five files, this moved training from six states to 20-24 and the
test interval from roughly 02:42-08:08 UTC to 11:49-12:39 UTC. The new tail was
provider-confirmed pre-market, so every trade was correctly masked, all 30 smile
comparisons tied at zero, and the simplicity rule selected full-feature flat
GRUs at about 102 microseconds median actor latency. Both research and sandbox
paths returned 0% with no fills or fees. This is currentness and safety evidence,
not alpha or a useful feature comparison; the running collector must accumulate
regular-session states before economic evaluation. Default arena invocations
now write collision-resistant timestamped run directories, while the interface
shows each held-out time range and preserves prior runs for drill-down.
Walk-forward, universe, and arena schemas advance to v63, v46, and v7; package
version is 0.80.0.

v0.81 adds a pre-training economic-evidence gate to the default arena. The
latest-fold correction exposed that a fully pre-market validation/test tail can
only produce masked no-op decisions, making 225 training replicates incapable
of generating useful agent evidence. The arena now checks every validation and
test state for provider-confirmed regular session, a covered underlying quote no
older than the environment threshold, and at least one positive non-crossed
option quote. Unready tickers record exact counts and timestamps in
`agent-arena.json`; if every ticker is waiting, the command completes without
calling the trainer. Streamlit shows ready ticker count plus regular, fresh, and
executable validation/test counts while retaining the last successful agents.
The Agent Lab makes those retained policies tangible: one card per ticker, a
registry with checkpoint and model identity, the full recurrent/GNN candidate
fleet, and a per-step decision tape that distinguishes research HOLD, unfilled
or blocked intent, and sandbox-enforced HOLD.

The first live preflight found 0/5 tickers ready: all underlying timestamps were
fresh, but every tail state was pre-market, and AAPL/NVDA also had no executable
option quote in those partitions. The first gate avoided training in 7.04
seconds. Loading only raw materially deduplicated snapshots before the gate then
reduced the same run to 1.48 seconds, 79% faster than the initial gate and about
97% faster than the roughly 47-second full arena. Full causal feature engineering
is deferred until a ticker passes. This is compute and evidence hygiene, not
alpha. The current persisted arena exposes five guarded agents, 75 candidate
configurations (30 surface-GNN), 225 independently trained seed replicates, and
15 explicit held-out HOLD decisions; the guard, not missing model artifacts,
accounts for the zero operational fills. Arena schema advances to v8; package
version is 0.81.0.

v0.82 adds cadence-normalized IV-surface dynamics as an explicit, ablatable
agent input. Front ATM IV, 25-delta risk reversal, 25-delta butterfly, and ATM
term-slope changes are divided by the strictly causal elapsed snapshot time and
bounded at plus or minus two volatility units per hour. Existing change and
coverage fields remain intact, so missing history never becomes an observed zero
velocity and the recurrent model can still learn the raw move separately. This
is a compact representation hypothesis motivated by recent IV-surface-feedback
work, not an alpha claim.

The default arena adds six exact `surface_velocity` removals across flat and
surface-GNN GRU/LSTM/gated-mixture sparse policies, increasing the declared
tournament from 15 to 21 configurations per ticker and from 225 to 315 training-
seed replicates across five tickers. A same-machine random-policy benchmark found
noise-level actor differences: 106.7 microseconds for the full flat GRU versus
107.5 for its physically compacted ablation, while the full model used 104 more
parameters. Surface-GNN parameter count was unchanged because market inputs are
masked in place. Feature engineering for 30 deduplicated AAPL states measured
1.17 seconds median and remains outside the per-decision path. A five-ticker
quality profile found 28-32 gap-covered states and 9-12 ATM/wing-change-covered
states per ticker. Term-slope velocity was unavailable for four tickers and
covered twice for GOOG; no velocity was non-finite or clipped, no contract key
duplicated within a snapshot, and each market value was consistent across its
contract rows. Sparse term coverage is a declared limitation, not a numeric zero
signal. A declared AAPL `--allow-unready-tail` plumbing run completed all 63
seed-trained replicas and 21 candidates without failure, selected the full flat
factorized GRU at 105.4 microseconds, and populated six exact velocity ablation
pairs. Every validation comparison and held-out return tied at zero because the
tail was PRE, so the sandbox abstained. The immediately following locked default
run stopped all five tickers in 1.6 seconds with five expected readiness statuses
and no trainer invocation. Environment,
feature-vector, checkpoint, walk-forward, universe, and arena schemas advance to
v29, `dimensionless.v24`, v56, v64, v47, and v9; package version is 0.82.0.

v0.83 connects continuous collection to tangible training. A user LaunchAgent
checks the shared arena contract once per minute, launches the isolated
five-ticker tournament once per eligible New York session, and exposes atomic
waiting/running/error/complete/up-to-date state in the Agent Lab. Session,
ordered symbols, and a versioned run contract prevent duplicate 315-replica
jobs. Package version is 0.83.0.

v0.84 corrects the eligibility boundary discovered while watching the first
regular snapshots arrive. A regular seven-state validation/test tail did not
prove that the expanding six-state training partition contained any executable
states; it could remain entirely pre-market, making policy training actionless.
The locked arena now filters before splitting and needs thirteen eligible states
per ticker: six training, three validation, and four test. The live UI shows
source, eligible, required, and excluded counts. Arena and watcher schemas
advance to v10 and status/run v2; package version is 0.84.0.

v0.71 adds critic-only LayerNorm as a separately selectable training
hypothesis for GRU, LSTM, hybrid, and gated-mixture PPO/REINFORCE models. The
normalizer sits between the recurrent representation and value head; policy and
auxiliary heads keep the unmodified representation, and actor-only deployment
never calls it. This isolates the critic-conditioning intervention described in
[TOPPO](https://arxiv.org/abs/2605.11473) without importing its PopArt,
FairGrad, PCGrad, task heads, or Meta-World+ performance claims.

```bash
train-universe-walk-forward \
  --critic-layer-norm \
  --critic-layer-norm-ablation
```

The ablation flag adds an otherwise matched disabled candidate to both
single-ticker and shared-universe selection, records validation reward and score
lift, and prefers the simpler disabled critic on an exact score tie. LayerNorm
adds `2 * critic_width` trainable parameters: 256 for a width-128 flat mixture,
and no actor operation. Identically seeded enabled and disabled models produced
exactly equal actor logits. A seven-repeat local actor benchmark measured
216.395 versus 210.125 microseconds median-of-medians; because the actor graphs
are identical, that ordering is timing noise rather than a speedup. A batch-16,
eight-step actor-critic forward measured 1,290.979 versus 1,307.333 microseconds,
a machine-local 1.27% training-forward cost.

In a real two-fold, one-episode AAPL width-eight smoke, enabled and disabled
candidates both earned zero validation reward in both folds, so the declared
tie-break selected the disabled critic twice. The normalized candidate also had
higher value-residual RMS and critic-head gradient norm in this tiny run. This
verifies the experiment surface but gives no reason to promote LayerNorm and no
evidence of alpha. Checkpoint, single-ticker walk-forward, and universe
walk-forward schemas advance to v53, v57, and v41; environment v26 and feature
schema `dimensionless.v21` are unchanged. Package version is 0.71.0.

v0.70 narrows the deployment cost of contiguous flat feature removals. When a
named ablation masks one uninterrupted raw-input interval, streaming policies
now copy the two surviving NumPy slices instead of performing a generic indexed
gather. Noncontiguous masks retain the generic path, graph encoders remain
unchanged, and both paths still validate the raw feature width before passing a
precompacted tensor to the recurrent model.

On the width-128 flat AAPL GRU with the contiguous four-input `smile_fit`
removal, an alternating seven-repeat benchmark measured 126.917 versus 126.875
microseconds median-of-medians for batch one and 807.375 versus 800.917
microseconds for synchronized batch 16. That is effectively unchanged batch-one
latency and a machine-local 0.80% batch improvement, not a general model-speed
claim. A reusable `np.take(..., out=...)` buffer experiment was rejected because
it made batch one 6.32% and batch 16 11.26% slower in the preceding matched
measurement. Actions, training behavior, model/checkpoint contracts, and schemas
are unchanged; package version is 0.70.0.

v0.69 adds a compact executable front-smile fit without adding actor-side
surface reconstruction. For the nearest expiry, it combines current OTM calls
and puts within 0.35 absolute forward log-moneyness, requires at least five
unique points with two unique coordinates on each side of ATM, standardizes the
coordinates, and requires observed support through +/-0.05 before fitting one
quadratic. Four market scalars expose curvature over +/-0.05 forward
log-moneyness, relative fit RMSE, the leave-one-out relative ATM residual, and
binary fit coverage. Only positive bid/ask pairs with `bid <= ask` enter the
fit; unavailable geometry remains zero with zero coverage.

```bash
train-walk-forward --symbol AAPL --ablation smile_fit
```

This is a validation hypothesis motivated by the reported short-horizon return
information in cross-strike smile curvature, deviations, and ATM richness in
[Intraday Volatility-Smile Geometry and Option Returns (revised 2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5893362).
It does not reproduce that paper's sample, timing, or portfolio construction and
does not establish alpha. `smile_fit` is a required removal candidate; the
feature block never controls execution or quote-validity masks.
Yahoo does not provide synchronized exchange timestamps for every option
bid/ask in this feed, so "executable" here means a positive internally valid
top-of-book pair at capture time, not proof of contemporaneous fillability.

On the current 20-snapshot AAPL integration sample, 10 snapshots had a complete
executable composite fit and the other 10 stayed uncovered. That is a data-path
check, not an independent return experiment. At 32 slots and hidden width 128,
the four scalars add 1,544 parameters to a flat GRU (625,655 versus 624,111,
about 0.25%). An alternating nine-repeat actor benchmark measured 125.13
microseconds median-of-medians for the full model and 127.61 for the compact
ablation; the latter includes feature-compaction overhead, so the ordering is
not an intrinsic speed claim. The quadratic fit happens during one-time dataset
engineering, not actor inference. Environment, feature-vector, checkpoint,
single-ticker walk-forward, and universe walk-forward schemas advance to v26,
`dimensionless.v21`, v52, v56, and v40. Package version is 0.69.0.

A two-fold, one-episode CLI smoke produced both full and `smile_fit`-removed
candidates, persisted the four masked inputs and validation lift, selected one
winner before constructing each test environment, and wrote the v56 artifact.
Every validation reward was zero on this tiny path, so this proves workflow
plumbing only and provides no evidence for retaining the features.

v0.68 adds a compact systematic-market context without adding a per-ticker
request or a learned encoder. The collector fetches one timestamped SPY
one-minute close per cycle by default and persists the same observation and
source metadata with each ticker. The policy receives one-step benchmark
return, 4/16-observation benchmark cumulative return and realized volatility,
ticker-minus-benchmark return, quote age, and coverage once in its market
vector. Strictly advancing provider time is required for a benchmark return;
legacy, repeated, backward, or untimestamped observations remain neutral with
zero coverage. Changing the declared benchmark resets its rolling history.

```bash
collect-options --once --benchmark-symbol SPY
train-walk-forward --symbol AAPL --ablation systematic_context
```

This is a cheap regime proxy motivated by the revised evidence that more robust
option momentum is concentrated in a systematic component. It is not that
paper's latent factor decomposition and does not establish alpha. The named
ablation must earn its extra input width on validation after costs. Benchmark
failure is fail-soft, and benchmark state never changes action masks or
execution. Environment, feature-vector, checkpoint, single-ticker
walk-forward, and universe walk-forward schemas advance to v25,
`dimensionless.v20`, v51, v55, and v39. Package version is 0.68.0.

At 32 slots and hidden width 128, the 12-scalar block added 4,632 parameters to
a flat GRU (624,111 versus 619,479, about 0.75%). An alternating nine-repeat
local AAPL actor benchmark measured 124.10 microseconds median-of-medians for
the full input and 125.83 for the compact ablation. Moving flat ablation
compaction to the NumPy policy boundary reduced the ablated path from 132.54 to
125.83 microseconds, removing about 5.1% from that path without changing
actions. The remaining 1.4% difference is small machine-specific overhead, not
evidence that the wider model is intrinsically faster.

v0.67 adds an optional train-only market-neutrality objective without changing
the policy input, recurrent architecture, or actor inference path. For the
post-transition portfolio it computes signed Delta notional weight
`w_delta = portfolio_delta * spot / NAV` and trains on
`r_train = r_executable - coefficient * min(abs(w_delta), 10)`. The cap bounds
bad early-policy leverage, the coefficient defaults to zero, and metrics retain
raw reward, shaped reward, coverage, exposure, and the penalty separately.
Validation and test always use unshaped executable net-liquidation returns.

```bash
train-walk-forward \
  --symbol AAPL \
  --delta-neutrality-coefficient 0.0001 \
  --delta-neutrality-ablation
```

Use `--delta-neutrality-ablation` with a positive
`--delta-neutrality-coefficient` to add a matched zero-coefficient candidate.
The candidate has the same parameters and actor operations; single-ticker and
shared-universe artifacts report validation reward and score lift against the
disabled objective. An exact score tie selects the disabled candidate because
an additional training hyperparameter must earn its place. This implements a
small options-specific test of the composite market-neutral recurrent-RL idea in
[AlphaZeroBeta (2026)](https://arxiv.org/abs/2607.18001); its equity-index
results do not establish portability or alpha here.

Every evaluation report now includes underlying return/volatility, strategy
return beta and correlation to the underlying with explicit coverage, and mean
and maximum absolute Delta-notional weight. Validation selection can predeclare
`--selection-abs-beta-penalty` and
`--selection-delta-notional-penalty`; a positive coefficient rejects an
uncovered validation metric instead of treating missing history as zero.
These diagnostics distinguish directional beta from an option-volatility
hypothesis, but short paths remain descriptive rather than statistically useful.

The fixed surface graph also reuses one nonpersistent diagonal mask instead of
allocating `torch.eye` on every decision. A profiler confirms the actor path no
longer executes `aten::eye`; a matched seven-repeat AAPL surface-graph mixture
benchmark measured 367.19 microseconds median-of-medians versus the prior
369.35, a 0.6% reduction within run noise. Checkpoint, single-ticker
walk-forward, and universe walk-forward schemas advance to v50, v54, and v38;
environment v24 and `dimensionless.v19` remain unchanged.

The three training CLIs now expose the same core optimizer controls:
`--learning-rate`, `--ppo-epochs`, `--minibatch-size`, `--clip-ratio`,
`--value-clip`, `--target-kl`, `--value-coefficient`, and
`--gradient-clip`. This makes cheap smoke runs and predeclared optimizer
experiments reproducible without constructing `TrainingConfig` in Python.

A real 18-snapshot AAPL CLI smoke trained the enabled and disabled objectives
with identical 3,363-parameter flat GRUs. Both produced zero unshaped
validation reward/score, so the disabled candidate won the declared tie and
the held-out report retained full Delta-weight coverage. A requested absolute-
beta selection penalty correctly failed because that short validation segment
had no covered return beta; the Delta-notional selection path completed. This
is integration and fail-closed coverage evidence, not neutrality or alpha.

v0.66 adds a train-only delta-hedged option-return target to the recurrent
auxiliary objective. For matched executable contracts it computes the
cross-sectional median of
`((future_mid - current_mid) - current_delta * spot_move) / current_spot`,
then applies the existing bounded signed-log transform. The hedge uses only the
Delta available at the policy-state endpoint and is not rebalanced using future
information. Both endpoints require explicit regular-session state and covered
provider underlying ages no greater than 1,200 seconds; pre/post-market and
unknown/stale provenance are masked. It is a volatility-specific representation target, not reward,
execution P&L, or evidence of alpha; financing, dividends, interim hedging, and
precise option-quote timestamp alignment remain outside this approximation.

The target follows evidence that volatility-surface representations can contain
information about future delta-hedged option returns
([Höfler, revised 2025](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4869272))
while respecting newer warnings that hedge timing and microstructure can create
spurious premiums
([Eberbach et al., revised 2025](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5192589)).
Use `--auxiliary-target-ablation medianContractDeltaHedgedSpotReturn` in a
walk-forward run to add a matched candidate that removes only this loss target;
both single-ticker and shared-universe artifacts report its validation lift
against the full target set.

The added output contributes 129 train-only parameters at recurrent width 128.
Actor weights and operations are identical. A seven-repeat, batch-one CPU
benchmark measured 369.35 microseconds median-of-medians with nine targets and
364.90 with the previous eight-target head, a 1.2% difference within run noise
despite no auxiliary-head call. Checkpoint, single-ticker walk-forward, and
universe walk-forward schemas advance to v49, v53, and v37; environment and
feature schemas remain v24 and `dimensionless.v19`.

In the current 16-snapshot AAPL sample, a quote-and-spot-only calculation would
have labeled 9 of 15 transitions available. All were pre-market or lacked the
required regular/fresh endpoint contract, so the final provenance mask correctly
excludes all 15 from this target. Synthetic regular-session integration evidence
confirms the target becomes available and updates its head row when those
requirements are met.

v0.65 adds `surface_graph_set`, an opt-in, permutation-equivariant graph policy
whose topology follows the option surface rather than quote values. It builds
same-side nearest-neighbor edges in standardized forward-log-moneyness and DTE,
then links every valid contract to its nearest opposite-side coordinate
counterpart. Implied volatility remains node content and cannot rewire the
graph. One shared neighbor transform handles both edge classes, keeping the
representation compact; this is a trading-policy representation hypothesis,
not put-call-parity enforcement or evidence of alpha.

The design is motivated by the irregular quote geometry and dynamic sampling
studied in
[Operator Deep Smoothing for Implied Volatility](https://arxiv.org/abs/2406.11520),
but it consumes only observed contracts and never reconstructs executable
quotes. The original IV/delta/moneyness/DTE similarity graph and zero-neighbor
Deep Sets path remain explicit ablations. On the current 32-slot AAPL layout, a
matched batch-one CPU benchmark measured 298.75 microseconds median for the
structured graph versus 273.83 for the similarity graph and 179.13 for Deep
Sets. Sharing one message operator cut the structured version by about 7.7% and
2,336 parameters relative to a two-operator prototype. These are machine-specific
engineering measurements; the structured encoder must win validation after costs
and satisfy the existing latency gate. Checkpoint, single-ticker walk-forward,
and universe walk-forward schemas advance to v48, v52, and v36; environment and
feature schemas remain v24 and `dimensionless.v19`.

v0.64 makes underlying price provenance and freshness causal. Yahoo's explicit
`PRE`, `REGULAR`, and `POST` states now select the matching provider price/time
pair instead of always attaching `regularMarketPrice` to every session. CSVs
persist `underlyingPriceSource`, `underlyingQuoteTime`, and
`underlyingQuoteTimeSource`; a provider timestamp advance is a material market
event, while capture-time-only repetition is still deduplicated.

The market vector adds `underlyingQuoteAgeSeconds` and
`underlyingQuoteAgeCoverage`, with a named `data_freshness` ablation and fixed
log transform. Explicit quote ages above the configurable 1,200-second default
mask all simulated option and underlying fills while retaining hold and full recurrent
and critic chronology. Missing legacy provenance has zero coverage and remains
research-demo tradeable with a warning. The Streamlit sandbox exposes the same
age and gate. This is a necessary data-quality control, not evidence that Yahoo
quotes are executable or that the threshold creates alpha.

The design follows recent evidence that reporting latency can reorder market
events and cause look-ahead errors
([Battalio et al., 2026](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5907665))
and option-market evidence that stale quotes distort observed co-movements used
in pricing and hedging
([Fahlenbrach and Sandas](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=690722)).
Environment, feature, checkpoint, single-ticker walk-forward, and universe
walk-forward schemas advance to v24, `dimensionless.v19`, v47, v51, and v35.

v0.63 makes actor credit assignment aware of whether the policy had a genuine
choice. Session-forced holds and any other state where every decoder factor is
singleton no longer enter actor-advantage normalization, the policy surrogate,
entropy, approximate KL, or clip fraction. Joint PPO/REINFORCE masks entire
choice-free transitions; the dimensionwise PPO ablation masks individual
singleton factors. This follows the constrained-policy lesson that training
must operate on the actual masked distribution, supported by
[Huang and Ontañón](https://arxiv.org/abs/2006.14171) and the more recent
[state-dependent relevance work](https://arxiv.org/abs/2406.03704).

Nothing is removed from chronological learning: every forced transition still
advances GRU/LSTM state and participates in rewards, GAE or returns, critic
loss, and available auxiliary targets. Episode metrics now record actor-choice
steps, forced-hold steps, and their fraction; checkpoint manifests persist the
full support contract. This is training-only and adds no deployment inference
operation. Checkpoint, single-ticker walk-forward, and universe walk-forward
schemas advance to v46, v50, and v34; environment and feature schemas remain
v23 and `dimensionless.v18`.

All-forced minibatches also skip categorical construction, log-probability,
importance-ratio, and entropy work. A matched seven-repeat benchmark of 20
one-step closed-session PPO runs measured 25.86 ms per run versus 25.98 ms on
committed v0.62. The removed operations are semantically unnecessary, but the
0.5% end-to-end difference is not a measurable speedup because recurrent,
critic, setup, and evaluation work dominate this tiny run.

v0.62 adds a causal market-session boundary from Yahoo's option-chain
[`underlying.marketState`](https://github.com/ranaroussi/yfinance/blob/main/yfinance/ticker.py).
The collector normalizes and persists the provider
state, treats state transitions as material snapshots, and atomically adds the
column to legacy per-ticker CSVs. It does not guess exchange state from a clock,
timezone, weekday, or holiday calendar.

The policy market vector adds `regularMarketSession` and
`marketStateCoverage`. An explicit `PRE`, `POST`, `CLOSED`, or other recognized
non-regular state masks all option and underlying trades while retaining hold;
the Streamlit sandbox applies the same rule. Legacy rows become `UNKNOWN` with
zero coverage and remain research-demo tradeable for compatibility, with a
visible warning. This fallback cannot support paper-alpha claims. The two
policy inputs have a named `market_session` ablation, but the independent
execution mask remains active during that ablation. Environment, feature,
checkpoint, single-ticker walk-forward, and universe walk-forward schemas
advance to v23, `dimensionless.v18`, v45, v49, and v33.

The post-change live cycle migrated all 50 ticker CSVs, appended 20,538 rows,
reported `PRE` for every latest snapshot, and completed with zero failures. A
matched seven-repeat, 10,000-call AAPL policy-vector benchmark measured 33.61
microseconds median for v0.62 versus 33.77 for committed v0.61 (1,393 versus
1,391 inputs). The two scalar features therefore added no measurable
preprocessing regression in this smoke; these timings are machine-local and
say nothing about strategy return.

v0.61 adds training-only critic-balance diagnostics before introducing another
normalization method. Every PPO and REINFORCE episode now records reward RMS,
return-target mean/standard deviation/RMS/maximum magnitude, pre-update value
residual RMS, and pre-clipping actor-head and critic-head gradient norms. The
head gradients are exact and cheap because the policy and value output modules
are disjoint; they already include the configured loss coefficients and do not
attribute gradients inside the shared recurrent/GNN trunk.

`critic_balance_diagnostics(metrics)` aggregates transition-scale measurements
by transition count and gradient measurements by optimizer-update count for each
ticker. Checkpoints and every single-ticker or universe walk-forward candidate
retain the per-symbol evidence, cross-symbol positive-scale ratios, zero-return
symbols, and a predeclared 10x imbalance flag. The flag is diagnostic only: it
cannot alter checkpoint, architecture, training-seed, validation, or held-out
selection. A triggered flag recommends a separately named normalization
ablation; it never silently changes the learner.

A real five-ticker `MU/GEV/KLAC/AMD/AMAT` flat-mixture, single-leg smoke used two
four-transition episodes per ticker. Return-target RMS varied 3.57x and
critic-head gradient norm varied 9.25x, so the 10x trigger remained false. The
sample contains only 13-15 mostly overnight snapshots per ticker and is too
small to reject or promote critic normalization. Computing both gradient norms
on an already-backpropagated 32-hidden-unit mixture took 16.28 microseconds per
optimizer update in a local 10,000-call benchmark and adds no inference work or
extra backward pass. These measurements are diagnostics, not evidence of alpha.
Checkpoint, single-ticker walk-forward, and universe walk-forward schemas advance
to v44, v48, and v32; feature and environment schemas remain
`dimensionless.v17` and v22.

v0.60 adds an actor-only batched runtime for synchronized paper agents and
counterfactual rollouts. Deployment no longer executes the value or auxiliary
head when only an action is requested, while training retains the full
actor-critic path. A fixed batch carries independent GRU, LSTM, hybrid, or
mixture hidden-state columns through one model forward:

```python
from trading_bot.training import batched_recurrent_policy, load_checkpoint

model, manifest = load_checkpoint("artifacts/policy.pt")
agents = batched_recurrent_policy(
    model,
    sequence_length=manifest["training"]["sequence_length"],
    batch_size=3,
)
actions = agents((aapl_observation, msft_observation, nvda_observation))

# MSFT starts a new episode; the other two recurrent cursors are unchanged.
agents.reset([1])
branch = agents.fork()
portable_state = agents.snapshot()
```

Every batch call is transactional with respect to recurrent state: it validates
batch width, feature layout, model mode, and every cursor's chronology before
running the actor or advancing any cursor. Partial reset explicitly zeros only
the selected hidden-state columns. Batched snapshots validate their own v1
schema, model contract, cursor counts, finite tensor layout, timestamp/step
consistency, and zero state for reset cursors. All cursors must supply one
observation per synchronized call; this is not an asynchronous scheduler.

On a local one-thread flat-GRU benchmark over seven matched 3,000-iteration
runs, skipping the critic reduced median single-agent latency from 93.51 to
88.76 microseconds, a 5.4% speedup. One batch of eight agents took 316.09
microseconds versus 716.11 serially (2.27x throughput); 32 agents took 1,076.50
microseconds versus 2,845.35 serially (2.64x). Per-agent batched latency was
39.51 and 33.64 microseconds respectively. These machine-specific measurements
do not imply trading alpha. The batch-one inference schema advances to v2,
single-ticker walk-forward to v47, and universe walk-forward to v31; checkpoint,
feature, and environment schemas remain v43, `dimensionless.v17`, and v22.

v0.59 replaces the opaque recurrent evaluation closure with an explicit
`StreamingRecurrentPolicy` runtime for GRU, LSTM, concatenated-hybrid, and
gated-mixture agents. It remains callable by the existing environment and adds
episode `reset()`, independent `fork()`, and device-portable `snapshot()` /
`restore()` operations:

```python
from trading_bot.training import load_checkpoint, recurrent_policy

model, manifest = load_checkpoint("artifacts/policy.pt")
policy = recurrent_policy(model, manifest["training"]["sequence_length"])
action = policy(observation)

branch_state = policy.snapshot()  # cloned CPU tensors; no shared storage
branch = policy.fork()             # same read-only model, independent cursor
policy.reset()                     # required before a new episode or ticker
policy.restore(branch_state)       # resume the original causal branch
```

The runtime requires `model.eval()` so dropout cannot silently make decisions
nondeterministic. Strict chronology is on by default and rejects duplicate or
backward observations before mutating hidden state. Snapshots carry a v1 schema,
step/timestamp consistency, finite tensor checks, exact recurrent tensor shapes,
and a SHA-256 contract over the full `RecurrentConfig`. A snapshot must still be
restored only into the same checkpoint weights; the configuration hash is not a
weight identity. The inference benchmark explicitly disables chronology because
it intentionally reuses one observation to measure model latency.

On a local one-thread CPU benchmark over seven matched 3,000-call runs, guarded
streaming measured 95.85 microseconds per call versus 94.51 without chronology,
a 1.33-microsecond median overhead. Resetting an existing cursor took 0.057
microseconds versus 2.211 to construct a wrapper, about 38.7x cheaper. These are
machine-specific engineering measurements, not evidence of alpha.

v0.58 makes entropy regularization invariant to padded and hold-only action
rows. For every masked categorical factor with `K > 1` feasible choices, the
trainer computes exact entropy divided by `log(K)` and averages only those
explorable factors within each decision, then averages decisions. The result
stays in `[0, 1]`; factors with one choice
contribute neither a fake zero nor a divide-by-zero. This gives the same
exploration scale to sparse and dense option surfaces and works unchanged with
PPO, REINFORCE, GRU, LSTM, hybrid, gated mixture, graph encoders, and the
single-leg joint decoder.

Raw entropy, normalized entropy, and the explorable-factor fraction remain in
every episode record. `--entropy-objective-ablation` adds otherwise matched
legacy `raw_mean` candidates in both walk-forward runners and reports their
validation-only lift. An exact score tie retains feasible-normalized entropy
before latency comparison because this objective changes no inference work.
On a local 512-transition, 33-row tensor benchmark, computing normalized and
raw diagnostics took 1,187.21 microseconds versus 810.98 for raw entropy alone,
about 0.74 microseconds of extra training work per transition.

The real two-seed GOOG smoke gave both objectives zero robust validation score
and the raw-mean ablation zero lift. The normalized checkpoint won the declared
tie and its two-transition held-out path made no trades or return. This validates
the invariant and selection path; it is not evidence of alpha. Checkpoint,
single-ticker walk-forward, and universe walk-forward schemas advance to v43,
v46, and v30. Feature and environment schemas remain `dimensionless.v17` and
v22.

v0.57 corrects the default factorized PPO objective to use the likelihood of
the complete action vector. Because the row policies are conditionally
independent, the exact joint log likelihood is the sum of their masked
categorical log likelihoods; PPO now forms and clips one importance ratio from
that sum. REINFORCE likewise uses the joint score function. The single-leg
decoder was already an exact joint categorical and retains that behavior.

The earlier per-row clipped surrogate remains available as the explicit
`dimensionwise` research ablation through
`--factorized-objective-ablation`. Single-ticker and shared-universe artifacts
report the effective objective and its validation-only lift against the joint
default. An exact validation-score tie retains the joint objective before
latency comparison because the objectives do not alter inference operations.
Probability-identity and gradient tests cover the product/sum
relationship, and every GRU, LSTM, hybrid, and gated-mixture learner now records
one PPO importance ratio or one REINFORCE score-function likelihood per
transition under the default. This changes training
semantics without adding model parameters or inference work; it also reduces
the stored ratio/surrogate tensor width from 33 to one on the default 32-slot
environment. In a local CPU tensor-only benchmark with 4,096 transitions and
33 action rows, the clipped-surrogate kernel measured 44.45 microseconds median
with the joint ratio versus 287.62 microseconds dimension-wise, while ratio
storage fell from 135,168 to 4,096 elements. This excludes network forward and
backward work, so it is not an end-to-end training-speed claim.

A real two-seed GOOG walk-forward smoke gave both objectives zero robust
validation score and the dimension-wise ablation zero lift. The new tie rule
retained the joint checkpoint; its two-transition held-out path made no trades
and returned zero. This verifies objective selection and provenance but is not
evidence of trading alpha.

Checkpoint, single-ticker walk-forward, and universe walk-forward schemas
advance to v42, v45, and v29. Feature and environment schemas remain
`dimensionless.v17` and v22.

v0.56 adds optional volatility-stratified recurrent training starts for PPO and
REINFORCE across GRU, LSTM, hybrid, mixture, graph, and shared-universe models.
Sampling uses only fully covered backward-looking realized volatility inside
the training partition. Every episode records requested/effective sampling,
the selected regime bin, and available-bin count; checkpoint sidecars and all
candidate replicates retain aggregate bin evidence. A matched uniform-start
tournament candidate reports its validation lift, and exact ties retain the
configured stratified method only after validation, latency, capacity, and
training-work comparisons.

Synthetic tests exercise three distinct strata and deterministic sampling. The
current local GOOG integration data has only 15 usable snapshots and identically
zero covered four-snapshot realized volatility, so the real two-seed smoke
correctly recorded uniform fallback for both candidates. Both validation scores
and the selected held-out return were zero with no trades. This exposes the
current data limitation and validates fallback/artifact plumbing; it does not
show regime-sampling lift or alpha.

Checkpoint, single-ticker walk-forward, and universe walk-forward schemas
advance to v41, v44, and v28. Feature and environment schemas remain
`dimensionless.v17` and v22 because inference state is unchanged.

v0.55 makes state-dependent trading capacity visible to the recurrent policy
and value estimator without appending the full action mask. Two bounded values
per option slot summarize the currently feasible buy and sell quantity buckets;
two portfolio values do the same for underlying-share actions. The summaries
are regenerated from the exact causal mask at every snapshot, remain secondary
to that mask, and are removable through `action_feasibility`.

At 32 slots and hidden size 128, the full flat GRU has 1,391 inputs and 617,806
parameters; compacting the 66 feasibility inputs leaves 1,325 active inputs and
592,330 parameters. In the two-seed tiny AAPL smoke, robust validation scores
tied at zero. Full-model seed medians were 130.42 and 129.73 microseconds versus
137.21 and 134.02 for the smaller gathered-input ablation, so the latency
tie-break retained the faster full model. The held-out path again made zero
trades and zero return. This validates the state, ablation, and selection path,
not improved sample efficiency or alpha.

The feature, environment, checkpoint, single-ticker walk-forward, and universe
walk-forward schemas advance to `dimensionless.v17`, v22, v40, v43, and v27.
Model IDs now include the feature-vector schema in their digest, preventing a
new layout from reusing an old checkpoint filename in the same output directory.

v0.54 added explicit position-lifecycle state and made latency an honest
equal-score tournament tie-break. Each visible held contract reports
`positionAgeSteps` and `positionLastTradeAgeSteps`; same-sign adds preserve the
first clock, all adjustments reset the second, and a long/short crossing resets
both. This exposes holding and rebalance cadence to truncated recurrent chunks
without requiring the GRU or LSTM to reconstruct the entire trade history.

Flat-model feature ablations now physically remove masked inputs from LayerNorm
and the recurrent input matrix. At 32 slots and hidden size 128, removing the 64
lifecycle inputs cut the flat GRU from 592,330 to 567,626 parameters. It did not
improve batch-one latency on this machine: across independently trained seeds 7
and 1007, the full model measured 129.79/128.60 microseconds median versus
131.79/132.71 for the compact lifecycle ablation. Because validation scores tied
at zero, the new worst-seed latency tie-break correctly retained the faster full
model. The held-out smoke made zero trades and zero return, so lifecycle state
remains an ablation hypothesis rather than alpha evidence.

The feature, environment, checkpoint, single-ticker walk-forward, and universe
walk-forward schemas advance to `dimensionless.v16`, v21, v39, v42, and v26.
The external action contract is unchanged.

Collection intervals are not assumed to be regular. The market vector includes
the positive elapsed seconds from the immediately prior snapshot and a separate
coverage bit; `dimensionless.v17` log-compresses the interval before it reaches
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
  --selection-turnover-penalty 0.01 \
  --selection-abs-beta-penalty 0.01 \
  --selection-delta-notional-penalty 0.01
```

The declared score is validation reward minus each coefficient times maximum
drawdown, downside deviation, turnover, absolute underlying beta, or mean
absolute Delta-notional weight. All quantities are dimensionless. Beta and
Delta-notional selection penalties require covered validation metrics; they
fail closed on insufficient history rather than awarding false neutrality.
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

For long, non-stationary training partitions, opt into causal regime-balanced
starts with `--start-sampling volatility_stratified`. The sampler sorts only
fully covered training-partition starts by `realizedVol16` (or the declared
`--volatility-regime-window 4`), divides them into a configurable number of
quantile strata, chooses a stratum uniformly, and then chooses a start within
it. `--volatility-regime-bins` defaults to three. If there are too few covered
or distinct values, it records and uses a uniform fallback. Validation and test
remain complete chronological paths. Add `--start-sampling-ablation` to compare
an otherwise matched uniform-start candidate using validation only. This changes
training exposure, not the policy architecture or inference latency.

Add `--encoder graph` to run masked message passing over the option surface,
`--encoder graph_set` to use a permutation-equivariant similarity-graph set
policy, `--encoder surface_graph_set` to use causal surface-coordinate
relations, or `--encoder attention_set` to learn masked cross-contract relations
before temporal encoding:

```bash
train-demo --symbol AAPL --encoder graph --kind hybrid --episodes 25
train-demo --symbol AAPL --encoder graph_set --kind hybrid --episodes 25
train-demo --symbol AAPL --encoder surface_graph_set --kind hybrid --episodes 25
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

`surface_graph_set` keeps the same masked pooling and shared scoring contract,
but constructs local edges only between calls or only between puts using
`forwardLogMoneyness` and `dteDays`. Each node also selects its closest valid
opposite-side contract in those coordinates; the union is symmetrized and
passed through one shared neighbor projection. Delta's sign identifies option
side, including negative-zero puts, while IV is message content rather than an
edge coordinate. With `--graph-neighbors 0`, this encoder retains counterpart
edges; use zero-neighbor `graph_set` for the true no-adjacency Deep Sets
baseline. The counterpart edge is structural context only because these equity
options may be American-style and the environment does not impose exact parity.

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
`0.01`. The default `feasible_normalized` objective divides each masked
factor's entropy by its current maximum `log(K)` and excludes factors with only
one feasible choice before averaging decisions, so padding and unavailable
contracts cannot dilute the
bonus. Override these with `--initial-hold-bias`, `--entropy-coefficient`, and
`--entropy-objective`. Episode metrics retain raw and normalized entropy,
explorable-factor coverage, requested option orders,
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
  --candidate surface_graph_set:hybrid:ppo:3 \
  --candidate attention_set:hybrid:ppo \
  --candidate flat:gru:ppo:single_leg \
  --candidate graph_set:hybrid:ppo:0:single_leg \
  --hidden-size 256 \
  --parameter-budget 20000 \
  --latency-warmup-iterations 10 \
  --latency-measured-iterations 100 \
  --max-median-inference-latency-us 500 \
  --start-sampling volatility_stratified \
  --start-sampling-ablation \
  --factorized-objective-ablation \
  --entropy-objective-ablation \
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
`--latency-warmup-iterations` and `--latency-measured-iterations`. Timing never
outranks validation score and is not portable across hardware or runtime
configurations. It is the first deterministic tie-break only when seed-robust
validation scores are equal, preventing a smaller-but-slower model from winning
on parameter count. Use it to expose graph construction or recurrent execution
cost. When a deployment SLA is known in advance,
`--max-median-inference-latency-us` makes that predeclared ceiling an
eligibility constraint: candidates above it retain their configuration,
validation evidence, measured latency, and exclusion reason, but cannot win or
reach test. The run fails if every candidate exceeds the ceiling. The default
has no hard ceiling, while the equal-score latency tie-break remains active.

Repeat `--ablation GROUP` to add one matched feature-removal candidate per
architecture while retaining each full-feature candidate. Available groups are
`slot_identity`, `position_state`, `position_lifecycle`, `action_feasibility`, `contract_dynamics`, `static_arbitrage`,
`time_context`, `price_trend`, `surface_wings`, `term_structure`,
`surface_dynamics`, `volatility_regime`, `volatility_normalization`,
`data_quality`, and `derived_contract_surface`. Masking happens inside the
checkpointed model after the versioned transform. Flat encoders gather only
active inputs, while graph encoders zero masked inputs before graph construction;
training and restored inference use identical semantics. Artifacts report each ablated candidate's validation
score and raw-reward lift versus its full-feature counterpart plus active and
masked input counts.
Only the validation winner reaches test, preventing feature research from
becoming repeated holdout peeking.

Factorized PPO defaults to one clipped importance ratio for the complete action
vector. Add `--factorized-objective-ablation` to train otherwise matched legacy
`dimensionwise` candidates and report their validation lift. This ablation is
created only for factorized PPO candidates; single-leg PPO and every REINFORCE
candidate continue to use an exact joint likelihood. The optional
`--factorized-ppo-objective dimensionwise` switch exists for declared legacy
reproduction, not as the recommended training default. An exact validation tie
retains the joint candidate because this training-only choice has no deployment
latency tradeoff.

Add `--entropy-objective-ablation` to compare the default mask-density-invariant
entropy against the unnormalized explorable-factor mean. The flag requires a positive
entropy coefficient and creates a matched candidate for every declared model.
Artifacts record both objective values and validation lift; an exact tie retains
`feasible_normalized` because the choice adds no inference operations.

When auxiliary training is enabled, add `--auxiliary-ablation` to create an
otherwise matched candidate with coefficient zero. Both candidates start from
the same seeded initialization and train on the same fold; artifacts report the
disabled candidate's validation reward and selection-score lift relative to the
enabled version. This comparison is also available in the universe runner.
For multi-horizon experiments, add `--auxiliary-horizon-ablation` to include a
matched one-step candidate in the same validation tournament. Artifacts retain
effective horizons and the one-step candidate's lift relative to the configured
multi-horizon model.
Repeat `--auxiliary-target-ablation TARGET` to add a matched candidate that
removes only one named output from the Smooth-L1 loss mask while preserving the
same policy, recurrent state, head shape, initialization, and inference path.
For the volatility-specific target use:

```bash
train-walk-forward \
  --symbol AAPL \
  --auxiliary-coefficient 0.05 \
  --auxiliary-target-ablation medianContractDeltaHedgedSpotReturn
```

The excluded target has zero training coverage in its candidate artifact, and
the candidate reports reward and selection-score lift against the full target
set. Exact ties retain the full target set because both models have identical
deployment cost.
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
