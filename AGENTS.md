# Trading Bot Agent Guide

## Mission

Build an options research and trading system incrementally. The current system
collects option data, calculates Greeks, and supports local paper trades. It
does not place live trades.

## Working principles

- Surface assumptions that affect financial meaning or live-trading safety.
- Implement only the current milestone; avoid speculative abstractions.
- Keep changes inside the relevant domain module.
- Define a measurable check before changing behavior and run it afterward.
- Never imply that Yahoo Finance data is suitable for live execution.

## Repository layout

- `src/trading_bot/market_data/`: ticker universe, Yahoo option retrieval,
  risk-free rates, and recurring CSV collection.
- `src/trading_bot/analytics/`: deterministic financial calculations.
- `src/trading_bot/execution/`: the paper broker and portfolio valuation. It
  must remain independent of any future live-broker adapter.
- `src/trading_bot/interface/`: the Streamlit data explorer and its data loader.
- `src/trading_bot/training/`: versioned research-demo schemas, manifests,
  snapshot loader, fixed-slot environment, deterministic baselines, and
  evaluation reports. `features.py` computes causal features and `sequence.py`
  builds chronological windows. `recurrent.py` is an optional PyTorch GRU,
  LSTM, or hybrid actor-critic with flat or graph contract encoding. `trainer.py`
  owns factorized PPO/GAE optimization, deterministic evaluation, safe
  checkpoint restoration, and provenance. `walk_forward.py` owns validation-only
  model selection, held-out test evaluation, fold artifacts, and its CLI. These
  modules stay outside ordinary collector imports.
- `tests/`: offline tests; market-data calls must be mocked here.
- `data/`: generated append-only CSVs, one per ticker; intentionally git-ignored.

Add future code by domain rather than to a generic utilities module:

- `training/` for datasets, features, experiment configuration, and model runs.
- `execution/live/` for future broker adapters and live-order workflows.
- `statistics/` for backtests, portfolio metrics, and performance attribution.

Create those packages only when implementing their first real behavior.

## Current data workflow

1. `market_data.universe` defines the top 50 U.S. company tickers. The snapshot
   is dated 2026-07-21 and sourced from CompaniesMarketCap's U.S. ranking.
2. `market_data.rates` fetches `^IRX` once per cycle as a short-term risk-free
   proxy.
3. `market_data.option_chain` retrieves a configurable number of expirations,
   calls, puts, underlying price, and dividend yield. Collection defaults to
   three expirations; one is the low-latency mode and zero in the CLI means all.
   Yahoo's option payload reports dividend yield in percentage units, so the
   adapter divides it by 100.
4. `analytics.greeks` calculates Black-Scholes-Merton Greeks.
5. `market_data.collector` appends the enriched rows to `data/<TICKER>.csv`.
6. `interface.app` displays the latest saved snapshot; it never fetches markets.
7. `execution.paper_broker` stores fake cash, long positions, and fills in
   `data/paper_portfolio.db`; `execution.valuation` marks positions from CSVs.
8. `training.env.OptionsEnv` exposes the current CSVs through a Gymnasium-style
   `reset`/`step` API. It is `research_demo` only and must not be used as a
   historical-performance benchmark.

A failure for one ticker must be logged without stopping the other tickers. A
risk-free-rate failure stops that cycle because silently using a stale or fixed
rate would make the calculated data misleading.

## CSV contract

CSV files are append-only snapshots. Do not remove or reorder columns without a
migration. `collector._migrate_csv` upgrades older files atomically before an
append. Important model/input columns are:

- `collectedAt`, `symbol`, `expiration`, `optionType`
- `underlyingPrice`, `riskFreeRate`, `riskFreeRateSource`, `dividendYield`
- `timeToExpiryYears`, `impliedVolatility`, `greekModel`
- `delta`, `gamma`, `theta`, `vega`

Training-time surface features include forward log-moneyness, extrinsic value,
ATM IV/skew, ATM term slope, put-call IV spread, and parity residual. They are
derived within a single captured timestamp and are not persisted into the raw
CSV contract. The market vector also contains front-expiry ATM IV and its
difference from causal 4/16-snapshot realized volatility. Keep those global
regime features out of each contract node.

Greek conventions:

- Expiration is 4:00 PM `America/New_York`, ACT/365.
- Theta is per calendar day.
- Vega is per one IV percentage point.
- Invalid or expired inputs produce missing Greeks rather than unstable values.
- The model is an approximation for American-style equity options.

## Paper-execution contract

`PaperBroker` is the stable boundary for the interface and future agents.

- Starting cash is $100,000 when the database is first created.
- Orders are marketable paper fills: buy at the saved ask, sell at the saved bid.
- One contract has a multiplier of 100.
- Positions are long-only; selling more than the owned quantity is rejected.
- Orders with non-positive price/strike/quantity are rejected.
- Buys with insufficient cash are rejected.
- SQLite `BEGIN IMMEDIATE` transactions serialize account updates.
- The ledger records every fill and realized P&L; positions use average cost.
- Portfolio marks use bid/ask midpoint, falling back to last price.
- No module in this repository may route `PaperBroker` calls to a real broker.

Future agents should read quotes from `interface.data.load_latest_snapshot`,
make a decision, then call `PaperBroker.buy` or `PaperBroker.sell` with explicit
contract metadata and price. Do not let an agent invent or silently substitute
a missing quote.

## RL research-demo contract

`training.OptionsEnv` is the first executable RL surface. It is intentionally
small and stable:

- `reset(seed=None, options=None)` returns `(Observation, info)`.
- `step(action)` returns `(Observation, reward, terminated, truncated, info)`.
- `Observation.contracts` has fixed shape `(K, features)` and carries
  `contract_ids`, `valid_mask`, and `action_mask`.
- `Observation.portfolio` contains cash, invested cost, NAV, portfolio
  Delta/Gamma/Theta/Vega, and the underlying-share position. Total Delta must
  include the shares.
- `Observation.action_mask` has `K+1` rows: `K` option slots and one final
  underlying slot. Action `0` means hold; `1..Q` are buy buckets and
  `Q+1..2Q` are sell buckets. Legacy length-`K` arrays imply underlying hold.
- Option buckets represent contracts. Underlying buckets represent multiples
  of `underlying_lot_size`, may open bounded shorts, and must obey cash, Delta,
  and `max_abs_underlying_shares` constraints.
- Masks are generated from the pre-step state and include quote validity,
  fee-adjusted affordability, held quantity, and optional absolute Greek limits.
- Multiple orders in one action are revalidated sequentially so cash cannot go
  negative even when individual pre-step actions were affordable.
- Execute the underlying leg first, then option slots in ascending order. Masks
  describe the pre-step state; every leg must still be revalidated against the
  running cash and Greek state.
- `info` retains executions, invalid-action count, P&L, fees, trade notional,
  and reward components.
- `reward_components` must sum to the returned scalar reward. Gross P&L includes
  spread/mark effects, while commission and invalid-action penalties are
  separate components.
- A missing contract is never silently transferred to another slot.
- Non-held slots are surface-stratified: one near-ATM contract from each
  expiration/type group is selected before deeper strikes, with spread and open
  interest as deterministic tie-breakers.
- `step()` must execute against the exact cached slots and mask returned by the
  preceding observation; only the next state may rerank the surface.
- An episode with `N` snapshots has exactly `N-1` tradable transitions and
  truncates on arrival at the last snapshot; never permit an unmarkable extra
  fill at the terminal timestamp.
- Multi-order actions revalidate Greek budgets sequentially. If market drift
  puts a portfolio over a limit, actions that reduce the absolute exposure must
  remain permitted even when they do not immediately return below the limit.

The current environment is a deterministic accounting and API scaffold, not a
historical simulator. Do not add a `historical` mode until the data manifest
contains point-in-time all-expiry quotes, depth/quality fields, lifecycle data,
and an explicit source/license.

Engineered features must be causal: current rows may use current cross-sectional
values and the immediately prior snapshot, but never future rows. Sequence
windows are chronological and unpadded. GRU/LSTM/hybrid code is optional
(`.[ml]`) and must preserve a no-PyTorch collector path. Checkpoints must retain
the environment fingerprint, full model and training configuration, metrics,
and the `research_demo` label.

An environment manifest hashes only its selected ticker CSV. Do not broaden the
fingerprint to unrelated ticker files: doing so adds startup I/O and invalidates
otherwise reproducible experiments whenever the collector updates another
symbol.

`sequence.observation_vector` is the versioned policy boundary. Under
`dimensionless.v4`, price-like fields are relative to spot, contract Gamma is
the Delta change for a 10% spot move, portfolio and Greek exposures are relative
to NAV/deployed capital, underlying shares are represented by NAV weight, DTE
is expressed in years, and heavy-tailed fields are compressed and clipped. Raw
volume and open interest
must not be reintroduced beside their log features without ablation evidence.
Any transform change requires a new feature-vector schema, scale-invariance and
finite-value tests, and a checkpoint-schema bump; old weights must never be
silently loaded against a changed feature layout.

Underlying return and 4/16-snapshot realized-volatility estimates are market
features, not contract features; never duplicate global state across every
slot. Realized volatility uses only timestamped prices at or before the current
snapshot, annualizes by actual elapsed time, and carries a coverage fraction so
zero history cannot masquerade as zero volatility.

The trainer uses stateful factorized per-slot PPO ratios, GAE, policy/value
clipping, target-KL stopping, entropy regularization, and gradient clipping.
Policy heads initialize with a trainable hold-logit prior because a near-uniform
33-row categorical policy creates pathological turnover before learning begins.
The default entropy coefficient is `1e-4`, calibrated to return-scale rewards.
Do not hard-cap active rows or post-process sampled actions without deriving the
matching joint likelihood; that would invalidate PPO ratios. Preserve requested
option/underlying order counts and action-density metrics in every episode.
Rollouts and deterministic evaluation must carry the actual GRU/LSTM hidden
state one snapshot at a time. PPO minibatches are composed from contiguous
truncated-backpropagation chunks initialized with the old policy's causal
hidden state. `sequence_length` is the gradient chunk bound. Do not restore
left-zero-padded sliding windows: they create fictitious history, discard
state older than the window, and repeat recurrent work at every inference step.
Training rollouts default to seeded, uniformly sampled windows of at most 128
transitions inside the training dataset. Persist each start/end index in episode
metrics. Random starts may never cross the supplied partition, affect validation
or test evaluation, or replace the deterministic full-partition selection run.
`train-demo` model selection is deterministic but in-sample and must remain
labeled `in_sample_research_demo`. When `selection_env` is supplied, selection
must use only that validation environment and be labeled
`validation_research_demo`. Checkpoints must load with PyTorch
`weights_only=True`; never weaken this to unrestricted pickle loading.

The graph encoder uses only valid option slots, symmetric nearest-neighbor edges,
and self edges. Padded contracts must neither send nor receive messages. Keep the
dense implementation while the slot count is small; require profiling evidence
before adding a graph-framework dependency.

Evaluation changes must preserve chronological order. `walk_forward_splits`
uses half-open train/validation/test ranges with explicit embargoes and may
return no folds when history is insufficient. Never relax requested sizes to
manufacture a result. `evaluate_cost_stress` must run identical policy logic
under each scenario; default stress doubles both executable spread and
commission. Episode reports retain return, drawdown, volatility/downside,
turnover, costs, execution quality, and final/peak Greek exposure diagnostics.
Held-out statistical comparisons pair agent and baseline returns by exact
arrival timestamp and test seed, then use circular moving blocks. Do not pool
duplicate deterministic seeds as independent observations. The default minimum
is 20 transitions; shorter paths must report `insufficient_history` and null
intervals. Bootstrap results are post-selection evidence only and may never
affect features, hyperparameters, early stopping, or checkpoint choice.
The long-volatility baseline is causal: require its configured realized-vol
coverage and edge over front ATM IV, select feasible front-ATM positive- and
negative-Delta legs, enter once, then hedge residual Delta on later snapshots.
Persist its horizon, threshold, coverage, and quantity in each fold. Do not call
it a straddle when the nearest feasible call and put have different strikes,
and do not infer a short-volatility result from a long-only environment.

Underlying fills use the captured spot with explicit synthetic slippage and
per-share commission because the current CSV has no underlying bid/ask. Keep
the assumption visible. Short shares are capped but the demo does not model
borrow, margin, dividends, or funding, so results remain research-only.

The recurrent policy has `K+1` action slots but the graph encoder still has
exactly `K` contract nodes. Keep `RecurrentConfig.slot_count` equal to
`env.slot_count` and `action_slot_count` equal to `env.action_shape[0]`.

`run_walk_forward_training` is the executable research boundary. For every
fold, train only on `train`, choose and restore weights only from `validation`,
then evaluate `test`. Architecture tournaments must give candidates the same
fold and seed, rank validation reward only, and break exact ties by parameter
count then stable model ID. Instantiate the test environment only after the
winner is fixed, save only the winning checkpoint, and never attach test metrics
to losing candidates. The test range may populate reports and provenance only
after selection; it must never affect features, hyperparameters, early stopping,
or checkpoint choice. Persist all candidate configs, validation scores,
parameter counts, all three dataset fingerprints, and exact split indices. An
insufficient dataset is a hard failure, not permission to shrink partitions.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
uv run --extra dev python -m pytest -q
python -c 'from pathlib import Path; from trading_bot.training import OptionsEnv; print(OptionsEnv.from_directory(Path("data"), "AAPL").manifest.fingerprint)'
collect-options --once --expirations 3
collect-options
streamlit run src/trading_bot/interface/app.py
option-chain AAPL
train-demo --symbol AAPL --encoder graph --kind hybrid --episodes 25
train-walk-forward --symbol AAPL --min-train-size 500 --validation-size 100 --test-size 100 --embargo 8 --candidate flat:gru --candidate graph:hybrid
```

The collector defaults to three expirations per ticker, one cycle every 900
seconds, and a one-second delay between tickers. Use `--expirations 1` when
remote request latency matters more than term-structure coverage.

## Verification expectations

- Unit tests must remain network-free and deterministic.
- Changes to Greeks require a published numeric test vector and unit checks.
- Changes to persistence require an append/migration test.
- Paper execution changes require isolated tests for cash, position quantity,
  ledger writes, insufficient funds, and overselling.
- Market-data changes require a one-cycle live smoke test when network access is
  available; report success and failure counts.
- Interface changes require checking that at least one real ticker CSV renders.
- Before any new live-execution package can place orders, require explicit user
  approval plus risk limits, idempotent client order IDs, paper/live environment
  separation, and kill-switch tests.

## Performance policy

Profile before introducing native code. The current collection cycle is limited
by remote HTTP latency, so Rust or C++ would not materially improve throughput.
Keep orchestration and deep-learning integration in Python. If profiling later
finds a CPU-bound hot path, prefer an isolated Rust extension with benchmarks
and a Python fallback; use C++ only when required by an existing library.

## Known limitations and next decisions

- The top-50 universe is a dated snapshot and must be refreshed deliberately.
- Collection defaults to the nearest three listed expirations; this is still
  sparse relative to a licensed full-surface historical feed.
- `^IRX / 100` is a quoted 13-week bill-yield approximation, not a
  maturity-matched zero curve.
- Dividend yield falls back to zero when Yahoo omits it.
- CSV storage is appropriate for this stage; reassess Parquet or a database only
  when measured data volume or query needs justify it.
- The paper ledger uses SQLite because account updates require transactions.
- The current local AAPL sample is sufficient for integration smoke tests, not
  statistical training claims. Follow `docs/research-roadmap.md` gates before
  treating a model improvement as alpha.
