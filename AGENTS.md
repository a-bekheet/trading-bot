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
  owns research-demo optimization and checkpoint provenance. Both stay outside
  ordinary collector imports.
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
CSV contract.

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
- Action `0` means hold; `1..Q` means buy `Q` buckets; `Q+1..2Q` means sell.
- Masks are generated from the pre-step state and include quote validity,
  fee-adjusted affordability, and held quantity.
- Multiple orders in one action are revalidated sequentially so cash cannot go
  negative even when individual pre-step actions were affordable.
- `info` retains executions, invalid-action count, P&L, fees, trade notional,
  and reward components.
- A missing contract is never silently transferred to another slot.
- Non-held slots are surface-stratified: one near-ATM contract from each
  expiration/type group is selected before deeper strikes, with spread and open
  interest as deterministic tie-breakers.

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

The graph encoder uses only valid option slots, symmetric nearest-neighbor edges,
and self edges. Padded contracts must neither send nor receive messages. Keep the
dense implementation while the slot count is small; require profiling evidence
before adding a graph-framework dependency.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -c 'from pathlib import Path; from trading_bot.training import OptionsEnv; print(OptionsEnv.from_directory(Path("data"), "AAPL").manifest.fingerprint)'
collect-options --once --expirations 3
collect-options
streamlit run src/trading_bot/interface/app.py
option-chain AAPL
train-demo --symbol AAPL --encoder graph --kind hybrid --episodes 25
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
- Only the nearest listed expiration is collected.
- `^IRX / 100` is a quoted 13-week bill-yield approximation, not a
  maturity-matched zero curve.
- Dividend yield falls back to zero when Yahoo omits it.
- CSV storage is appropriate for this stage; reassess Parquet or a database only
  when measured data volume or query needs justify it.
- The paper ledger uses SQLite because account updates require transactions.
- Define the broker, paper-trading environment, model target, and backtest rules
  before adding execution or deep-learning workflows.
