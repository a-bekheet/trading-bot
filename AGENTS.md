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
  LSTM, concatenated hybrid, or gated-mixture actor-critic with flat,
  flattened-graph, graph-set, or masked attention-set contract
  encoding. `trainer.py`
  owns factorized and exact single-leg-joint PPO/GAE optimization,
  deterministic evaluation, safe
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
5. `market_data.collector` appends the enriched rows to `data/<TICKER>.csv`
   only when the raw quote/rate surface materially changes. It also holds a
   per-output-directory process lock and atomically updates
   `data/_collector_status.json` throughout each cycle.
6. `market_data.status` validates heartbeat freshness, cycle failures, and
   continuous-process liveness. `market_data.service` manages the optional
   macOS LaunchAgent used for unattended restart-on-failure collection.
7. `interface.app` displays the latest saved snapshot; it never fetches markets.
8. `execution.paper_broker` stores fake cash, long positions, and fills in
   `data/paper_portfolio.db`; `execution.valuation` marks positions from CSVs.
9. `training.env.OptionsEnv` exposes the current CSVs through a Gymnasium-style
   `reset`/`step` API. It is `research_demo` only and must not be used as a
   historical-performance benchmark.

A failure for one ticker must be logged without stopping the other tickers. A
risk-free-rate failure stops that cycle because silently using a stale or fixed
rate would make the calculated data misleading.

## CSV contract

CSV files are append-only snapshots. Do not remove or reorder columns without a
migration. `collector._migrate_csv` upgrades older files atomically before an
append. Important model/input columns are:

Snapshot identity excludes `collectedAt`, time-to-expiry, and derived Greeks.
Do not let recomputation against unchanged stale quotes create a new market
state. Identity must retain raw quote fields, contract membership, spot,
dividend yield, and risk-free-rate inputs so any material change is persisted.
The persisted Greek-model identifier is also material so a deliberate model
version change can coexist with the older calculation.
The training loader must apply the same rule to consecutive legacy snapshots;
after filtering, causal gaps are measured from the last materially distinct
surface. Do not delete old CSV rows as part of this filter.

- `collectedAt`, `symbol`, `expiration`, `optionType`
- `underlyingPrice`, `riskFreeRate`, `riskFreeRateSource`, `dividendYield`
- `timeToExpiryYears`, `impliedVolatility`, `greekModel`
- `delta`, `gamma`, `theta`, `vega`

Training-time surface features include forward log-moneyness, extrinsic value,
ATM IV/skew, ATM term slope, put-call IV spread, and parity residual. They are
derived within a single captured timestamp and are not persisted into the raw
CSV contract. The market vector also contains front-expiry ATM IV, executable
25-delta risk reversal and butterfly, ATM/wing/quote/Greek coverage, and the
ATM-IV difference from causal 4/16-snapshot realized volatility. Wing factors
must exclude zero-bid, crossed, or otherwise unexecutable quotes. Keep global
regime and quality features out of each contract node.

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
  Delta/Gamma/Theta/Vega, the underlying-share position, reserved cash
  collateral, and reserved covered shares. Total Delta must include the shares.
- Every valid option row exposes current `positionQuantity`,
  `positionAveragePrice`, and `positionUnrealizedReturn`. These are causal agent
  state, not collector columns. Quantity and average price come from the
  environment ledger; unrealized return uses the current executable sell price.
  Unheld rows are zero. Route all three through the named `position_state`
  ablation and never infer holdings only from aggregate Greeks or action masks.
- `Observation.action_mask` has `K+1` rows: `K` option slots and one final
  underlying slot. Action `0` means hold; `1..Q` are buy buckets and
  `Q+1..2Q` are sell buckets. Legacy length-`K` arrays imply underlying hold.
- Option buckets represent contracts. Underlying buckets represent multiples
  of `underlying_lot_size`, may open bounded shorts, and must obey cash, Delta,
  and `max_abs_underlying_shares` constraints.
- Option positions are long-only unless
  `allow_collateralized_option_shorts=True`. The opt-in mode permits covered
  calls and cash-secured puts only: reserve 100 owned shares per short call and
  `strike * 100` cash per short put. Never reuse reserved collateral or permit
  naked option exposure.
- A short option requires a recognized call/put type, finite positive strike,
  and parseable expiration. It must remain inside the underlying-share bound
  after possible put assignment.
- Masks are generated from the pre-step state and include quote validity,
  fee-adjusted affordability, signed position transitions, collateral, and
  optional absolute Greek limits.
- Multiple orders in one action are revalidated sequentially so cash cannot go
  negative even when individual pre-step actions were affordable.
- Execute the underlying leg first, then option slots in ascending order. Masks
  describe the pre-step state; every leg must still be revalidated against the
  running cash and Greek state.
- `info` retains executions, invalid-action count, P&L, fees, trade notional,
  reward components, and slot retention/churn diagnostics.
- `reward_components` must sum to the returned scalar reward. Gross P&L includes
  spread/mark effects, while commission and invalid-action penalties are
  separate components. Optional downside shaping is the negative coefficient
  times the negative part of the current net P&L return. Optional drawdown
  shaping charges only the increase in running maximum NAV drawdown, not the
  full current drawdown on every step. This keeps the signal path-causal and
  makes its episode sum equal negative coefficient times maximum drawdown.
- Reset peak NAV and maximum drawdown at every episode boundary. Persist both
  reward coefficients in the environment manifest, expose current/maximum
  drawdown and drawdown increase in `info.path_risk`, and retain all five reward
  components in rollout metrics. Coefficients must be finite, non-negative, and
  default to zero so old training behavior and policy inference remain intact.
- A surviving contract keeps its exact slot unless a reappearing held position
  needs visibility to remain sellable. A missing contract vacates only its own
  slot; ordinary replacements never shift other identities. Currently visible
  held contracts take priority when home-slot histories collide.
- Initial and replacement slots are surface-stratified: one near-ATM contract from each
  expiration/type group is selected before deeper strikes, with spread and open
  interest as deterministic tie-breakers.
- `step()` must execute against the exact cached slots and mask returned by the
  preceding observation. Under stable assignment, only vacant next-state slots
  may rank replacements. `ranked` is an explicit comparison mode.
- `slotContinuity` is zero on reset/replacement/padding and one only when the
  same contract occupied that index in the preceding observation. Persist
  retained, changed, comparable, and churn counts through training artifacts.
- An episode with `N` snapshots has exactly `N-1` tradable transitions and
  truncates on arrival at the last snapshot; never permit an unmarkable extra
  fill at the terminal timestamp.
- Multi-order actions revalidate Greek budgets sequentially. If market drift
  puts a portfolio over a limit, actions that reduce the absolute exposure must
  remain permitted even when they do not immediately return below the limit.
- Settle expiry only on the first observed date strictly after the listed
  expiration. Physically assign in-the-money short equity calls/puts at the
  strike using 100 shares per contract; expire out-of-the-money positions
  worthless. Long intrinsic value is cash-settled in this demo. Do not imply
  that early assignment, exercise notices, dividends, or corporate actions are
  modeled.

The current environment is a deterministic accounting and API scaffold, not a
historical simulator. Do not add a `historical` mode until the data manifest
contains point-in-time all-expiry quotes, depth/quality fields, lifecycle data,
and an explicit source/license.

Engineered features must be causal: current rows may use current cross-sectional
values and the immediately prior snapshot, but never future rows. Sequence
windows are chronological and unpadded. GRU/LSTM/hybrid/mixture code is optional
(`.[ml]`) and must preserve a no-PyTorch collector path. Checkpoints must retain
the environment fingerprint, full model and training configuration, metrics,
and the `research_demo` label.

An environment manifest hashes only its selected ticker CSV. Do not broaden the
fingerprint to unrelated ticker files: doing so adds startup I/O and invalidates
otherwise reproducible experiments whenever the collector updates another
symbol.

`sequence.observation_vector` is the versioned policy boundary. Under
`dimensionless.v13`, price-like fields are relative to spot, contract Gamma is
the Delta change for a 10% spot move, portfolio and Greek exposures are relative
to NAV/deployed capital, underlying shares and covered-share reserves are
represented by NAV weight, cash collateral is NAV-scaled, DTE is expressed in
years, held quantity and unrealized return are signed-log compressed, and
heavy-tailed fields are compressed and clipped. Raw
volume and open interest
must not be reintroduced beside their log features without ablation evidence.
Any transform change requires a new feature-vector schema, scale-invariance and
finite-value tests, and a checkpoint-schema bump; old weights must never be
silently loaded against a changed feature layout.
Transform-only optimizations must preserve the exact versioned numerical
contract, including NaN-to-zero and infinity clipping at plus/minus 10. Prefer
batched column operations and direct float32 output assembly; do not restore
generic multi-pass sanitation without a matched benchmark and edge-case tests.

Per-contract dynamics must match `contractSymbol` against the immediately prior
snapshot. Mid-quote return and relative-spread change require positive,
non-crossed bid/ask quotes at both endpoints; IV change also requires positive
IV at both endpoints. Missing or invalid history must produce zero change and a
zero coverage bit. Do not make these quote-derived signals depend on last-trade
price, and do not substitute unsigned volume or model-derived Delta changes for
signed order flow. Keep the five fields in the named `contract_dynamics`
ablation so their per-slot cost and lift remain measurable.

Market-level term-structure slope/curvature must use executable near-ATM points
from the current snapshot only. Surface-change features may subtract only the
immediately prior engineered snapshot. Persist separate coverage for ATM-level,
wing, term-slope, term-curvature, and prior/current change availability; a
missing expiration or wing must never become an unexplained numeric zero.
Keep these factors once in `MARKET_FEATURES`, not repeated per contract, and
route `term_structure` and `surface_dynamics` through named walk-forward
ablations before attributing lift.

Underlying return, 4/16-snapshot cumulative log returns, and 4/16-snapshot
realized-volatility estimates are market features, not contract features;
never duplicate global state across every slot. Both multi-snapshot summaries
use only valid timestamped prices at or before the current snapshot and share
the same explicit coverage. Realized volatility annualizes by actual elapsed
time. Partial history must never masquerade as a complete trend or volatility
estimate. Route only the cumulative-return values through the named
`price_trend` ablation so their marginal value is separable from volatility and
history availability. Keep the shared history-coverage scalars unmasked in both
`price_trend` and `volatility_regime` comparisons; duplicating coverage inputs
or hiding availability while leaving a dependent signal would be misleading.

`snapshotGapSeconds` is the positive elapsed wall time from the immediately
prior snapshot and is log-compressed at the policy boundary. Pair it with
`snapshotGapCoverage`; the first snapshot, missing/malformed timestamps, and
non-increasing timestamps must remain neutral and uncovered. Keep the pair once
in the market vector and route it through the named `time_context` ablation.
Auxiliary horizons remain snapshot counts rather than elapsed-time horizons.

Front 25-delta risk reversal is call IV minus put IV; butterfly is mean wing IV
minus executable ATM IV. Both are computed cross-sectionally from the current
snapshot only and require explicit coverage. Never impute a missing wing with
zero or use an unexecutable quote to manufacture a surface factor. A wing must
be within 0.15 Delta of its target and ATM must be within 0.10 absolute forward
log-moneyness; otherwise reduce coverage and leave the factor neutral.

The trainer supports stateful PPO and Monte-Carlo REINFORCE with a learned value
baseline. The default factorized decoder uses per-slot likelihood ratios. The
optional `single_leg` decoder uses one exact joint categorical over global hold
plus every feasible row/non-hold-action pair, structurally permitting at most
one order per snapshot. It is an explicit lower-complexity action-space
candidate, not post-processing or an inferred execution cap. PPO uses GAE,
policy/value clipping, and target-KL stopping. By default, both algorithms use
`gamma ** (elapsed_seconds / discount_reference_seconds)` for continuation;
PPO applies the same physical-time composition to GAE lambda. Never apply one
fixed discount per irregular snapshot without declaring
`time_aware_discounting=False`. Persist the reference interval, observed
transition durations, and effective gamma/lambda ranges; use the matched
fixed-step discount ablation for comparison. REINFORCE uses discounted trajectory returns,
bootstraps bounded nonterminal rollouts only, and performs exactly one on-policy
optimizer pass; never reuse its trajectory across epochs. Both algorithms use
entropy regularization, gradient clipping, and contiguous recurrent chunks.
Policy heads initialize with a trainable hold-logit prior because a near-uniform
33-row categorical policy creates pathological turnover before learning begins.
The default entropy coefficient is `1e-4`, calibrated to return-scale rewards.
Do not hard-cap active rows or post-process sampled actions without deriving the
matching joint likelihood; that would invalidate PPO ratios. Preserve requested
option/underlying order counts and action-density metrics in every episode.
Persist the decoder and number of likelihood factors. `single_leg` must encode
masks before sampling, decode to the existing environment action array, reject
training actions with multiple non-hold rows, and retain one scalar log
probability/entropy per step. It cannot express same-snapshot spreads or hedges,
so keep factorized decoding available and select between them on validation.
ATM-IV and volatility-premium rolling z-scores filter valid values from the 16
strictly prior snapshots. Require four prior values, use population mean/standard
deviation, clip to ±8, and emit zero when current coverage is absent or prior
dispersion is unusable. Persist separate prior-history coverage. Engineer the
full chronological ticker dataset before splitting so validation/test may use
only genuinely earlier market context; `subset()` must preserve those causal
precomputed values. Never refit normalization inside a validation or test slice
and never let its future rows affect an earlier score. The named
`volatility_normalization` ablation masks only both z-scores and preserves their
coverage fields.
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
Before a sampled nonzero start, warm recurrent state on at most the configured
number of immediately preceding observations. Advance that prefix only with
hold actions so the zero-position random-window contract remains coherent;
exclude its rewards and gradients, batch its recurrent evaluation, and detach
the resulting hidden state before the first optimized transition. Never cross
the training boundary or use validation/test context. Persist requested and
actual burn-in lengths and prefix start indices. Default to eight steps, allow
zero explicitly, and expose a matched validation-only disabled ablation. On a
validation tie, prefer the context-aware candidate over the disabled ablation.
Selection patience counts evaluated checkpoints, not episodes, and may use only
the configured in-sample or validation selection environment. Persist whether
each run stopped early, its completed episode count, patience, and minimum
improvement. `None` disables patience; test results may never reset it or resume
training.

`kind="hybrid"` concatenates full-width GRU and LSTM outputs and is a distinct
checkpoint contract. `kind="mixture"` runs the same causal experts but applies
one state-dependent scalar sigmoid gate per timestamp and convex-combines their
outputs at the original hidden width. Initialize that gate to an exact 0.5
blend, preserve both experts' hidden states, and never reinterpret an old hybrid
checkpoint as a mixture. The mixture's smaller heads do not imply lower latency;
keep both parameter and batch-one inference measurements in tournaments.
The declared selection score is total reward minus non-negative configured
coefficients times maximum drawdown, downside deviation, and turnover. Use this
one score consistently for best-episode restoration, patience, architecture,
algorithm, and feature-ablation ranking. Persist raw reward, every component,
and all coefficients. Defaults are zero; never tune coefficients from test.
Training-time drawdown and downside reward coefficients are independent from
these validation-selection coefficients. Experiments may enable either layer
or both, but must declare and persist the choice; never silently treat selection
penalties as training reward or hide intentional double risk penalization.
`train-demo` model selection is deterministic but in-sample and must remain
labeled `in_sample_research_demo`. When `selection_env` is supplied, selection
must use only that validation environment and be labeled
`validation_research_demo`. Checkpoints must load with PyTorch
`weights_only=True`; never weaken this to unrestricted pickle loading.

Auxiliary dynamics prediction is optional representation supervision, not
reward shaping. At step t the recurrent state may predict cumulative
dimensionless market changes at configured positive, increasing snapshot
horizons only when both endpoints were observed inside the same training
rollout and partition. Contract targets must match identifiers at both
endpoints, require positive non-crossed bid/ask quotes, use a permutation-
invariant cross-sectional median, require at least 50% current-cross-section
coverage, and mask unavailable IV separately. Never
derive them from last trade or slot position. Mask incomplete rollout tails and
require point-in-time coverage at both endpoints for sparse IV-surface targets;
never teach a missing wing, expiration, or contract match as zero. Keep the
prediction head out of
`forward`, `sample_action`, streaming evaluation, and latency benchmarks so it
cannot alter PPO likelihoods or deployment latency. Persist its targets,
snapshot horizons, coefficient, masked loss/MAE, and nested horizon/target
coverage in checkpoints. Any claimed benefit requires both the matched
one-step `--auxiliary-horizon-ablation` and zero-coefficient
`--auxiliary-ablation` comparisons on validation before the single selected
policy reaches test. Snapshot horizons are not elapsed-time horizons.

Shared-policy training accepts unique-symbol environment pools with identical
feature and action layouts. Schedule seeded shuffled ticker cycles and require
at least one episode per ticker. Reset recurrent state, portfolio state, and
rollout bounds at every ticker boundary; never concatenate trajectories or
advantages across symbols. Evaluate each selection environment with a fresh
policy state, persist every per-symbol report and environment fingerprint, and
label multi-ticker scopes `in_sample_universe_research_demo` or
`validation_universe_research_demo`. Aggregate per-ticker selection scores as
`(1-w) * mean + w * worst - d * std`, using only predeclared worst-ticker weight
and dispersion penalty. A symbol embedding is not part of the current contract;
the dimensionless shared policy must remain usable on unseen tickers.

`run_universe_walk_forward_training` is the shared-policy research boundary.
Apply identical ordinal split indices to each ticker, but additionally require
global wall-clock separation: maximum train arrival below minimum validation
arrival and maximum validation arrival below minimum test arrival. Persist
these four timestamps, every per-symbol partition fingerprint, source length,
common length, and ignored tail count. Train candidates only on the training
pool, aggregate only per-symbol validation reports, and instantiate all test
environments only after architecture, latency eligibility, early stopping, and
checkpoint restoration are fixed. Evaluate the winner independently per
ticker; do not pool ticker paths in the moving-block bootstrap. Any aggregate
held-out universe result must remain labeled descriptive, never inferential.

The graph encoder uses only valid option slots, symmetric nearest-neighbor edges,
and self edges. Padded contracts must neither send nor receive messages. Keep the
dense implementation while the slot count is small; require profiling evidence
before adding a graph-framework dependency.

The `graph_set` encoder is the slot-agnostic graph policy. Its temporal input
uses masked mean/max pooling plus valid-node coverage, every option row uses the
same contract scorer, and the underlying row uses a separate scorer. Permuting
valid contract slots must permute their option logits while leaving the value,
underlying logits, and auxiliary predictions unchanged. Padded node content must
remain inert. Preserve these invariants with tests whenever graph construction,
pooling, action layout, or recurrent state changes. Do not assume `graph_set` is
always faster: enforce a predeclared latency budget when speed affects selection.
For `single_leg`, treat each option's contiguous non-hold joint-logit block as
the equivariant row output; global hold and the underlying block stay invariant.
With `graph_neighbors=0`, construct only pointwise self layers: do not allocate
adjacency/degree tensors, call graph multiplication, or register dead neighbor
weights. This self-only Deep Sets configuration must remain selectable alongside
the full neighbor graph in one validation-only tournament.

The `attention_set` encoder is the learned-relation counterpart. It must have no
slot or positional embedding: masked self-attention operates only among valid
contracts, followed by the same invariant pooling and shared contract scorer.
Permuting valid contracts must permute option logits and leave value,
underlying logits, and auxiliary predictions unchanged for both action
decoders. Padded node values must be inert, and an all-invalid surface must stay
finite. `attention_heads` is positive, divides `graph_hidden_size`, and is
persisted in checkpoint and walk-forward artifacts; `graph_neighbors` is
normalized to zero because it has no meaning for learned global attention.
Keep `graph_set` with zero and fixed neighbors as explicit baselines, and never
promote attention from reconstruction literature or a training-loss result.

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
and do not infer a short-volatility result merely because the opt-in liability
surface exists.

The `cash_secured_short_put_delta_hedge` baseline is the required carry hurdle.
Require covered realized-volatility and front-ATM-IV inputs, enter only when IV
exceeds realized volatility by its declared edge, sell one feasible
front-expiry ATM put, then reduce signed Delta with underlying actions. It must
remain a no-op when collateralized shorts are disabled. Persist its horizon,
coverage, edge, and quantity in every fold; include it in timestamp-paired
held-out comparisons and run fresh policy state under both base and doubled
costs. Never bypass the environment's fills, collateral, Greek limits, or
assignment lifecycle.

The underlying-trend baseline is also causal: require configured price-history
coverage, derive direction from the selected 4/16-snapshot cumulative log
return, and rebalance toward a bounded share target through the environment's
feasible underlying actions. Never buy again when already at target. Persist
its window, threshold, coverage, and quantity in every fold and include it in
the same timestamp-paired held-out comparisons as other baselines.

Underlying fills use the captured spot with explicit synthetic slippage and
per-share commission because the current CSV has no underlying bid/ask. Keep
the assumption visible. Short shares are capped but the demo does not model
borrow, margin, dividends, or funding, so results remain research-only.

The recurrent policy has `K+1` action slots but the graph encoder still has
exactly `K` contract nodes. Keep `RecurrentConfig.slot_count` equal to
`env.slot_count` and `action_slot_count` equal to `env.action_shape[0]`.

`run_walk_forward_training` is the executable research boundary. For every
fold, train only on `train`, choose and restore weights only from `validation`,
then evaluate `test`. Architecture tournaments must give PPO and REINFORCE
candidates the same fold and seed, rank the declared validation selection score
only, and break exact ties by parameter count, active input count, optimizer
updates, then stable model ID. Instantiate the test environment
only after the winner is fixed, save only the winning checkpoint, and never
attach test metrics to losing candidates. The test range may populate reports
and provenance only after selection; it must never affect features,
hyperparameters, early stopping,
or checkpoint choice. Persist all candidate configs, validation scores,
parameter counts, all three dataset fingerprints, and exact split indices. An
insufficient dataset is a hard failure, not permission to shrink partitions.
Candidate episode budgets may end early through the same validation-only
selection patience, and the comparison artifact must expose completed episodes
so compute differences are auditable.

When architecture candidates declare a parameter budget, treat the requested
recurrent hidden size as a cap and choose the widest size whose exact trainable
parameter count does not exceed that budget. Resolve capacity from the training
environment's observation/action layout only, cache the result across folds,
and fail if even hidden size one is too large. Validation and test values must
not participate in capacity resolution. Persist the requested spec, resolved
recurrent config, exact parameter count, and budget headroom, and verify the
trained model matches that count before selection.

Benchmark each candidate's streaming batch-one inference on a training-range
observation after training. Record median, p95, and mean latency together with
device, PyTorch version, thread count, warm-up iterations, and measured
iterations. Restore the model's prior train/eval mode afterward. Timing is
machine-specific diagnostic evidence and must not silently enter validation
ranking, tie-breaking, or held-out selection. An explicit predeclared median
latency ceiling may filter deployment-ineligible candidates before winner
selection; preserve their validation evidence and exclusion reason, allow only
eligible candidates to reach test, and fail if none remain. With no ceiling,
timing must remain informational.

Feature-removal candidates must use the named, non-overlapping groups in
`sequence.FEATURE_ABLATION_GROUPS`. Apply masks inside the recurrent model after
the versioned transform and persist exact flattened indices in
`RecurrentConfig`; external preprocessing would make restored checkpoints
ambiguous. CLI ablations retain a matched full-feature candidate, report
validation score and raw-reward lift versus it, and obey the same one-winner
test boundary.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
uv run --extra dev python -m pytest -q
python -c 'from pathlib import Path; from trading_bot.training import OptionsEnv; print(OptionsEnv.from_directory(Path("data"), "AAPL").manifest.fingerprint)'
collect-options --once --expirations 3
collect-options
collector-status
collector-status --json
collector-service install
streamlit run src/trading_bot/interface/app.py
option-chain AAPL
train-demo --symbol AAPL --encoder graph --kind hybrid --episodes 25
train-demo --symbol AAPL --encoder graph_set --kind hybrid --episodes 25
train-demo --symbol AAPL --encoder graph_set --kind mixture --episodes 25
train-demo --symbol AAPL --encoder attention_set --attention-heads 4 --kind hybrid --episodes 25
train-demo --symbol AAPL --allow-collateralized-option-shorts --episodes 25
train-walk-forward --symbol AAPL --min-train-size 500 --validation-size 100 --test-size 100 --embargo 8 --candidate flat:gru --candidate graph_set:hybrid:ppo:0 --candidate attention_set:hybrid:ppo
train-walk-forward --symbol AAPL --allow-collateralized-option-shorts --short-volatility-min-edge 0.02 --min-train-size 500 --validation-size 100 --test-size 100
```

The collector defaults to three expirations per ticker, one cycle every 900
seconds, and a one-second delay between tickers. Use `--expirations 1` when
remote request latency matters more than term-structure coverage.

On macOS, `collector-service install` writes a user LaunchAgent with absolute
repository, interpreter, output, and log paths, then bootstraps it. It must use
the virtual-environment launcher path without resolving its interpreter
symlink. The LaunchAgent restarts nonzero exits; the collector's advisory lock
prevents duplicate writers. `collector-service uninstall` is intentionally an
explicit operation.

## Verification expectations

- Unit tests must remain network-free and deterministic.
- Changes to Greeks require a published numeric test vector and unit checks.
- Changes to persistence require an append/migration test.
- Collector persistence changes require changed/unchanged identity tests,
  process-lock coverage, and heartbeat/status health tests.
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
The training environment must keep snapshot rows as lightweight column-array
views through slot assignment, valuation, masks, and fills. Do not reintroduce
`DataFrame.iterrows()`, per-contract `Series` construction, or repeated indexed
DataFrames in the transition loop. Preserve first-occurrence duplicate-symbol
semantics and verify both accounting equivalence and full-episode latency after
changing this path.

## Known limitations and next decisions

- The top-50 universe is a dated snapshot and must be refreshed deliberately.
- Data gathered while the underlying raw surface is unchanged is intentionally
  represented by one state plus the next observed time gap, not repeated Greek
  recomputations. A single retained state is insufficient for training even if
  the source CSV contains many stale capture timestamps.
- Collection defaults to the nearest three listed expirations; this is still
  sparse relative to a licensed full-surface historical feed.
- `^IRX / 100` is a quoted 13-week bill-yield approximation, not a
  maturity-matched zero curve.
- Dividend yield falls back to zero when Yahoo omits it.
- CSV storage is appropriate for this stage; reassess Parquet or a database only
  when measured data volume or query needs justify it.
- The paper ledger uses SQLite because account updates require transactions.
- The RL short-option lifecycle has no early assignment, exercise instruction,
  dividend, or corporate-action model. Post-expiry physical assignment is a
  conservative deterministic research approximation, not broker fidelity.
- The current local AAPL sample is sufficient for integration smoke tests, not
  statistical training claims. Follow `docs/research-roadmap.md` gates before
  treating a model improvement as alpha.
