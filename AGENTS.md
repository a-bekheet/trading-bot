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
- `src/trading_bot/execution/`: the manual paper broker, portfolio valuation,
  isolated agent store, selected-checkpoint runtime, change-aware watcher, and
  its macOS service wrapper. It must remain independent of any future
  live-broker adapter.
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
  modules stay outside ordinary collector imports. `arena.py` applies one fixed
  GRU/LSTM/gated-mixture comparison contract across independent tickers, writes
  one normal walk-forward summary per ticker, and records partial failures in a
  small arena index; it must not merge ticker timelines or peek across tests.
  `arena_watch.py` owns readiness polling and once-per-session invocation;
  `arena_service.py` is only the macOS LaunchAgent lifecycle wrapper.
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
3. `market_data.benchmark` fetches one provider-timestamped SPY one-minute
   close per cycle by default. `--benchmark-symbol` may change the declared
   proxy. Benchmark failure is optional and must not stop ticker collection;
   missing context remains zero with zero coverage.
4. `market_data.option_chain` retrieves a configurable number of expirations,
   calls, puts, underlying price, dividend yield, and the provider's explicit
   market-session state. Collection defaults to
   three expirations; one is the low-latency mode and zero in the CLI means all.
   Yahoo's option payload reports dividend yield in percentage units, so the
   adapter divides it by 100.
5. `analytics.greeks` calculates Black-Scholes-Merton Greeks.
6. `market_data.collector` appends the enriched rows to `data/<TICKER>.csv`
   only when the raw quote/rate surface materially changes. It also holds a
   per-output-directory process lock and atomically updates
   `data/_collector_status.json` throughout each cycle.
7. `market_data.status` validates heartbeat freshness, cycle failures, and
   continuous-process liveness. `market_data.service` manages the optional
   macOS LaunchAgent used for unattended restart-on-failure collection.
8. `interface.app` displays saved walk-forward agent evidence and the latest
   market snapshot; it never fetches markets. Agent results come only from
   JSON artifacts under `data/agent_runs/` or `data/models/walk-forward/`.
   Keep the five-task information architecture stable: Overview for operating
   health, Agent Desk for one policy, Trade for the order ticket and chain,
   Portfolio for positions and fills, and Research for dense training evidence.
   Operator tabs should show conclusions and next actions before raw tables;
   detailed readiness, candidate fleets, ablations, and provenance belong in
   Research or expanders.
   Candidate rankings are validation-only; only the fixed winner is labeled
   held out. The UI must keep exploratory sample-size and legacy execution-
   provenance warnings visible. It disables paper orders when the provider
   explicitly reports a non-regular session and warns when legacy data has no
   session-state coverage.
9. `execution.paper_broker` stores fake cash, long positions, and fills in
   `data/paper_portfolio.db`; `execution.valuation` marks positions from CSVs.
10. `execution.agent_runtime` restores each newest selected checkpoint into an
    isolated account in `data/agent_paper.db`. It processes only eligible
    post-evaluation snapshots, atomically binds portfolio and recurrent state to
    the checkpoint SHA-256, and substitutes HOLD unless validation activated
    the policy. `agent_watch` invokes it only when data or run artifacts change;
    `agent_service` is the macOS LaunchAgent wrapper.
11. `training.env.OptionsEnv` exposes the current CSVs through a Gymnasium-style
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
The declared benchmark symbol, price, provider timestamp, and sources are also
material. Fetch the benchmark once per cycle, never once per ticker, and never
persist an untimestamped fallback merely to manufacture market context.
The persisted Greek-model identifier is also material so a deliberate model
version change can coexist with the older calculation. Provider `marketState`
is material: a regular-to-closed transition must be stored even when quotes do
not move. Select `underlyingPrice` and `underlyingQuoteTime` as a matching
provider session pair and persist both source fields; never attach a pre/post
timestamp to a regular price. A provider quote-time advance is material even
when price is unchanged. Never infer session state from local clock time,
weekdays, or a holiday calendar; absent and unrecognized values migrate to the
explicit `UNKNOWN` fallback.
The training loader must apply the same rule to consecutive legacy snapshots;
after filtering, causal gaps are measured from the last materially distinct
surface. Do not delete old CSV rows as part of this filter.

- `collectedAt`, `symbol`, `expiration`, `optionType`
- `underlyingPrice`, `underlyingPriceSource`, `underlyingQuoteTime`,
  `underlyingQuoteTimeSource`, `marketState`, `riskFreeRate`,
  `riskFreeRateSource`, `dividendYield`
- `benchmarkSymbol`, `benchmarkPrice`, `benchmarkPriceSource`,
  `benchmarkQuoteTime`, `benchmarkQuoteTimeSource`
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

An executable option quote is defined by finite, positive, non-crossed bid and
ask prices. `lastPrice` is not part of that predicate and must not gate fills,
action validity, `executableQuoteCoverage`, or surface-factor coverage. It may
remain raw display context or a declared fallback when no executable book exists.

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
- RL portfolio NAV defaults to net liquidation value: long options use the
  stressed executable bid, shorts use the stressed executable ask, and both
  reserve estimated closing commission. Underlying positions use exit-side
  synthetic slippage and closing commission. `midpoint` is an explicit legacy
  comparison mode, never an interchangeable result.
- When a held option quote disappears, carry its last executable liquidation
  mark rather than replacing it with entry cost. A position opened from an
  executable quote must always initialize that fallback.
- No module in this repository may route `PaperBroker` calls to a real broker.

Future agents should read quotes from `interface.data.load_latest_snapshot`,
make a decision, then call `PaperBroker.buy` or `PaperBroker.sell` with explicit
contract metadata and price. Do not let an agent invent or silently substitute
a missing quote.

Selected recurrent agents use the stricter `AgentPaperStore` boundary instead
of the manual `PaperBroker` account:

- One deployment is the tuple of stable agent ID and exact checkpoint SHA-256.
- Each deployment owns independent cash, positions, underlying shares,
  collateral, risk state, recurrent hidden state, and snapshot cursor.
- Portfolio and recurrent state are committed in the same SQLite transaction as
  every new decision. `(deployment_id, snapshot_timestamp)` is unique, so a
  restart cannot replay a paper fill.
- Hold the database-specific advisory lock for the entire fleet cycle. A manual
  command and background watcher must never restore and overwrite the same
  deployment concurrently.
- Restore only JSON-safe state and validate the complete environment contract.
  Never use pickle for runtime cursor persistence.
- Warm recurrent state on at most the checkpoint sequence length ending at the
  held-out cutoff. Never execute or record those warm-up outputs.
- Decisions begin strictly after the selected fold's held-out end. Current
  checkpoint provenance and run-summary model ID/activation must agree.
- Label the newest decision reward `same_snapshot_execution_only`; there is no
  invented future mark. Earlier decisions processed with a real next snapshot
  use `through_next_eligible_snapshot`. Paper equity, not a provisional reward,
  is the current account result.
- Store pre-trade decision NAV for every action. Leave the newest economic
  outcome pending, then atomically finalize it at the next real eligible mark as
  `outcome_nav / decision_nav - 1`. Never treat the duplicated terminal frame
  used for current-snapshot execution as a realized forward return.
- Record research orders even for guarded agents, but replace the executable
  action vector with all-zero HOLD unless the validation gate activated it.
- Only provider-confirmed regular snapshots with fresh underlying provenance
  and an executable option quote may advance the runtime.
- Old or incompatible checkpoints fail closed. Do not fall back to an older
  model merely to show activity.
- The agent database is simulated execution state and must never be wired to a
  live broker adapter without a separately authorized milestone.

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
- Explicit provider states other than `REGULAR` mask every option and
  underlying trade while leaving hold available. Legacy `UNKNOWN` state has
  zero `marketStateCoverage` and remains tradeable only to keep old research
  demos loadable; results from that fallback are not paper-alpha evidence.
  `regularMarketSession` and `marketStateCoverage` belong to the named
  `market_session` ablation, but masking those policy inputs must never disable
  the independent execution-safety mask.
- Explicit provider quote age above the environment's configured maximum masks
  every non-hold option and underlying action. Missing or future-skewed quote
  time has zero `underlyingQuoteAgeCoverage`; legacy missing coverage remains
  research-demo tradeable and must be reported. The age/coverage inputs belong
  to `data_freshness`, but its ablation must never disable the execution guard.
- Multiple orders in one action are revalidated sequentially so cash cannot go
  negative even when individual pre-step actions were affordable.
- Execute the underlying leg first, then option slots in ascending order. Masks
  describe the pre-step state; every leg must still be revalidated against the
  running cash and Greek state.
- `info` retains executions, invalid-action count, P&L, fees, trade notional,
  reward components, and slot retention/churn diagnostics.
- `reward_components` must sum to the returned scalar reward. Gross P&L includes
  spread/mark effects and the change in estimated future closing costs, while
  realized commission and invalid-action penalties are separate components.
  Closing at unchanged quotes under liquidation valuation must not create an
  artificial NAV or reward jump. Optional downside shaping is the negative
  coefficient times the negative part of the current net P&L return. Optional drawdown
  shaping charges only the increase in running maximum NAV drawdown, not the
  full current drawdown on every step. This keeps the signal path-causal and
  makes its episode sum equal negative coefficient times maximum drawdown.
- Reset peak NAV and maximum drawdown at every episode boundary. Persist both
  reward coefficients in the environment manifest, expose current/maximum
  drawdown and drawdown increase in `info.path_risk`, and retain all five reward
  components in rollout metrics. Coefficients must be finite, non-negative, and
  default to zero so shaping is disabled unless explicitly requested.
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

The deployed recurrent policy boundary is `StreamingRecurrentPolicy`, returned
by `recurrent_policy(...)`. Treat its hidden state as part of the causal agent
state:

- Put the model in evaluation mode before constructing the runtime. The runtime
  rejects training mode at construction and before every decision so dropout
  cannot create silent nondeterminism.
- Use one cursor per episode and ticker. Call `reset()` at every boundary;
  hidden state must never leak across independent paths, symbols, folds, or
  portfolio resets.
- Leave strict chronology enabled for agents. Duplicate or backward timestamps
  are rejected before inference. `enforce_chronology=False` is reserved for the
  latency benchmark that deliberately repeats one training observation.
- Use `fork()` for counterfactual branches. It shares the same read-only model
  but clones the recurrent cursor, so advancing one branch cannot advance the
  other.
- `snapshot()` clones hidden tensors to CPU. `restore()` clones them onto the
  model device and validates the state schema, full `RecurrentConfig` hash,
  recurrent kind/shape, finite values, and step/timestamp consistency.
- Restore state only with the exact same checkpoint weights. The model-contract
  hash prevents configuration mismatches but intentionally does not claim to be
  a checkpoint-weight identity. Never accept an untrusted pickle merely to move
  a recurrent state.
- Keep `reset()` allocation-free. Reusing a policy cursor at a valid episode
  boundary is preferred to rebuilding or reloading the model.
- Use `BatchedStreamingRecurrentPolicy` only when every fixed batch member can
  provide one observation per synchronized actor pass. It is a vectorized
  policy surface, not an asynchronous scheduler.
- Before reusing a batch column for a new episode or ticker, call
  `reset([index])`. Partial reset must zero only that hidden-state column and
  leave every other cursor unchanged. A batched snapshot with a reset cursor
  must contain an exactly zero state for that column.
- Validate every observation and timestamp before batched inference. One bad
  cursor must fail the whole call without partially advancing any hidden state.
- Deployment policies use the actor-only `model.act(...)` path. It must remain
  action-equivalent to deterministic full actor-critic inference and must not
  execute value or auxiliary heads. Training rollouts still require
  `sample_action(...)` because GAE and the value loss need critic estimates.
- For flat feature-removal candidates, validate the raw external vector width,
  compact it with the model's fixed active indices before creating the Torch
  tensor, and let the recurrent model accept that precompacted width. Keep raw
  training tensors supported and action-equivalent; do not reintroduce a Torch
  `index_select` into every streaming actor call.
- When those fixed masked indices form one contiguous interval, compact with the
  two surviving NumPy slices; otherwise use the fixed advanced-index gather.
  Do not replace either path with Python-filled reusable buffers or
  `np.take(..., out=...)` without a new matched benchmark: that approach was
  slower for both batch-one and synchronized batch-16 actors here.
- Walk-forward latency gates measure actor-only batch-one deployment. Batched
  throughput is separate operational evidence and must never replace the
  batch-one gate unless the intended deployment itself has a fixed batch.

An environment manifest hashes only its selected ticker CSV. Do not broaden the
fingerprint to unrelated ticker files: doing so adds startup I/O and invalidates
otherwise reproducible experiments whenever the collector updates another
symbol.

Walk-forward summaries are also the interface contract for tangible agents.
Persist held-out timestamps, step returns, NAVs, decisions, and fills only after
validation fixes the winning model. Preserve test-slice market-state and quote-
time coverage beside those paths. The interface may aggregate validation
candidates into a leaderboard, but it must never relabel validation metrics as
held-out performance or imply alpha when block-bootstrap evidence is
insufficient. Treat each newest per-ticker winner as a persisted agent with a
stable model-derived ID, checkpoint, recurrent core, encoder topology,
activation state, latency, and test time range. Show every stored decision,
including HOLD, and keep the research action separate from the sandbox action
when the validation gate substitutes no-op. Guarded agents are real saved
policies, not missing agents, and GNN challengers must remain visible even when
a flat policy wins. Training and market fetching stay outside Streamlit reruns.

`agent-arena` is the reproducible integration-demo entry point. Keep candidate
families, split sizes, training budget, costs, and risk rules identical across
its tickers unless the changed contract is explicit in the artifact. Every flat
GRU/LSTM/gated-mixture family receives both factorized multi-leg and exact
single-leg candidates so action sparsity is validation-selected rather than
silently fixed. The default arena also includes GRU/LSTM/gated-mixture
`surface_graph_set` candidates with one same-side surface neighbor,
opposite-side counterpart edges, and the exact single-leg decoder. Preserve the
six flat candidates as measured controls; do not report a GNN validation win
without its parameter count, actor latency, and held-out result. A failure
for one ticker must be written with its exception type and message while other
tickers continue. Do not average returns into a portfolio claim: each row is an
independent, validation-selected research-demo path with its own provenance.
The default arena also carries six exact `contract_smile_residual` removals:
flat and `surface_graph_set` GRU/LSTM/gated-mixture agents with the same sparse
decoder and otherwise identical model settings. Keep these matched identities
intact so `validation_score_lift_vs_full` is populated. The interface must show
whether each winner enabled or ablated the signal and summarize only validation
lift; never use held-out results to decide whether the feature remains enabled.
The arena additionally carries six exact `surface_velocity` removals with the
same flat/GNN, recurrent-family, decoder, seed, and budget identities. Keep the
raw change and coverage inputs enabled in these removals so the comparison asks
only whether explicit elapsed-time normalization helps.

The default arena uses training-seed offsets 0, 1, and 2. Model selection first
finds the highest seed-robust validation score, then retains every eligible
candidate within `max(best_seed_score_std / sqrt(seed_count), 1e-4)` of that
leader. Only inside this simplest-competitive pool may the established
ablation, worst-seed actor-latency, parameter, active-input, and training-cost
tie-breaks choose a winner. Store the raw leader, tolerance components,
competitive IDs, and score sacrificed in every artifact. Single-ticker and
universe commands default the materiality floor to zero unless explicitly
declared; never tune it from held-out results.

The fast default arena must use `latest_fold_only=True`. Build its single split
backward from the newest available test tail, preserve both embargoes, and assign
all earlier eligible history to expanding training. Do not recreate this with a
large step size: that selects the earliest fold and silently leaves newly
collected states unused. General walk-forward commands remain multi-fold unless
`--latest-fold-only` is explicit. Persist the mode and exact boundaries in every
summary, and show test timestamps in the interface. Default `agent-arena` runs
must use timestamped subdirectories beneath `data/agent_runs/recurrent-arena`;
an explicit `--output-dir` may reproduce or replace a caller-owned target.

The default arena must preflight all three partitions before model construction.
Filter materially deduplicated raw snapshots to states that are
provider-confirmed `REGULAR`, have a covered underlying quote age within the
environment threshold, and contain at least one positive non-crossed option
bid/ask. Build the split from only those eligible states so pre-market history
cannot satisfy the training minimum while execution is disabled. The default
contract therefore needs at least thirteen eligible states: six training, three
validation, and four test. Do not engineer policy features until the ticker
passes. Persist source, eligible, required, and exclusion counts together with
per-partition readiness and time bounds. Readiness-only failures are expected
status, not a crashed job, and must remain visible in Streamlit while the most
recent successful agents stay inspectable. `--allow-unready-tail` is an explicit
plumbing override that retains the unfiltered source dataset and must never
support an economic or alpha claim.

`arena-watch` must reuse `arena_walk_forward_config()` so readiness and the
launched job cannot drift. It may launch the strict arena only when all watched
tickers have thirteen eligible states and resolve to the same New York session
date. Persist an atomic heartbeat before and after training, retain the latest
successful artifact across waiting/errors, and key completion by session,
ordered symbol set, and watcher run-contract version. Never pass
`--allow-unready-tail`. Hold
one advisory lock for the watcher lifetime so manual and LaunchAgent instances
cannot train the same session concurrently. The service checks frequently but
trains at most once per completed contract/session; Streamlit only reads its
status file and must not initiate training.

Validation selection does not automatically authorize sandbox execution. After
the research winner is fixed, evaluate deterministic no-op on validation only
and require the winner's seed-robust score to strictly exceed the no-op score
plus `activation_min_score_advantage`. The default arena margin is `1e-4`; the
general walk-forward commands default to zero. Persist the aligned validation
report, scores, margin, decision, and sandbox policy. Held-out learned-agent
evidence remains visible for research, but the operational projection must use
no-op return, fills, fees, and zero actor latency when the gate abstains.
Promotion additionally requires that this validation activation gate passed.

The interface promotion status is a strict research-to-deployment gate, not a
model-selection input. `Promotion ready` requires every held-out path to be
positive, beat no-op with statistical support, remain positive under doubled
costs, have at least statistically evaluated history, use provider-confirmed
regular-session data, and contain no invalid actions. Missing evidence fails
closed. Keep all failed reasons visible in the drill-down and never weaken a
gate merely to produce a deployable-looking result.

`sequence.observation_vector` is the versioned policy boundary. Under
`dimensionless.v24`, price-like fields are relative to spot, contract Gamma is
the Delta change for a 10% spot move, portfolio and Greek exposures are relative
to NAV/deployed capital, underlying shares and covered-share reserves are
represented by NAV weight, cash collateral is NAV-scaled, DTE is expressed in
years, held quantity and unrealized return are signed-log compressed, and
heavy-tailed fields, including provider quote age, are compressed and clipped. Raw
volume and open interest
must not be reintroduced beside their log features without ablation evidence.
Any transform change requires a new feature-vector schema, scale-invariance and
finite-value tests, and a checkpoint-schema bump; old weights must never be
silently loaded against a changed feature layout.

Capture time belongs once in the market vector as
`easternCaptureDayFraction` plus `easternCaptureClockCoverage`. Convert the
current `collectedAt` instant from UTC to `America/New_York` with DST support;
invalid or absent timestamps are zero with zero coverage. Keep both inputs in
the named `intraday_clock` ablation. This is policy context only: never use it
to infer provider market state, holidays, early closes, quote time, or execution
permission, and never let removing it alter the independent session mask.

Systematic market context belongs once in the market vector. It contains only
the provider-timestamped benchmark return, 4/16-observation cumulative return
and realized volatility, ticker-minus-benchmark return, quote age, and coverage.
Require strictly advancing benchmark provider time for one-step and relative
returns; repeated or backward timestamps are unavailable, not observed zeros.
Reset rolling benchmark history when the declared symbol changes so unlike
price levels can never form a synthetic return.
Keep this block removable through `systematic_context`. It is a cheap regime
proxy, not a factor decomposition, hedge instrument, or alpha claim, and it
must never control execution masks.

Fitted smile geometry also belongs once in the market vector. Build it only
from current, executable front-expiry OTM quotes inside 0.35 absolute forward
log-moneyness. Require at least five unique points and two unique coordinates
on each side of ATM, require observed support through +/-0.05, standardize
coordinates before the quadratic solve, and expose
fixed-radius curvature, relative RMSE, leave-one-out relative ATM residual, and
binary coverage. Missing support is unavailable, not a flat smile. Keep the
block removable through `smile_fit`; it is a noisy surface diagnostic and alpha
hypothesis, never a reconstructed quote or execution authority. Yahoo option
bid/ask rows have no synchronized exchange timestamp, so numeric quote validity
is not proof of contemporaneous fillability.

Contract smile residuals are node-local observed-surface diagnostics. Within
the current snapshot, group only positive, non-crossed executable quotes by the
same expiration and option side, restrict absolute forward log-moneyness to
0.35, and require at least five rows with three unique coordinates. Standardize
the coordinate before a quadratic solve, convert the in-sample residual to its
leverage-adjusted leave-one-out value, divide by observed IV, bound it, and emit
separate coverage. Missing support is zero with zero coverage. Keep both fields
in `contract_smile_residual`; flat ablation compacts them while graph ablation
masks them without rewiring strike/expiry edges. Never use this residual as a
reconstructed volatility, quote, fair value, fill price, or execution permit.

Every visible held contract exposes snapshots since opening and since its most
recent trade. Same-sign adds retain the opening index, every fill resets the
last-trade index, and crossing through zero starts a new position lifecycle.
Both clocks are zero for unheld contracts, causal, log-compressed, and isolated
in the named `position_lifecycle` ablation. Preserve these rules when changing
fills, partial closes, settlement, stable slots, or observation construction.

The recurrent input must expose bounded state-dependent action capacity without
duplicating the full action mask. Each valid contract reports the fraction of
buy and sell quantity buckets allowed by the exact current per-row mask; the portfolio
reports the corresponding underlying fractions. Compute them only after all
cash, collateral, quote, position, Greek, and underlying-limit checks. The mask
remains authoritative for sampling and execution. Keep the summaries in the
named `action_feasibility` ablation, including both contract and portfolio
indices, never treat them as proof that simultaneous factorized orders are
jointly feasible, and never let a summary make an invalid action executable.

Static-arbitrage diagnostics must remain current-snapshot, bid/ask-aware data
quality signals. Compare only positive, non-crossed quotes with the same
expiration and option side; sort unique strikes deterministically and retain
the first duplicate. Normalize violation magnitudes by spot, keep explicit
coverage bits, and ablate scores without masking coverage. Do not label a
positive score as free or fillable alpha: American exercise, dividends,
fractional uneven-strike weights, stale quotes, and missing depth remain outside
this diagnostic. Do not add calendar constraints until the data contract has a
point-in-time forward/dividend surface suitable for them.
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

Cadence-normalized surface velocity is a separate representation hypothesis.
Divide only covered front-ATM, 25-delta risk-reversal/butterfly, and ATM-term-
slope changes by the same causal positive `snapshotGapSeconds` expressed in
hours, then bound each rate at plus or minus two volatility units per hour.
Missing or invalid gap coverage must produce zero velocity while the existing
change and coverage fields remain visible. Keep all four rates in the named
`surface_velocity` ablation; never call a validation lift alpha, and do not add
the rates to auxiliary targets because they are deterministic transforms of
targets already present there.

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
baseline. Keep learning rate, PPO epochs, minibatch size, policy/value clipping,
target KL, value coefficient, and gradient clipping exposed consistently by
direct, single-ticker walk-forward, and universe walk-forward CLIs and persisted
through `TrainingConfig`. The default factorized decoder multiplies the independent masked-row
probabilities, equivalently summing their log probabilities, and uses one exact
joint action likelihood per transition. PPO must clip one ratio formed from
that joint likelihood; REINFORCE must use the joint log likelihood in its score
function. Per-row PPO ratios are the named `dimensionwise` research ablation,
not the default, and must be compared against the joint objective on validation.
Persist the effective objective, PPO importance-ratio count, REINFORCE
score-function likelihood count, and validation lift.
On an exact validation-score tie, retain the joint objective before consulting
latency because the two objectives do not change the inference graph.
The optional `single_leg` decoder uses one exact joint categorical over global hold
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
The default `feasible_normalized` entropy objective operates on the already
masked categorical distribution. For each likelihood factor, count only finite
masked logits as feasible actions. When `K > 1`, divide exact categorical
entropy by `log(K)`; average only those explorable factors. A factor with
`K <= 1` is excluded. First average within each decision, then average decisions
with at least one explorable factor so dense surfaces do not receive more weight
than sparse ones. This makes the objective bounded in `[0, 1]` and invariant to
padded or hold-only slots. Never normalize by the global encoded action count,
normalize before masking, or let right-padding enter the denominator. Preserve
raw entropy, normalized entropy, and the explorable-factor fraction in episode
metrics. Keep `raw_mean` only as the named validation ablation, require a
positive coefficient for that comparison, and prefer `feasible_normalized` on
an exact score tie because neither objective changes inference.
Actor credit assignment must use the same exact support. A transition is an
actor-choice step only when at least one decoder factor has more than one
feasible action. Normalize advantages only across those steps. The default
joint PPO/REINFORCE objective averages only actor-choice transitions; the
dimensionwise PPO ablation averages only explorable factors. Apply the same
support to entropy, approximate-KL, and clip-fraction aggregation. Forced-hold
transitions still advance recurrent state and remain in rewards, GAE/returns,
critic loss, and available auxiliary targets. Never drop them from chronology
or treat their absence from the actor loss as an absent market state. Persist
choice/forced counts and the exact credit-assignment contract.
Do not hard-cap active rows or post-process sampled actions without deriving the
matching joint likelihood; that would invalidate PPO ratios. Preserve requested
option/underlying order counts and action-density metrics in every episode.
Persist the decoder, number of likelihood factors, likelihood aggregation,
importance-ratio count, and score-function likelihood count. `single_leg` must encode
masks before sampling, decode to the existing environment action array, reject
training actions with multiple non-hold rows, and retain one scalar log
probability/entropy per step. It cannot express same-snapshot spreads or hedges,
so keep factorized decoding available and select between them on validation.
For deterministic actor-only single-leg inference, global hold is intrinsically
safe: bypass per-row safe-hold cloning, mask only the flattened non-hold logits
in place, and decode the winning joint index. Keep stochastic training and
full-logit evaluation on the exact masked categorical path, and preserve action
equivalence between both paths with flat and set-encoder tests.
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
Optional `volatility_stratified` starts may use only the declared backward-only
4- or 16-snapshot realized-volatility feature already computed causally for the
training partition. Require full coverage, at least one candidate per requested
bin, and at least as many distinct values as bins. Sort deterministically into
quantile strata, sample strata uniformly and starts uniformly within a stratum,
and fall back explicitly to uniform when those conditions fail. Persist the
requested/effective mode, bin, bin count, and replicate-level counts. Compare a
matched uniform candidate on validation before retaining stratification; never
rebalance validation or held-out paths, and never describe a fallback run as
stratified evidence.
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
coefficients times maximum drawdown, downside deviation, turnover, absolute
underlying-return beta, and mean absolute Delta-notional weight. Use this
one score consistently for best-episode restoration, patience, architecture,
algorithm, and feature-ablation ranking. Persist raw reward, every component,
and all coefficients. A positive beta or Delta-notional coefficient requires
covered validation evidence and must fail rather than score missing history as
zero. Defaults are zero; never tune coefficients from test.
Training-time drawdown and downside reward coefficients are independent from
these validation-selection coefficients. Experiments may enable either layer
or both, but must declare and persist the choice; never silently treat selection
penalties as training reward or hide intentional double risk penalization.

Delta-neutrality shaping is a separate train-only objective. Compute signed
portfolio Delta notional as `portfolio_delta * current_spot / current_NAV`
after each transition, require finite positive spot and NAV, and subtract the
configured coefficient times its absolute value capped at 10. Persist raw and
shaped reward, the separate component, raw exposure statistics, coverage, and
cap semantics. Validation and test rewards remain unshaped executable
net-liquidation returns. A matched `--delta-neutrality-ablation` must set only
the coefficient to zero while preserving model parameters and inference work;
report its validation lift, and prefer the disabled objective on an exact score
tie. Never call point-in-time Delta weight realized beta or alpha.
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
coverage, and mask unavailable IV separately. The delta-hedged target must use
the current endpoint's Delta against the later spot move, normalize by current
positive spot, and never use future Delta or intermediate rehedging. Require
explicit regular-session state, state coverage, underlying-age coverage, and an
underlying age no greater than the versioned 1,200-second target ceiling at both
endpoints. Mask pre/post-market, stale, and unknown provenance. Treat it as
a bounded representation target only: it excludes financing, dividends,
intermediate hedge costs, and execution attribution. Never
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
policy reaches test. Any claim specific to one target additionally requires a
matched `--auxiliary-target-ablation TARGET`; exclusions alter only the loss
mask, must be unique, cannot remove every target, and must remain in model IDs,
training metrics, and checkpoint manifests. When validation scores tie exactly,
prefer the full target set before latency-noise tie-breaks because deployment
operations and parameters are identical. Snapshot horizons are not
elapsed-time horizons.

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

Every training episode must retain critic-balance evidence: reward RMS,
return-target level/dispersion/RMS/maximum magnitude, pre-update value-residual
RMS, and the actor-head and critic-head gradient norms measured after backward
but before global clipping. These norms are coefficient-weighted gradients of
the disjoint output heads only. Never describe them as shared-trunk gradient
attribution or a full per-task gradient Gram matrix.

Aggregate scale metrics by transition count and head-gradient metrics by actual
optimizer updates for each ticker. Preserve zero-return tickers instead of
dropping them from the evidence. A cross-ticker ratio at or above the declared
10x engineering threshold, or a mixture of positive and zero return-target
scales, may recommend a normalization ablation, but the diagnostic must remain
selection-inert. It cannot change a checkpoint, candidate, seed, or held-out
result.

Critic LayerNorm is implemented as the first separate candidate. It normalizes
the recurrent representation only on the value branch immediately before the
linear value head. Never route its output into the policy or auxiliary head, and
never call it from `model.act(...)` or `policy_sequence(...)`. Require critic
width at least two, keep it disabled by default, expose a matched disabled
candidate through `--critic-layer-norm-ablation`, persist validation lift, and
prefer disabled on an exact selection tie. This is one critic-conditioning
ablation, not TOPPO. PopArt target normalization and per-task gradient balancing
remain distinct future candidates and must not be bundled with it.

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

The `surface_graph_set` encoder is the structured option-surface candidate. Its
same-side local topology uses only cross-sectionally standardized
`forwardLogMoneyness` and `dteDays`; implied volatility is node content and must
not rewire the graph. Every valid node also selects its nearest valid
opposite-side coordinate counterpart. Use Delta's sign bit for side so a
negative-zero put remains distinct from a call. Symmetrize the union, add self
edges, and use one shared neighbor transform for both relation classes unless a
measured validation and latency experiment earns more complexity. Preserve the
same permutation, padding, action-decoder, recurrent-streaming, and empty-surface
invariants as `graph_set`. `graph_neighbors=0` retains counterpart edges for
this encoder; only zero-neighbor `graph_set` is the no-adjacency Deep Sets
baseline. Treat counterpart edges as representation structure, not put-call
parity enforcement, reconstructed prices, or executable quotes.
Fixed graph encoders must reuse the device-portable nonpersistent diagonal
mask; do not allocate `torch.eye` per forward or add the mask to checkpoint
weights.

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
turnover, costs, execution quality, final/peak Greek exposure diagnostics,
underlying return/volatility, aligned strategy beta/correlation with coverage,
and mean/maximum absolute Delta-notional weight with coverage.
Held-out statistical comparisons pair agent and baseline returns by exact
arrival timestamp, then use circular moving blocks. A deterministic policy on
one deterministic CSV path must accept exactly one held-out seed: changing the
seed label does not create a new path or an independent observation. Multiple
training seeds must produce independently trained checkpoints before they can
measure learned-policy variability; never replace that experiment with repeated
evaluation of one checkpoint. Training-seed offsets must be predeclared. Rank an
architecture by its configured mean/worst/dispersion aggregate, require every
replicate to satisfy the latency ceiling, and deploy the run closest to the median
validation score rather than cherry-picking the best seed. When robust validation
scores tie exactly, compare the worst training-seed median inference latency
before parameter count, active input count, or training work. Persist every seed
score, the representative rule, actual path count, seed repetitions, and
bootstrap independence unit in every fold. The default minimum
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
only, and apply the declared objective-ablation preferences before worst-seed
actor latency, parameter count, active input count, optimizer updates, and
stable model ID. Instantiate the test environment
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
observation after training using the actor-only deployment path. Do not execute
the critic or auxiliary head inside this measurement. Record median, p95, and
mean latency together with device, PyTorch version, thread count, warm-up
iterations, measured iterations, and evaluated heads. Restore the model's prior
train/eval mode after success or failure. Timing is
machine-specific and must never outrank validation evidence. It is the first
deterministic tie-break only when robust validation scores are exactly equal, so
a smaller but slower candidate does not win on parameter count. An explicit predeclared median
latency ceiling may filter deployment-ineligible candidates before winner
selection; preserve their validation evidence and exclusion reason, allow only
eligible candidates to reach test, and fail if none remain. With no ceiling,
timing remains inactive unless validation scores tie.

Feature-removal candidates must use the named, non-overlapping groups in
`sequence.FEATURE_ABLATION_GROUPS`. Apply masks inside the recurrent model after
the versioned transform and persist exact flattened indices in
`RecurrentConfig`; external preprocessing would make restored checkpoints
ambiguous. CLI ablations retain a matched full-feature candidate, report
validation score and raw-reward lift versus it, and obey the same one-winner
test boundary. Groups may span market, contract, and portfolio sections; their
flattened indices must remain exact and non-overlapping. Flat encoders must
gather active columns before LayerNorm and the
recurrent input matrix so removal changes actual capacity; graph encoders retain
the full structured layout and zero masked relations before graph construction.
Persist masked and active counts plus the execution mode in checkpoint evidence.
Model-spec identifiers must hash the feature-vector schema as well as the model
specification so schema changes cannot silently reuse checkpoint filenames.

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
arena-watch --once
arena-service install
streamlit run src/trading_bot/interface/app.py
option-chain AAPL
train-demo --symbol AAPL --encoder graph --kind hybrid --episodes 25
train-demo --symbol AAPL --encoder graph_set --kind hybrid --episodes 25
train-demo --symbol AAPL --encoder graph_set --kind mixture --episodes 25
train-demo --symbol AAPL --encoder surface_graph_set --kind hybrid --episodes 25
train-demo --symbol AAPL --encoder attention_set --attention-heads 4 --kind hybrid --episodes 25
train-demo --symbol AAPL --allow-collateralized-option-shorts --episodes 25
train-walk-forward --symbol AAPL --min-train-size 500 --validation-size 100 --test-size 100 --embargo 8 --candidate flat:gru --candidate graph_set:hybrid:ppo:0 --candidate surface_graph_set:hybrid:ppo:3 --candidate attention_set:hybrid:ppo
train-walk-forward --symbol AAPL --delta-neutrality-coefficient 0.0001 --delta-neutrality-ablation --selection-abs-beta-penalty 0.01 --selection-delta-notional-penalty 0.01 --min-train-size 500 --validation-size 100 --test-size 100 --embargo 8
train-walk-forward --symbol AAPL --auxiliary-coefficient 0.05 --auxiliary-target-ablation medianContractDeltaHedgedSpotReturn --min-train-size 500 --validation-size 100 --test-size 100 --embargo 8
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

`arena-service install` similarly writes a user LaunchAgent with absolute
repository, interpreter, data, and log paths. Its watcher performs cheap raw
readiness checks, records `data/_arena_watch_status.json`, and launches the
locked default arena only once per eligible New York session. Keep training in
a subprocess so a failed or memory-heavy tournament cannot corrupt the watcher;
`arena-service uninstall` is the explicit teardown.

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
For stable assignment on sparse surfaces, return once every visible unique
contract is already assigned; padded vacancies are not evidence that ranking is
needed. A newly visible contract must still trigger ranking before vacancies are
filled. Expiry settlement may bypass timestamp parsing only when there are no
option positions; never apply that shortcut to an open long or collateralized
short lifecycle.

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
- RL liquidation value uses saved top-of-book quotes and assumes the configured
  quantity can exit there. It does not model depth, partial fills, market impact,
  or adverse selection; stale fallback marks remain explicitly approximate.
- The RL short-option lifecycle has no early assignment, exercise instruction,
  dividend, or corporate-action model. Post-expiry physical assignment is a
  conservative deterministic research approximation, not broker fidelity.
- The current local AAPL sample is sufficient for integration smoke tests, not
  statistical training claims. Follow `docs/research-roadmap.md` gates before
  treating a model improvement as alpha.
- SPY is only the default systematic-context proxy. Its return does not isolate
  a stock option's systematic component, and the collector has no benchmark
  option surface, factor model, or synchronized exchange feed.
