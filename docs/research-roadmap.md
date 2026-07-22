# Alpha research roadmap

Last literature review: 2026-07-22

## Standard of evidence

In this project, "alpha" means an out-of-sample improvement over implementable
baselines after bid/ask spread, commissions, turnover, position limits, and
risk are included. A higher in-sample reward or one favorable seed is not
alpha. Yahoo snapshots remain useful for API and training-pipeline smoke tests,
but they are not enough for a historical performance claim.

Collector v0.36 makes this boundary stricter at ingestion. Consecutive surfaces
whose raw quotes, contract membership, spot, dividend input, and risk-free rate
are identical are one market state even if later timestamps produce different
time-to-expiry and model Greeks. Both persistence and legacy CSV loading enforce
that identity. This removes synthetic closed-market transitions but can expose
that a seemingly long CSV contains almost no usable temporal history; collect
during live market evolution before running selection gates.

Every candidate must eventually pass:

1. Fixed chronological train, validation, and untouched test periods, followed
   by rolling walk-forward evaluation.
2. No-op, Black-Scholes delta hedge, and simple rules-based volatility or
   moneyness baselines using the same information and execution model.
3. Multiple genuinely distinct seeds or paths, paired moving-block confidence
   intervals, turnover and drawdown reporting, and doubled-cost stress.
4. Ablation against a smaller flat GRU. LSTM, hybrid GRU+LSTM, GNN, and surface
   latent models earn their complexity only through out-of-sample improvement.
5. A feature-availability audit proving that every input existed before the
   simulated decision timestamp.

## What recent work changes

| Evidence | Useful idea for this repository | Decision |
| --- | --- | --- |
| [Recurrent Experience Replay in Distributed Reinforcement Learning (ICLR 2019)](https://deepmind.google/research/publications/recurrent-experience-replay-in-distributed-reinforcement-learning/) | Recurrent training on partial sequences must address inaccurate boundary hidden states; a prefix can reconstruct state before loss-bearing transitions. | Random training windows now use a bounded causal no-op prefix, one batched no-gradient recurrent call, explicit metrics, and a validation-only disabled ablation. This is on-policy context reconstruction, not replay. |
| [AlphaZeroBeta: Deep Reinforcement Learning for Market-Neutral Portfolios (2026)](https://arxiv.org/abs/2607.18001) | A current finance study combines recurrent PPO, transaction-cost-aware objectives, and rolling walk-forward evaluation, but its reported equity-index results do not establish option alpha here. | Keep recurrent PPO in the tournament, require cost stress and walk-forward evidence, and treat market-neutrality controls as a later declared objective rather than importing performance claims. |
| [Sizing the Risk: Kelly, VIX, and Hybrid Approaches in Put-Writing (2025)](https://arxiv.org/abs/2508.16598) | The preprint treats implied-versus-realized volatility and volatility-regime scaling as interacting inputs for put-writing size, but its SPXW backtest does not establish portability to equity options. | Add bounded prior-only ATM-IV and volatility-premium normalization as two compact state candidates. Keep sizing bounded by collateral and require the named normalization ablation, costs, and held-out folds before retaining either signal. |
| [Deep Reinforcement Learning Algorithms for Option Hedging (2025)](https://arxiv.org/abs/2504.05521) | PPO is competitive, but Monte-Carlo policy gradients can be a strong hedge benchmark and sparse terminal rewards matter. | Keep PPO; add delta-hedge and Monte-Carlo policy-gradient comparisons before claiming algorithmic lift. |
| [Risk-Sensitive Contract-unified RL for Option Hedging (2024)](https://arxiv.org/abs/2411.09659) | Learning tail risk of terminal hedging P&L can improve the objective beyond mean reward and allow a policy to span contract conditions. | The collateralized liability foundation now exists. Add CVaR or learned P&L-distribution objectives only after enough independent paths and lifecycle validation exist; the current tiny research demo cannot identify tail risk. |
| [ATM S&P 500 options hedging with DRL (2025)](https://arxiv.org/abs/2510.09247) | Moneyness, maturity, realized volatility, current hedge state, walk-forward testing, and transaction-cost stress are central. | Causal realized-volatility horizons, walk-forward evaluation, and explicit per-contract position quantity/cost/P&L state implement this lesson. |
| [Deep Hedging with Reinforcement Learning (2025)](https://arxiv.org/abs/2512.12420) | Normalize exposures, combine IV term structure/skew with realized volatility, enforce realistic limits, and quantify uncertainty; attractive point estimates often lose significance. | `dimensionless.v13`, compact ATM-IV-minus-realized-volatility and term/dynamics state, stable contract/position identity, Greek budgets, collateral state, and paired moving-block intervals implement the state/risk lesson. |
| [Deep Hedging with Options Using the Implied Volatility Surface (revised 2025)](https://arxiv.org/abs/2504.06208) | Joint return/surface dynamics, multiple hedge instruments, variance-risk-premium state, and transaction costs can create useful state-dependent no-trade regions. | Keep whole-surface factors, option-plus-share actions, and sparse action priors. The collateralized short-put carry baseline now prevents attributing a simple IV-versus-realized rule to RL. |
| [IV-surface feedback for deep option hedging (revised 2026)](https://arxiv.org/abs/2407.21138) | A compact surface factorization includes ATM level, maturity and moneyness slopes, smile attenuation, smirk, and their dynamics; bounded recurrent hybrids outperform standalone networks in its numerical study. | Executable 25-delta risk-reversal/butterfly, ATM term slope/curvature, and one-snapshot factor changes now have explicit coverage once per market snapshot; test them through named tournament ablations. |
| [Shortfall-aware RL option hedging (2026)](https://arxiv.org/abs/2601.01709) | Better static IV fit need not produce better dynamic hedging; replication-error and shortfall objectives under costs are separate evidence. | Keep realized path diagnostics primary. The liability surface is implemented; defer shortfall/CVaR training until enough independent paths make tail estimates meaningful. |
| [Autonomous AI Agents for Option Hedging (2026)](https://arxiv.org/abs/2603.06587) | Listed-option experiments emphasize realized path shortfall frequency and Expected Shortfall rather than static fit. | Preserve executable current position state now; add terminal shortfall distributions only after liability episodes and enough independent held-out paths make tail estimates meaningful. |
| [CANDID DAC (2024)](https://arxiv.org/abs/2407.05789) | Independent policies over coupled action dimensions can struggle; sequential policies coordinate dimensions without enumerating the joint action space. | Compare the factorized decoder with an exact single-leg joint categorical now. Defer autoregressive multi-leg decoding until this simpler restriction earns validation lift; never post-process sampled rows in a way that breaks PPO likelihoods. |
| [Structured Policy Initialization for Large Discrete Actions (2026)](https://arxiv.org/abs/2601.04441) | Independence can create incoherent combinatorial actions, while learning full action structure can be slow and unstable; a pretrained structure model can improve convergence. | Keep the exact single-leg structural baseline lightweight. Consider learned multi-leg action structure only after sufficient trajectories exist for pretraining and the single-leg restriction is demonstrably too limiting. |
| [Meta-learning neural processes for IV surfaces (2025)](https://arxiv.org/abs/2509.11928) | Log-moneyness/time-to-expiry surface coordinates, cross-day learning, and model-based priors help sparse reconstruction. | Treat a SABR-prior or attention surface encoder as a later experiment, after full-surface history and arbitrage checks exist. |
| [Deep option pricing with market IV surfaces (updated 2026)](https://arxiv.org/abs/2509.05911) | A low-dimensional whole-surface latent representation may retain most surface information. | Benchmark causal PCA first; try VAE/attention compression only if it beats the simpler representation out of sample. |
| [When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning (2024)](https://arxiv.org/abs/2406.17718) and [Bridging State and History Representations (2024)](https://arxiv.org/abs/2401.08898) | Predictive auxiliary objectives can improve RL history representations, but their value depends on observation structure and distractions rather than being universal. | A masked multi-horizon Smooth-L1 head supervises compact market changes plus permutation-invariant median matched-contract quote/IV dynamics only on training transitions. Keep it only through matched one-step and disabled validation ablations. |
| [Data-Efficient RL with Self-Predictive Representations (2020)](https://arxiv.org/abs/2007.05929) | Predicting multiple future steps can improve representation learning under limited interaction, though its evidence comes from visual-control domains rather than markets. | Support predeclared cumulative snapshot horizons with endpoint masks and compare multi-horizon, one-step, and disabled heads using validation only. |
| [Still Competitive: Revisiting Recurrent Models for Irregular Time Series Prediction (2025)](https://arxiv.org/abs/2510.16161) | Explicit time-triggered mechanisms can make simple recurrent models competitive on irregular series at low overhead. | Add causal prior-snapshot elapsed time plus coverage once in the market vector before considering a continuous-time recurrent cell; validate through the named `time_context` ablation. |
| [Time-Aware Q-Networks (2021)](https://arxiv.org/abs/2105.02580) and [Semi-Markov Offline RL (2022)](https://arxiv.org/abs/2203.09365) | Irregular decision intervals affect both state estimation and the discount applied to future value; treating variable-duration transitions as fixed-step MDP transitions can change the learned objective. | PPO and REINFORCE now compose gamma, and GAE lambda where applicable, over elapsed wall-clock time relative to a declared reference interval. Retain fixed-step semantics as a matched validation ablation. |
| [Multi-Horizon Echo State Network Prediction of Intraday Stock Returns (2025)](https://arxiv.org/abs/2504.19623) | Compact recurrent models can predict intraday returns at multiple horizons without the cost of a large generic architecture. | Expose causal 4/16-snapshot cumulative log returns to the existing GRU/LSTM families before adding another recurrent family; require the named `price_trend` ablation. |
| [A New Option Momentum: The Role of the Systematic Component (revised 2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4404190) | Transaction-cost-robust option momentum is concentrated in a systematic component, while past prices have limited influence relative to risk and quality characteristics. | Treat price trend as a small optional state contribution, keep surface risk/quality features, and remove trend unless it earns validation lift after costs. |
| [Intraday Volatility-Smile Geometry and Option Returns (revised 2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5893362) | Intraday smile geometry contains return information beyond lagged option returns and liquidity controls in the study sample. | The repository already exposes compact ATM, wing, and term geometry. Add matched executable contract quote/IV changes as controls, then require both geometry and dynamics removal tests rather than expanding the recurrent model on the paper's result alone. |
| [Binary Tree Option Pricing Under Market Microstructure Effects (2025)](https://arxiv.org/abs/2507.16701) | Bid/ask structure and serially dependent returns matter for option pricing under microstructure effects. | Use bid/ask midpoint return and relative-spread change only when the same contract has executable quotes at both endpoints. Keep coverage explicit and avoid last-trade momentum. |
| [Hybrid Recurrent Expert Gating for Financial Time Series (2026)](https://doi.org/10.1016/j.procs.2026.06.366) | A learnable gate can vary the contribution of recurrent experts across nonstationary regimes, but its daily-price forecasting evidence does not establish trading or RL benefit. | Add a compact scalar-gated GRU-LSTM mixture as a declared candidate while preserving GRU, LSTM, and concatenated-hybrid baselines, parameter budgets, latency gates, and held-out selection. |

This is not an exclusive reading list. Profiling, microstructure knowledge,
negative experimental results, and newly published work should change the
priorities when they provide stronger evidence.

## Implemented research foundations

As of 2026-07-22, the repository has backward-only 4/16-snapshot realized
volatility with coverage masks, embargoed expanding/rolling fold generation,
normal versus doubled-cost execution scenarios, and episode-level return,
drawdown, risk, turnover, cost, and Greek-exposure diagnostics. These are
evaluation tools, not evidence that the small local AAPL sample observed during
implementation has enough history for a valid walk-forward result.

The executable walk-forward runner now trains each recurrent PPO model on the
training range, selects checkpoints exclusively on validation reward, and only
then evaluates the held-out test range against no-op, first-feasible,
buy-first-then-Delta-hedge, and doubled-cost scenarios. The Delta comparator now
uses a real bounded underlying-share action with explicit synthetic costs. It
remains a research approximation until historical underlying bid/ask, borrow,
margin, dividend, and funding data are available.

The runner can now compare flat, flattened-graph, and graph-set GRU, LSTM,
concatenated-hybrid, and gated-mixture candidates within each fold. All
candidates share the fold and seed;
architecture selection uses
the declared validation selection score with a deterministic simplicity
tie-break, and only the winner reaches held-out evaluation. A common trainable
parameter ceiling can now resolve the widest fitting recurrent state per
candidate from the training layout alone, preventing a nominally equal hidden
width from giving LSTM, hybrid, and graph models materially different capacity.
Requested and resolved capacity, exact count, and headroom remain auditable.
This supports disciplined recurrent/GNN ablation without turning the test
partition into a model-selection leaderboard.
The candidate count and search space must still be declared up front; repeatedly
expanding them after seeing test results would invalidate the holdout.

Recurrent PPO now carries causal hidden state during rollouts and inference,
then trains on contiguous truncated-backpropagation chunks initialized from
the old policy state. This removes fictitious zero-padded history and avoids
recomputing a full temporal window at every decision. It improves temporal
correctness and latency; it is not evidence of alpha.

The same recurrent architectures now support a Monte-Carlo REINFORCE
comparator with a learned value baseline. It uses one on-policy pass over
discounted returns and shares the validation-only tournament boundary with
PPO, allowing algorithmic lift to be tested without exposing every learner to
the held-out range.

The trainer now supports one ticker-invariant recurrent policy over the full
top-50 environment pool. Seeded shuffled cycles give each symbol balanced
rollout coverage without crossing recurrent, portfolio, reward, or advantage
boundaries. Selection retains each ticker report and can penalize cross-ticker
score dispersion or blend mean performance toward the worst ticker. The
current executable command is in-sample integration evidence; a shared
walk-forward runner must precede any cross-ticker alpha claim.

That universe walk-forward runner is now executable. It combines independent
per-ticker chronological splits with a global wall-clock separation check,
selects one architecture/checkpoint from robust aggregate validation only, and
opens every held-out ticker only after selection. Test reports, baselines, cost
stress, and dependence-aware comparisons stay separate by ticker; the universe
summary is descriptive. Meaningful alpha evidence still requires substantially
more snapshots than the current integration history and predeclared repeated
folds.

Selection can now use a declared validation-only reward-minus-risk score with
separate maximum-drawdown, downside-deviation, and turnover coefficients. The
score controls early stopping, checkpoint restoration, algorithms,
architectures, and feature ablations consistently while retaining every raw
component. Coefficients remain zero by default and must be fixed without test
feedback.

Training can now independently use path-causal downside and incremental
maximum-drawdown reward components. The former penalizes only negative net
transition returns; the latter telescopes to negative maximum drawdown over an
episode instead of repeatedly charging an underwater policy. This is a small,
auditable bridge from mean-return training toward the shortfall-aware literature:
it does not estimate tail distributions, CVaR, or Expected Shortfall. Those
claims remain deferred until substantially more independent paths are available.
Zero coefficients preserve the original objective and leave inference inputs
and latency unchanged.

Seeded random training windows now reconstruct bounded recurrent context before
the loss-bearing segment. The prefix uses causal training observations and hold
actions, carries no gradient or reward, and is evaluated in one batched call.
Both single-ticker and shared-universe tournaments can add a matched disabled
candidate and report its validation score/reward lift. This reduces a known
GRU/LSTM boundary artifact; only held-out ablation can determine whether it
improves trading performance.

The market state now includes front-expiry ATM IV and its difference from
backward-only 4/16-snapshot realized volatility, each paired with the existing
history coverage. PPO training samples seeded bounded windows across the
training partition instead of replaying only its first regime. Both choices
improve sample efficiency; their value still requires walk-forward ablation.

Two additional market scalars normalize front ATM IV and the short-window
implied-minus-realized premium using valid values from 16 strictly prior
snapshots.
They require four observations, retain separate coverage, clip extreme scores,
and preserve precomputed causal history across fold subsets. The named
`volatility_normalization` candidate masks only these scores. On the current
ten-snapshot AAPL sample the ATM score is populated, while the premium score is
neutral because its short prior history has no usable dispersion. That is
integration evidence, not an alpha result. A matched flat-mixture smoke tied at
zero validation score and selected the masked candidate by active-input count.

The same backward-only price history now supplies 4/16-snapshot cumulative log
returns. They reuse the volatility windows and coverage masks, add no per-node
state, and are signed-log transformed at the policy boundary. The
`price_trend` candidate masks just these two values, preserving history
availability and volatility context so validation measures their marginal
contribution. The volatility removal candidate likewise leaves the shared
history-availability scalars intact rather than duplicating or hiding them.

The current AAPL integration sample cannot test the trend hypothesis: all
retained snapshots were after the close, spot remained 327.74, and both return
horizons were zero. A tiny matched walk-forward smoke tied at zero validation
reward, selected the `price_trend`-masked GRU by active-input count, and produced
no trend-baseline trades. The implementation is ready for market-hours data,
but this negative result provides no reason to promote the inputs.

A deterministic underlying-trend comparator consumes the same covered return,
targets a bounded long/flat/short share position, and rebalances only when the
target changes. It uses the environment's action masks, slippage, commission,
and position limits. This makes any learned directional benefit compete against
a cheap implementable rule, although missing borrow, funding, and dividend
accounting keep its short leg research-only.

The recurrent state now also receives the positive elapsed time from the
immediately prior snapshot plus explicit availability. This disambiguates
irregular collection gaps with two market scalars and no continuous-time model
dependency. Snapshot auxiliary horizons deliberately retain their existing
count-based meaning; the `time_context` removal candidate must establish any
validation benefit. In the current 22-snapshot AAPL integration sample, 21
intervals were available and ranged from about 53 to 967 seconds. A tiny matched
walk-forward smoke tied at zero validation reward and selected the masked model
by the declared active-input tie-break, so the feature is implemented but not
empirically promoted.

The compact market state also includes executable front-expiry 25-delta risk
reversal and butterfly factors. Explicit ATM/wing, quote, and Greek coverage
prevent a missing or zero-bid surface from looking like a real zero signal.
These factors use only the current cross-section and add six scalar inputs,
leaving the per-contract graph width unchanged.

The policy head now has a trainable hold-logit prior and reward-scale entropy
coefficient. This reduced untrained requested action density on the current
AAPL surface without imposing a hard order cap or changing PPO likelihoods.
Episode provenance reports requested option and hedge actions separately.

An optional exact single-leg decoder now replaces the 33 independent row
categoricals with one masked categorical over hold or one row/action pair. It
trains under both PPO and REINFORCE with GRU, LSTM, hybrid, mixture, flat, graph, and
graph-set encoders; graph-set option scores remain permutation equivariant. It
reduces the default flat-hybrid head by 8,224 parameters but measured roughly
11% slower in batch-one deterministic inference because the joint category must
be decoded into the fixed environment array. The current tiny AAPL tournament
tied at zero validation reward and selected it only by parameter count, so the
factorized decoder remains the default and any coordination benefit is unproven.

The gated recurrent mixture keeps independent GRU and LSTM causal states but
compresses their outputs through one state-dependent scalar convex gate before
all heads. Equal initialization avoids favoring either expert. It reduces
parameters modestly versus concatenation but measured slower in batch-one CPU
inference on both flat and graph-set layouts, so it remains a tournament
candidate subject to the same latency ceiling rather than replacing GRU or the
concatenated hybrid.

The `dimensionless.v13` policy transform now batches signed contract columns,
uses clipping for infinity handling, replaces NaNs in one pass, and assembles
the float32 vector directly. This reduced local preprocessing median by 37% and
cut matched batch-one medians across flat and graph-set hybrid/mixture models
without changing policy features or checkpoint compatibility. It is latency
headroom for future experiments, not evidence that any model earns more return.

The RL environment now has an explicit, opt-in option-liability foundation.
It permits only covered calls and fully cash-secured puts, exposes locked cash
and shares to the policy, prevents collateral reuse across simultaneous orders,
supports signed close/cross accounting, and physically assigns in-the-money
short equity options on the first observed post-expiry date. The constraints
follow the conservative covered/cash-secured concepts in
[FINRA Rule 4210](https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210?page=1),
not a broker-specific portfolio-margin engine. Early assignment and corporate
actions remain absent, so this enables controlled liability experiments rather
than live-execution fidelity.

Walk-forward artifacts now include post-selection, paired circular moving-block
comparisons of the agent against every baseline. They report cumulative
log-return lift, confidence bounds, and the fraction of bootstrap estimates
above zero per held-out seed. Folds below the minimum sample count produce no
bounds, preventing tiny integration datasets from masquerading as statistical
evidence.

A causal long-volatility baseline now converts the IV-minus-realized feature
into an executable comparator: after sufficient history, it opens feasible
front-ATM positive- and negative-Delta legs when realized volatility clears IV
by a configured edge, then reduces residual Delta with shares. It remains
long-only and has no lifecycle exit, so it tests whether the learned agent beats
a simple underpriced-volatility rule rather than a complete volatility book.

The matched short-side hurdle is now executable. It waits for covered ATM IV
to exceed backward-only realized volatility, sells only a feasible cash-secured
front-expiry ATM put, and hedges residual Delta with the existing underlying
action. Single-ticker and universe folds persist its configuration, compare its
arrival-aligned returns to the selected agent, and report fresh base and
doubled-cost episodes. It deliberately holds through expiry so assignment is
not hidden. This is a simple carry control with residual model limitations—not
evidence of a variance-risk-premium alpha.

Training targets now respect irregular transition duration. PPO/GAE and
REINFORCE convert their configured per-reference-interval factors to
`base ** (elapsed_seconds / reference_seconds)`, while inference stays
unchanged. Each episode records transition-time and effective-factor ranges,
and walk-forward tournaments can include a matched fixed-step discount
candidate. This corrects objective semantics; it is not evidence of alpha.

## Prioritized implementation sequence

### 1. Make evaluation credible

- Store sufficient timestamped, point-in-time option and underlying history.
- Run the implemented full training runner after sufficient history is stored.
- Retain the implemented backward-only realized-volatility horizons and
  explicit history-coverage masks.
- Report NAV return, downside deviation, Sharpe/Sortino, maximum drawdown,
  turnover, fees, invalid actions, and all four Greek exposure paths.
- Retain the implemented paired moving-block intervals and normal/doubled
  spread-and-fee scenarios; add multiple-testing control when comparing many
  model families.

### 2. Strengthen baselines

- No trade and first-feasible policies test environment mechanics.
- Retain the implemented Black-Scholes Delta hedge as a comparator and extend
  it to the new option-liability episodes when historical depth permits.
- Retain the implemented long-volatility IV-versus-realized rule and the
  deterministic collateralized short-put carry comparator using the same
  assignment, collateral, costs, and risk limits as the learned agent.
- Retain the implemented underlying-trend comparator so a recurrent policy
  cannot receive credit for reproducing a trivial covered-return rule.
- Retain the implemented recurrent Monte-Carlo REINFORCE-with-value-baseline
  trainer as an algorithmic comparator to PPO.

### 3. Improve the state without inflating latency

- Retain the implemented column-array environment row path and profile complete
  episodes before adding caches or native extensions. The current measured
  bottleneck was pandas object allocation rather than numerical compute; do not
  trade deterministic first-quote, accounting, or action-mask semantics for a
  synthetic microbenchmark win.
- Keep the 33-field contract state under `dimensionless.v13` as the current
  model: current per-slot quantity, average entry price, and executable
  unrealized return prevent portfolio-state aliasing, while matched bid/ask and
  IV dynamics separate observed change from missing history. Test these through
  the named `position_state` and `contract_dynamics` removal candidates;
  volatility-regime state still belongs once in the market vector.
- Retain prior-snapshot elapsed time only if its named `time_context` removal
  candidate does not win validation; do not silently reinterpret snapshot
  horizons as wall-clock horizons.
- Retain elapsed-time-aware gamma/lambda only if it survives the separate
  fixed-step discount ablation; declare the reference interval before training
  and never tune it on held-out results.
- Extend the implemented realized-volatility state only through ablation-tested
  regime features.
- Retain 4/16-snapshot cumulative return only if the `price_trend` removal
  candidate fails to improve validation selection score after costs.
- Retain the implemented ATM/wing, executable-quote, and Greek coverage instead
  of substituting plausible-looking market values.
- Ablate the implemented executable ATM term slope/curvature and prior-snapshot
  ATM, wing, and term-slope changes before retaining them in a paper strategy.
- Compare the implemented multi-horizon recurrent auxiliary loss against both
  its matched one-step and zero-coefficient candidates; never infer benefit
  from training-loss reduction alone.
- Retain episode-stable contract indices and explicit continuity unless the
  ranked-slot comparison wins validation; inspect churn before interpreting any
  recurrent or graph result.
- Use the implemented walk-forward removal candidates to measure named feature
  groups on validation without exposing every ablation to test; add permutation
  diagnostics only as post-selection sensitivity evidence.

### 4. Earn relational and surface complexity

The dense GNN connects valid contracts by IV, delta, log-moneyness, and DTE
before the GRU/LSTM temporal layer. Its role is cross-contract structure; the
recurrent layer handles time. The implemented `graph_set` variant replaces the
slot-dependent flattened policy with validity-masked mean/max pooling and a
shared per-contract action scorer. This makes option outputs equivariant to slot
permutations, makes global outputs invariant, and cuts policy parameters without
introducing a graph-framework dependency. A zero-neighbor configuration now
removes adjacency construction, graph multiplication, and neighbor weights while
retaining shared pointwise encoding and invariant pooling. It is the low-latency
[Deep Sets](https://arxiv.org/abs/1703.06114) baseline for deciding whether
learned cross-contract messages earn their cost. The architecture supplies the
desired set symmetry, not evidence of trading performance. Next experiments
should compare:

- Flat GRU versus GNN-GRU at matched parameter and latency budgets.
- Zero-neighbor Deep Sets versus neighbor-message graph sets in the same
  validation-only tournament.
- Hand-built neighbor graphs versus learned attention with validity masks.
- Per-ticker training versus the shared graph-set policy with ticker/regime
  context.
- Raw normalized surface features versus causal PCA, then a compact VAE or
  neural-process surface latent only after the data volume supports it.

The flat, flattened-graph, graph-set, and recurrent-family tournament plumbing,
exact parameter-cap matching, and a standardized streaming batch-one inference
benchmark are implemented. Each fold reports median, p95, and mean latency with
runtime context from a training observation. Timing is informational by default; a
predeclared median ceiling can exclude deployment-ineligible candidates before
the validation winner reaches test, with exclusions kept in the artifact. The
next valid experiment needs sufficiently long point-in-time history and should
predeclare any such ceiling. Equal parameter counts do not imply equal graph
construction or recurrent execution cost, and timings cannot be compared
across different machines.
Validation-patience stopping now avoids continuing stalled candidates through
their entire requested budget and records completed episodes per architecture.
This is a compute optimization, not evidence that shorter training improves
returns; serious comparisons should also report equal-budget results.

Named feature-removal candidates now mask surface wings, volatility regime,
data quality, or derived contract-surface inputs inside the recurrent model.
Each is paired with its full-feature architecture and records validation reward
lift; only one validation winner reaches the held-out range.

Do not add a graph framework while 32 dense slots remain simpler and profiling
does not show a net benefit.
Do not train a VAE across a random split of surface days; that would leak future
surface regimes into its representation.

## Promotion gate

A candidate can move from `research_demo` toward a paper strategy only when its
untouched and walk-forward results both beat the designated baseline after
costs, the improvement survives confidence intervals and cost stress, no
feature or universe leakage is found, and its latency fits the intended
decision interval. Live execution remains a separate, explicitly authorized
safety milestone.
