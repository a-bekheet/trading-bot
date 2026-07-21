# Alpha research roadmap

Last literature review: 2026-07-22

## Standard of evidence

In this project, "alpha" means an out-of-sample improvement over implementable
baselines after bid/ask spread, commissions, turnover, position limits, and
risk are included. A higher in-sample reward or one favorable seed is not
alpha. Yahoo snapshots remain useful for API and training-pipeline smoke tests,
but they are not enough for a historical performance claim.

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
| [Deep Reinforcement Learning Algorithms for Option Hedging (2025)](https://arxiv.org/abs/2504.05521) | PPO is competitive, but Monte-Carlo policy gradients can be a strong hedge benchmark and sparse terminal rewards matter. | Keep PPO; add delta-hedge and Monte-Carlo policy-gradient comparisons before claiming algorithmic lift. |
| [Risk-Sensitive Contract-unified RL for Option Hedging (2024)](https://arxiv.org/abs/2411.09659) | Learning tail risk of terminal hedging P&L can improve the objective beyond mean reward and allow a policy to span contract conditions. | Add CVaR or learned P&L-distribution objectives only after explicit short-option liability episodes and enough independent paths exist; the current tiny research demo cannot identify tail risk. |
| [ATM S&P 500 options hedging with DRL (2025)](https://arxiv.org/abs/2510.09247) | Moneyness, maturity, realized volatility, current hedge state, walk-forward testing, and transaction-cost stress are central. | Add causal realized-volatility horizons and a formal walk-forward runner. |
| [Deep Hedging with Reinforcement Learning (2025)](https://arxiv.org/abs/2512.12420) | Normalize exposures, combine IV term structure/skew with realized volatility, enforce realistic limits, and quantify uncertainty; attractive point estimates often lose significance. | `dimensionless.v5`, compact ATM-IV-minus-realized-volatility state, Greek budgets, and paired moving-block intervals implement the state/risk lesson. |
| [IV-surface feedback for deep option hedging (revised 2026)](https://arxiv.org/abs/2407.21138) | A compact surface factorization includes ATM level, maturity and moneyness slopes, smile attenuation, and smirk; bounded recurrent hybrids outperform standalone networks in its numerical study. | Add executable 25-delta risk-reversal and butterfly factors with coverage once per market snapshot; test them through the architecture tournament. |
| [Shortfall-aware RL option hedging (2026)](https://arxiv.org/abs/2601.01709) | Better static IV fit need not produce better dynamic hedging; replication-error and shortfall objectives under costs are separate evidence. | Keep realized path diagnostics primary. Defer shortfall/CVaR training until explicit option-liability episodes and enough independent paths exist. |
| [CANDID DAC (2024)](https://arxiv.org/abs/2407.05789) | Independent policies over coupled action dimensions can struggle; sequential policies coordinate dimensions without enumerating the joint action space. | Use a sparse trainable hold prior now. Benchmark an autoregressive multi-leg option policy later; never post-process sampled rows in a way that breaks PPO likelihoods. |
| [Meta-learning neural processes for IV surfaces (2025)](https://arxiv.org/abs/2509.11928) | Log-moneyness/time-to-expiry surface coordinates, cross-day learning, and model-based priors help sparse reconstruction. | Treat a SABR-prior or attention surface encoder as a later experiment, after full-surface history and arbitrage checks exist. |
| [Deep option pricing with market IV surfaces (updated 2026)](https://arxiv.org/abs/2509.05911) | A low-dimensional whole-surface latent representation may retain most surface information. | Benchmark causal PCA first; try VAE/attention compression only if it beats the simpler representation out of sample. |

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

The runner can now compare flat/graph GRU, LSTM, and hybrid candidates within
each fold. All candidates share the fold and seed, architecture selection uses
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

Selection can now use a declared validation-only reward-minus-risk score with
separate maximum-drawdown, downside-deviation, and turnover coefficients. The
score controls early stopping, checkpoint restoration, algorithms,
architectures, and feature ablations consistently while retaining every raw
component. Coefficients remain zero by default and must be fixed without test
feedback.

The market state now includes front-expiry ATM IV and its difference from
backward-only 4/16-snapshot realized volatility, each paired with the existing
history coverage. PPO training samples seeded bounded windows across the
training partition instead of replaying only its first regime. Both choices
improve sample efficiency; their value still requires walk-forward ablation.

The compact market state also includes executable front-expiry 25-delta risk
reversal and butterfly factors. Explicit ATM/wing, quote, and Greek coverage
prevent a missing or zero-bid surface from looking like a real zero signal.
These factors use only the current cross-section and add six scalar inputs,
leaving the per-contract graph width unchanged.

The policy head now has a trainable hold-logit prior and reward-scale entropy
coefficient. This reduced untrained requested action density on the current
AAPL surface without imposing a hard order cap or changing PPO likelihoods.
Episode provenance reports requested option and hedge actions separately.

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
  it to explicit option-liability episodes when the historical dataset permits.
- Retain the implemented long-volatility IV-versus-realized rule; add a
  collateralized short-volatility/carry comparator only after margin, assignment,
  and option-liability accounting exist.
- Retain the implemented recurrent Monte-Carlo REINFORCE-with-value-baseline
  trainer as an algorithmic comparator to PPO.

### 3. Improve the state without inflating latency

- Keep the 25-field `dimensionless.v5` contract state as the minimum model;
  volatility-regime state belongs once in the market vector.
- Extend the implemented realized-volatility state only through ablation-tested
  regime features.
- Retain the implemented ATM/wing, executable-quote, and Greek coverage instead
  of substituting plausible-looking market values.
- Use the implemented walk-forward removal candidates to measure named feature
  groups on validation without exposing every ablation to test; add permutation
  diagnostics only as post-selection sensitivity evidence.

### 4. Earn relational and surface complexity

The current dense GNN connects valid contracts by IV, delta, log-moneyness, and
DTE before the GRU/LSTM temporal layer. Its role is cross-contract structure;
the recurrent layer handles time. Next experiments should compare:

- Flat GRU versus GNN-GRU at matched parameter and latency budgets.
- Hand-built neighbor graphs versus learned attention with validity masks.
- Per-ticker training versus a shared graph policy with ticker/regime context.
- Raw normalized surface features versus causal PCA, then a compact VAE or
  neural-process surface latent only after the data volume supports it.

The flat/graph and recurrent-family tournament plumbing, exact parameter-cap
matching, and a standardized streaming batch-one inference benchmark are
implemented. Each fold reports median, p95, and mean latency with runtime
context from a training observation; timing does not affect model selection.
The next valid experiment needs sufficiently long point-in-time history and
should predeclare how latency evidence will constrain deployable candidates.
Equal parameter counts do not imply equal graph construction or recurrent
execution cost, and timings cannot be compared across different machines.
Validation-patience stopping now avoids continuing stalled candidates through
their entire requested budget and records completed episodes per architecture.
This is a compute optimization, not evidence that shorter training improves
returns; serious comparisons should also report equal-budget results.

Named feature-removal candidates now mask surface wings, volatility regime,
data quality, or derived contract-surface inputs inside the recurrent model.
Each is paired with its full-feature architecture and records validation reward
lift; only one validation winner reaches the held-out range.

Do not add a graph framework while 32 dense slots remain faster and simpler.
Do not train a VAE across a random split of surface days; that would leak future
surface regimes into its representation.

## Promotion gate

A candidate can move from `research_demo` toward a paper strategy only when its
untouched and walk-forward results both beat the designated baseline after
costs, the improvement survives confidence intervals and cost stress, no
feature or universe leakage is found, and its latency fits the intended
decision interval. Live execution remains a separate, explicitly authorized
safety milestone.
