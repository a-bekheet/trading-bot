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
3. Multiple seeds, bootstrap confidence intervals, turnover and drawdown
   reporting, and doubled-cost stress.
4. Ablation against a smaller flat GRU. LSTM, hybrid GRU+LSTM, GNN, and surface
   latent models earn their complexity only through out-of-sample improvement.
5. A feature-availability audit proving that every input existed before the
   simulated decision timestamp.

## What recent work changes

| Evidence | Useful idea for this repository | Decision |
| --- | --- | --- |
| [Deep Reinforcement Learning Algorithms for Option Hedging (2025)](https://arxiv.org/abs/2504.05521) | PPO is competitive, but Monte-Carlo policy gradients can be a strong hedge benchmark and sparse terminal rewards matter. | Keep PPO; add delta-hedge and Monte-Carlo policy-gradient comparisons before claiming algorithmic lift. |
| [ATM S&P 500 options hedging with DRL (2025)](https://arxiv.org/abs/2510.09247) | Moneyness, maturity, realized volatility, current hedge state, walk-forward testing, and transaction-cost stress are central. | Add causal realized-volatility horizons and a formal walk-forward runner. |
| [Deep Hedging with Reinforcement Learning (2025)](https://arxiv.org/abs/2512.12420) | Normalize exposures, enforce realistic limits, compare against simple investments, and quantify uncertainty; attractive point estimates often lose significance. | `dimensionless.v2` and Greek budgets implement the state/risk lesson; bootstrap intervals remain an evaluation gate. |
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
then evaluates the held-out test range against no-op, first-feasible, and
doubled-cost scenarios. The next baseline gap is an implementable delta hedge;
that requires adding an underlying-asset action rather than pretending an option
order is an equivalent hedge.

## Prioritized implementation sequence

### 1. Make evaluation credible

- Store sufficient timestamped, point-in-time option and underlying history.
- Run the implemented full training runner after sufficient history is stored.
- Retain the implemented backward-only realized-volatility horizons and
  explicit history-coverage masks.
- Report NAV return, downside deviation, Sharpe/Sortino, maximum drawdown,
  turnover, fees, invalid actions, and all four Greek exposure paths.
- Add block-bootstrap intervals; normal/doubled spread-and-fee scenarios are
  already executable.

### 2. Strengthen baselines

- No trade and first-feasible policies already test environment mechanics.
- Add a Black-Scholes delta hedge with the same action and risk constraints.
- Add simple IV mean-reversion/carry rules that use only available quotes.
- Add a Monte-Carlo policy-gradient trainer as an algorithmic comparator.

### 3. Improve the state without inflating latency

- Keep the 25-field `dimensionless.v2` contract state as the minimum model.
- Extend the implemented realized-volatility state only through ablation-tested
  regime features.
- Include explicit missingness/quote-quality masks instead of substituting
  plausible-looking market values.
- Measure feature value with walk-forward permutation and removal ablations.

### 4. Earn relational and surface complexity

The current dense GNN connects valid contracts by IV, delta, log-moneyness, and
DTE before the GRU/LSTM temporal layer. Its role is cross-contract structure;
the recurrent layer handles time. Next experiments should compare:

- Flat GRU versus GNN-GRU at matched parameter and latency budgets.
- Hand-built neighbor graphs versus learned attention with validity masks.
- Per-ticker training versus a shared graph policy with ticker/regime context.
- Raw normalized surface features versus causal PCA, then a compact VAE or
  neural-process surface latent only after the data volume supports it.

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
