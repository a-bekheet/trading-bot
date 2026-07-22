"""Minimal reproducible policy evaluation for the research demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np

from trading_bot.training.env import OptionsEnv, PORTFOLIO_GREEK_SLICE
from trading_bot.training.schemas import Observation


Policy = Callable[[Observation], np.ndarray]


@dataclass(frozen=True)
class EpisodeReport:
    seed: int
    steps: int
    total_reward: float
    initial_nav: float
    final_nav: float
    total_return: float
    max_drawdown: float
    mean_step_return: float
    step_volatility: float
    downside_deviation: float
    step_sharpe: float
    step_sortino: float
    turnover: float
    fees: float
    invalid_actions: int
    executions: int
    max_abs_delta: float
    max_abs_gamma: float
    max_abs_theta: float
    max_abs_vega: float
    final_delta: float
    final_gamma: float
    final_theta: float
    final_vega: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeTrace:
    """Episode metrics plus the aligned held-out return path."""

    report: EpisodeReport
    timestamps: tuple[str, ...]
    step_returns: tuple[float, ...]


@dataclass(frozen=True)
class BootstrapComparison:
    """Paired dependence-aware uncertainty for agent minus baseline returns."""

    status: str
    metric: str
    observations: int
    bootstrap_samples: int
    block_length: int
    confidence_level: float
    point_estimate: float
    ci_lower: float | None
    ci_upper: float | None
    bootstrap_fraction_positive: float | None
    supports_improvement: bool

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode_trace(
    env: OptionsEnv,
    policy: Policy,
    seed: int = 0,
) -> EpisodeTrace:
    """Run one policy and retain returns aligned to arrival timestamps."""
    observation, _ = env.reset(seed=seed)
    navs = [float(observation.portfolio[2])]
    timestamps = []
    total_reward = fees = 0.0
    invalid_actions = executions = steps = 0
    trade_notional = 0.0
    max_abs_greeks = np.abs(
        observation.portfolio[PORTFOLIO_GREEK_SLICE]
    ).astype(float)
    while True:
        observation, reward, terminated, truncated, info = env.step(policy(observation))
        total_reward += reward
        fees += float(info["fees"])
        invalid_actions += int(info["invalid_action_count"])
        executions += len(info["executions"])
        trade_notional += float(info["trade_notional"])
        max_abs_greeks = np.maximum(
            max_abs_greeks,
            np.abs(observation.portfolio[PORTFOLIO_GREEK_SLICE]),
        )
        navs.append(float(observation.portfolio[2]))
        timestamps.append(observation.timestamp)
        steps += 1
        if terminated or truncated:
            break
    peak = navs[0]
    drawdowns = []
    for nav in navs:
        peak = max(peak, nav)
        drawdowns.append((peak - nav) / peak if peak else 1.0)
    nav_array = np.asarray(navs, dtype=np.float64)
    step_returns = np.divide(
        np.diff(nav_array),
        nav_array[:-1],
        out=np.zeros(len(nav_array) - 1, dtype=np.float64),
        where=nav_array[:-1] != 0,
    )
    mean_return = float(step_returns.mean()) if len(step_returns) else 0.0
    volatility = float(step_returns.std()) if len(step_returns) else 0.0
    downside = float(
        np.sqrt(np.square(np.minimum(step_returns, 0.0)).mean())
    ) if len(step_returns) else 0.0
    final_greeks = observation.portfolio[PORTFOLIO_GREEK_SLICE]
    report = EpisodeReport(
        seed=seed,
        steps=steps,
        total_reward=total_reward,
        initial_nav=navs[0],
        final_nav=navs[-1],
        total_return=(navs[-1] / navs[0] - 1) if navs[0] else -1.0,
        max_drawdown=max(drawdowns),
        mean_step_return=mean_return,
        step_volatility=volatility,
        downside_deviation=downside,
        step_sharpe=mean_return / volatility if volatility > 1e-12 else 0.0,
        step_sortino=mean_return / downside if downside > 1e-12 else 0.0,
        turnover=trade_notional / navs[0] if navs[0] else 0.0,
        fees=fees,
        invalid_actions=invalid_actions,
        executions=executions,
        max_abs_delta=float(max_abs_greeks[0]),
        max_abs_gamma=float(max_abs_greeks[1]),
        max_abs_theta=float(max_abs_greeks[2]),
        max_abs_vega=float(max_abs_greeks[3]),
        final_delta=float(final_greeks[0]),
        final_gamma=float(final_greeks[1]),
        final_theta=float(final_greeks[2]),
        final_vega=float(final_greeks[3]),
    )
    return EpisodeTrace(
        report=report,
        timestamps=tuple(timestamps),
        step_returns=tuple(float(value) for value in step_returns),
    )


def run_episode(env: OptionsEnv, policy: Policy, seed: int = 0) -> EpisodeReport:
    return run_episode_trace(env, policy, seed).report


def paired_moving_block_bootstrap(
    candidate_returns,
    baseline_returns,
    *,
    samples: int = 2_000,
    block_length: int | None = None,
    confidence_level: float = 0.95,
    min_observations: int = 20,
    seed: int = 70_001,
) -> BootstrapComparison:
    """Estimate paired cumulative-log-return lift with circular time blocks."""
    candidate = np.asarray(candidate_returns, dtype=np.float64)
    baseline = np.asarray(baseline_returns, dtype=np.float64)
    if candidate.ndim != 1 or baseline.ndim != 1:
        raise ValueError("paired bootstrap returns must be one-dimensional")
    if candidate.shape != baseline.shape:
        raise ValueError("paired bootstrap paths must have equal length")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if min_observations < 2:
        raise ValueError("min_observations must be at least two")
    if block_length is not None and block_length < 1:
        raise ValueError("block_length must be positive when provided")
    if (
        not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
        or np.any(candidate <= -1)
        or np.any(baseline <= -1)
    ):
        raise ValueError("paired bootstrap returns must be finite and greater than -1")

    observations = len(candidate)
    log_difference = np.log1p(candidate) - np.log1p(baseline)
    point_estimate = float(log_difference.sum())
    if not observations:
        effective_block = 0
    elif block_length is not None:
        effective_block = min(block_length, observations)
    else:
        effective_block = max(
            1,
            min(observations, int(round(np.sqrt(observations)))),
        )
    common = {
        "metric": "cumulative_log_return_difference",
        "observations": observations,
        "bootstrap_samples": samples,
        "block_length": effective_block,
        "confidence_level": confidence_level,
        "point_estimate": point_estimate,
    }
    if observations < min_observations:
        return BootstrapComparison(
            status="insufficient_history",
            ci_lower=None,
            ci_upper=None,
            bootstrap_fraction_positive=None,
            supports_improvement=False,
            **common,
        )

    blocks = int(np.ceil(observations / effective_block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, observations, size=(samples, blocks))
    offsets = np.arange(effective_block)
    indices = (starts[..., None] + offsets) % observations
    indices = indices.reshape(samples, -1)[:, :observations]
    estimates = log_difference[indices].sum(axis=1)
    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(estimates, (tail, 1 - tail))
    fraction_positive = float(np.mean(estimates > 0))
    return BootstrapComparison(
        status="ok",
        ci_lower=float(lower),
        ci_upper=float(upper),
        bootstrap_fraction_positive=fraction_positive,
        supports_improvement=bool(lower > 0),
        **common,
    )


def evaluate_policy(env_factory: Callable[[], OptionsEnv], policy: Policy, seeds=(0, 1, 2)) -> list[EpisodeReport]:
    """Run the same policy under independent deterministic environment seeds."""
    return [run_episode(env_factory(), policy, seed) for seed in seeds]


@dataclass(frozen=True)
class CostScenario:
    name: str
    spread_multiplier: float = 1.0
    commission_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cost scenario name cannot be empty")
        if self.spread_multiplier < 0 or self.commission_multiplier < 0:
            raise ValueError("cost multipliers cannot be negative")


DEFAULT_COST_SCENARIOS = (
    CostScenario("base"),
    CostScenario("double_costs", spread_multiplier=2.0, commission_multiplier=2.0),
)


def cost_stressed_environment(
    source: OptionsEnv,
    scenario: CostScenario,
) -> OptionsEnv:
    commission = source.commission_per_contract * scenario.commission_multiplier
    spread = source.spread_multiplier * scenario.spread_multiplier
    underlying_commission = (
        source.underlying_commission_per_share * scenario.commission_multiplier
    )
    underlying_slippage = (
        source.underlying_slippage_bps * scenario.spread_multiplier
    )
    manifest = replace(
        source.manifest,
        commission_per_contract=commission,
        spread_multiplier=spread,
        portfolio_valuation=source.portfolio_valuation,
        underlying_commission_per_share=underlying_commission,
        underlying_slippage_bps=underlying_slippage,
    )
    return OptionsEnv(
        source.dataset,
        manifest=manifest,
        slot_count=source.slot_count,
        slot_assignment=source.slot_assignment,
        max_quantity=source.max_quantity,
        allow_collateralized_option_shorts=(
            source.allow_collateralized_option_shorts
        ),
        starting_cash=source.starting_cash,
        commission_per_contract=commission,
        spread_multiplier=spread,
        portfolio_valuation=source.portfolio_valuation,
        underlying_lot_size=source.underlying_lot_size,
        max_abs_underlying_shares=source.max_abs_underlying_shares,
        underlying_commission_per_share=underlying_commission,
        underlying_slippage_bps=underlying_slippage,
        invalid_action_penalty=source.invalid_action_penalty,
        reward_drawdown_penalty=source.reward_drawdown_penalty,
        reward_downside_penalty=source.reward_downside_penalty,
        max_abs_delta=source.risk_limits["delta"],
        max_abs_gamma=source.risk_limits["gamma"],
        max_abs_theta=source.risk_limits["theta"],
        max_abs_vega=source.risk_limits["vega"],
    )


def evaluate_cost_stress(
    env: OptionsEnv,
    policy: Policy,
    *,
    scenarios: tuple[CostScenario, ...] = DEFAULT_COST_SCENARIOS,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, list[EpisodeReport]]:
    """Evaluate a stateless policy under identical normal and stressed costs."""
    if len({scenario.name for scenario in scenarios}) != len(scenarios):
        raise ValueError("cost scenario names must be unique")
    return {
        scenario.name: [
            run_episode(cost_stressed_environment(env, scenario), policy, seed)
            for seed in seeds
        ]
        for scenario in scenarios
    }
