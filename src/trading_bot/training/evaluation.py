"""Minimal reproducible policy evaluation for the research demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np

from trading_bot.training.env import OptionsEnv
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

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode(env: OptionsEnv, policy: Policy, seed: int = 0) -> EpisodeReport:
    observation, _ = env.reset(seed=seed)
    navs = [float(observation.portfolio[2])]
    total_reward = fees = 0.0
    invalid_actions = executions = steps = 0
    trade_notional = 0.0
    max_abs_greeks = np.abs(observation.portfolio[3:]).astype(float)
    while True:
        observation, reward, terminated, truncated, info = env.step(policy(observation))
        total_reward += reward
        fees += float(info["fees"])
        invalid_actions += int(info["invalid_action_count"])
        executions += len(info["executions"])
        trade_notional += float(info["trade_notional"])
        max_abs_greeks = np.maximum(
            max_abs_greeks,
            np.abs(observation.portfolio[3:]),
        )
        navs.append(float(observation.portfolio[2]))
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
    return EpisodeReport(
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
    manifest = replace(
        source.manifest,
        commission_per_contract=commission,
        spread_multiplier=spread,
    )
    return OptionsEnv(
        source.dataset,
        manifest=manifest,
        slot_count=source.slot_count,
        max_quantity=source.max_quantity,
        starting_cash=source.starting_cash,
        commission_per_contract=commission,
        spread_multiplier=spread,
        invalid_action_penalty=source.invalid_action_penalty,
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
