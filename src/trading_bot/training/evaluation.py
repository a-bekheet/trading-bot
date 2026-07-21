"""Minimal reproducible policy evaluation for the research demo."""

from __future__ import annotations

from dataclasses import dataclass, asdict
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
    final_nav: float
    max_drawdown: float
    fees: float
    invalid_actions: int
    executions: int

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode(env: OptionsEnv, policy: Policy, seed: int = 0) -> EpisodeReport:
    observation, _ = env.reset(seed=seed)
    navs = [float(observation.portfolio[2])]
    total_reward = fees = 0.0
    invalid_actions = executions = steps = 0
    while True:
        observation, reward, terminated, truncated, info = env.step(policy(observation))
        total_reward += reward
        fees += float(info["fees"])
        invalid_actions += int(info["invalid_action_count"])
        executions += len(info["executions"])
        navs.append(float(observation.portfolio[2]))
        steps += 1
        if terminated or truncated:
            break
    peak = navs[0]
    drawdowns = []
    for nav in navs:
        peak = max(peak, nav)
        drawdowns.append((peak - nav) / peak if peak else 1.0)
    return EpisodeReport(seed, steps, total_reward, navs[-1], max(drawdowns), fees, invalid_actions, executions)


def evaluate_policy(env_factory: Callable[[], OptionsEnv], policy: Policy, seeds=(0, 1, 2)) -> list[EpisodeReport]:
    """Run the same policy under independent deterministic environment seeds."""
    return [run_episode(env_factory(), policy, seed) for seed in seeds]
