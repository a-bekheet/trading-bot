"""Small deterministic baselines for environment smoke tests."""

from __future__ import annotations

import numpy as np

from trading_bot.training.schemas import Observation


def no_op(observation: Observation) -> np.ndarray:
    """Always hold."""
    return np.zeros(observation.action_mask.shape[0], dtype=int)


def first_feasible(observation: Observation) -> np.ndarray:
    """Buy one feasible contract in the first available slot, otherwise hold."""
    action = no_op(observation)
    for slot in range(observation.action_mask.shape[0]):
        feasible = np.flatnonzero(observation.action_mask[slot, 1:])
        if len(feasible):
            action[slot] = int(feasible[0] + 1)
            break
    return action
