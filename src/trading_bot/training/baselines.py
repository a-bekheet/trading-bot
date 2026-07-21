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
    for slot in range(len(observation.contract_ids)):
        feasible = np.flatnonzero(observation.action_mask[slot, 1:])
        if len(feasible):
            action[slot] = int(feasible[0] + 1)
            break
    return action


def delta_neutral(observation: Observation) -> np.ndarray:
    """Trade underlying shares to reduce current absolute portfolio Delta."""
    action = no_op(observation)
    underlying_slot = len(observation.contract_ids)
    quantities = np.asarray(observation.underlying_action_quantities, dtype=int)
    if underlying_slot >= observation.action_mask.shape[0] or len(quantities) < 2:
        return action
    feasible = observation.action_mask[underlying_slot]
    delta = float(observation.portfolio[3])
    candidates = np.flatnonzero(feasible)
    if not len(candidates):
        return action
    best = int(candidates[np.argmin(np.abs(delta + quantities[candidates]))])
    if abs(delta + quantities[best]) + 1e-12 < abs(delta):
        action[underlying_slot] = best
    return action


class BuyFirstThenDeltaHedge:
    """Open one feasible option, then reduce its portfolio Delta with shares."""

    def __init__(self) -> None:
        self._opened = False

    def __call__(self, observation: Observation) -> np.ndarray:
        if not self._opened:
            action = first_feasible(observation)
            self._opened = bool(action[:len(observation.contract_ids)].any())
            return action
        return delta_neutral(observation)


def buy_first_then_delta_hedge() -> BuyFirstThenDeltaHedge:
    """Return fresh stateful policy state for one evaluation episode."""
    return BuyFirstThenDeltaHedge()
