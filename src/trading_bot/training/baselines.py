"""Small deterministic baselines for environment smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_bot.training.env import CONTRACT_FEATURES, MARKET_FEATURES
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


@dataclass(frozen=True)
class LongVolatilityConfig:
    """Causal entry rule for the long-volatility benchmark."""

    realized_window: int = 16
    min_coverage: float = 0.75
    min_volatility_edge: float = 0.02
    quantity: int = 1

    def __post_init__(self) -> None:
        if self.realized_window not in {4, 16}:
            raise ValueError("realized_window must be 4 or 16")
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be between zero and one")
        if self.min_volatility_edge < 0:
            raise ValueError("min_volatility_edge cannot be negative")
        if self.quantity < 1:
            raise ValueError("quantity must be positive")


class LongVolatilityThenDeltaHedge:
    """Buy a cheap front-ATM call/put pair, then hedge residual Delta."""

    def __init__(self, config: LongVolatilityConfig | None = None) -> None:
        self.config = config or LongVolatilityConfig()
        self._opened = False

    def _signal_present(self, observation: Observation) -> bool:
        if observation.market.size != len(MARKET_FEATURES):
            return False
        market = {
            name: float(observation.market[index])
            for index, name in enumerate(MARKET_FEATURES)
        }
        window = self.config.realized_window
        coverage = market[f"realizedVol{window}Coverage"]
        realized = market[f"realizedVol{window}"]
        implied = market["frontAtmIv"]
        return (
            np.isfinite((coverage, realized, implied)).all()
            and coverage >= self.config.min_coverage
            and implied > 0
            and realized - implied >= self.config.min_volatility_edge
        )

    def _front_atm_pair(self, observation: Observation) -> tuple[int, int] | None:
        if observation.contracts.shape[1] != len(CONTRACT_FEATURES):
            return None
        quantity = self.config.quantity
        if quantity >= observation.action_mask.shape[1]:
            return None
        valid = np.flatnonzero(
            observation.valid_mask
            & observation.action_mask[:len(observation.contract_ids), quantity]
        )
        if not len(valid):
            return None
        feature = {
            name: CONTRACT_FEATURES.index(name)
            for name in (
                "delta",
                "dteDays",
                "logMoneyness",
                "spreadPct",
                "openInterestLog",
            )
        }
        contracts = observation.contracts
        dte = contracts[valid, feature["dteDays"]]
        finite_dte = np.isfinite(dte) & (dte >= 0)
        if not finite_dte.any():
            return None
        valid = valid[finite_dte]
        front_dte = float(np.min(contracts[valid, feature["dteDays"]]))
        valid = valid[
            np.isclose(
                contracts[valid, feature["dteDays"]],
                front_dte,
            )
        ]

        def best(side: str) -> int | None:
            delta = contracts[valid, feature["delta"]]
            side_mask = delta > 0 if side == "call" else delta < 0
            candidates = valid[side_mask]
            if not len(candidates):
                return None
            ranks = np.lexsort(
                (
                    candidates,
                    -contracts[candidates, feature["openInterestLog"]],
                    contracts[candidates, feature["spreadPct"]],
                    np.abs(contracts[candidates, feature["logMoneyness"]]),
                )
            )
            return int(candidates[ranks[0]])

        call_slot, put_slot = best("call"), best("put")
        if call_slot is None or put_slot is None or call_slot == put_slot:
            return None
        return call_slot, put_slot

    def __call__(self, observation: Observation) -> np.ndarray:
        if self._opened:
            return delta_neutral(observation)
        action = no_op(observation)
        if not self._signal_present(observation):
            return action
        pair = self._front_atm_pair(observation)
        if pair is None:
            return action
        action[list(pair)] = self.config.quantity
        self._opened = True
        return action


def long_volatility_delta_hedge(
    config: LongVolatilityConfig | None = None,
) -> LongVolatilityThenDeltaHedge:
    """Return fresh long-volatility benchmark state for one episode."""
    return LongVolatilityThenDeltaHedge(config)
