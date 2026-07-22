"""Small deterministic baselines for environment smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_bot.training.env import CONTRACT_FEATURES, MARKET_FEATURES
from trading_bot.training.schemas import Observation


_CONTRACT_INDEX = {
    name: CONTRACT_FEATURES.index(name)
    for name in (
        "delta",
        "dteDays",
        "logMoneyness",
        "spreadPct",
        "openInterestLog",
    )
}
_MARKET_INDEX = {
    name: MARKET_FEATURES.index(name)
    for name in (
        "frontAtmIv",
        "frontAtmIvCoverage",
        "realizedVol4",
        "realizedVol4Coverage",
        "realizedVol16",
        "realizedVol16Coverage",
    )
}


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
        if (
            not np.isfinite(self.min_volatility_edge)
            or self.min_volatility_edge < 0
        ):
            raise ValueError(
                "min_volatility_edge must be finite and nonnegative"
            )
        if self.quantity < 1:
            raise ValueError("quantity must be positive")


class LongVolatilityThenDeltaHedge:
    """Buy a cheap front-ATM call/put pair, then hedge residual Delta."""

    def __init__(self, config: LongVolatilityConfig | None = None) -> None:
        self.config = config or LongVolatilityConfig()
        self._opened = False

    def _signal_present(self, observation: Observation) -> bool:
        edge = _implied_minus_realized(
            observation,
            self.config.realized_window,
            self.config.min_coverage,
        )
        return (
            edge is not None
            and -edge >= self.config.min_volatility_edge
        )

    def _front_atm_pair(self, observation: Observation) -> tuple[int, int] | None:
        quantity = self.config.quantity
        maximum_quantity = (observation.action_mask.shape[1] - 1) // 2
        if quantity > maximum_quantity:
            return None
        valid = _front_expiry_feasible_slots(observation, quantity)
        call_slot = _best_atm_side(observation, valid, "call")
        put_slot = _best_atm_side(observation, valid, "put")
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


def _implied_minus_realized(
    observation: Observation,
    realized_window: int,
    min_coverage: float,
) -> float | None:
    """Return a current-snapshot IV edge only when both inputs are covered."""
    if observation.market.size != len(MARKET_FEATURES):
        return None
    market = observation.market
    realized = float(market[_MARKET_INDEX[f"realizedVol{realized_window}"]])
    realized_coverage = float(
        market[_MARKET_INDEX[f"realizedVol{realized_window}Coverage"]]
    )
    implied = float(market[_MARKET_INDEX["frontAtmIv"]])
    implied_coverage = float(market[_MARKET_INDEX["frontAtmIvCoverage"]])
    if (
        not np.isfinite(
            (realized, realized_coverage, implied, implied_coverage)
        ).all()
        or realized_coverage < min_coverage
        or implied_coverage <= 0
        or implied <= 0
    ):
        return None
    return implied - realized


def _front_expiry_feasible_slots(
    observation: Observation,
    action_index: int,
) -> np.ndarray:
    """Return feasible slots from the nearest valid expiration."""
    if (
        observation.contracts.ndim != 2
        or observation.contracts.shape[1] != len(CONTRACT_FEATURES)
        or action_index < 1
        or action_index >= observation.action_mask.shape[1]
    ):
        return np.empty(0, dtype=int)
    option_rows = len(observation.contract_ids)
    valid = np.flatnonzero(
        observation.valid_mask
        & observation.action_mask[:option_rows, action_index]
    )
    if not len(valid):
        return valid
    dte = observation.contracts[valid, _CONTRACT_INDEX["dteDays"]]
    finite = np.isfinite(dte) & (dte >= 0)
    if not finite.any():
        return np.empty(0, dtype=int)
    valid = valid[finite]
    front_dte = float(np.min(
        observation.contracts[valid, _CONTRACT_INDEX["dteDays"]]
    ))
    return valid[np.isclose(
        observation.contracts[valid, _CONTRACT_INDEX["dteDays"]],
        front_dte,
    )]


def _best_atm_side(
    observation: Observation,
    valid: np.ndarray,
    side: str,
) -> int | None:
    """Rank one call or put by ATM distance, spread, depth, then slot."""
    if side not in {"call", "put"}:
        raise ValueError("side must be call or put")
    contracts = observation.contracts
    delta = contracts[valid, _CONTRACT_INDEX["delta"]]
    side_mask = delta > 0 if side == "call" else delta < 0
    candidates = valid[side_mask]
    if not len(candidates):
        return None
    ranks = np.lexsort((
        candidates,
        -contracts[candidates, _CONTRACT_INDEX["openInterestLog"]],
        contracts[candidates, _CONTRACT_INDEX["spreadPct"]],
        np.abs(contracts[candidates, _CONTRACT_INDEX["logMoneyness"]]),
    ))
    return int(candidates[ranks[0]])


@dataclass(frozen=True)
class ShortVolatilityConfig:
    """Causal entry rule for a cash-secured short-put carry benchmark."""

    realized_window: int = 16
    min_coverage: float = 0.75
    min_volatility_edge: float = 0.02
    quantity: int = 1

    def __post_init__(self) -> None:
        if self.realized_window not in {4, 16}:
            raise ValueError("realized_window must be 4 or 16")
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be between zero and one")
        if (
            not np.isfinite(self.min_volatility_edge)
            or self.min_volatility_edge < 0
        ):
            raise ValueError(
                "min_volatility_edge must be finite and nonnegative"
            )
        if self.quantity < 1:
            raise ValueError("quantity must be positive")


class CashSecuredShortPutThenDeltaHedge:
    """Sell one rich front-ATM put, then reduce its Delta with shares."""

    def __init__(self, config: ShortVolatilityConfig | None = None) -> None:
        self.config = config or ShortVolatilityConfig()
        self._opened = False

    def __call__(self, observation: Observation) -> np.ndarray:
        if self._opened:
            return delta_neutral(observation)
        action = no_op(observation)
        edge = _implied_minus_realized(
            observation,
            self.config.realized_window,
            self.config.min_coverage,
        )
        if edge is None or edge < self.config.min_volatility_edge:
            return action
        maximum_quantity = (observation.action_mask.shape[1] - 1) // 2
        if self.config.quantity > maximum_quantity:
            return action
        sell_index = maximum_quantity + self.config.quantity
        valid = _front_expiry_feasible_slots(observation, sell_index)
        put_slot = _best_atm_side(observation, valid, "put")
        if put_slot is None:
            return action
        action[put_slot] = sell_index
        self._opened = True
        return action


def cash_secured_short_put_delta_hedge(
    config: ShortVolatilityConfig | None = None,
) -> CashSecuredShortPutThenDeltaHedge:
    """Return a fresh collateralized short-volatility benchmark."""
    return CashSecuredShortPutThenDeltaHedge(config)


@dataclass(frozen=True)
class UnderlyingTrendConfig:
    """Causal target-position rule for the underlying trend benchmark."""

    return_window: int = 16
    min_coverage: float = 0.75
    min_abs_log_return: float = 0.0
    quantity: int = 1

    def __post_init__(self) -> None:
        if self.return_window not in {4, 16}:
            raise ValueError("return_window must be 4 or 16")
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be between zero and one")
        if (
            not np.isfinite(self.min_abs_log_return)
            or self.min_abs_log_return < 0
        ):
            raise ValueError("min_abs_log_return must be finite and nonnegative")
        if self.quantity < 1:
            raise ValueError("quantity must be positive")


class UnderlyingTrend:
    """Rebalance shares toward a small position signed by causal price trend."""

    def __init__(self, config: UnderlyingTrendConfig | None = None) -> None:
        self.config = config or UnderlyingTrendConfig()

    def __call__(self, observation: Observation) -> np.ndarray:
        action = no_op(observation)
        underlying_slot = len(observation.contract_ids)
        quantities = np.asarray(observation.underlying_action_quantities, dtype=int)
        maximum_quantity = (len(quantities) - 1) // 2
        if (
            observation.market.size != len(MARKET_FEATURES)
            or observation.portfolio.size < 8
            or underlying_slot >= observation.action_mask.shape[0]
            or self.config.quantity > maximum_quantity
        ):
            return action
        coverage = float(observation.market[MARKET_FEATURES.index(
            f"realizedVol{self.config.return_window}Coverage"
        )])
        trend = float(observation.market[MARKET_FEATURES.index(
            f"underlyingLogReturn{self.config.return_window}"
        )])
        if not np.isfinite((coverage, trend)).all() or coverage < self.config.min_coverage:
            return action

        magnitude = int(abs(quantities[self.config.quantity]))
        threshold = self.config.min_abs_log_return
        target = magnitude if trend > threshold else -magnitude if trend < -threshold else 0
        current = float(observation.portfolio[7])
        if not np.isfinite(current):
            return action
        feasible = np.flatnonzero(observation.action_mask[underlying_slot])
        if not len(feasible):
            return action
        distances = np.abs(current + quantities[feasible] - target)
        best = int(feasible[np.argmin(distances)])
        if distances.min() + 1e-12 < abs(current - target):
            action[underlying_slot] = best
        return action


def underlying_trend(
    config: UnderlyingTrendConfig | None = None,
) -> UnderlyingTrend:
    """Return a deterministic causal underlying-trend comparator."""
    return UnderlyingTrend(config)
