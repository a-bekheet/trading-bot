"""Fixed-shape sequence windows for recurrent policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_bot.training.env import CONTRACT_FEATURES, MARKET_FEATURES
from trading_bot.training.features import REALIZED_VOL_WINDOWS
from trading_bot.training.schemas import Observation


def _dimensionless_components(
    observation: Observation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-rule, point-in-time scaling with no fitted future state."""
    market = np.asarray(observation.market, dtype=np.float64).copy()
    contracts = np.asarray(observation.contracts, dtype=np.float64).copy()
    portfolio = np.asarray(observation.portfolio, dtype=np.float64).copy()
    if contracts.ndim != 2 or contracts.shape[1] != len(CONTRACT_FEATURES):
        return market, contracts, portfolio

    spot = abs(float(market[0])) if len(market) else 0.0
    spot_scale = max(spot, 1e-8)
    indices = {name: index for index, name in enumerate(CONTRACT_FEATURES)}
    for name in ("strike", "lastPrice", "bid", "ask", "midPrice", "spread"):
        contracts[:, indices[name]] /= spot_scale
    # Gamma times a 10% spot move is the approximate corresponding Delta
    # change and remains comparable across differently priced underlyings.
    contracts[:, indices["gamma"]] *= spot_scale * 0.1
    contracts[:, indices["theta"]] /= spot_scale
    contracts[:, indices["vega"]] /= spot_scale
    contracts[:, indices["dteDays"]] /= 365.0
    contracts[:, indices["volumeLog"]] /= 10.0
    contracts[:, indices["openInterestLog"]] /= 10.0
    contracts[:, indices["quoteAgeSeconds"]] = (
        np.log1p(np.maximum(contracts[:, indices["quoteAgeSeconds"]], 0.0)) / 10.0
    )

    if len(market):
        # Contract prices and Greeks are expressed relative to spot below, so
        # retaining the absolute dollar price only makes policies ticker-scale
        # dependent. Keep a unit reference for valid positive spot values.
        market[0] = 1.0 if market[0] > 0 else 0.0
    if len(market) == len(MARKET_FEATURES):
        market_indices = {
            name: index for index, name in enumerate(MARKET_FEATURES)
        }
        return_index = market_indices["underlyingReturn"]
        market[return_index] = (
            np.sign(market[return_index])
            * np.log1p(abs(market[return_index]) * 100.0)
        )
        for window in REALIZED_VOL_WINDOWS:
            volatility_index = market_indices[f"realizedVol{window}"]
            market[volatility_index] = np.log1p(
                max(market[volatility_index], 0.0)
            )
    if len(portfolio) == 8:
        nav = float(portfolio[2])
        nav_scale = max(abs(nav), 1.0)
        underlying_notional = abs(float(portfolio[7])) * spot_scale
        deployed_capital = max(
            abs(float(portfolio[0]))
            + abs(float(portfolio[1]))
            + underlying_notional,
            1.0,
        )
        portfolio = np.array(
            [
                portfolio[0] / nav_scale,
                portfolio[1] / nav_scale,
                nav / deployed_capital,
                portfolio[3] * spot_scale / nav_scale,
                portfolio[4] * spot_scale**2 / nav_scale,
                portfolio[5] / nav_scale,
                portfolio[6] / nav_scale,
                portfolio[7] * spot_scale / nav_scale,
            ],
            dtype=np.float64,
        )

    market = np.nan_to_num(market, nan=0.0, posinf=10.0, neginf=-10.0)
    contracts = np.nan_to_num(contracts, nan=0.0, posinf=10.0, neginf=-10.0)
    portfolio = np.nan_to_num(portfolio, nan=0.0, posinf=10.0, neginf=-10.0)
    return (
        np.clip(market, -10.0, 10.0),
        np.clip(contracts, -10.0, 10.0),
        np.clip(portfolio, -10.0, 10.0),
    )


def observation_vector(observation: Observation) -> np.ndarray:
    """Flatten one observation using the versioned dimensionless transform."""
    market, contracts, portfolio = _dimensionless_components(observation)
    return np.concatenate(
        (
            market.ravel(),
            contracts.ravel(),
            portfolio.ravel(),
            observation.valid_mask.astype(np.float64).ravel(),
        )
    ).astype(np.float32)


@dataclass(frozen=True)
class SequenceWindow:
    features: np.ndarray
    actions: np.ndarray | None = None
    rewards: np.ndarray | None = None


def build_windows(
    observations: list[Observation],
    *,
    window: int,
    actions: list[np.ndarray] | None = None,
    rewards: list[float] | None = None,
) -> list[SequenceWindow]:
    """Build chronological, non-padded windows; no future rows are included."""
    if window < 1:
        raise ValueError("window must be positive")
    if len(observations) < window:
        return []
    vectors = np.stack([observation_vector(item) for item in observations])
    result = []
    for end in range(window, len(observations) + 1):
        start = end - window
        result.append(
            SequenceWindow(
                features=vectors[start:end],
                actions=np.stack(actions[start:end]) if actions is not None else None,
                rewards=np.asarray(rewards[start:end], dtype=np.float32) if rewards is not None else None,
            )
        )
    return result
