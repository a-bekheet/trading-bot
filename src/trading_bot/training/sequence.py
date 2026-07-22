"""Fixed-shape sequence windows for recurrent policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_bot.training.env import CONTRACT_FEATURES, MARKET_FEATURES
from trading_bot.training.features import REALIZED_VOL_WINDOWS
from trading_bot.training.schemas import Observation


FEATURE_ABLATION_GROUPS = {
    "slot_identity": (
        "slotContinuity",
    ),
    "surface_wings": (
        "front25DeltaRiskReversal",
        "front25DeltaButterfly",
        "front25DeltaCoverage",
    ),
    "term_structure": (
        "atmTermStructureSlope",
        "atmTermStructureCurvature",
        "atmTermSlopeCoverage",
        "atmTermCurvatureCoverage",
    ),
    "surface_dynamics": (
        "frontAtmIvChange",
        "frontAtmIvChangeCoverage",
        "front25DeltaRiskReversalChange",
        "front25DeltaButterflyChange",
        "frontWingChangeCoverage",
        "atmTermStructureSlopeChange",
        "atmTermSlopeChangeCoverage",
    ),
    "volatility_regime": (
        "realizedVol4",
        "realizedVol4Coverage",
        "realizedVol16",
        "realizedVol16Coverage",
        "frontAtmIv",
        "frontAtmIvCoverage",
        "atmIvMinusRealizedVol4",
        "atmIvMinusRealizedVol16",
    ),
    "data_quality": (
        "executableQuoteCoverage",
        "greekCoverage",
    ),
    "derived_contract_surface": (
        "forwardLogMoneyness",
        "extrinsicValuePct",
        "atmIv",
        "ivSkew",
        "atmTermSlope",
        "putCallIvSpread",
        "parityResidual",
    ),
}


AUXILIARY_TARGET_FEATURES = (
    "underlyingReturn",
    "frontAtmIvChange",
    "front25DeltaRiskReversalChange",
    "front25DeltaButterflyChange",
    "atmTermStructureSlopeChange",
)

_AUXILIARY_LEVEL_FEATURES = (
    "underlyingPrice",
    "frontAtmIv",
    "front25DeltaRiskReversal",
    "front25DeltaButterfly",
    "atmTermStructureSlope",
)

_AUXILIARY_LEVEL_COVERAGE = {
    "frontAtmIv": ("frontAtmIvCoverage", 0.0),
    "front25DeltaRiskReversal": ("front25DeltaCoverage", 1.0),
    "front25DeltaButterfly": ("front25DeltaCoverage", 1.0),
    "atmTermStructureSlope": ("atmTermSlopeCoverage", 1.0),
}


def feature_ablation_indices(
    groups: tuple[str, ...],
    slot_count: int,
) -> tuple[int, ...]:
    """Map named feature groups to stable flattened observation indices."""
    if slot_count < 1:
        raise ValueError("slot_count must be positive")
    if len(set(groups)) != len(groups):
        raise ValueError("feature ablation groups must be unique")
    unknown = set(groups) - set(FEATURE_ABLATION_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature ablation groups: {sorted(unknown)}")

    indices = set()
    contract_start = len(MARKET_FEATURES)
    for group in groups:
        for name in FEATURE_ABLATION_GROUPS[group]:
            if name in MARKET_FEATURES:
                indices.add(MARKET_FEATURES.index(name))
                continue
            contract_index = CONTRACT_FEATURES.index(name)
            for slot in range(slot_count):
                indices.add(
                    contract_start
                    + slot * len(CONTRACT_FEATURES)
                    + contract_index
                )
    return tuple(sorted(indices))


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
        front_iv_index = market_indices["frontAtmIv"]
        market[front_iv_index] = np.log1p(max(market[front_iv_index], 0.0))
        for name in (
            "front25DeltaRiskReversal",
            "front25DeltaButterfly",
            "atmTermStructureSlope",
            "atmTermStructureCurvature",
            "frontAtmIvChange",
            "front25DeltaRiskReversalChange",
            "front25DeltaButterflyChange",
            "atmTermStructureSlopeChange",
        ):
            index = market_indices[name]
            value = market[index]
            market[index] = np.sign(value) * np.log1p(abs(value))
        for window in REALIZED_VOL_WINDOWS:
            spread_index = market_indices[f"atmIvMinusRealizedVol{window}"]
            spread = market[spread_index]
            market[spread_index] = np.sign(spread) * np.log1p(abs(spread))
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


def auxiliary_market_change_targets(
    current: Observation,
    future: Observation,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded point-to-point market changes and coverage.

    Consecutive observations reproduce the existing one-step target semantics.
    Non-consecutive observations create cumulative changes for an explicitly
    requested future horizon without adding any future value to policy inputs.
    """
    current_market = np.asarray(current.market, dtype=np.float64)
    future_market = np.asarray(future.market, dtype=np.float64)
    if (
        len(current_market) != len(MARKET_FEATURES)
        or len(future_market) != len(MARKET_FEATURES)
    ):
        raise ValueError("observation market layout does not match auxiliary targets")
    indices = {name: index for index, name in enumerate(MARKET_FEATURES)}
    values = np.zeros(len(AUXILIARY_TARGET_FEATURES), dtype=np.float64)
    available = np.zeros(len(AUXILIARY_TARGET_FEATURES), dtype=np.float32)

    current_spot = current_market[indices["underlyingPrice"]]
    future_spot = future_market[indices["underlyingPrice"]]
    if (
        np.isfinite(current_spot)
        and current_spot > 0
        and np.isfinite(future_spot)
        and future_spot > 0
    ):
        change = future_spot / current_spot - 1.0
        values[0] = np.sign(change) * np.log1p(abs(change) * 100.0)
        available[0] = 1.0

    for target_index, level_name in enumerate(
        _AUXILIARY_LEVEL_FEATURES[1:],
        start=1,
    ):
        coverage_name, threshold = _AUXILIARY_LEVEL_COVERAGE[level_name]
        current_coverage = current_market[indices[coverage_name]]
        future_coverage = future_market[indices[coverage_name]]
        current_level = current_market[indices[level_name]]
        future_level = future_market[indices[level_name]]
        coverage_available = (
            np.isfinite(current_coverage)
            and np.isfinite(future_coverage)
            and (
                min(current_coverage, future_coverage) > 0
                if threshold == 0
                else min(current_coverage, future_coverage) >= threshold
            )
        )
        if (
            coverage_available
            and np.isfinite(current_level)
            and np.isfinite(future_level)
        ):
            change = future_level - current_level
            values[target_index] = np.sign(change) * np.log1p(abs(change))
            available[target_index] = 1.0

    values = np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(values, -10.0, 10.0).astype(np.float32), available


def multi_horizon_auxiliary_targets(
    observations: list[Observation],
    horizons: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Align cumulative future-market targets to preceding policy states."""
    normalized = tuple(horizons)
    if (
        not normalized
        or any(not isinstance(item, int) or isinstance(item, bool) for item in normalized)
        or any(item < 1 for item in normalized)
        or tuple(sorted(set(normalized))) != normalized
    ):
        raise ValueError("auxiliary horizons must be unique positive increasing integers")
    transition_count = max(len(observations) - 1, 0)
    width = len(normalized) * len(AUXILIARY_TARGET_FEATURES)
    values = np.zeros((transition_count, width), dtype=np.float32)
    available = np.zeros((transition_count, width), dtype=np.float32)
    for step in range(transition_count):
        for horizon_index, horizon in enumerate(normalized):
            future_index = step + horizon
            if future_index >= len(observations):
                continue
            target, mask = auxiliary_market_change_targets(
                observations[step],
                observations[future_index],
            )
            start = horizon_index * len(AUXILIARY_TARGET_FEATURES)
            end = start + len(AUXILIARY_TARGET_FEATURES)
            values[step, start:end] = target
            available[step, start:end] = mask
    return values, available


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
