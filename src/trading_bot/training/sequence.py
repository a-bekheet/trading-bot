"""Fixed-shape sequence windows for recurrent policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_bot.market_data.freshness import (
    DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS,
)
from trading_bot.training.env import (
    ACTION_FEASIBILITY_CONTRACT_FEATURES,
    CONTRACT_FEATURES,
    MARKET_FEATURES,
    PORTFOLIO_FEATURES,
)
from trading_bot.training.features import (
    BENCHMARK_CONTEXT_FEATURES,
    CONTRACT_SMILE_RESIDUAL_FEATURES,
    CONTRACT_DYNAMICS_FEATURES,
    INTRADAY_CLOCK_FEATURES,
    REALIZED_VOL_WINDOWS,
    SMILE_FIT_FEATURES,
    SURFACE_VELOCITY_FEATURES,
)
from trading_bot.training.schemas import Observation


FEATURE_ABLATION_GROUPS = {
    "slot_identity": ("slotContinuity",),
    "position_state": (
        "positionQuantity",
        "positionAveragePrice",
        "positionUnrealizedReturn",
    ),
    "position_lifecycle": (
        "positionAgeSteps",
        "positionLastTradeAgeSteps",
    ),
    "action_feasibility": (
        *ACTION_FEASIBILITY_CONTRACT_FEATURES,
        "underlyingBuyFeasibleFraction",
        "underlyingSellFeasibleFraction",
    ),
    "contract_dynamics": CONTRACT_DYNAMICS_FEATURES,
    "contract_smile_residual": CONTRACT_SMILE_RESIDUAL_FEATURES,
    "static_arbitrage": (
        "verticalArbitrageViolationPct",
        "butterflyArbitrageViolationPct",
    ),
    "time_context": (
        "snapshotGapSeconds",
        "snapshotGapCoverage",
    ),
    "intraday_clock": INTRADAY_CLOCK_FEATURES,
    "market_session": (
        "regularMarketSession",
        "marketStateCoverage",
    ),
    "data_freshness": (
        "underlyingQuoteAgeSeconds",
        "underlyingQuoteAgeCoverage",
    ),
    "systematic_context": BENCHMARK_CONTEXT_FEATURES,
    "price_trend": (
        "underlyingLogReturn4",
        "underlyingLogReturn16",
    ),
    "surface_wings": (
        "front25DeltaRiskReversal",
        "front25DeltaButterfly",
        "front25DeltaCoverage",
    ),
    "smile_fit": SMILE_FIT_FEATURES,
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
    "surface_velocity": SURFACE_VELOCITY_FEATURES,
    "volatility_regime": (
        "realizedVol4",
        "realizedVol16",
        "frontAtmIv",
        "frontAtmIvCoverage",
        "atmIvMinusRealizedVol4",
        "atmIvMinusRealizedVol16",
    ),
    "volatility_normalization": (
        "frontAtmIvZScore16",
        "volatilityRiskPremiumZScore16",
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


MARKET_AUXILIARY_TARGET_FEATURES = (
    "underlyingReturn",
    "frontAtmIvChange",
    "front25DeltaRiskReversalChange",
    "front25DeltaButterflyChange",
    "atmTermStructureSlopeChange",
)
CONTRACT_AUXILIARY_TARGET_FEATURES = (
    "medianContractMidPriceLogReturn",
    "medianContractSpreadPctChange",
    "medianContractIvChange",
    "medianContractDeltaHedgedSpotReturn",
)
CONTRACT_AUXILIARY_MIN_COVERAGE = 0.5
DELTA_HEDGED_AUXILIARY_MAX_UNDERLYING_QUOTE_AGE_SECONDS = (
    DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS
)
AUXILIARY_TARGET_FEATURES = (
    *MARKET_AUXILIARY_TARGET_FEATURES,
    *CONTRACT_AUXILIARY_TARGET_FEATURES,
)


def normalize_auxiliary_target_exclusions(
    exclusions: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate a stable train-only auxiliary target-removal contract."""
    normalized = tuple(exclusions)
    if len(set(normalized)) != len(normalized):
        raise ValueError("auxiliary target exclusions must be unique")
    unknown = set(normalized) - set(AUXILIARY_TARGET_FEATURES)
    if unknown:
        raise ValueError(f"unknown auxiliary target exclusions: {sorted(unknown)}")
    if len(normalized) == len(AUXILIARY_TARGET_FEATURES):
        raise ValueError("auxiliary target exclusions cannot remove every target")
    return normalized


def auxiliary_target_exclusion_indices(
    exclusions: tuple[str, ...],
    horizons: tuple[int, ...],
) -> tuple[int, ...]:
    """Return flattened multi-horizon indices disabled only in the loss mask."""
    normalized = normalize_auxiliary_target_exclusions(exclusions)
    target_indices = tuple(AUXILIARY_TARGET_FEATURES.index(name) for name in normalized)
    width = len(AUXILIARY_TARGET_FEATURES)
    return tuple(
        horizon_index * width + target_index
        for horizon_index in range(len(horizons))
        for target_index in target_indices
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

_CONTRACT_FEATURE_INDEX = {name: index for index, name in enumerate(CONTRACT_FEATURES)}
_MARKET_FEATURE_INDEX = {name: index for index, name in enumerate(MARKET_FEATURES)}


def _delta_hedged_endpoint_is_observable(market: np.ndarray) -> bool:
    """Require explicit regular-session and fresh-underlying provenance."""
    required = {
        "regularMarketSession",
        "marketStateCoverage",
        "underlyingQuoteAgeSeconds",
        "underlyingQuoteAgeCoverage",
    }
    if len(market) != len(MARKET_FEATURES) or not required.issubset(
        _MARKET_FEATURE_INDEX
    ):
        return False
    regular = market[_MARKET_FEATURE_INDEX["regularMarketSession"]]
    state_coverage = market[_MARKET_FEATURE_INDEX["marketStateCoverage"]]
    quote_age = market[_MARKET_FEATURE_INDEX["underlyingQuoteAgeSeconds"]]
    quote_age_coverage = market[_MARKET_FEATURE_INDEX["underlyingQuoteAgeCoverage"]]
    return bool(
        np.isfinite(regular)
        and regular >= 0.5
        and np.isfinite(state_coverage)
        and state_coverage >= 0.5
        and np.isfinite(quote_age)
        and 0 <= quote_age <= DELTA_HEDGED_AUXILIARY_MAX_UNDERLYING_QUOTE_AGE_SECONDS
        and np.isfinite(quote_age_coverage)
        and quote_age_coverage >= 0.5
    )


_SPOT_SCALED_CONTRACT_INDICES = np.asarray(
    [
        _CONTRACT_FEATURE_INDEX[name]
        for name in (
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "midPrice",
            "spread",
            "positionAveragePrice",
        )
    ]
)
_SIGNED_SURFACE_MARKET_FEATURES = (
    "front25DeltaRiskReversal",
    "front25DeltaButterfly",
    "frontSmileCurvature",
    "frontAtmSmileResidualPct",
    "atmTermStructureSlope",
    "atmTermStructureCurvature",
    "frontAtmIvChange",
    "front25DeltaRiskReversalChange",
    "front25DeltaButterflyChange",
    "atmTermStructureSlopeChange",
    *SURFACE_VELOCITY_FEATURES,
)
_SIGNED_CONTRACT_FEATURES = (
    "positionQuantity",
    "positionUnrealizedReturn",
    "midPriceLogReturn",
    "spreadPctChange",
    "ivChange",
    "smileResidualPct",
)
_SIGNED_CONTRACT_INDICES = np.asarray(
    [_CONTRACT_FEATURE_INDEX[name] for name in _SIGNED_CONTRACT_FEATURES]
)
_SIGNED_CONTRACT_SCALES = np.asarray(
    (
        1.0,
        1.0,
        100.0,
        10.0,
        10.0,
        10.0,
    )
)


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
    portfolio_start = contract_start + slot_count * len(CONTRACT_FEATURES)
    for group in groups:
        for name in FEATURE_ABLATION_GROUPS[group]:
            if name in MARKET_FEATURES:
                indices.add(MARKET_FEATURES.index(name))
                continue
            if name in PORTFOLIO_FEATURES:
                indices.add(portfolio_start + PORTFOLIO_FEATURES.index(name))
                continue
            contract_index = _CONTRACT_FEATURE_INDEX[name]
            for slot in range(slot_count):
                indices.add(
                    contract_start + slot * len(CONTRACT_FEATURES) + contract_index
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
    indices = _CONTRACT_FEATURE_INDEX
    contracts[:, _SPOT_SCALED_CONTRACT_INDICES] = (
        contracts[:, _SPOT_SCALED_CONTRACT_INDICES] / spot_scale
    )
    signed_contract_values = contracts[:, _SIGNED_CONTRACT_INDICES]
    contracts[:, _SIGNED_CONTRACT_INDICES] = np.sign(signed_contract_values) * np.log1p(
        abs(signed_contract_values) * _SIGNED_CONTRACT_SCALES
    )
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
    for name in ("positionAgeSteps", "positionLastTradeAgeSteps"):
        contracts[:, indices[name]] = (
            np.log1p(np.maximum(contracts[:, indices[name]], 0.0)) / 10.0
        )

    if len(market):
        # Contract prices and Greeks are expressed relative to spot below, so
        # retaining the absolute dollar price only makes policies ticker-scale
        # dependent. Keep a unit reference for valid positive spot values.
        market[0] = 1.0 if market[0] > 0 else 0.0
    if len(market) == len(MARKET_FEATURES):
        market_indices = _MARKET_FEATURE_INDEX
        return_index = market_indices["underlyingReturn"]
        market[return_index] = np.sign(market[return_index]) * np.log1p(
            abs(market[return_index]) * 100.0
        )
        for name in ("benchmarkReturn", "relativeUnderlyingReturn"):
            index = market_indices[name]
            value = market[index]
            market[index] = np.sign(value) * np.log1p(abs(value) * 100.0)
        gap_index = market_indices["snapshotGapSeconds"]
        market[gap_index] = np.log1p(max(market[gap_index], 0.0)) / 10.0
        quote_age_index = market_indices["underlyingQuoteAgeSeconds"]
        market[quote_age_index] = np.log1p(max(market[quote_age_index], 0.0)) / 10.0
        benchmark_age_index = market_indices["benchmarkQuoteAgeSeconds"]
        market[benchmark_age_index] = (
            np.log1p(max(market[benchmark_age_index], 0.0)) / 10.0
        )
        for window in REALIZED_VOL_WINDOWS:
            volatility_index = market_indices[f"realizedVol{window}"]
            market[volatility_index] = np.log1p(max(market[volatility_index], 0.0))
            trend_index = market_indices[f"underlyingLogReturn{window}"]
            trend = market[trend_index]
            market[trend_index] = np.sign(trend) * np.log1p(abs(trend) * 100.0)
            benchmark_volatility_index = market_indices[f"benchmarkRealizedVol{window}"]
            market[benchmark_volatility_index] = np.log1p(
                max(market[benchmark_volatility_index], 0.0)
            )
            benchmark_trend_index = market_indices[f"benchmarkLogReturn{window}"]
            benchmark_trend = market[benchmark_trend_index]
            market[benchmark_trend_index] = np.sign(benchmark_trend) * np.log1p(
                abs(benchmark_trend) * 100.0
            )
        front_iv_index = market_indices["frontAtmIv"]
        market[front_iv_index] = np.log1p(max(market[front_iv_index], 0.0))
        smile_rmse_index = market_indices["frontSmileFitRmsePct"]
        market[smile_rmse_index] = np.log1p(max(market[smile_rmse_index], 0.0))
        for name in _SIGNED_SURFACE_MARKET_FEATURES:
            index = market_indices[name]
            value = market[index]
            market[index] = np.sign(value) * np.log1p(abs(value))
        for window in REALIZED_VOL_WINDOWS:
            spread_index = market_indices[f"atmIvMinusRealizedVol{window}"]
            spread = market[spread_index]
            market[spread_index] = np.sign(spread) * np.log1p(abs(spread))
    if len(portfolio) in {8, 10, 12}:
        nav = float(portfolio[2])
        nav_scale = max(abs(nav), 1.0)
        underlying_notional = abs(float(portfolio[7])) * spot_scale
        deployed_capital = max(
            abs(float(portfolio[0])) + abs(float(portfolio[1])) + underlying_notional,
            1.0,
        )
        values = [
            portfolio[0] / nav_scale,
            portfolio[1] / nav_scale,
            nav / deployed_capital,
            portfolio[3] * spot_scale / nav_scale,
            portfolio[4] * spot_scale**2 / nav_scale,
            portfolio[5] / nav_scale,
            portfolio[6] / nav_scale,
            portfolio[7] * spot_scale / nav_scale,
        ]
        if len(portfolio) == 10:
            values.extend(
                (
                    portfolio[8] / nav_scale,
                    portfolio[9] * spot_scale / nav_scale,
                )
            )
        elif len(portfolio) == 12:
            values.extend(
                (
                    portfolio[8] / nav_scale,
                    portfolio[9] * spot_scale / nav_scale,
                    portfolio[10],
                    portfolio[11],
                )
            )
        portfolio = np.asarray(values, dtype=np.float64)

    for values in (market, contracts, portfolio):
        # Clipping already maps infinities to the finite policy boundary. A
        # dedicated NaN pass avoids nan_to_num's two redundant infinity scans.
        values[np.isnan(values)] = 0.0
        np.clip(values, -10.0, 10.0, out=values)
    return market, contracts, portfolio


def observation_vector(observation: Observation) -> np.ndarray:
    """Flatten one observation using the versioned dimensionless transform."""
    market, contracts, portfolio = _dimensionless_components(observation)
    market_size = market.size
    contract_size = contracts.size
    portfolio_size = portfolio.size
    result = np.empty(
        market_size + contract_size + portfolio_size + observation.valid_mask.size,
        dtype=np.float32,
    )
    contract_start = market_size
    portfolio_start = contract_start + contract_size
    mask_start = portfolio_start + portfolio_size
    result[:contract_start] = market.ravel()
    result[contract_start:portfolio_start] = contracts.ravel()
    result[portfolio_start:mask_start] = portfolio.ravel()
    result[mask_start:] = observation.valid_mask.ravel()
    return result


def cross_sectional_contract_change_targets(
    current: Observation,
    future: Observation,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robust matched-contract changes independent of slot ordering."""
    values = np.zeros(len(CONTRACT_AUXILIARY_TARGET_FEATURES), dtype=np.float64)
    available = np.zeros(
        len(CONTRACT_AUXILIARY_TARGET_FEATURES),
        dtype=np.float32,
    )
    current_contracts = np.asarray(current.contracts, dtype=np.float64)
    future_contracts = np.asarray(future.contracts, dtype=np.float64)
    if (
        current_contracts.ndim != 2
        or future_contracts.ndim != 2
        or current_contracts.shape[1] != len(CONTRACT_FEATURES)
        or future_contracts.shape[1] != len(CONTRACT_FEATURES)
        or len(current.contract_ids) != len(current_contracts)
        or len(future.contract_ids) != len(future_contracts)
        or len(current.valid_mask) != len(current_contracts)
        or len(future.valid_mask) != len(future_contracts)
    ):
        return values.astype(np.float32), available

    future_by_id: dict[str, int] = {}
    for index, contract_id in enumerate(future.contract_ids):
        if contract_id is not None and bool(future.valid_mask[index]):
            future_by_id.setdefault(contract_id, index)
    current_valid_ids = {
        contract_id
        for index, contract_id in enumerate(current.contract_ids)
        if contract_id is not None and bool(current.valid_mask[index])
    }
    current_indices = []
    future_indices = []
    matched_ids: set[str] = set()
    for index, contract_id in enumerate(current.contract_ids):
        if (
            contract_id is None
            or contract_id in matched_ids
            or not bool(current.valid_mask[index])
            or contract_id not in future_by_id
        ):
            continue
        matched_ids.add(contract_id)
        current_indices.append(index)
        future_indices.append(future_by_id[contract_id])
    current_valid_count = len(current_valid_ids)
    if (
        not current_indices
        or len(current_indices) / current_valid_count < CONTRACT_AUXILIARY_MIN_COVERAGE
    ):
        return values.astype(np.float32), available

    current_rows = current_contracts[np.asarray(current_indices)]
    future_rows = future_contracts[np.asarray(future_indices)]
    bid_index = _CONTRACT_FEATURE_INDEX["bid"]
    ask_index = _CONTRACT_FEATURE_INDEX["ask"]
    iv_index = _CONTRACT_FEATURE_INDEX["impliedVolatility"]
    delta_index = _CONTRACT_FEATURE_INDEX["delta"]
    current_bid = current_rows[:, bid_index]
    current_ask = current_rows[:, ask_index]
    future_bid = future_rows[:, bid_index]
    future_ask = future_rows[:, ask_index]
    quote_available = (
        np.isfinite(current_bid)
        & np.isfinite(current_ask)
        & np.isfinite(future_bid)
        & np.isfinite(future_ask)
        & (current_bid > 0)
        & (current_ask > 0)
        & (future_bid > 0)
        & (future_ask > 0)
        & (current_bid <= current_ask)
        & (future_bid <= future_ask)
    )
    quote_coverage = float(quote_available.sum()) / current_valid_count
    if quote_coverage >= CONTRACT_AUXILIARY_MIN_COVERAGE:
        current_mid = (current_bid[quote_available] + current_ask[quote_available]) / 2
        future_mid = (future_bid[quote_available] + future_ask[quote_available]) / 2
        mid_returns = np.log(future_mid / current_mid)
        spread_changes = (
            future_ask[quote_available] - future_bid[quote_available]
        ) / future_mid - (
            current_ask[quote_available] - current_bid[quote_available]
        ) / current_mid
        median_mid_return = float(np.median(mid_returns))
        median_spread_change = float(np.median(spread_changes))
        values[0] = np.sign(median_mid_return) * np.log1p(
            abs(median_mid_return) * 100.0
        )
        values[1] = np.sign(median_spread_change) * np.log1p(
            abs(median_spread_change) * 10.0
        )
        available[:2] = 1.0

    current_iv = current_rows[:, iv_index]
    future_iv = future_rows[:, iv_index]
    iv_available = (
        quote_available
        & np.isfinite(current_iv)
        & np.isfinite(future_iv)
        & (current_iv > 0)
        & (future_iv > 0)
    )
    iv_coverage = float(iv_available.sum()) / current_valid_count
    if iv_coverage >= CONTRACT_AUXILIARY_MIN_COVERAGE:
        median_iv_change = float(
            np.median(future_iv[iv_available] - current_iv[iv_available])
        )
        values[2] = np.sign(median_iv_change) * np.log1p(abs(median_iv_change) * 10.0)
        available[2] = 1.0

    current_market = np.asarray(current.market, dtype=np.float64)
    future_market = np.asarray(future.market, dtype=np.float64)
    if len(current_market) == len(MARKET_FEATURES) and len(future_market) == len(
        MARKET_FEATURES
    ):
        spot_index = _MARKET_FEATURE_INDEX["underlyingPrice"]
        current_spot = current_market[spot_index]
        future_spot = future_market[spot_index]
        current_delta = current_rows[:, delta_index]
        hedge_available = quote_available & np.isfinite(current_delta)
        hedge_coverage = float(hedge_available.sum()) / current_valid_count
        if (
            hedge_coverage >= CONTRACT_AUXILIARY_MIN_COVERAGE
            and _delta_hedged_endpoint_is_observable(current_market)
            and _delta_hedged_endpoint_is_observable(future_market)
            and np.isfinite(current_spot)
            and current_spot > 0
            and np.isfinite(future_spot)
            and future_spot > 0
        ):
            current_mid = (
                current_bid[hedge_available] + current_ask[hedge_available]
            ) / 2
            future_mid = (future_bid[hedge_available] + future_ask[hedge_available]) / 2
            delta_hedged_spot_returns = (
                future_mid
                - current_mid
                - current_delta[hedge_available] * (future_spot - current_spot)
            ) / current_spot
            median_delta_hedged_return = float(np.median(delta_hedged_spot_returns))
            values[3] = np.sign(median_delta_hedged_return) * np.log1p(
                abs(median_delta_hedged_return) * 100.0
            )
            available[3] = 1.0

    values = np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(values, -10.0, 10.0).astype(np.float32), available


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
    if len(current_market) != len(MARKET_FEATURES) or len(future_market) != len(
        MARKET_FEATURES
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

    contract_values, contract_available = cross_sectional_contract_change_targets(
        current,
        future,
    )
    contract_start = len(MARKET_AUXILIARY_TARGET_FEATURES)
    values[contract_start:] = contract_values
    available[contract_start:] = contract_available

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
        or any(
            not isinstance(item, int) or isinstance(item, bool) for item in normalized
        )
        or any(item < 1 for item in normalized)
        or tuple(sorted(set(normalized))) != normalized
    ):
        raise ValueError(
            "auxiliary horizons must be unique positive increasing integers"
        )
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
                rewards=np.asarray(rewards[start:end], dtype=np.float32)
                if rewards is not None
                else None,
            )
        )
    return result
