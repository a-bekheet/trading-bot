"""Causal, point-in-time engineered option features."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


REALIZED_VOL_WINDOWS = (4, 16)
VOLATILITY_REGIME_WINDOW = 16
VOLATILITY_REGIME_MIN_HISTORY = 4
VOLATILITY_REGIME_ZSCORE_CLIP = 8.0
FRONT_ATM_LOG_MONEYNESS_TOLERANCE = 0.10
FRONT_WING_DELTA_TOLERANCE = 0.15
MARKET_ENGINEERED_FEATURES = (
    "underlyingReturn",
    "snapshotGapSeconds",
    "snapshotGapCoverage",
    "realizedVol4",
    "realizedVol4Coverage",
    "underlyingLogReturn4",
    "realizedVol16",
    "realizedVol16Coverage",
    "underlyingLogReturn16",
    "frontAtmIv",
    "frontAtmIvCoverage",
    "front25DeltaRiskReversal",
    "front25DeltaButterfly",
    "front25DeltaCoverage",
    "atmTermStructureSlope",
    "atmTermStructureCurvature",
    "atmTermSlopeCoverage",
    "atmTermCurvatureCoverage",
    "frontAtmIvChange",
    "frontAtmIvChangeCoverage",
    "front25DeltaRiskReversalChange",
    "front25DeltaButterflyChange",
    "frontWingChangeCoverage",
    "atmTermStructureSlopeChange",
    "atmTermSlopeChangeCoverage",
    "executableQuoteCoverage",
    "greekCoverage",
    "atmIvMinusRealizedVol4",
    "atmIvMinusRealizedVol16",
    "frontAtmIvZScore16",
    "frontAtmIvZScore16Coverage",
    "volatilityRiskPremiumZScore16",
    "volatilityRiskPremiumZScore16Coverage",
)
CONTRACT_DYNAMICS_FEATURES = (
    "midPriceLogReturn",
    "spreadPctChange",
    "quoteChangeCoverage",
    "ivChange",
    "ivChangeCoverage",
)
CONTRACT_ARBITRAGE_FEATURES = (
    "verticalArbitrageViolationPct",
    "verticalArbitrageCoverage",
    "butterflyArbitrageViolationPct",
    "butterflyArbitrageCoverage",
)
ENGINEERED_FEATURES = (
    "midPrice", "spread", "spreadPct", "logMoneyness", "dteDays",
    "volumeLog", "openInterestLog", "quoteAgeSeconds",
    *CONTRACT_DYNAMICS_FEATURES,
    *CONTRACT_ARBITRAGE_FEATURES,
    "forwardLogMoneyness", "extrinsicValuePct", "atmIv", "ivSkew",
    "atmTermSlope", "putCallIvSpread", "parityResidual",
    *MARKET_ENGINEERED_FEATURES,
)


def _executable_arbitrage_features(result: pd.DataFrame) -> None:
    """Add current-snapshot bid/ask-aware strike-arbitrage diagnostics."""
    values = {
        name: np.zeros(len(result), dtype=np.float64)
        for name in CONTRACT_ARBITRAGE_FEATURES
    }
    required = {
        "expiration", "optionType", "strike", "underlyingPrice", "bid", "ask",
    }
    if not required.issubset(result.columns) or result.empty:
        for name, feature_values in values.items():
            result[name] = feature_values
        return

    surface = pd.DataFrame({
        "position": np.arange(len(result)),
        "expiration": result["expiration"].astype(str).to_numpy(),
        "optionType": result["optionType"].astype(str).str.lower().to_numpy(),
        "strike": pd.to_numeric(result["strike"], errors="coerce").to_numpy(),
        "spot": pd.to_numeric(
            result["underlyingPrice"],
            errors="coerce",
        ).to_numpy(),
        "bid": pd.to_numeric(result["bid"], errors="coerce").to_numpy(),
        "ask": pd.to_numeric(result["ask"], errors="coerce").to_numpy(),
    })
    surface = surface[
        ~surface["expiration"].str.strip().str.lower().isin(
            ("", "nan", "nat", "none")
        )
        & surface["optionType"].isin(("call", "put"))
        & np.isfinite(surface["strike"])
        & (surface["strike"] > 0)
        & np.isfinite(surface["spot"])
        & (surface["spot"] > 0)
        & np.isfinite(surface["bid"])
        & np.isfinite(surface["ask"])
        & (surface["bid"] > 0)
        & (surface["ask"] > 0)
        & (surface["bid"] <= surface["ask"])
    ].drop_duplicates(
        ["expiration", "optionType", "strike"],
        keep="first",
    )

    vertical = values["verticalArbitrageViolationPct"]
    vertical_coverage = values["verticalArbitrageCoverage"]
    butterfly = values["butterflyArbitrageViolationPct"]
    butterfly_coverage = values["butterflyArbitrageCoverage"]
    for (_, option_type), group in surface.groupby(
        ["expiration", "optionType"],
        sort=False,
    ):
        ordered = group.sort_values("strike", kind="mergesort")
        positions = ordered["position"].to_numpy(dtype=np.int64)
        strikes = ordered["strike"].to_numpy(dtype=np.float64)
        bids = ordered["bid"].to_numpy(dtype=np.float64)
        asks = ordered["ask"].to_numpy(dtype=np.float64)
        spot_scale = max(
            abs(float(np.median(ordered["spot"].to_numpy(dtype=np.float64)))),
            1e-8,
        )

        if len(ordered) >= 2:
            widths = strikes[1:] - strikes[:-1]
            left_positions = positions[:-1]
            right_positions = positions[1:]
            vertical_coverage[left_positions] = 1.0
            vertical_coverage[right_positions] = 1.0
            if option_type == "call":
                ordering_violation = bids[1:] - asks[:-1]
                width_violation = bids[:-1] - asks[1:] - widths
            else:
                ordering_violation = bids[:-1] - asks[1:]
                width_violation = bids[1:] - asks[:-1] - widths
            violation = np.maximum.reduce((
                ordering_violation,
                width_violation,
                np.zeros_like(widths),
            )) / spot_scale
            np.maximum.at(vertical, left_positions, violation)
            np.maximum.at(vertical, right_positions, violation)

        if len(ordered) >= 3:
            full_widths = strikes[2:] - strikes[:-2]
            left_weights = (strikes[2:] - strikes[1:-1]) / full_widths
            right_weights = (strikes[1:-1] - strikes[:-2]) / full_widths
            left_positions = positions[:-2]
            middle_positions = positions[1:-1]
            right_positions = positions[2:]
            butterfly_coverage[left_positions] = 1.0
            butterfly_coverage[middle_positions] = 1.0
            butterfly_coverage[right_positions] = 1.0
            violation = np.maximum(
                bids[1:-1]
                - left_weights * asks[:-2]
                - right_weights * asks[2:],
                0.0,
            ) / spot_scale
            np.maximum.at(butterfly, left_positions, violation)
            np.maximum.at(butterfly, middle_positions, violation)
            np.maximum.at(butterfly, right_positions, violation)

    for name, feature_values in values.items():
        result[name] = feature_values


def _contract_dynamics_features(
    result: pd.DataFrame,
    previous: pd.DataFrame | None,
) -> None:
    """Add matched-contract changes with explicit point-in-time coverage."""
    for name in CONTRACT_DYNAMICS_FEATURES:
        result[name] = 0.0
    if previous is None or previous.empty or "contractSymbol" not in previous:
        return

    prior = previous.drop_duplicates("contractSymbol", keep="first").set_index(
        "contractSymbol"
    )
    aligned_prior = prior.reindex(result["contractSymbol"].to_numpy())

    def numeric_values(frame: pd.DataFrame, name: str) -> np.ndarray:
        if name not in frame:
            return np.full(len(result), np.nan, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)

    prior_bid = numeric_values(aligned_prior, "bid")
    prior_ask = numeric_values(aligned_prior, "ask")
    current_bid = numeric_values(result, "bid")
    current_ask = numeric_values(result, "ask")
    current_quote_valid = (
        np.isfinite(current_bid)
        & np.isfinite(current_ask)
        & (current_bid > 0)
        & (current_ask > 0)
        & (current_bid <= current_ask)
    )
    prior_quote_valid = (
        np.isfinite(prior_bid)
        & np.isfinite(prior_ask)
        & (prior_bid > 0)
        & (prior_ask > 0)
        & (prior_bid <= prior_ask)
    )
    quote_coverage = current_quote_valid & prior_quote_valid
    current_mid = (current_bid + current_ask) / 2
    prior_mid = (prior_bid + prior_ask) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        current_spread_pct = (current_ask - current_bid) / current_mid
        prior_spread_pct = (prior_ask - prior_bid) / prior_mid
    result["quoteChangeCoverage"] = quote_coverage.astype(float)
    covered_quotes = quote_coverage.astype(bool)
    mid_price_log_return = np.zeros(len(result), dtype=float)
    spread_pct_change = np.zeros(len(result), dtype=float)
    mid_price_log_return[covered_quotes] = np.log(
        current_mid[covered_quotes] / prior_mid[covered_quotes]
    )
    spread_pct_change[covered_quotes] = (
        current_spread_pct[covered_quotes] - prior_spread_pct[covered_quotes]
    )
    result["midPriceLogReturn"] = mid_price_log_return
    result["spreadPctChange"] = spread_pct_change

    current_iv = numeric_values(result, "impliedVolatility")
    prior_iv = numeric_values(aligned_prior, "impliedVolatility")
    iv_coverage = (
        quote_coverage
        & np.isfinite(current_iv)
        & np.isfinite(prior_iv)
        & (current_iv > 0)
        & (prior_iv > 0)
    )
    result["ivChangeCoverage"] = iv_coverage.astype(float)
    covered_iv = iv_coverage.astype(bool)
    iv_change = np.zeros(len(result), dtype=float)
    iv_change[covered_iv] = (
        current_iv[covered_iv] - prior_iv[covered_iv]
    )
    result["ivChange"] = iv_change


def realized_volatility_features(
    spot_history: Sequence[tuple[pd.Timestamp, float]],
) -> dict[str, float]:
    """Backward-only return/volatility summaries and history coverage."""
    result = {
        f"realizedVol{window}": 0.0
        for window in REALIZED_VOL_WINDOWS
    }
    result.update({
        f"realizedVol{window}Coverage": 0.0
        for window in REALIZED_VOL_WINDOWS
    })
    result.update({
        f"underlyingLogReturn{window}": 0.0
        for window in REALIZED_VOL_WINDOWS
    })
    if len(spot_history) < 2:
        return result

    recent = spot_history[-(max(REALIZED_VOL_WINDOWS) + 1):]
    timestamps = pd.to_datetime([item[0] for item in recent], utc=True)
    spots = np.asarray([item[1] for item in recent], dtype=np.float64)
    elapsed_seconds = np.diff(timestamps.view("int64")) / 1e9
    valid = (
        np.isfinite(spots[:-1])
        & np.isfinite(spots[1:])
        & (spots[:-1] > 0)
        & (spots[1:] > 0)
        & (elapsed_seconds > 0)
    )
    log_returns = np.zeros(len(elapsed_seconds), dtype=np.float64)
    log_returns[valid] = np.log(spots[1:][valid] / spots[:-1][valid])
    seconds_per_year = 365.25 * 24 * 60 * 60
    for window in REALIZED_VOL_WINDOWS:
        window_valid = valid[-window:]
        count = int(window_valid.sum())
        result[f"realizedVol{window}Coverage"] = count / window
        if not count:
            continue
        returns = log_returns[-window:][window_valid]
        result[f"underlyingLogReturn{window}"] = float(returns.sum())
        years = elapsed_seconds[-window:][window_valid].sum() / seconds_per_year
        if years > 0:
            result[f"realizedVol{window}"] = float(
                np.sqrt(np.square(returns).sum() / years)
            )
    return result


def snapshot_gap_features(
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
) -> dict[str, float]:
    """Return elapsed wall time from the immediately prior snapshot."""
    neutral = {"snapshotGapSeconds": 0.0, "snapshotGapCoverage": 0.0}
    if previous is None or previous.empty or current.empty:
        return neutral

    def first_timestamp(frame: pd.DataFrame) -> pd.Timestamp:
        if "collectedAt" not in frame:
            return pd.NaT
        return pd.to_datetime(frame["collectedAt"].iloc[0], errors="coerce", utc=True)

    current_timestamp = first_timestamp(current)
    previous_timestamp = first_timestamp(previous)
    if pd.isna(current_timestamp) or pd.isna(previous_timestamp):
        return neutral
    elapsed = float((current_timestamp - previous_timestamp).total_seconds())
    if not math.isfinite(elapsed) or elapsed <= 0:
        return neutral
    return {"snapshotGapSeconds": elapsed, "snapshotGapCoverage": 1.0}


def volatility_regime_observation(
    frame: pd.DataFrame,
) -> tuple[float | None, float | None]:
    """Extract valid ATM-IV and short-window volatility-premium levels."""
    if frame.empty:
        return None, None

    def scalar(name: str) -> float | None:
        if name not in frame:
            return None
        try:
            value = float(frame[name].iloc[0])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    front_coverage = scalar("frontAtmIvCoverage")
    front_atm_iv = scalar("frontAtmIv")
    if (
        front_coverage is None
        or front_coverage <= 0
        or front_atm_iv is None
        or front_atm_iv <= 0
    ):
        front_atm_iv = None

    realized_coverage = scalar("realizedVol4Coverage")
    premium = scalar("atmIvMinusRealizedVol4")
    if (
        front_atm_iv is None
        or realized_coverage is None
        or realized_coverage < 1
        or premium is None
    ):
        premium = None
    return front_atm_iv, premium


def volatility_regime_zscore_features(
    current: pd.DataFrame,
    history: Sequence[tuple[float | None, float | None]],
) -> dict[str, float]:
    """Normalize current volatility levels against prior-only causal history."""
    result = {
        "frontAtmIvZScore16": 0.0,
        "frontAtmIvZScore16Coverage": 0.0,
        "volatilityRiskPremiumZScore16": 0.0,
        "volatilityRiskPremiumZScore16Coverage": 0.0,
    }
    current_values = volatility_regime_observation(current)
    recent = history[-VOLATILITY_REGIME_WINDOW:]
    for value_index, (value_name, coverage_name) in enumerate((
        ("frontAtmIvZScore16", "frontAtmIvZScore16Coverage"),
        (
            "volatilityRiskPremiumZScore16",
            "volatilityRiskPremiumZScore16Coverage",
        ),
    )):
        current_value = current_values[value_index]
        if current_value is None:
            continue
        prior = np.asarray([
            values[value_index]
            for values in recent
            if values[value_index] is not None
        ], dtype=np.float64)
        result[coverage_name] = len(prior) / VOLATILITY_REGIME_WINDOW
        if len(prior) < VOLATILITY_REGIME_MIN_HISTORY:
            continue
        scale = float(prior.std())
        if not math.isfinite(scale) or scale <= 1e-12:
            continue
        score = (current_value - float(prior.mean())) / scale
        result[value_name] = float(np.clip(
            score,
            -VOLATILITY_REGIME_ZSCORE_CLIP,
            VOLATILITY_REGIME_ZSCORE_CLIP,
        ))
    return result


def _surface_features(result: pd.DataFrame) -> None:
    """Add causal cross-sectional option-surface features in place."""
    neutral = (
        "forwardLogMoneyness", "extrinsicValuePct", "atmIv", "ivSkew",
        "atmTermSlope", "putCallIvSpread", "parityResidual",
    )
    if "optionType" not in result or "expiration" not in result:
        for name in neutral:
            result[name] = 0.0
        return

    option_type = result["optionType"].astype(str).str.lower()
    expiration = result["expiration"].astype(str)
    strike = pd.to_numeric(result["strike"], errors="coerce").fillna(0.0)
    spot = pd.to_numeric(result["underlyingPrice"], errors="coerce").fillna(0.0)
    iv = pd.to_numeric(result["impliedVolatility"], errors="coerce").fillna(0.0)
    years = pd.to_numeric(
        result.get("timeToExpiryYears", result["dteDays"] / 365),
        errors="coerce",
    ).clip(lower=0).fillna(0.0)
    rate = pd.to_numeric(result.get("riskFreeRate", 0.0), errors="coerce")
    dividend = pd.to_numeric(result.get("dividendYield", 0.0), errors="coerce")
    if not isinstance(rate, pd.Series):
        rate = pd.Series(float(rate), index=result.index)
    if not isinstance(dividend, pd.Series):
        dividend = pd.Series(float(dividend), index=result.index)
    rate = rate.fillna(0.0)
    dividend = dividend.fillna(0.0)

    result["forwardLogMoneyness"] = (
        result["logMoneyness"] + (rate - dividend) * years
    )
    intrinsic = np.select(
        (option_type.eq("call"), option_type.eq("put")),
        ((spot - strike).clip(lower=0), (strike - spot).clip(lower=0)),
        default=0.0,
    )
    result["extrinsicValuePct"] = (
        (result["midPrice"] - intrinsic).clip(lower=0)
        / spot.replace(0, np.nan)
    ).fillna(0.0)

    grouping = [expiration, option_type]
    distance = result["forwardLogMoneyness"].abs()
    atm_indices = distance.groupby(grouping).idxmin()
    atm_rows = pd.DataFrame(
        {
            "expiration": expiration.loc[atm_indices].to_numpy(),
            "optionType": option_type.loc[atm_indices].to_numpy(),
            "atmIv": iv.loc[atm_indices].to_numpy(),
            "atmYears": years.loc[atm_indices].to_numpy(),
        }
    )
    atm_lookup = atm_rows.set_index(["expiration", "optionType"])["atmIv"]
    row_keys = pd.MultiIndex.from_arrays((expiration, option_type))
    result["atmIv"] = atm_lookup.reindex(row_keys).to_numpy()
    result["ivSkew"] = iv - result["atmIv"]

    front = atm_rows.sort_values("atmYears").groupby("optionType", sort=False).first()
    front_iv = option_type.map(front["atmIv"])
    front_years = option_type.map(front["atmYears"])
    term_distance = np.sqrt(years) - np.sqrt(front_years)
    result["atmTermSlope"] = (
        (result["atmIv"] - front_iv) / term_distance.replace(0, np.nan)
    ).fillna(0.0)

    surface = pd.DataFrame(
        {
            "expiration": expiration,
            "strike": strike,
            "optionType": option_type,
            "iv": iv,
            "mid": result["midPrice"],
        }
    )
    pairs = surface.pivot_table(
        index=["expiration", "strike"],
        columns="optionType",
        values=["iv", "mid"],
        aggfunc="mean",
    )
    pair_keys = pd.MultiIndex.from_arrays((expiration, strike))

    def paired_value(value: str, side: str) -> np.ndarray:
        key = (value, side)
        if key not in pairs:
            return np.full(len(result), np.nan)
        return pairs[key].reindex(pair_keys).to_numpy()

    call_iv = paired_value("iv", "call")
    put_iv = paired_value("iv", "put")
    call_mid = paired_value("mid", "call")
    put_mid = paired_value("mid", "put")
    result["putCallIvSpread"] = np.nan_to_num(call_iv - put_iv)
    discounted_spot = spot * np.exp(-dividend * years)
    discounted_strike = strike * np.exp(-rate * years)
    parity = call_mid - put_mid - (discounted_spot - discounted_strike)
    result["parityResidual"] = (
        pd.Series(parity, index=result.index) / spot.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _market_surface_features(result: pd.DataFrame) -> None:
    """Add compact, causal surface factors and explicit quality coverage."""
    neutral = (
        "frontAtmIv",
        "frontAtmIvCoverage",
        "front25DeltaRiskReversal",
        "front25DeltaButterfly",
        "front25DeltaCoverage",
        "atmTermStructureSlope",
        "atmTermStructureCurvature",
        "atmTermSlopeCoverage",
        "atmTermCurvatureCoverage",
        "executableQuoteCoverage",
        "greekCoverage",
        "atmIvMinusRealizedVol4",
        "atmIvMinusRealizedVol16",
    )
    for name in neutral:
        result[name] = 0.0

    quote_required = {"bid", "ask", "lastPrice"}
    if quote_required.issubset(result.columns):
        bid = pd.to_numeric(result["bid"], errors="coerce")
        ask = pd.to_numeric(result["ask"], errors="coerce")
        last = pd.to_numeric(result["lastPrice"], errors="coerce")
        executable = (
            np.isfinite(bid)
            & np.isfinite(ask)
            & np.isfinite(last)
            & (bid > 0)
            & (ask > 0)
            & (last > 0)
            & (bid <= ask)
        )
        result["executableQuoteCoverage"] = float(executable.mean())
    else:
        executable = pd.Series(False, index=result.index)

    greek_names = ("delta", "gamma", "theta", "vega")
    if set(greek_names).issubset(result.columns):
        greeks = result.loc[:, greek_names].apply(
            pd.to_numeric,
            errors="coerce",
        )
        result["greekCoverage"] = float(
            np.isfinite(greeks.to_numpy(dtype=np.float64)).all(axis=1).mean()
        )

    required = {
        "expiration",
        "optionType",
        "impliedVolatility",
        "forwardLogMoneyness",
        "dteDays",
    }
    if not required.issubset(result.columns):
        return

    surface = pd.DataFrame({
        "expiration": result["expiration"].astype(str),
        "optionType": result["optionType"].astype(str).str.lower(),
        "iv": pd.to_numeric(result["impliedVolatility"], errors="coerce"),
        "delta": pd.to_numeric(result.get("delta"), errors="coerce"),
        "forwardLogMoneyness": pd.to_numeric(
            result["forwardLogMoneyness"],
            errors="coerce",
        ),
        "dteDays": pd.to_numeric(result["dteDays"], errors="coerce"),
        "executable": executable,
    })
    surface = surface[
        surface["executable"]
        & np.isfinite(surface["iv"])
        & (surface["iv"] > 0)
        & np.isfinite(surface["dteDays"])
        & (surface["dteDays"] >= 0)
    ]
    if surface.empty:
        return

    front_dte = float(surface["dteDays"].min())
    front = surface[np.isclose(surface["dteDays"], front_dte)]
    term_points = []
    for _, expiry_rows in surface.groupby("expiration", sort=False):
        expiry_atm_values = []
        for side in ("call", "put"):
            side_rows = expiry_rows[
                expiry_rows["optionType"].eq(side)
                & np.isfinite(expiry_rows["forwardLogMoneyness"])
            ]
            if side_rows.empty:
                continue
            atm_index = side_rows["forwardLogMoneyness"].abs().idxmin()
            if (
                abs(float(side_rows.loc[atm_index, "forwardLogMoneyness"]))
                <= FRONT_ATM_LOG_MONEYNESS_TOLERANCE
            ):
                expiry_atm_values.append(float(side_rows.loc[atm_index, "iv"]))
        if expiry_atm_values:
            expiry_dte = float(expiry_rows["dteDays"].min())
            term_points.append((
                math.sqrt(max(expiry_dte / 365.25, 0.0)),
                float(np.median(expiry_atm_values)),
            ))
    term_points = sorted(set(term_points))
    result["atmTermSlopeCoverage"] = min(len(term_points) / 2, 1.0)
    result["atmTermCurvatureCoverage"] = min(len(term_points) / 3, 1.0)
    if len(term_points) >= 2:
        term_x = np.asarray([item[0] for item in term_points])
        term_iv = np.asarray([item[1] for item in term_points])
        if np.ptp(term_x) > 1e-12:
            result["atmTermStructureSlope"] = float(
                np.polyfit(term_x, term_iv, 1)[0]
            )
    if len(term_points) >= 3:
        first = term_points[0]
        middle = term_points[len(term_points) // 2]
        last = term_points[-1]
        left_width = middle[0] - first[0]
        right_width = last[0] - middle[0]
        total_width = last[0] - first[0]
        if min(left_width, right_width, total_width) > 1e-12:
            left_slope = (middle[1] - first[1]) / left_width
            right_slope = (last[1] - middle[1]) / right_width
            result["atmTermStructureCurvature"] = (
                right_slope - left_slope
            ) / total_width

    atm_values = []
    for side in ("call", "put"):
        side_rows = front[
            front["optionType"].eq(side)
            & np.isfinite(front["forwardLogMoneyness"])
        ]
        if side_rows.empty:
            continue
        atm_index = side_rows["forwardLogMoneyness"].abs().idxmin()
        if (
            abs(float(side_rows.loc[atm_index, "forwardLogMoneyness"]))
            > FRONT_ATM_LOG_MONEYNESS_TOLERANCE
        ):
            continue
        atm_values.append(float(side_rows.loc[atm_index, "iv"]))
    result["frontAtmIvCoverage"] = len(atm_values) / 2
    if not atm_values:
        return
    front_atm_iv = float(np.median(atm_values))
    if not np.isfinite(front_atm_iv) or front_atm_iv <= 0:
        return
    result["frontAtmIv"] = front_atm_iv

    wing_values: dict[str, float] = {}
    for side, target_delta in (("call", 0.25), ("put", -0.25)):
        side_rows = front[
            front["optionType"].eq(side)
            & np.isfinite(front["delta"])
        ]
        if side_rows.empty:
            continue
        delta_distance = (side_rows["delta"] - target_delta).abs()
        wing_index = delta_distance.idxmin()
        if float(delta_distance.loc[wing_index]) > FRONT_WING_DELTA_TOLERANCE:
            continue
        wing_values[side] = float(side_rows.loc[wing_index, "iv"])
    result["front25DeltaCoverage"] = len(wing_values) / 2
    if len(wing_values) == 2:
        call_iv = wing_values["call"]
        put_iv = wing_values["put"]
        result["front25DeltaRiskReversal"] = call_iv - put_iv
        result["front25DeltaButterfly"] = (
            (call_iv + put_iv) / 2 - front_atm_iv
        )

    for window in REALIZED_VOL_WINDOWS:
        coverage = float(result[f"realizedVol{window}Coverage"].iloc[0])
        if coverage <= 0:
            continue
        realized = float(result[f"realizedVol{window}"].iloc[0])
        result[f"atmIvMinusRealizedVol{window}"] = front_atm_iv - realized


def _surface_dynamics_features(
    result: pd.DataFrame,
    previous: pd.DataFrame | None,
) -> None:
    """Add one-snapshot surface-factor changes with prior/current coverage."""
    values = {
        "frontAtmIvChange": 0.0,
        "frontAtmIvChangeCoverage": 0.0,
        "front25DeltaRiskReversalChange": 0.0,
        "front25DeltaButterflyChange": 0.0,
        "frontWingChangeCoverage": 0.0,
        "atmTermStructureSlopeChange": 0.0,
        "atmTermSlopeChangeCoverage": 0.0,
    }
    if previous is None or previous.empty:
        result[list(values)] = tuple(values.values())
        return

    def scalar(frame: pd.DataFrame, name: str) -> float | None:
        if name not in frame:
            return None
        try:
            value = float(frame[name].iloc[0])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    current_atm_coverage = scalar(result, "frontAtmIvCoverage")
    previous_atm_coverage = scalar(previous, "frontAtmIvCoverage")
    if (
        current_atm_coverage is not None
        and previous_atm_coverage is not None
    ):
        coverage = min(current_atm_coverage, previous_atm_coverage)
        values["frontAtmIvChangeCoverage"] = coverage
        current_atm = scalar(result, "frontAtmIv")
        previous_atm = scalar(previous, "frontAtmIv")
        if coverage > 0 and current_atm is not None and previous_atm is not None:
            values["frontAtmIvChange"] = current_atm - previous_atm

    current_wing_coverage = scalar(result, "front25DeltaCoverage")
    previous_wing_coverage = scalar(previous, "front25DeltaCoverage")
    if (
        current_wing_coverage is not None
        and previous_wing_coverage is not None
        and current_wing_coverage >= 1
        and previous_wing_coverage >= 1
    ):
        values["frontWingChangeCoverage"] = 1.0
        for name in (
            "front25DeltaRiskReversal",
            "front25DeltaButterfly",
        ):
            current_value = scalar(result, name)
            previous_value = scalar(previous, name)
            if current_value is not None and previous_value is not None:
                values[f"{name}Change"] = current_value - previous_value

    current_term_coverage = scalar(result, "atmTermSlopeCoverage")
    previous_term_coverage = scalar(previous, "atmTermSlopeCoverage")
    if (
        current_term_coverage is not None
        and previous_term_coverage is not None
    ):
        coverage = min(current_term_coverage, previous_term_coverage)
        values["atmTermSlopeChangeCoverage"] = coverage
        current_slope = scalar(result, "atmTermStructureSlope")
        previous_slope = scalar(previous, "atmTermStructureSlope")
        if (
            coverage >= 1
            and current_slope is not None
            and previous_slope is not None
        ):
            values["atmTermStructureSlopeChange"] = (
                current_slope - previous_slope
            )
    result[list(values)] = tuple(values.values())


def engineer_snapshot(
    frame: pd.DataFrame,
    previous: pd.DataFrame | None = None,
    *,
    spot_history: Sequence[tuple[pd.Timestamp, float]] = (),
    volatility_history: Sequence[
        tuple[float | None, float | None]
    ] = (),
) -> pd.DataFrame:
    """Add only current or prior-snapshot features; never reads future rows."""
    result = frame.copy()
    bid = pd.to_numeric(result["bid"], errors="coerce").fillna(0.0)
    ask = pd.to_numeric(result["ask"], errors="coerce").fillna(0.0)
    mid = ((bid + ask) / 2).where((bid > 0) & (ask > 0), pd.to_numeric(result["lastPrice"], errors="coerce"))
    spot = pd.to_numeric(result["underlyingPrice"], errors="coerce").replace(0, np.nan)
    strike = pd.to_numeric(result["strike"], errors="coerce").replace(0, np.nan)
    timestamp = pd.to_datetime(result["collectedAt"], utc=True)
    expiration = pd.to_datetime(result["expiration"], errors="coerce", utc=True)
    last_trade_series = result["lastTradeDate"] if "lastTradeDate" in result else pd.Series(pd.NaT, index=result.index)
    last_trade = pd.to_datetime(last_trade_series, errors="coerce", utc=True)
    result["midPrice"] = mid.fillna(0.0)
    result["spread"] = (ask - bid).clip(lower=0).fillna(0.0)
    result["spreadPct"] = (result["spread"] / result["midPrice"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["logMoneyness"] = np.log(spot / strike).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["dteDays"] = ((expiration - timestamp).dt.total_seconds() / 86400).clip(lower=0).fillna(0.0)
    volume = result["volume"] if "volume" in result else pd.Series(0.0, index=result.index)
    open_interest = result["openInterest"] if "openInterest" in result else pd.Series(0.0, index=result.index)
    result["volumeLog"] = np.log1p(pd.to_numeric(volume, errors="coerce").clip(lower=0)).fillna(0.0)
    result["openInterestLog"] = np.log1p(pd.to_numeric(open_interest, errors="coerce").clip(lower=0)).fillna(0.0)
    result["quoteAgeSeconds"] = ((timestamp - last_trade).dt.total_seconds()).clip(lower=0).fillna(0.0)
    result["underlyingReturn"] = 0.0
    if previous is not None and not previous.empty:
        previous_spot = float(previous["underlyingPrice"].iloc[0])
        current_spot = float(spot.iloc[0]) if np.isfinite(spot.iloc[0]) else previous_spot
        result["underlyingReturn"] = current_spot / previous_spot - 1 if previous_spot else 0.0
    _contract_dynamics_features(result, previous)
    _executable_arbitrage_features(result)
    for name, value in snapshot_gap_features(result, previous).items():
        result[name] = value
    for name, value in realized_volatility_features(spot_history).items():
        result[name] = value
    _surface_features(result)
    _market_surface_features(result)
    _surface_dynamics_features(result, previous)
    for name, value in volatility_regime_zscore_features(
        result,
        volatility_history,
    ).items():
        result[name] = value
    result[list(ENGINEERED_FEATURES)] = result[list(ENGINEERED_FEATURES)].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return result
