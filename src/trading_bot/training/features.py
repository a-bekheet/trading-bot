"""Causal, point-in-time engineered option features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


REALIZED_VOL_WINDOWS = (4, 16)
MARKET_ENGINEERED_FEATURES = (
    "underlyingReturn",
    "realizedVol4",
    "realizedVol4Coverage",
    "realizedVol16",
    "realizedVol16Coverage",
)
ENGINEERED_FEATURES = (
    "midPrice", "spread", "spreadPct", "logMoneyness", "dteDays",
    "volumeLog", "openInterestLog", "quoteAgeSeconds", "ivChange",
    "forwardLogMoneyness", "extrinsicValuePct", "atmIv", "ivSkew",
    "atmTermSlope", "putCallIvSpread", "parityResidual",
    *MARKET_ENGINEERED_FEATURES,
)


def realized_volatility_features(
    spot_history: Sequence[tuple[pd.Timestamp, float]],
) -> dict[str, float]:
    """Annualized backward-only realized volatility and history coverage."""
    result = {
        f"realizedVol{window}": 0.0
        for window in REALIZED_VOL_WINDOWS
    }
    result.update({
        f"realizedVol{window}Coverage": 0.0
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
        years = elapsed_seconds[-window:][window_valid].sum() / seconds_per_year
        if years > 0:
            result[f"realizedVol{window}"] = float(
                np.sqrt(np.square(returns).sum() / years)
            )
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


def engineer_snapshot(
    frame: pd.DataFrame,
    previous: pd.DataFrame | None = None,
    *,
    spot_history: Sequence[tuple[pd.Timestamp, float]] = (),
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
    result["ivChange"] = 0.0
    if previous is not None and not previous.empty:
        previous_spot = float(previous["underlyingPrice"].iloc[0])
        current_spot = float(spot.iloc[0]) if np.isfinite(spot.iloc[0]) else previous_spot
        result["underlyingReturn"] = current_spot / previous_spot - 1 if previous_spot else 0.0
        prior_iv = previous.set_index("contractSymbol")["impliedVolatility"]
        result["ivChange"] = (
            pd.to_numeric(result["impliedVolatility"], errors="coerce")
            - pd.to_numeric(result["contractSymbol"].map(prior_iv), errors="coerce")
        ).fillna(0.0)
    for name, value in realized_volatility_features(spot_history).items():
        result[name] = value
    _surface_features(result)
    result[list(ENGINEERED_FEATURES)] = result[list(ENGINEERED_FEATURES)].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return result
