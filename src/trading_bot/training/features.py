"""Causal, point-in-time engineered option features."""

from __future__ import annotations

import numpy as np
import pandas as pd


ENGINEERED_FEATURES = (
    "midPrice", "spread", "spreadPct", "logMoneyness", "dteDays",
    "volumeLog", "openInterestLog", "quoteAgeSeconds", "underlyingReturn",
    "ivChange",
)


def engineer_snapshot(frame: pd.DataFrame, previous: pd.DataFrame | None = None) -> pd.DataFrame:
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
    return result
