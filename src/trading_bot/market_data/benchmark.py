"""Fetch one timestamped benchmark quote for shared market context."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import yfinance as yf


DEFAULT_BENCHMARK_SYMBOL = "SPY"


@dataclass(frozen=True, slots=True)
class BenchmarkSnapshot:
    """One causal benchmark observation shared by a collection cycle."""

    symbol: str
    price: float
    price_source: str
    quote_time: str | None
    quote_time_source: str | None


def _positive_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _utc_timestamp(value) -> str | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        return None
    return timestamp.tz_convert("UTC").isoformat()


def fetch_benchmark_snapshot(
    symbol: str = DEFAULT_BENCHMARK_SYMBOL,
) -> BenchmarkSnapshot:
    """Return the newest positive one-minute close with provider time."""
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("benchmark symbol cannot be empty")
    ticker = yf.Ticker(normalized)
    history = ticker.history(
        period="1d",
        interval="1m",
        auto_adjust=False,
        prepost=True,
    )
    if not history.empty and "Close" in history:
        close = pd.to_numeric(history["Close"], errors="coerce")
        valid = close.map(lambda value: _positive_number(value) is not None)
        if bool(valid.any()):
            position = valid.to_numpy().nonzero()[0][-1]
            price = float(close.iloc[position])
            quote_time = _utc_timestamp(history.index[position])
            if quote_time is not None:
                return BenchmarkSnapshot(
                    symbol=normalized,
                    price=price,
                    price_source="yfinance.history.1m.Close",
                    quote_time=quote_time,
                    quote_time_source="yfinance.history.1m.index",
                )

    raise ValueError(f"No timestamped benchmark price found for {normalized}")
