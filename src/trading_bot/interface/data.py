"""Read collector CSVs for the user interface."""

from pathlib import Path

import pandas as pd

from trading_bot.market_data.freshness import (
    DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS,
    underlying_quote_age,
)
from trading_bot.market_data.market_state import (
    market_state_features,
    normalize_market_state,
)


def available_tickers(data_dir: Path) -> list[str]:
    return sorted(path.stem for path in data_dir.glob("*.csv"))


def load_latest_snapshot(data_dir: Path, symbol: str) -> pd.DataFrame:
    data = pd.read_csv(data_dir / f"{symbol}.csv")
    if data.empty:
        return data
    return data[data["collectedAt"] == data["collectedAt"].iloc[-1]].copy()


def market_session_status(snapshot: pd.DataFrame) -> dict[str, object]:
    """Describe whether a snapshot can support simulated execution."""
    value = (
        snapshot["marketState"].iloc[0]
        if not snapshot.empty and "marketState" in snapshot
        else None
    )
    state = normalize_market_state(value)
    regular, coverage = market_state_features(value)
    return {
        "provider_state": state,
        "regular": bool(regular),
        "coverage": coverage,
        "trading_enabled": coverage < 0.5 or regular >= 0.5,
    }


def market_data_freshness_status(
    snapshot: pd.DataFrame,
    max_age_seconds: float = DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS,
) -> dict[str, object]:
    """Describe quote age and its independent simulated-execution guard."""
    first = snapshot.iloc[0] if not snapshot.empty else {}
    quote_time = first.get("underlyingQuoteTime")
    age, coverage = underlying_quote_age(first.get("collectedAt"), quote_time)
    return {
        "quote_time": quote_time,
        "quote_time_source": first.get("underlyingQuoteTimeSource"),
        "price_source": first.get("underlyingPriceSource"),
        "age_seconds": age,
        "coverage": coverage,
        "max_age_seconds": max_age_seconds,
        "trading_enabled": coverage < 0.5 or age <= max_age_seconds,
    }
