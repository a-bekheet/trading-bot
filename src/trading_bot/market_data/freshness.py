"""Causal provider-quote selection and freshness measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from trading_bot.market_data.market_state import normalize_market_state


# One 15-minute collection interval plus five minutes of scheduling/provider
# tolerance. Explicitly missing timestamps retain legacy behavior through the
# coverage feature; they are never silently classified as fresh.
DEFAULT_MAX_UNDERLYING_QUOTE_AGE_SECONDS = 20 * 60.0


@dataclass(frozen=True, slots=True)
class UnderlyingQuote:
    """One price and its matching provider timestamp/provenance."""

    price: float
    price_source: str
    quote_time: str | None
    quote_time_source: str | None


_SESSION_PREFIX = {
    "PREPRE": "preMarket",
    "PRE": "preMarket",
    "REGULAR": "regularMarket",
    "POST": "postMarket",
    "POSTPOST": "postMarket",
}
_FALLBACK_PREFIXES = ("regularMarket", "preMarket", "postMarket")


def _positive_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _provider_time(value: object) -> tuple[float, str] | None:
    seconds = _positive_number(value)
    if seconds is None:
        return None
    try:
        timestamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return seconds, timestamp.isoformat()


def select_underlying_quote(
    underlying: Mapping[str, object],
    market_state: object,
) -> UnderlyingQuote | None:
    """Select a price/timestamp pair consistent with provider market state.

    For an explicit live session, its price is preferred even when its
    timestamp is absent; this avoids pairing a pre/post price with a regular
    market timestamp. Closed or unknown states use the most recently timestamped
    complete provider pair, with deterministic regular/pre/post fallback.
    """
    state = normalize_market_state(market_state)
    preferred = _SESSION_PREFIX.get(state)
    if preferred is not None:
        price_source = f"{preferred}Price"
        price = _positive_number(underlying.get(price_source))
        if price is not None:
            time_source = f"{preferred}Time"
            provider_time = _provider_time(underlying.get(time_source))
            return UnderlyingQuote(
                price=price,
                price_source=price_source,
                quote_time=(provider_time[1] if provider_time else None),
                quote_time_source=(time_source if provider_time else None),
            )

    prefixes = tuple(
        prefix for prefix in _FALLBACK_PREFIXES if prefix != preferred
    )

    candidates: list[tuple[float, int, UnderlyingQuote]] = []
    untimed: list[UnderlyingQuote] = []
    for priority, prefix in enumerate(prefixes):
        price_source = f"{prefix}Price"
        price = _positive_number(underlying.get(price_source))
        if price is None:
            continue
        time_source = f"{prefix}Time"
        provider_time = _provider_time(underlying.get(time_source))
        quote = UnderlyingQuote(
            price=price,
            price_source=price_source,
            quote_time=(provider_time[1] if provider_time else None),
            quote_time_source=(time_source if provider_time else None),
        )
        if provider_time is None:
            untimed.append(quote)
        else:
            candidates.append((provider_time[0], -priority, quote))
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    return untimed[0] if untimed else None


def _utc_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            text = str(value).strip()
            if not text or text.lower() in {"nan", "nat", "none"}:
                return None
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def underlying_quote_age(
    collected_at: object,
    quote_time: object,
) -> tuple[float, float]:
    """Return nonnegative ``(age_seconds, coverage)`` without clock leakage."""
    collected = _utc_datetime(collected_at)
    quoted = _utc_datetime(quote_time)
    if collected is None or quoted is None:
        return 0.0, 0.0
    elapsed = (collected - quoted).total_seconds()
    if not math.isfinite(elapsed) or elapsed < 0:
        return 0.0, 0.0
    return float(elapsed), 1.0
