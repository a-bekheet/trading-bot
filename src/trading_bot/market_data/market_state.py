"""Normalize provider market-session state without clock-based inference."""

from __future__ import annotations


MARKET_STATE_UNKNOWN = "UNKNOWN"
REGULAR_MARKET_STATE = "REGULAR"
RECOGNIZED_MARKET_STATES = frozenset({
    "PREPRE",
    "PRE",
    REGULAR_MARKET_STATE,
    "POST",
    "POSTPOST",
    "CLOSED",
})


def normalize_market_state(value: object) -> str:
    """Return a stable provider state or ``UNKNOWN`` for absent/new values."""
    if value is None:
        return MARKET_STATE_UNKNOWN
    state = str(value).strip().upper()
    return state if state in RECOGNIZED_MARKET_STATES else MARKET_STATE_UNKNOWN


def market_state_features(value: object) -> tuple[float, float]:
    """Return ``(is_regular, coverage)`` for policy-safe numeric features."""
    state = normalize_market_state(value)
    if state == MARKET_STATE_UNKNOWN:
        return 0.0, 0.0
    return float(state == REGULAR_MARKET_STATE), 1.0
