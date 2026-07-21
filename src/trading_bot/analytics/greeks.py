"""Black-Scholes-Merton Greek calculations."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timezone
from statistics import NormalDist
from zoneinfo import ZoneInfo


NORMAL = NormalDist()
SECONDS_PER_YEAR = 365 * 24 * 60 * 60


def years_to_expiration(expiration: str, collected_at: datetime) -> float:
    """Return ACT/365 years until 4:00 PM New York time on expiration day."""
    expiry = datetime.combine(
        date.fromisoformat(expiration),
        time(16, 0),
        tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(timezone.utc)
    captured = collected_at.astimezone(timezone.utc)
    return (expiry - captured).total_seconds() / SECONDS_PER_YEAR


def black_scholes_greeks(
    option_type: str,
    spot: float,
    strike: float,
    years: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """Calculate delta, gamma, daily theta, and vega per IV percentage point."""
    values = (spot, strike, years, volatility)
    if option_type not in {"call", "put"} or not all(
        math.isfinite(value) and value > 0 for value in values
    ):
        return {name: math.nan for name in ("delta", "gamma", "theta", "vega")}

    sqrt_years = math.sqrt(years)
    discount = math.exp(-rate * years)
    dividend_discount = math.exp(-dividend_yield * years)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + volatility**2 / 2) * years
    ) / (volatility * sqrt_years)
    d2 = d1 - volatility * sqrt_years
    density = math.exp(-(d1**2) / 2) / math.sqrt(2 * math.pi)

    gamma = dividend_discount * density / (spot * volatility * sqrt_years)
    vega = spot * dividend_discount * density * sqrt_years / 100
    common_theta = -(spot * dividend_discount * density * volatility) / (
        2 * sqrt_years
    )

    if option_type == "call":
        delta = dividend_discount * NORMAL.cdf(d1)
        annual_theta = (
            common_theta
            - rate * strike * discount * NORMAL.cdf(d2)
            + dividend_yield * spot * dividend_discount * NORMAL.cdf(d1)
        )
    else:
        delta = dividend_discount * (NORMAL.cdf(d1) - 1)
        annual_theta = (
            common_theta
            + rate * strike * discount * NORMAL.cdf(-d2)
            - dividend_yield * spot * dividend_discount * NORMAL.cdf(-d1)
        )

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": annual_theta / 365,
        "vega": vega,
    }
