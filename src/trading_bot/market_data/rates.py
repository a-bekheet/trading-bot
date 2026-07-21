"""Risk-free-rate retrieval."""

import math

import yfinance as yf


RISK_FREE_RATE_SOURCE = "Yahoo ^IRX 13-week Treasury bill yield"


def fetch_risk_free_rate() -> float:
    """Return the latest ^IRX yield as an annual decimal rate."""
    closes = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
    if closes.empty:
        raise ValueError("No ^IRX risk-free-rate data available")
    rate = float(closes.iloc[-1]) / 100
    if not math.isfinite(rate) or rate < 0:
        raise ValueError(f"Invalid ^IRX risk-free rate: {rate}")
    return rate
