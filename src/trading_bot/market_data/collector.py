"""Continuously save Greek-enriched option data for top U.S. companies."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trading_bot.analytics.greeks import black_scholes_greeks, years_to_expiration
from trading_bot.market_data.option_chain import fetch_option_chain
from trading_bot.market_data.rates import RISK_FREE_RATE_SOURCE, fetch_risk_free_rate
from trading_bot.market_data.universe import TOP_50_TICKERS


CSV_COLUMNS = (
    "collectedAt", "symbol", "expiration", "optionType", "contractSymbol",
    "lastTradeDate", "strike", "lastPrice", "bid", "ask", "change",
    "percentChange", "volume", "openInterest", "impliedVolatility", "inTheMoney",
    "contractSize", "currency", "underlyingPrice", "riskFreeRate",
    "riskFreeRateSource", "dividendYield", "timeToExpiryYears", "delta", "gamma",
    "theta", "vega", "greekModel",
)


def _add_greeks(
    frame: pd.DataFrame,
    option_type: str,
    spot: float,
    years: float,
    rate: float,
    dividend_yield: float,
) -> pd.DataFrame:
    enriched = frame.copy()
    values = [
        black_scholes_greeks(
            option_type,
            spot,
            float(row.strike),
            years,
            rate,
            float(row.impliedVolatility),
            dividend_yield,
        )
        for row in enriched.itertuples()
    ]
    for name in ("delta", "gamma", "theta", "vega"):
        enriched[name] = [greeks[name] for greeks in values]
    return enriched


def _migrate_csv(path: Path) -> None:
    """Atomically add new columns to an older collector CSV."""
    if not path.exists():
        return
    current_columns = tuple(pd.read_csv(path, nrows=0).columns)
    if current_columns == CSV_COLUMNS:
        return
    existing = pd.read_csv(path).reindex(columns=CSV_COLUMNS)
    temporary = path.with_suffix(".csv.tmp")
    existing.to_csv(temporary, index=False)
    temporary.replace(path)


def save_snapshot(
    symbol: str,
    output_dir: Path,
    risk_free_rate: float,
    collected_at: datetime | None = None,
) -> tuple[Path, int]:
    """Append one ticker's nearest-expiration chain to its CSV file."""
    expiration, chain, spot, dividend_yield = fetch_option_chain(symbol)
    captured = collected_at or datetime.now(timezone.utc)
    years = years_to_expiration(expiration, captured)

    calls = _add_greeks(chain.calls, "call", spot, years, risk_free_rate, dividend_yield)
    puts = _add_greeks(chain.puts, "put", spot, years, risk_free_rate, dividend_yield)
    metadata = {
        "collectedAt": captured.isoformat(),
        "symbol": symbol,
        "expiration": expiration,
        "underlyingPrice": spot,
        "riskFreeRate": risk_free_rate,
        "riskFreeRateSource": RISK_FREE_RATE_SOURCE,
        "dividendYield": dividend_yield,
        "timeToExpiryYears": years,
        "greekModel": "black-scholes-merton",
    }
    calls = calls.assign(optionType="call", **metadata)
    puts = puts.assign(optionType="put", **metadata)
    snapshot = pd.concat((calls, puts), ignore_index=True).reindex(columns=CSV_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}.csv"
    _migrate_csv(path)
    snapshot.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path, len(snapshot)


def collect_all(output_dir: Path, ticker_delay: float = 1.0) -> tuple[int, int]:
    """Collect every ticker, continuing when an individual ticker fails."""
    try:
        risk_free_rate = fetch_risk_free_rate()
        logging.info("risk-free rate: %.6f (%s)", risk_free_rate, RISK_FREE_RATE_SOURCE)
    except Exception as error:
        logging.error("risk-free rate: %s", error)
        return 0, len(TOP_50_TICKERS)

    successes = 0
    failures = 0
    for index, symbol in enumerate(TOP_50_TICKERS):
        try:
            path, row_count = save_snapshot(symbol, output_dir, risk_free_rate)
            logging.info("%s: appended %d rows to %s", symbol, row_count, path)
            successes += 1
        except Exception as error:
            logging.error("%s: %s", symbol, error)
            failures += 1
        if ticker_delay and index < len(TOP_50_TICKERS) - 1:
            time.sleep(ticker_delay)
    return successes, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--ticker-delay", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1 or args.ticker_delay < 0:
        parser.error("--interval must be positive and --ticker-delay cannot be negative")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    while True:
        successes, failures = collect_all(args.output_dir, args.ticker_delay)
        logging.info("cycle complete: %d succeeded, %d failed", successes, failures)
        if args.once:
            return 1 if failures else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
