"""Retrieve the nearest option chain for a stock ticker."""

import argparse
import math

import yfinance as yf


def _number(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def fetch_option_chain(symbol: str):
    """Return expiration, chain, spot price, and dividend yield for a ticker."""
    symbol = symbol.strip().upper()
    ticker = yf.Ticker(symbol)
    expirations = ticker.options

    if not expirations:
        raise ValueError(f"No listed options found for {symbol}")

    expiration = expirations[0]
    chain = ticker.option_chain(expiration)
    underlying = chain.underlying or {}
    spot = _number(underlying.get("regularMarketPrice"))
    if spot <= 0:
        spot = _number(ticker.fast_info.get("last_price"))
    if spot <= 0:
        raise ValueError(f"No underlying price found for {symbol}")

    # Yahoo's option payload expresses dividendYield in percentage units.
    dividend_yield = _number(underlying.get("dividendYield")) / 100
    return expiration, chain, spot, dividend_yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Stock ticker, for example AAPL")
    args = parser.parse_args()

    try:
        expiration, chain, _, _ = fetch_option_chain(args.symbol)
    except Exception as error:
        parser.exit(1, f"error: {error}\n")

    print(f"{args.symbol.upper()} option chain for {expiration}")
    print("\nCALLS")
    print(chain.calls.to_string(index=False))
    print("\nPUTS")
    print(chain.puts.to_string(index=False))


if __name__ == "__main__":
    main()
