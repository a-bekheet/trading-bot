"""Retrieve one or more option expirations for a stock ticker."""

import argparse
import math

import yfinance as yf


def _number(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def fetch_option_chains(symbol: str, expiration_count: int | None = 1):
    """Return selected chains plus shared spot and dividend inputs.

    ``expiration_count=None`` selects every listed expiration. The same ticker
    object and shared market inputs are reused across all selected chains.
    """
    symbol = symbol.strip().upper()
    if expiration_count is not None and expiration_count < 1:
        raise ValueError("expiration_count must be positive or None")
    ticker = yf.Ticker(symbol)
    expirations = ticker.options

    if not expirations:
        raise ValueError(f"No listed options found for {symbol}")

    selected = expirations if expiration_count is None else expirations[:expiration_count]
    chains = tuple((expiration, ticker.option_chain(expiration)) for expiration in selected)
    underlying = next(
        (chain.underlying for _, chain in chains if getattr(chain, "underlying", None)),
        {},
    )
    spot = _number(underlying.get("regularMarketPrice"))
    if spot <= 0:
        spot = _number(ticker.fast_info.get("last_price"))
    if spot <= 0:
        raise ValueError(f"No underlying price found for {symbol}")

    # Yahoo's option payload expresses dividendYield in percentage units.
    dividend_yield = _number(underlying.get("dividendYield")) / 100
    return chains, spot, dividend_yield


def fetch_option_chain(symbol: str):
    """Backward-compatible nearest-expiration chain helper."""
    chains, spot, dividend_yield = fetch_option_chains(symbol, expiration_count=1)
    expiration, chain = chains[0]
    return expiration, chain, spot, dividend_yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Stock ticker, for example AAPL")
    parser.add_argument(
        "--expirations",
        type=int,
        default=1,
        help="number of expirations to print; 0 selects all (default: 1)",
    )
    args = parser.parse_args()
    if args.expirations < 0:
        parser.error("--expirations cannot be negative")

    try:
        chains, _, _ = fetch_option_chains(
            args.symbol,
            expiration_count=args.expirations or None,
        )
    except Exception as error:
        parser.exit(1, f"error: {error}\n")

    for expiration, chain in chains:
        print(f"{args.symbol.upper()} option chain for {expiration}")
        print("\nCALLS")
        print(chain.calls.to_string(index=False))
        print("\nPUTS")
        print(chain.puts.to_string(index=False))


if __name__ == "__main__":
    main()
