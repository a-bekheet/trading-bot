"""Continuously save Greek-enriched option data for top U.S. companies."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from trading_bot.analytics.greeks import black_scholes_greeks, years_to_expiration
from trading_bot.market_data.benchmark import (
    DEFAULT_BENCHMARK_SYMBOL,
    BenchmarkSnapshot,
    fetch_benchmark_snapshot,
)
from trading_bot.market_data.option_chain import fetch_option_chain_snapshot
from trading_bot.market_data.rates import RISK_FREE_RATE_SOURCE, fetch_risk_free_rate
from trading_bot.market_data.snapshot_identity import (
    material_snapshot_fingerprint,
    persisted_material_snapshot_fingerprint,
)
from trading_bot.market_data.universe import TOP_50_TICKERS


CSV_COLUMNS = (
    "collectedAt", "symbol", "expiration", "optionType", "contractSymbol",
    "lastTradeDate", "strike", "lastPrice", "bid", "ask", "change",
    "percentChange", "volume", "openInterest", "impliedVolatility", "inTheMoney",
    "contractSize", "currency", "underlyingPrice", "underlyingPriceSource",
    "underlyingQuoteTime", "underlyingQuoteTimeSource", "marketState", "riskFreeRate",
    "riskFreeRateSource", "dividendYield", "benchmarkSymbol", "benchmarkPrice",
    "benchmarkPriceSource", "benchmarkQuoteTime", "benchmarkQuoteTimeSource",
    "timeToExpiryYears", "delta", "gamma", "theta", "vega", "greekModel",
)
COLLECTOR_STATUS_SCHEMA_VERSION = "collector.status.v2"
SNAPSHOT_STATE_SCHEMA_VERSION = "collector.snapshot-state.v7"
COLLECTOR_STATUS_FILENAME = "_collector_status.json"


@dataclass
class CycleResult:
    """Observable outcome of one top-50 collection cycle."""

    cycle_started_at: str
    continuous: bool = False
    status: str = "running"
    last_heartbeat_at: str = ""
    cycle_completed_at: str | None = None
    next_cycle_at: str | None = None
    pid: int = field(default_factory=os.getpid)
    ticker_total: int = field(default_factory=lambda: len(TOP_50_TICKERS))
    tickers_attempted: int = 0
    successes: int = 0
    failures: int = 0
    appended: int = 0
    unchanged: int = 0
    rows_appended: int = 0
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL
    benchmark_price: float | None = None
    benchmark_quote_time: str | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def heartbeat(self) -> None:
        self.last_heartbeat_at = _utc_now().isoformat()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": COLLECTOR_STATUS_SCHEMA_VERSION,
            **asdict(self),
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_cycle_status(output_dir: Path, result: CycleResult) -> None:
    result.heartbeat()
    _write_json_atomic(output_dir / COLLECTOR_STATUS_FILENAME, result.to_dict())


@contextmanager
def collector_lock(output_dir: Path) -> Iterator[None]:
    """Hold an advisory process lock for one collector instance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".collector.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another collector already holds {path}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _snapshot_state_path(output_dir: Path, symbol: str) -> Path:
    return output_dir / ".snapshot_state" / f"{symbol}.json"


def _load_snapshot_state(path: Path, csv_path: Path) -> str | None:
    if not path.exists() or not csv_path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        stat = csv_path.stat()
        if (
            state.get("schema_version") == SNAPSHOT_STATE_SCHEMA_VERSION
            and state.get("csv_size") == stat.st_size
            and state.get("csv_mtime_ns") == stat.st_mtime_ns
        ):
            return str(state["fingerprint"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def _latest_persisted_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    existing = pd.read_csv(path)
    if existing.empty or "collectedAt" not in existing:
        return None
    timestamps = pd.to_datetime(existing["collectedAt"], utc=True, errors="coerce")
    latest = timestamps.max()
    if pd.isna(latest):
        return None
    return material_snapshot_fingerprint(existing.loc[timestamps.eq(latest)])


def _save_snapshot_state(
    state_path: Path,
    csv_path: Path,
    fingerprint: str,
    captured: datetime,
    row_count: int,
) -> None:
    stat = csv_path.stat()
    _write_json_atomic(state_path, {
        "schema_version": SNAPSHOT_STATE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "last_observed_at": captured.isoformat(),
        "row_count": row_count,
        "csv_size": stat.st_size,
        "csv_mtime_ns": stat.st_mtime_ns,
    })


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
    expiration_count: int | None = 3,
    benchmark_snapshot: BenchmarkSnapshot | None = None,
) -> tuple[Path, int]:
    """Append one materially changed surface; return zero rows if unchanged."""
    market_snapshot = fetch_option_chain_snapshot(symbol, expiration_count)
    chains = market_snapshot.chains
    spot = market_snapshot.spot
    dividend_yield = market_snapshot.dividend_yield
    captured = collected_at or datetime.now(timezone.utc)
    frames = []
    for expiration, chain in chains:
        years = years_to_expiration(expiration, captured)
        calls = _add_greeks(
            chain.calls, "call", spot, years, risk_free_rate, dividend_yield
        )
        puts = _add_greeks(
            chain.puts, "put", spot, years, risk_free_rate, dividend_yield
        )
        metadata = {
            "collectedAt": captured.isoformat(),
            "symbol": symbol,
            "expiration": expiration,
            "underlyingPrice": spot,
            "underlyingPriceSource": market_snapshot.underlying_price_source,
            "underlyingQuoteTime": market_snapshot.underlying_quote_time,
            "underlyingQuoteTimeSource": (
                market_snapshot.underlying_quote_time_source
            ),
            "marketState": market_snapshot.market_state,
            "riskFreeRate": risk_free_rate,
            "riskFreeRateSource": RISK_FREE_RATE_SOURCE,
            "dividendYield": dividend_yield,
            "benchmarkSymbol": (
                benchmark_snapshot.symbol if benchmark_snapshot else ""
            ),
            "benchmarkPrice": (
                benchmark_snapshot.price if benchmark_snapshot else None
            ),
            "benchmarkPriceSource": (
                benchmark_snapshot.price_source if benchmark_snapshot else ""
            ),
            "benchmarkQuoteTime": (
                benchmark_snapshot.quote_time if benchmark_snapshot else None
            ),
            "benchmarkQuoteTimeSource": (
                benchmark_snapshot.quote_time_source
                if benchmark_snapshot
                else None
            ),
            "timeToExpiryYears": years,
            "greekModel": "black-scholes-merton",
        }
        frames.extend(
            (
                calls.assign(optionType="call", **metadata),
                puts.assign(optionType="put", **metadata),
            )
        )
    snapshot = pd.concat(frames, ignore_index=True).reindex(columns=CSV_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}.csv"
    _migrate_csv(path)
    fingerprint = persisted_material_snapshot_fingerprint(snapshot)
    state_path = _snapshot_state_path(output_dir, symbol)
    previous_fingerprint = _load_snapshot_state(state_path, path)
    if previous_fingerprint is None:
        previous_fingerprint = _latest_persisted_fingerprint(path)
    if previous_fingerprint == fingerprint:
        if path.exists():
            _save_snapshot_state(
                state_path,
                path,
                fingerprint,
                captured,
                len(snapshot),
            )
        return path, 0
    snapshot.to_csv(path, mode="a", header=not path.exists(), index=False)
    _save_snapshot_state(
        state_path,
        path,
        fingerprint,
        captured,
        len(snapshot),
    )
    return path, len(snapshot)


def collect_cycle(
    output_dir: Path,
    ticker_delay: float = 1.0,
    expiration_count: int | None = 3,
    *,
    continuous: bool = False,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
) -> CycleResult:
    """Collect every ticker and persist a queryable heartbeat throughout."""
    result = CycleResult(
        cycle_started_at=_utc_now().isoformat(),
        continuous=continuous,
        benchmark_symbol=benchmark_symbol.strip().upper(),
    )
    _write_cycle_status(output_dir, result)
    try:
        risk_free_rate = fetch_risk_free_rate()
        logging.info("risk-free rate: %.6f (%s)", risk_free_rate, RISK_FREE_RATE_SOURCE)
    except Exception as error:
        message = str(error)
        logging.error("risk-free rate: %s", message)
        result.failures = len(TOP_50_TICKERS)
        result.errors["risk_free_rate"] = message
        result.status = "complete"
        result.cycle_completed_at = _utc_now().isoformat()
        _write_cycle_status(output_dir, result)
        return result

    benchmark_snapshot = None
    try:
        benchmark_snapshot = fetch_benchmark_snapshot(benchmark_symbol)
        result.benchmark_symbol = benchmark_snapshot.symbol
        result.benchmark_price = benchmark_snapshot.price
        result.benchmark_quote_time = benchmark_snapshot.quote_time
        logging.info(
            "benchmark: %s %.6f (%s)",
            benchmark_snapshot.symbol,
            benchmark_snapshot.price,
            benchmark_snapshot.price_source,
        )
    except Exception as error:
        message = str(error)
        logging.error("benchmark: %s", message)
        result.errors["benchmark"] = message
        _write_cycle_status(output_dir, result)

    for index, symbol in enumerate(TOP_50_TICKERS):
        result.tickers_attempted += 1
        try:
            path, row_count = save_snapshot(
                symbol,
                output_dir,
                risk_free_rate,
                expiration_count=expiration_count,
                benchmark_snapshot=benchmark_snapshot,
            )
            result.successes += 1
            if row_count:
                result.appended += 1
                result.rows_appended += row_count
                logging.info("%s: appended %d rows to %s", symbol, row_count, path)
            else:
                result.unchanged += 1
                logging.info("%s: unchanged; skipped append to %s", symbol, path)
        except Exception as error:
            message = str(error)
            logging.error("%s: %s", symbol, message)
            result.failures += 1
            result.errors[symbol] = message
        _write_cycle_status(output_dir, result)
        if ticker_delay and index < len(TOP_50_TICKERS) - 1:
            time.sleep(ticker_delay)

    result.status = "complete"
    result.cycle_completed_at = _utc_now().isoformat()
    _write_cycle_status(output_dir, result)
    return result


def collect_all(
    output_dir: Path,
    ticker_delay: float = 1.0,
    expiration_count: int | None = 3,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
) -> tuple[int, int]:
    """Collect every ticker, continuing when an individual ticker fails."""
    result = collect_cycle(
        output_dir,
        ticker_delay,
        expiration_count,
        benchmark_symbol=benchmark_symbol,
    )
    return result.successes, result.failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--ticker-delay", type=float, default=1.0)
    parser.add_argument(
        "--benchmark-symbol",
        default=DEFAULT_BENCHMARK_SYMBOL,
        help="shared benchmark quote fetched once per cycle (default: SPY)",
    )
    parser.add_argument(
        "--expirations",
        type=int,
        default=3,
        help="expirations per ticker; 0 selects all (default: 3)",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.interval < 1
        or args.ticker_delay < 0
        or args.expirations < 0
        or not args.benchmark_symbol.strip()
    ):
        parser.error(
            "--interval must be positive; delays and expiration count cannot "
            "be negative; benchmark symbol cannot be empty"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        with collector_lock(args.output_dir):
            while True:
                result = collect_cycle(
                    args.output_dir,
                    args.ticker_delay,
                    expiration_count=args.expirations or None,
                    continuous=not args.once,
                    benchmark_symbol=args.benchmark_symbol,
                )
                logging.info(
                    "cycle complete: %d succeeded, %d failed, %d appended, "
                    "%d unchanged",
                    result.successes,
                    result.failures,
                    result.appended,
                    result.unchanged,
                )
                if args.once:
                    return 1 if result.failures else 0
                result.status = "sleeping"
                result.next_cycle_at = datetime.fromtimestamp(
                    time.time() + args.interval,
                    tz=timezone.utc,
                ).isoformat()
                _write_cycle_status(args.output_dir, result)
                time.sleep(args.interval)
    except RuntimeError as error:
        logging.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
