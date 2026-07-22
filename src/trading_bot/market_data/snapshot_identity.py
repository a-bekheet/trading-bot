"""Stable identities for materially distinct option-chain snapshots."""

from __future__ import annotations

import hashlib
from io import StringIO

import pandas as pd


# Collector-derived clock/model outputs are deliberately absent. A later
# collection of the same stale quotes must not become a new market state merely
# because time-to-expiry and Black-Scholes-Merton Greeks were recomputed.
MATERIAL_SNAPSHOT_COLUMNS = (
    "symbol",
    "expiration",
    "optionType",
    "contractSymbol",
    "lastTradeDate",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "change",
    "percentChange",
    "volume",
    "openInterest",
    "impliedVolatility",
    "inTheMoney",
    "contractSize",
    "currency",
    "underlyingPrice",
    "riskFreeRate",
    "riskFreeRateSource",
    "dividendYield",
    "greekModel",
)


def material_snapshot_fingerprint(frame: pd.DataFrame) -> str:
    """Hash raw quote/model inputs while ignoring capture-time derivatives."""
    columns = tuple(name for name in MATERIAL_SNAPSHOT_COLUMNS if name in frame)
    if not columns:
        raise ValueError("snapshot has no material market columns")
    normalized = frame.loc[:, columns].copy()
    sort_columns = tuple(
        name
        for name in ("expiration", "optionType", "contractSymbol")
        if name in normalized
    )
    if sort_columns:
        normalized = normalized.sort_values(list(sort_columns), kind="stable")
    payload = normalized.to_csv(
        index=False,
        # Eight significant digits retain far more precision than executable
        # quote ticks while remaining stable across pandas' CSV float parser.
        float_format="%.8g",
        na_rep="<NA>",
        lineterminator="\n",
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def persisted_material_snapshot_fingerprint(frame: pd.DataFrame) -> str:
    """Hash the values pandas will recover from the collector's CSV append."""
    columns = tuple(name for name in MATERIAL_SNAPSHOT_COLUMNS if name in frame)
    if not columns:
        raise ValueError("snapshot has no material market columns")
    buffer = StringIO()
    frame.loc[:, columns].to_csv(buffer, index=False)
    buffer.seek(0)
    restored = pd.read_csv(buffer)
    return material_snapshot_fingerprint(restored)
