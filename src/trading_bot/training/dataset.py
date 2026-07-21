"""Point-in-time snapshot loader for the research demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading_bot.training.features import engineer_snapshot


REQUIRED_COLUMNS = {
    "collectedAt", "contractSymbol", "symbol", "expiration", "optionType",
    "strike", "bid", "ask", "lastPrice", "impliedVolatility", "underlyingPrice",
}


@dataclass(frozen=True)
class Snapshot:
    timestamp: str
    frame: pd.DataFrame


class SnapshotDataset:
    """Immutable-in-memory, timestamp-grouped demo snapshots."""

    def __init__(self, snapshots: tuple[Snapshot, ...], symbol: str):
        if not snapshots:
            raise ValueError("dataset contains no usable snapshots")
        self.snapshots = snapshots
        self.symbol = symbol

    @classmethod
    def from_directory(cls, data_dir: Path, symbol: str) -> "SnapshotDataset":
        path = data_dir / f"{symbol.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        # Raw pre-Greek rows are useful for display but cannot be an RL feature set.
        if "greekModel" in frame:
            frame = frame[frame["greekModel"].eq("black-scholes-merton")]
        frame = frame.copy()
        frame["collectedAt"] = pd.to_datetime(frame["collectedAt"], utc=True)
        frame = frame.sort_values(["collectedAt", "optionType", "expiration", "strike", "contractSymbol"])
        snapshots_list = []
        previous = None
        for timestamp, group in frame.groupby("collectedAt", sort=True):
            engineered = engineer_snapshot(group.reset_index(drop=True), previous)
            snapshots_list.append(Snapshot(timestamp=timestamp.isoformat(), frame=engineered))
            previous = group
        snapshots = tuple(snapshots_list)
        return cls(snapshots, symbol.upper())

    def __len__(self) -> int:
        return len(self.snapshots)
