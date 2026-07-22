"""Point-in-time snapshot loader for the research demo."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading_bot.market_data.snapshot_identity import material_snapshot_fingerprint
from trading_bot.training.features import (
    engineer_snapshot,
    volatility_regime_observation,
)


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
        self._fingerprint: str | None = None

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
        previous_material_fingerprint = None
        spot_history: list[tuple[pd.Timestamp, float]] = []
        volatility_history: list[tuple[float | None, float | None]] = []
        for timestamp, group in frame.groupby("collectedAt", sort=True):
            material_fingerprint = material_snapshot_fingerprint(group)
            if material_fingerprint == previous_material_fingerprint:
                continue
            spot = pd.to_numeric(group["underlyingPrice"], errors="coerce").iloc[0]
            spot_history.append((timestamp, float(spot)))
            engineered = engineer_snapshot(
                group.reset_index(drop=True),
                previous,
                spot_history=spot_history,
                volatility_history=volatility_history,
            )
            snapshots_list.append(Snapshot(timestamp=timestamp.isoformat(), frame=engineered))
            volatility_history.append(
                volatility_regime_observation(engineered)
            )
            previous = engineered
            previous_material_fingerprint = material_fingerprint
        snapshots = tuple(snapshots_list)
        return cls(snapshots, symbol.upper())

    def __len__(self) -> int:
        return len(self.snapshots)

    def subset(self, start: int, stop: int) -> "SnapshotDataset":
        """Return a chronological view while preserving precomputed past state."""
        if not 0 <= start < stop <= len(self.snapshots):
            raise ValueError("subset bounds are outside the dataset")
        return SnapshotDataset(self.snapshots[start:stop], self.symbol)

    @property
    def fingerprint(self) -> str:
        """Hash exact engineered snapshot contents for split provenance."""
        if self._fingerprint is None:
            digest = hashlib.sha256()
            digest.update(self.symbol.encode())
            for snapshot in self.snapshots:
                digest.update(snapshot.timestamp.encode())
                digest.update("\0".join(map(str, snapshot.frame.columns)).encode())
                digest.update(
                    pd.util.hash_pandas_object(
                        snapshot.frame,
                        index=True,
                    ).to_numpy().tobytes()
                )
            self._fingerprint = digest.hexdigest()
        return self._fingerprint
