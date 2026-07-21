"""Read collector CSVs for the user interface."""

from pathlib import Path

import pandas as pd


def available_tickers(data_dir: Path) -> list[str]:
    return sorted(path.stem for path in data_dir.glob("*.csv"))


def load_latest_snapshot(data_dir: Path, symbol: str) -> pd.DataFrame:
    data = pd.read_csv(data_dir / f"{symbol}.csv")
    if data.empty:
        return data
    return data[data["collectedAt"] == data["collectedAt"].iloc[-1]].copy()
