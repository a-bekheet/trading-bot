from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from trading_bot.interface.data import available_tickers, load_latest_snapshot


class InterfaceDataTests(TestCase):
    def test_loads_only_latest_snapshot(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            pd.DataFrame(
                [
                    {"collectedAt": "first", "symbol": "AAPL"},
                    {"collectedAt": "latest", "symbol": "AAPL"},
                    {"collectedAt": "latest", "symbol": "AAPL"},
                ]
            ).to_csv(data_dir / "AAPL.csv", index=False)

            tickers = available_tickers(data_dir)
            latest = load_latest_snapshot(data_dir, "AAPL")

        self.assertEqual(tickers, ["AAPL"])
        self.assertEqual(len(latest), 2)
        self.assertEqual(set(latest["collectedAt"]), {"latest"})
