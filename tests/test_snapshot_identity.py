from io import StringIO
from unittest import TestCase

import pandas as pd

from trading_bot.market_data.snapshot_identity import (
    material_snapshot_fingerprint,
    persisted_material_snapshot_fingerprint,
)


class SnapshotIdentityTests(TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        return pd.DataFrame([
            {
                "collectedAt": "2026-07-22T02:00:00Z",
                "symbol": "AAPL",
                "expiration": "2026-07-24",
                "optionType": "call",
                "contractSymbol": "AAPL-C2",
                "lastTradeDate": pd.Timestamp("2026-07-21T19:44:59Z"),
                "strike": 362.5,
                "bid": 0.01,
                "ask": 0.04,
                "impliedVolatility": 0.9531254687499999,
                "underlyingPrice": 327.74,
                "marketState": "REGULAR",
                "riskFreeRate": 0.037300000190734865,
                "greekModel": "black-scholes-merton",
                "timeToExpiryYears": 0.01,
                "delta": 0.2,
                "theta": -0.1,
            },
            {
                "collectedAt": "2026-07-22T02:00:00Z",
                "symbol": "AAPL",
                "expiration": "2026-07-22",
                "optionType": "put",
                "contractSymbol": "AAPL-P1",
                "lastTradeDate": pd.Timestamp("2026-07-21T19:59:59Z"),
                "strike": 327.5,
                "bid": 0.2,
                "ask": 0.22,
                "impliedVolatility": 0.201234567890625,
                "underlyingPrice": 327.74,
                "marketState": "REGULAR",
                "riskFreeRate": 0.037300000190734865,
                "greekModel": "black-scholes-merton",
                "timeToExpiryYears": 0.001,
                "delta": -0.4,
                "theta": -0.3,
            },
        ])

    def test_fingerprint_survives_csv_round_trip_and_row_reordering(self):
        frame = self.frame()
        restored = pd.read_csv(StringIO(frame.to_csv(index=False)))

        self.assertEqual(
            material_snapshot_fingerprint(frame),
            material_snapshot_fingerprint(restored.iloc[::-1]),
        )
        self.assertEqual(
            persisted_material_snapshot_fingerprint(frame),
            material_snapshot_fingerprint(restored),
        )

    def test_ignores_capture_derivatives_but_detects_quote_change(self):
        frame = self.frame()
        recomputed = frame.copy()
        recomputed["collectedAt"] = "2026-07-22T02:15:00Z"
        recomputed["timeToExpiryYears"] *= 0.9
        recomputed["delta"] += 0.01
        recomputed["theta"] -= 0.01
        changed_quote = recomputed.copy()
        changed_quote.loc[0, "bid"] += 0.01
        changed_session = recomputed.copy()
        changed_session["marketState"] = "CLOSED"

        original = material_snapshot_fingerprint(frame)
        self.assertEqual(original, material_snapshot_fingerprint(recomputed))
        self.assertNotEqual(original, material_snapshot_fingerprint(changed_quote))
        self.assertNotEqual(original, material_snapshot_fingerprint(changed_session))
