from unittest import TestCase

import numpy as np
import pandas as pd

from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.env import OptionsEnv
from trading_bot.training.manifest import EnvManifest


def demo_dataset() -> SnapshotDataset:
    rows = []
    for timestamp, bid, ask in (
        ("2026-07-21T14:00:00Z", 1.0, 1.2),
        ("2026-07-21T14:01:00Z", 1.5, 1.7),
    ):
        for contract_symbol, strike in (
            ("AAPL260821C00330000", 330.0),
            ("AAPL260821C00335000", 335.0),
        ):
            rows.append({
                "collectedAt": timestamp,
                "contractSymbol": contract_symbol,
                "symbol": "AAPL",
                "expiration": "2026-08-21",
                "optionType": "call",
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "lastPrice": bid,
                "impliedVolatility": 0.2,
                "underlyingPrice": 330.0,
                "riskFreeRate": 0.04,
                "delta": 0.5,
                "gamma": 0.01,
                "theta": -0.1,
                "vega": 0.2,
                "volume": 100,
                "openInterest": 200,
                "greekModel": "black-scholes-merton",
            })
    frame = pd.DataFrame(rows)
    return SnapshotDataset(
        tuple(
            Snapshot(
                timestamp=pd.to_datetime(timestamp, utc=True).isoformat(),
                frame=group.reset_index(drop=True),
            )
            for timestamp, group in frame.groupby("collectedAt", sort=True)
        ),
        "AAPL",
    )


class OptionsEnvTests(TestCase):
    def test_reset_is_deterministic_and_step_has_no_lookahead(self):
        env = OptionsEnv(
            demo_dataset(),
            manifest=EnvManifest(symbol="AAPL", slot_count=2),
            slot_count=2,
            starting_cash=1_000,
        )
        first, first_info = env.reset(seed=7)
        second, second_info = env.reset(seed=7)

        np.testing.assert_array_equal(first.contracts, second.contracts)
        self.assertEqual(first.timestamp, second.timestamp)
        self.assertEqual(first_info["manifest_fingerprint"], second_info["manifest_fingerprint"])

        action = np.zeros(2, dtype=int)
        action[0] = 1  # buy one contract; mask was generated from t=0
        next_observation, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(next_observation.timestamp, "2026-07-21T14:01:00+00:00")
        self.assertEqual(info["executions"][0]["price"], 1.2)
        self.assertEqual(info["invalid_action_count"], 0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertGreater(reward, 0)

        _, _, _, final_truncated, _ = env.step(np.zeros(2, dtype=int))
        self.assertTrue(final_truncated)

    def test_mask_rejects_aggregate_cash_violation(self):
        env = OptionsEnv(demo_dataset(), slot_count=2, starting_cash=130)
        observation, _ = env.reset()
        action = np.array([1, 1])

        _, _, _, _, info = env.step(action)

        self.assertEqual(info["invalid_action_count"], 1)
        self.assertGreaterEqual(env._cash, 0)

    def test_step_ranks_contract_slots_only_once_per_snapshot(self):
        env = OptionsEnv(demo_dataset(), slot_count=2, starting_cash=1_000)
        env.reset()
        calls = 0
        original = env._slots

        def counted_slots(frame):
            nonlocal calls
            calls += 1
            return original(frame)

        env._slots = counted_slots
        env.step(np.array([1, 1]))

        # One ranking for order execution and one for the next observation.
        self.assertEqual(calls, 2)

    def test_slots_cover_expirations_and_option_types_before_surface_depth(self):
        rows = []
        for expiration in ("2026-08-21", "2026-09-18"):
            for option_type in ("call", "put"):
                for strike in (95, 100, 105):
                    rows.append({
                        "contractSymbol": f"{expiration}-{option_type}-{strike}",
                        "expiration": expiration,
                        "optionType": option_type,
                        "strike": strike,
                        "underlyingPrice": 100,
                        "logMoneyness": np.log(100 / strike),
                        "spreadPct": 0.02,
                        "openInterest": 100,
                    })
        frame = pd.DataFrame(rows)
        dataset = SnapshotDataset(
            (Snapshot("2026-07-21T14:00:00+00:00", frame),),
            "AAPL",
        )
        env = OptionsEnv(dataset, slot_count=4)

        selected = env._slots(frame)

        self.assertEqual(
            {(row["expiration"], row["optionType"]) for row in selected},
            {
                ("2026-08-21", "call"),
                ("2026-08-21", "put"),
                ("2026-09-18", "call"),
                ("2026-09-18", "put"),
            },
        )
        self.assertTrue(all(row["strike"] == 100 for row in selected))
