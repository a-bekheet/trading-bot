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
        self.assertTrue(truncated)
        self.assertGreater(reward, 0)
        self.assertAlmostEqual(sum(info["reward_components"].values()), reward)
        self.assertEqual(next_observation.portfolio.shape, (7,))
        self.assertAlmostEqual(next_observation.portfolio[3], 50.0)
        self.assertAlmostEqual(info["greek_exposures"]["delta"], 50.0)


    def test_mask_rejects_aggregate_cash_violation(self):
        env = OptionsEnv(demo_dataset(), slot_count=2, starting_cash=130)
        observation, _ = env.reset()
        action = np.array([1, 1])

        _, _, _, _, info = env.step(action)

        self.assertEqual(info["invalid_action_count"], 1)
        self.assertGreaterEqual(env._cash, 0)

    def test_greek_limit_masks_and_revalidates_aggregate_orders(self):
        env = OptionsEnv(
            demo_dataset(),
            slot_count=2,
            starting_cash=1_000,
            max_quantity=2,
            max_abs_delta=60,
        )
        observation, reset_info = env.reset()

        self.assertTrue(observation.action_mask[0, 1])
        self.assertFalse(observation.action_mask[0, 2])
        self.assertEqual(reset_info["risk_limits"]["delta"], 60)
        self.assertEqual(reset_info["portfolio_features"][-4:], (
            "delta", "gamma", "theta", "vega",
        ))
        _, _, _, _, info = env.step(np.array([1, 1]))

        self.assertEqual(info["invalid_action_count"], 1)
        self.assertEqual(len(info["executions"]), 1)
        self.assertAlmostEqual(info["greek_exposures"]["delta"], 50.0)

    def test_rejects_nonpositive_greek_limit(self):
        with self.assertRaisesRegex(ValueError, "Greek risk limits"):
            OptionsEnv(demo_dataset(), max_abs_vega=0)

    def test_spread_stress_changes_fill_without_changing_market_quotes(self):
        env = OptionsEnv(
            demo_dataset(),
            slot_count=2,
            starting_cash=1_000,
            spread_multiplier=2.0,
        )
        observation, _ = env.reset()

        _, _, _, _, info = env.step(np.array([1, 0]))

        self.assertAlmostEqual(info["executions"][0]["price"], 1.3)
        self.assertAlmostEqual(
            observation.contracts[0, 2],
            1.0,
        )

    def test_rejects_negative_execution_costs(self):
        with self.assertRaisesRegex(ValueError, "execution costs"):
            OptionsEnv(demo_dataset(), spread_multiplier=-1)

    def test_rejects_stale_environment_manifest(self):
        with self.assertRaisesRegex(ValueError, "manifest schema"):
            OptionsEnv(
                demo_dataset(),
                manifest=EnvManifest(schema_version="research-demo.v3"),
            )

    def test_risk_reducing_sell_remains_allowed_after_greek_drift(self):
        source = demo_dataset()
        later = source.snapshots[1].frame.copy()
        later["delta"] = 2.0
        dataset = SnapshotDataset(
            (source.snapshots[0], Snapshot(source.snapshots[1].timestamp, later)),
            source.symbol,
        )
        env = OptionsEnv(
            dataset,
            slot_count=2,
            max_quantity=1,
            starting_cash=1_000,
            max_abs_delta=60,
        )
        env.reset()

        observation, _, _, _, _ = env.step(np.array([1, 0]))

        self.assertAlmostEqual(observation.portfolio[3], 200.0)
        self.assertTrue(observation.action_mask[0, 2])

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

        # The current policy-visible slots are cached; only the next state ranks.
        self.assertEqual(calls, 1)

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

    def test_single_snapshot_truncates_without_unmarkable_fill(self):
        source = demo_dataset()
        dataset = SnapshotDataset((source.snapshots[0],), source.symbol)
        env = OptionsEnv(dataset, slot_count=2, starting_cash=1_000)
        before, _ = env.reset()

        after, reward, terminated, truncated, info = env.step(np.array([1, 0]))

        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(reward, 0)
        self.assertEqual(info["executions"], [])
        np.testing.assert_array_equal(before.portfolio, after.portfolio)
