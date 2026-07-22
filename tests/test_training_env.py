from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.env import (
    CONTRACT_FEATURES,
    PORTFOLIO_FEATURES,
    OptionsEnv,
)
from trading_bot.training.manifest import EnvManifest
from trading_bot.training.sequence import observation_vector


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


def three_snapshot_dataset() -> SnapshotDataset:
    source = demo_dataset()
    final = source.snapshots[-1].frame.copy()
    final["collectedAt"] = "2026-07-21T14:02:00Z"
    final["bid"] = 1.6
    final["ask"] = 1.8
    return SnapshotDataset(
        (*source.snapshots, Snapshot("2026-07-21T14:02:00+00:00", final)),
        source.symbol,
    )


def slot_churn_dataset() -> SnapshotDataset:
    snapshots = []
    definitions = (
        ("2026-07-21T14:00:00Z", (("A", 100), ("B", 101))),
        ("2026-07-21T14:01:00Z", (("A", 100), ("B", 101))),
        ("2026-07-21T14:02:00Z", (("B", 101), ("C", 102))),
        ("2026-07-21T14:03:00Z", (("A", 100), ("B", 101), ("C", 102))),
    )
    spots = (100.1, 100.9, 101.5, 100.5)
    for (timestamp, contracts), spot in zip(definitions, spots, strict=True):
        rows = []
        for symbol, strike in contracts:
            rows.append({
                "collectedAt": timestamp,
                "contractSymbol": symbol,
                "symbol": "TEST",
                "expiration": "2026-08-21",
                "optionType": "call",
                "strike": strike,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "impliedVolatility": 0.2,
                "underlyingPrice": spot,
                "riskFreeRate": 0.04,
                "delta": 0.5,
                "gamma": 0.01,
                "theta": -0.1,
                "vega": 0.2,
            })
        snapshots.append(Snapshot(
            pd.to_datetime(timestamp, utc=True).isoformat(),
            pd.DataFrame(rows),
        ))
    return SnapshotDataset(tuple(snapshots), "TEST")


def option_lifecycle_dataset(
    option_type: str,
    *,
    spots: tuple[float, ...],
    timestamps: tuple[str, ...],
    expiration: str,
    contract_count: int = 1,
) -> SnapshotDataset:
    snapshots = []
    for timestamp, spot in zip(timestamps, spots, strict=True):
        rows = []
        for index in range(contract_count):
            rows.append({
                "collectedAt": timestamp,
                "contractSymbol": f"TEST-{option_type}-{index}",
                "symbol": "TEST",
                "expiration": expiration,
                "optionType": option_type,
                "strike": 100,
                "bid": 2.0,
                "ask": 2.2,
                "lastPrice": 2.1,
                "impliedVolatility": 0.2,
                "underlyingPrice": spot,
                "riskFreeRate": 0.04,
                "delta": 0.5 if option_type == "call" else -0.5,
                "gamma": 0.01,
                "theta": -0.1,
                "vega": 0.2,
            })
        snapshots.append(Snapshot(
            pd.to_datetime(timestamp, utc=True).isoformat(),
            pd.DataFrame(rows),
        ))
    return SnapshotDataset(tuple(snapshots), "TEST")


def drawdown_dataset() -> SnapshotDataset:
    source = demo_dataset().snapshots[0].frame
    snapshots = []
    for timestamp, bid, ask in (
        ("2026-07-21T14:00:00Z", 2.0, 2.2),
        ("2026-07-21T14:01:00Z", 1.5, 1.7),
        ("2026-07-21T14:02:00Z", 2.0, 2.2),
        ("2026-07-21T14:03:00Z", 1.6, 1.8),
    ):
        frame = source.copy()
        frame["collectedAt"] = timestamp
        frame["bid"] = bid
        frame["ask"] = ask
        frame["lastPrice"] = (bid + ask) / 2
        snapshots.append(Snapshot(timestamp, frame))
    return SnapshotDataset(tuple(snapshots), "AAPL")


class OptionsEnvTests(TestCase):
    def test_explicit_nonregular_session_masks_every_trade(self):
        source = demo_dataset()
        snapshots = []
        for snapshot in source.snapshots:
            frame = snapshot.frame.copy()
            frame["marketState"] = "CLOSED"
            snapshots.append(Snapshot(snapshot.timestamp, frame))
        env = OptionsEnv(
            SnapshotDataset(tuple(snapshots), source.symbol),
            slot_count=2,
        )

        observation, info = env.reset()

        self.assertTrue(observation.action_mask[:, 0].all())
        self.assertFalse(observation.action_mask[:, 1:].any())
        self.assertEqual(info["market_session"], {
            "provider_state": "CLOSED",
            "regular": False,
            "coverage": 1.0,
            "trading_enabled": False,
            "fallback": None,
        })

    def test_legacy_unknown_session_is_visible_and_remains_tradeable(self):
        env = OptionsEnv(demo_dataset(), slot_count=2)

        observation, info = env.reset()

        self.assertTrue(observation.action_mask[:, 1:].any())
        self.assertEqual(info["market_session"]["provider_state"], "UNKNOWN")
        self.assertTrue(info["market_session"]["trading_enabled"])
        self.assertEqual(
            info["market_session"]["fallback"],
            "legacy_unknown_permits_research_demo_fills",
        )

    def test_explicit_stale_underlying_quote_masks_every_trade(self):
        source = demo_dataset()
        snapshots = []
        for snapshot in source.snapshots:
            frame = snapshot.frame.copy()
            frame["marketState"] = "REGULAR"
            frame["underlyingPriceSource"] = "regularMarketPrice"
            frame["underlyingQuoteTimeSource"] = "regularMarketTime"
            frame["underlyingQuoteTime"] = "2026-07-21T13:00:00Z"
            snapshots.append(Snapshot(snapshot.timestamp, frame))
        env = OptionsEnv(
            SnapshotDataset(tuple(snapshots), source.symbol),
            slot_count=2,
            max_underlying_quote_age_seconds=1_200,
        )

        observation, info = env.reset()

        self.assertTrue(observation.action_mask[:, 0].all())
        self.assertFalse(observation.action_mask[:, 1:].any())
        self.assertEqual(info["market_data_freshness"], {
            "quote_time": "2026-07-21T13:00:00Z",
            "quote_time_source": "regularMarketTime",
            "price_source": "regularMarketPrice",
            "age_seconds": 3_600.0,
            "coverage": 1.0,
            "max_age_seconds": 1_200,
            "trading_enabled": False,
            "fallback": None,
        })

    def test_missing_quote_time_is_visible_and_legacy_tradeable(self):
        env = OptionsEnv(demo_dataset(), slot_count=2)

        observation, info = env.reset()

        self.assertTrue(observation.action_mask[:, 1:].any())
        self.assertEqual(info["market_data_freshness"]["coverage"], 0.0)
        self.assertTrue(info["market_data_freshness"]["trading_enabled"])
        self.assertEqual(
            info["market_data_freshness"]["fallback"],
            "legacy_unknown_permits_research_demo_fills",
        )

    def test_path_causal_reward_penalizes_downside_and_new_max_drawdown(self):
        env = OptionsEnv(
            drawdown_dataset(),
            slot_count=1,
            max_quantity=1,
            starting_cash=1_000,
            reward_drawdown_penalty=3.0,
            reward_downside_penalty=2.0,
        )
        observation, reset_info = env.reset(seed=11)
        self.assertEqual(reset_info["reward_objective"], {
            "invalid_action_penalty": 0.001,
            "drawdown_penalty": 3.0,
            "downside_penalty": 2.0,
        })
        self.assertEqual(reset_info["portfolio_valuation"], "liquidation")

        observation, first_reward, _, _, first = env.step(np.array([1, 0]))

        first_net = (
            first["reward_components"]["gross_pnl_return"]
            + first["reward_components"]["fees"]
        )
        self.assertAlmostEqual(first_net, -0.0713)
        self.assertAlmostEqual(first["path_risk"]["maximum_drawdown"], 0.0713)
        self.assertAlmostEqual(first["reward_components"]["drawdown"], -0.2139)
        self.assertAlmostEqual(first["reward_components"]["downside"], -0.1426)
        self.assertAlmostEqual(sum(first["reward_components"].values()), first_reward)

        observation, recovery_reward, _, _, recovery = env.step(
            np.array([0, 0])
        )

        self.assertGreater(recovery_reward, 0)
        self.assertEqual(recovery["path_risk"]["drawdown_increase"], 0)
        self.assertEqual(recovery["reward_components"]["drawdown"], 0)
        self.assertEqual(recovery["reward_components"]["downside"], 0)

        _, relapse_reward, _, truncated, relapse = env.step(np.array([0, 0]))

        self.assertTrue(truncated)
        relapse_net = (
            relapse["reward_components"]["gross_pnl_return"]
            + relapse["reward_components"]["fees"]
        )
        self.assertLess(relapse_reward, relapse_net)
        self.assertEqual(relapse["path_risk"]["drawdown_increase"], 0)
        self.assertEqual(relapse["reward_components"]["drawdown"], 0)
        self.assertAlmostEqual(
            relapse["reward_components"]["downside"],
            -2 * max(-relapse_net, 0),
        )

        env.reset(seed=11)
        _, repeated_reward, _, _, repeated = env.step(np.array([1, 0]))
        self.assertAlmostEqual(repeated_reward, first_reward)
        self.assertAlmostEqual(
            repeated["path_risk"]["drawdown_increase"],
            first["path_risk"]["drawdown_increase"],
        )

    def test_default_reward_objective_is_unchanged(self):
        default = OptionsEnv(
            drawdown_dataset(),
            slot_count=1,
            max_quantity=1,
            starting_cash=1_000,
        )
        shaped = OptionsEnv(
            drawdown_dataset(),
            slot_count=1,
            max_quantity=1,
            starting_cash=1_000,
            reward_drawdown_penalty=1.0,
            reward_downside_penalty=1.0,
        )
        default_observation, _ = default.reset()
        shaped_observation, _ = shaped.reset()

        np.testing.assert_array_equal(
            observation_vector(default_observation),
            observation_vector(shaped_observation),
        )
        np.testing.assert_array_equal(
            default_observation.action_mask,
            shaped_observation.action_mask,
        )

        default_next, default_reward, _, _, default_info = default.step(
            np.array([1, 0])
        )
        shaped_next, shaped_reward, _, _, _ = shaped.step(np.array([1, 0]))

        np.testing.assert_array_equal(
            observation_vector(default_next),
            observation_vector(shaped_next),
        )
        self.assertAlmostEqual(
            default_reward,
            default_info["pnl"] / 1_000,
        )
        self.assertEqual(default_info["reward_components"]["drawdown"], 0)
        self.assertEqual(default_info["reward_components"]["downside"], 0)
        self.assertLess(shaped_reward, default_reward)

    def test_liquidation_nav_charges_round_trip_cost_without_close_jump(self):
        env = OptionsEnv(
            option_lifecycle_dataset(
                "call",
                spots=(100, 100, 100),
                timestamps=(
                    "2026-07-20T14:00:00Z",
                    "2026-07-20T14:01:00Z",
                    "2026-07-20T14:02:00Z",
                ),
                expiration="2026-08-21",
            ),
            slot_count=1,
            max_quantity=1,
            starting_cash=1_000,
        )
        env.reset()

        opened, reward, _, _, _ = env.step(np.array([1, 0]))
        closed, close_reward, _, truncated, _ = env.step(np.array([2, 0]))

        self.assertAlmostEqual(opened.portfolio[2], 978.7)
        self.assertAlmostEqual(reward, -0.0213)
        self.assertTrue(truncated)
        self.assertAlmostEqual(closed.portfolio[2], opened.portfolio[2])
        self.assertAlmostEqual(close_reward, 0.0)

    def test_liquidation_nav_marks_short_at_ask_and_preserves_midpoint_mode(self):
        dataset = option_lifecycle_dataset(
            "put",
            spots=(100, 100),
            timestamps=(
                "2026-07-20T14:00:00Z",
                "2026-07-20T14:01:00Z",
            ),
            expiration="2026-08-21",
        )
        liquidation = OptionsEnv(
            dataset,
            slot_count=1,
            max_quantity=1,
            starting_cash=20_000,
            allow_collateralized_option_shorts=True,
        )
        midpoint = OptionsEnv(
            dataset,
            slot_count=1,
            max_quantity=1,
            starting_cash=20_000,
            allow_collateralized_option_shorts=True,
            portfolio_valuation="midpoint",
        )
        liquidation.reset()
        midpoint.reset()

        liquidated, _, _, _, liquidated_info = liquidation.step(
            np.array([2, 0])
        )
        midpoint_marked, _, _, _, midpoint_info = midpoint.step(
            np.array([2, 0])
        )

        self.assertAlmostEqual(liquidated.portfolio[2], 19_978.7)
        self.assertAlmostEqual(midpoint_marked.portfolio[2], 19_989.35)
        self.assertEqual(liquidated_info["portfolio_valuation"], "liquidation")
        self.assertEqual(midpoint_info["portfolio_valuation"], "midpoint")

    def test_liquidation_nav_uses_last_executable_mark_during_quote_gap(self):
        source = option_lifecycle_dataset(
            "call",
            spots=(100, 100),
            timestamps=(
                "2026-07-20T14:00:00Z",
                "2026-07-20T14:01:00Z",
            ),
            expiration="2026-08-21",
            contract_count=2,
        )
        missing_held_quote = source.snapshots[1].frame.iloc[[1]].copy()
        dataset = SnapshotDataset(
            (
                source.snapshots[0],
                Snapshot(source.snapshots[1].timestamp, missing_held_quote),
            ),
            source.symbol,
        )
        env = OptionsEnv(dataset, slot_count=1, max_quantity=1, starting_cash=1_000)
        env.reset()

        observation, _, _, _, _ = env.step(np.array([1, 0]))

        self.assertAlmostEqual(observation.portfolio[2], 978.7)

    def test_underlying_liquidation_nav_is_continuous_when_closed(self):
        dataset = option_lifecycle_dataset(
            "call",
            spots=(100, 100, 100),
            timestamps=(
                "2026-07-20T14:00:00Z",
                "2026-07-20T14:01:00Z",
                "2026-07-20T14:02:00Z",
            ),
            expiration="2026-08-21",
        )
        env = OptionsEnv(
            dataset,
            slot_count=1,
            max_quantity=1,
            starting_cash=10_000,
            underlying_lot_size=25,
            max_abs_underlying_shares=25,
            underlying_commission_per_share=0.01,
            underlying_slippage_bps=10,
        )
        env.reset()

        opened, _, _, _, _ = env.step(np.array([0, 1]))
        closed, reward, _, truncated, _ = env.step(np.array([0, 2]))

        self.assertAlmostEqual(opened.portfolio[2], 9_994.5)
        self.assertTrue(truncated)
        self.assertAlmostEqual(closed.portfolio[2], opened.portfolio[2])
        self.assertAlmostEqual(reward, 0.0)

    def test_collateralized_short_put_can_close_or_cross_without_reusing_cash(self):
        dataset = option_lifecycle_dataset(
            "put",
            spots=(100, 100, 100),
            timestamps=(
                "2026-07-20T14:00:00Z",
                "2026-07-20T14:01:00Z",
                "2026-07-20T14:02:00Z",
            ),
            expiration="2026-08-21",
        )
        legacy = OptionsEnv(
            dataset,
            slot_count=1,
            max_quantity=2,
            starting_cash=9_900,
        )
        legacy_observation, _ = legacy.reset()
        self.assertFalse(legacy_observation.action_mask[0, 3])

        env = OptionsEnv(
            dataset,
            slot_count=1,
            max_quantity=2,
            starting_cash=9_900,
            allow_collateralized_option_shorts=True,
        )
        observation, info = env.reset()
        quantity_index = CONTRACT_FEATURES.index("positionQuantity")
        average_index = CONTRACT_FEATURES.index("positionAveragePrice")
        unrealized_index = CONTRACT_FEATURES.index(
            "positionUnrealizedReturn"
        )

        self.assertTrue(observation.action_mask[0, 3])
        self.assertFalse(observation.action_mask[0, 4])
        self.assertEqual(info["collateral"]["reserved_cash"], 0)
        opened, _, _, _, opened_info = env.step(np.array([3, 0]))

        self.assertEqual(opened_info["invalid_action_count"], 0)
        self.assertEqual(opened.contracts[0, quantity_index], -1)
        self.assertEqual(opened.contracts[0, average_index], 2.0)
        self.assertAlmostEqual(
            opened.contracts[0, unrealized_index],
            -0.1,
        )
        self.assertAlmostEqual(opened.portfolio[8], 10_000)
        self.assertEqual(opened.portfolio[9], 0)
        self.assertFalse(opened.action_mask[-1, 1])

        crossed, _, _, truncated, crossed_info = env.step(
            np.array([2, 0])
        )

        self.assertTrue(truncated)
        self.assertEqual(crossed_info["invalid_action_count"], 0)
        self.assertEqual(crossed.contracts[0, quantity_index], 1)
        self.assertAlmostEqual(crossed.contracts[0, average_index], 2.2)
        self.assertEqual(crossed.portfolio[8], 0)

    def test_short_put_collateral_cannot_support_two_simultaneous_writers(self):
        dataset = option_lifecycle_dataset(
            "put",
            spots=(100, 100, 100),
            timestamps=(
                "2026-07-20T14:00:00Z",
                "2026-07-20T14:01:00Z",
                "2026-07-20T14:02:00Z",
            ),
            expiration="2026-08-21",
            contract_count=2,
        )
        env = OptionsEnv(
            dataset,
            slot_count=2,
            max_quantity=1,
            starting_cash=9_900,
            allow_collateralized_option_shorts=True,
        )
        observation, _ = env.reset()
        self.assertTrue(observation.action_mask[0, 2])
        self.assertTrue(observation.action_mask[1, 2])

        after, _, _, _, info = env.step(np.array([2, 2, 0]))

        self.assertEqual(info["invalid_action_count"], 1)
        self.assertEqual(len(info["executions"]), 1)
        self.assertEqual(after.portfolio[8], 10_000)
        self.assertGreaterEqual(info["collateral"]["available_cash"], 0)

    def test_cash_secured_put_is_physically_assigned_after_expiration(self):
        dataset = option_lifecycle_dataset(
            "put",
            spots=(100, 100, 80),
            timestamps=(
                "2026-07-20T14:00:00Z",
                "2026-07-21T15:00:00Z",
                "2026-07-22T14:00:00Z",
            ),
            expiration="2026-07-21",
        )
        env = OptionsEnv(
            dataset,
            slot_count=1,
            max_quantity=1,
            starting_cash=9_900,
            max_abs_underlying_shares=100,
            allow_collateralized_option_shorts=True,
        )
        env.reset()
        opened, _, _, _, _ = env.step(np.array([2, 0]))
        self.assertEqual(opened.portfolio[8], 10_000)
        self.assertEqual(opened.portfolio[7], 0)

        assigned, _, _, truncated, info = env.step(np.array([0, 0]))

        self.assertTrue(truncated)
        self.assertEqual(assigned.portfolio[7], 100)
        self.assertEqual(assigned.portfolio[8], 0)
        self.assertAlmostEqual(assigned.portfolio[0], 99.35)
        self.assertEqual(len(info["option_settlements"]), 1)
        settlement = info["option_settlements"][0]
        self.assertEqual(settlement["style"], "physical_assignment")
        self.assertEqual(settlement["position_quantity"], -1)
        self.assertEqual(settlement["intrinsic_value"], 20)

    def test_short_option_requires_a_valid_expiration(self):
        dataset = option_lifecycle_dataset(
            "put",
            spots=(100, 100),
            timestamps=(
                "2026-07-20T14:00:00Z",
                "2026-07-20T14:01:00Z",
            ),
            expiration="not-a-date",
        )
        env = OptionsEnv(
            dataset,
            slot_count=1,
            max_quantity=1,
            starting_cash=20_000,
            allow_collateralized_option_shorts=True,
        )

        observation, _ = env.reset()

        self.assertTrue(observation.action_mask[0, 1])
        self.assertFalse(observation.action_mask[0, 2])

    def test_short_option_respects_signed_greek_limit(self):
        dataset = option_lifecycle_dataset(
            "put",
            spots=(100, 100),
            timestamps=(
                "2026-07-20T14:00:00Z",
                "2026-07-20T14:01:00Z",
            ),
            expiration="2026-08-21",
        )
        env = OptionsEnv(
            dataset,
            slot_count=1,
            max_quantity=1,
            starting_cash=20_000,
            underlying_lot_size=25,
            allow_collateralized_option_shorts=True,
            max_abs_delta=40,
        )

        env.reset()
        observation, _, _, _, _ = env.step(np.array([0, 1]))

        self.assertTrue(observation.action_mask[0, 1])
        self.assertFalse(observation.action_mask[0, 2])

    def test_option_expiry_distinguishes_worthless_and_long_intrinsic(self):
        timestamps = (
            "2026-07-20T14:00:00Z",
            "2026-07-22T14:00:00Z",
        )
        short_put = OptionsEnv(
            option_lifecycle_dataset(
                "put",
                spots=(100, 120),
                timestamps=timestamps,
                expiration="2026-07-21",
            ),
            slot_count=1,
            max_quantity=1,
            starting_cash=9_900,
            allow_collateralized_option_shorts=True,
        )
        short_put.reset()

        expired, _, _, _, short_info = short_put.step(np.array([2, 0]))

        self.assertAlmostEqual(expired.portfolio[0], 10_099.35)
        self.assertEqual(expired.portfolio[8], 0)
        self.assertEqual(
            short_info["option_settlements"][0]["style"],
            "expired_worthless",
        )

        long_call = OptionsEnv(
            option_lifecycle_dataset(
                "call",
                spots=(100, 120),
                timestamps=timestamps,
                expiration="2026-07-21",
            ),
            slot_count=1,
            max_quantity=1,
            starting_cash=1_000,
        )
        long_call.reset()

        settled, _, _, _, long_info = long_call.step(np.array([1, 0]))

        self.assertAlmostEqual(settled.portfolio[0], 2_779.35)
        self.assertEqual(
            long_info["option_settlements"][0]["style"],
            "cash_intrinsic",
        )

    def test_covered_call_reserves_shares_and_is_assigned_after_expiration(self):
        dataset = option_lifecycle_dataset(
            "call",
            spots=(100, 100, 100, 120),
            timestamps=(
                "2026-07-20T14:00:00Z",
                "2026-07-20T14:01:00Z",
                "2026-07-20T14:02:00Z",
                "2026-07-22T14:00:00Z",
            ),
            expiration="2026-07-21",
            contract_count=2,
        )
        env = OptionsEnv(
            dataset,
            slot_count=2,
            max_quantity=1,
            starting_cash=25_000,
            underlying_lot_size=100,
            max_abs_underlying_shares=100,
            allow_collateralized_option_shorts=True,
        )
        initial, _ = env.reset()
        self.assertFalse(initial.action_mask[0, 2])
        self.assertFalse(initial.action_mask[1, 2])
        shares, _, _, _, _ = env.step(np.array([0, 0, 1]))
        self.assertEqual(shares.portfolio[7], 100)
        self.assertTrue(shares.action_mask[0, 2])
        self.assertTrue(shares.action_mask[1, 2])

        covered, _, _, _, info = env.step(np.array([2, 2, 0]))

        self.assertEqual(info["invalid_action_count"], 1)
        self.assertEqual(len(info["executions"]), 1)
        self.assertEqual(covered.portfolio[9], 100)
        self.assertFalse(covered.action_mask[-1, 2])
        assigned, _, _, truncated, info = env.step(np.array([0, 0, 0]))

        self.assertTrue(truncated)
        self.assertEqual(assigned.portfolio[7], 0)
        self.assertEqual(assigned.portfolio[9], 0)
        self.assertEqual(
            info["option_settlements"][0]["style"],
            "physical_assignment",
        )

    def test_stable_slots_preserve_identity_and_expose_replacements(self):
        env = OptionsEnv(slot_churn_dataset(), slot_count=2)
        continuity_index = CONTRACT_FEATURES.index("slotContinuity")

        first, first_info = env.reset(seed=5)
        second, _, _, truncated, second_info = env.step(np.zeros(2, dtype=int))
        third, _, _, _, third_info = env.step(np.zeros(2, dtype=int))

        self.assertEqual(first.contract_ids, ("A", "B"))
        np.testing.assert_array_equal(
            first.contracts[:, continuity_index],
            (0, 0),
        )
        self.assertEqual(first_info["slot_identity_status"], "no_prior_snapshot")
        self.assertEqual(second.contract_ids, ("A", "B"))
        np.testing.assert_array_equal(
            second.contracts[:, continuity_index],
            (1, 1),
        )
        self.assertFalse(truncated)
        self.assertEqual(second_info["slot_retained_count"], 2)
        self.assertEqual(second_info["slot_churn_rate"], 0.0)
        self.assertEqual(third.contract_ids, ("C", "B"))
        np.testing.assert_array_equal(
            third.contracts[:, continuity_index],
            (0, 1),
        )
        self.assertEqual(third_info["slot_retained_count"], 1)
        self.assertEqual(third_info["slot_changed_count"], 1)
        self.assertEqual(third_info["slot_comparable_count"], 2)
        self.assertEqual(third_info["slot_churn_rate"], 0.5)
        self.assertEqual(env.manifest.slot_assignment, "stable")

    def test_ranked_slot_fallback_reports_identity_churn(self):
        env = OptionsEnv(
            slot_churn_dataset(),
            slot_count=2,
            slot_assignment="ranked",
        )
        continuity_index = CONTRACT_FEATURES.index("slotContinuity")
        first, _ = env.reset()
        second, _, _, _, info = env.step(np.zeros(2, dtype=int))

        self.assertEqual(first.contract_ids, ("A", "B"))
        self.assertEqual(second.contract_ids, ("B", "A"))
        np.testing.assert_array_equal(
            second.contracts[:, continuity_index],
            (0, 0),
        )
        self.assertEqual(info["slot_identity_status"], "ranked")
        self.assertEqual(info["slot_churn_rate"], 1.0)
        reset, reset_info = env.reset(options={"start_index": 1})
        np.testing.assert_array_equal(
            reset.contracts[:, continuity_index],
            (0, 0),
        )
        self.assertEqual(reset_info["slot_identity_status"], "no_prior_snapshot")

    def test_held_contract_reclaims_home_slot_after_quote_gap(self):
        env = OptionsEnv(slot_churn_dataset(), slot_count=2)
        env.reset()

        env.step(np.array([1, 0]))
        missing, _, _, _, _ = env.step(np.zeros(2, dtype=int))
        restored, _, _, _, info = env.step(np.zeros(2, dtype=int))

        self.assertEqual(missing.contract_ids, ("C", "B"))
        self.assertEqual(restored.contract_ids, ("A", "B"))
        self.assertTrue(restored.action_mask[0, env.max_quantity + 1])
        self.assertEqual(info["slot_changed_count"], 1)

    def test_visible_held_contract_wins_home_collision(self):
        env = OptionsEnv(slot_churn_dataset(), slot_count=2)
        env.reset()

        env.step(np.array([1, 0]))
        env.step(np.zeros(2, dtype=int))
        restored, _, _, _, _ = env.step(np.array([1, 0]))

        self.assertEqual(restored.contract_ids, ("C", "A"))
        self.assertTrue(restored.action_mask[0, env.max_quantity + 1])
        self.assertTrue(restored.action_mask[1, env.max_quantity + 1])

    def test_rejects_unknown_slot_assignment(self):
        with self.assertRaisesRegex(ValueError, "slot_assignment"):
            OptionsEnv(demo_dataset(), slot_assignment="random")

    def test_reset_is_deterministic_and_step_has_no_lookahead(self):
        env = OptionsEnv(
            demo_dataset(),
            manifest=EnvManifest(symbol="AAPL", slot_count=2),
            slot_count=2,
            starting_cash=1_000,
        )
        first, first_info = env.reset(seed=7)
        second, second_info = env.reset(seed=7)
        quantity_index = CONTRACT_FEATURES.index("positionQuantity")
        average_index = CONTRACT_FEATURES.index("positionAveragePrice")
        unrealized_index = CONTRACT_FEATURES.index("positionUnrealizedReturn")

        np.testing.assert_array_equal(first.contracts, second.contracts)
        np.testing.assert_array_equal(first.contracts[:, quantity_index], 0)
        np.testing.assert_array_equal(first.contracts[:, average_index], 0)
        np.testing.assert_array_equal(first.contracts[:, unrealized_index], 0)
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
        self.assertEqual(next_observation.portfolio.shape, (12,))
        self.assertAlmostEqual(next_observation.portfolio[3], 50.0)
        self.assertAlmostEqual(info["greek_exposures"]["delta"], 50.0)
        np.testing.assert_array_equal(next_observation.portfolio[8:10], (0, 0))
        np.testing.assert_array_equal(next_observation.portfolio[10:], (0, 1))
        self.assertEqual(next_observation.contracts[0, quantity_index], 1)
        self.assertAlmostEqual(next_observation.contracts[0, average_index], 1.2)
        self.assertAlmostEqual(
            next_observation.contracts[0, unrealized_index],
            1.5 / 1.2 - 1,
        )

    def test_executable_bid_ask_remains_tradeable_without_last_price(self):
        source = demo_dataset()
        snapshots = tuple(
            Snapshot(
                snapshot.timestamp,
                snapshot.frame.assign(lastPrice=np.nan),
            )
            for snapshot in source.snapshots
        )
        env = OptionsEnv(
            SnapshotDataset(snapshots, source.symbol),
            slot_count=2,
            starting_cash=1_000,
        )

        observation, _ = env.reset()
        after, reward, _, truncated, info = env.step(np.array([1, 0, 0]))

        self.assertTrue(observation.valid_mask[0])
        self.assertTrue(observation.action_mask[0, 1])
        self.assertEqual(info["executions"][0]["price"], 1.2)
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(truncated)
        self.assertTrue(np.isfinite(after.portfolio).all())

        crossed_frame = snapshots[0].frame.assign(bid=2.0, ask=1.0)
        crossed = OptionsEnv(
            SnapshotDataset((Snapshot("crossed", crossed_frame),), "AAPL"),
            slot_count=2,
        ).reset()[0]
        self.assertFalse(crossed.valid_mask[0])
        self.assertFalse(crossed.action_mask[0].any())

    def test_position_lifecycle_clocks_survive_adds_and_reset_on_cross(self):
        dataset = option_lifecycle_dataset(
            "put",
            spots=(100, 100, 100, 100, 100),
            timestamps=(
                "2026-07-20T14:00:00Z",
                "2026-07-20T14:01:00Z",
                "2026-07-20T14:02:00Z",
                "2026-07-20T14:03:00Z",
                "2026-07-20T14:04:00Z",
            ),
            expiration="2026-08-21",
        )
        env = OptionsEnv(
            dataset,
            slot_count=1,
            max_quantity=3,
            starting_cash=20_000,
            allow_collateralized_option_shorts=True,
        )
        quantity = CONTRACT_FEATURES.index("positionQuantity")
        age = CONTRACT_FEATURES.index("positionAgeSteps")
        last_trade_age = CONTRACT_FEATURES.index(
            "positionLastTradeAgeSteps"
        )

        initial, _ = env.reset()
        opened, *_ = env.step(np.array([1, 0]))
        held, *_ = env.step(np.array([0, 0]))
        added, *_ = env.step(np.array([1, 0]))
        crossed, *_ = env.step(np.array([6, 0]))

        np.testing.assert_array_equal(
            initial.contracts[0, [quantity, age, last_trade_age]],
            (0, 0, 0),
        )
        np.testing.assert_array_equal(
            opened.contracts[0, [quantity, age, last_trade_age]],
            (1, 1, 1),
        )
        np.testing.assert_array_equal(
            held.contracts[0, [quantity, age, last_trade_age]],
            (1, 2, 2),
        )
        np.testing.assert_array_equal(
            added.contracts[0, [quantity, age, last_trade_age]],
            (2, 3, 1),
        )
        np.testing.assert_array_equal(
            crossed.contracts[0, [quantity, age, last_trade_age]],
            (-1, 1, 1),
        )

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
        self.assertEqual(reset_info["portfolio_features"][3:7], (
            "delta", "gamma", "theta", "vega",
        ))
        _, _, _, _, info = env.step(np.array([1, 1]))

        self.assertEqual(info["invalid_action_count"], 1)
        self.assertEqual(len(info["executions"]), 1)
        self.assertAlmostEqual(info["greek_exposures"]["delta"], 50.0)

    def test_compact_action_feasibility_state_matches_exact_masks(self):
        env = OptionsEnv(
            demo_dataset(),
            slot_count=2,
            starting_cash=1_000,
            max_quantity=2,
            max_abs_delta=60,
        )
        observation, _ = env.reset()
        buy = CONTRACT_FEATURES.index("buyFeasibleFraction")
        sell = CONTRACT_FEATURES.index("sellFeasibleFraction")
        underlying_buy = PORTFOLIO_FEATURES.index(
            "underlyingBuyFeasibleFraction"
        )
        underlying_sell = PORTFOLIO_FEATURES.index(
            "underlyingSellFeasibleFraction"
        )

        np.testing.assert_allclose(observation.contracts[:, buy], 0.5)
        np.testing.assert_allclose(observation.contracts[:, sell], 0.0)
        self.assertEqual(observation.portfolio[underlying_buy], 0.0)
        self.assertEqual(observation.portfolio[underlying_sell], 1.0)
        for slot in range(env.slot_count):
            self.assertEqual(
                observation.contracts[slot, buy],
                observation.action_mask[slot, 1:3].mean(),
            )
            self.assertEqual(
                observation.contracts[slot, sell],
                observation.action_mask[slot, 3:5].mean(),
            )

        next_observation, _, _, _, _ = env.step(np.array([1, 0]))

        self.assertEqual(next_observation.contracts[0, buy], 0.0)
        self.assertEqual(next_observation.contracts[0, sell], 0.5)
        self.assertEqual(next_observation.contracts[1, buy], 0.0)
        self.assertEqual(next_observation.contracts[1, sell], 0.0)

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
        for kwargs in (
            {"invalid_action_penalty": -1},
            {"reward_drawdown_penalty": -1},
            {"reward_downside_penalty": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError,
                "reward penalties",
            ):
                OptionsEnv(demo_dataset(), **kwargs)
        with self.assertRaisesRegex(ValueError, "portfolio_valuation"):
            OptionsEnv(demo_dataset(), portfolio_valuation="optimistic")

    def test_underlying_trade_updates_cash_nav_delta_and_position_limits(self):
        env = OptionsEnv(
            three_snapshot_dataset(),
            slot_count=2,
            max_quantity=2,
            starting_cash=1_000,
            underlying_lot_size=25,
            max_abs_underlying_shares=50,
            underlying_commission_per_share=0.01,
            underlying_slippage_bps=10,
        )
        observation, _ = env.reset()

        self.assertEqual(observation.action_mask.shape, (3, 5))
        np.testing.assert_array_equal(
            observation.underlying_action_quantities,
            np.array([0, 25, 50, -25, -50]),
        )
        next_observation, _, _, truncated, info = env.step(
            np.array([1, 0, 4])
        )

        self.assertFalse(truncated)
        self.assertEqual([item["instrument"] for item in info["executions"]], [
            "underlying",
            "option",
        ])
        self.assertAlmostEqual(info["executions"][0]["price"], 329.67)
        self.assertAlmostEqual(info["executions"][0]["fee"], 0.5)
        self.assertEqual(next_observation.portfolio[7], -50)
        self.assertAlmostEqual(next_observation.portfolio[3], 0)
        self.assertAlmostEqual(info["greek_exposures"]["delta"], 0)
        self.assertFalse(next_observation.action_mask[2, 3])
        self.assertFalse(next_observation.action_mask[2, 4])
        self.assertAlmostEqual(info["trade_notional"], 16_603.5)

    def test_rejects_stale_environment_manifest(self):
        with self.assertRaisesRegex(ValueError, "manifest schema"):
            OptionsEnv(
                demo_dataset(),
                manifest=EnvManifest(schema_version="research-demo.v4"),
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

    def test_sparse_stable_slots_skip_ranking_when_every_quote_is_assigned(self):
        env = OptionsEnv(demo_dataset(), slot_count=32, starting_cash=1_000)
        before, _ = env.reset()
        calls = 0
        original = env._ranked_slots

        def counted_rank(frame, rows=None):
            nonlocal calls
            calls += 1
            return original(frame, rows)

        env._ranked_slots = counted_rank
        after, _, _, _, _ = env.step(np.zeros(33, dtype=int))

        self.assertEqual(calls, 0)
        self.assertEqual(after.contract_ids[:2], before.contract_ids[:2])
        self.assertTrue(all(item is None for item in after.contract_ids[2:]))

    def test_sparse_stable_slots_rank_only_when_a_new_quote_appears(self):
        source = demo_dataset()
        dataset = SnapshotDataset(
            (
                Snapshot(
                    source.snapshots[0].timestamp,
                    source.snapshots[0].frame.iloc[:1].copy(),
                ),
                source.snapshots[1],
            ),
            source.symbol,
        )
        env = OptionsEnv(dataset, slot_count=4, starting_cash=1_000)
        env.reset()
        calls = 0
        original = env._ranked_slots

        def counted_rank(frame, rows=None):
            nonlocal calls
            calls += 1
            return original(frame, rows)

        env._ranked_slots = counted_rank
        observation, _, _, _, _ = env.step(np.zeros(5, dtype=int))

        self.assertEqual(calls, 1)
        self.assertEqual(
            {item for item in observation.contract_ids if item is not None},
            {
                "AAPL260821C00330000",
                "AAPL260821C00335000",
            },
        )

    def test_expiry_settlement_skips_timestamp_work_without_positions(self):
        env = OptionsEnv(demo_dataset(), slot_count=2)
        env.reset()

        with patch(
            "trading_bot.training.env.pd.to_datetime",
            side_effect=AssertionError("timestamp parser should not run"),
        ):
            settlements = env._settle_expired_positions(
                env.dataset.snapshots[1].frame
            )

        self.assertEqual(settlements, [])

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

        duplicated = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
        duplicate_safe = env._ranked_slots(duplicated)
        self.assertEqual(
            len({str(row["contractSymbol"]) for row in duplicate_safe}),
            len(duplicate_safe),
        )

    def test_duplicate_quotes_keep_first_row_for_slots_marks_and_greeks(self):
        source = demo_dataset()
        next_frame = source.snapshots[1].frame.copy()
        duplicate = next_frame.iloc[[0]].copy()
        duplicate["bid"] = 50.0
        duplicate["ask"] = 60.0
        duplicate["lastPrice"] = 55.0
        duplicate["delta"] = 9.0
        next_frame = pd.concat((next_frame, duplicate), ignore_index=True)
        dataset = SnapshotDataset(
            (
                source.snapshots[0],
                Snapshot(source.snapshots[1].timestamp, next_frame),
            ),
            source.symbol,
        )
        env = OptionsEnv(
            dataset,
            slot_count=1,
            max_quantity=1,
            starting_cash=1_000,
        )
        env.reset()

        observation, _, _, truncated, info = env.step(np.array([1, 0]))

        self.assertTrue(truncated)
        self.assertAlmostEqual(observation.portfolio[2], 1_028.7)
        self.assertAlmostEqual(observation.portfolio[3], 50.0)
        self.assertAlmostEqual(info["greek_exposures"]["delta"], 50.0)

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
