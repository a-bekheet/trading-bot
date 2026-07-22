import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from trading_bot.execution.agent_runtime import (
    agent_runtime_lock,
    discover_selected_deployments,
    recurrent_state_from_dict,
    recurrent_state_to_dict,
    run_paper_agents,
)
from trading_bot.execution.agent_store import AgentPaperStore
from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.trainer import StreamingRecurrentPolicy, TrainingConfig
from trading_bot.training.walk_forward import (
    ModelSpec,
    WalkForwardConfig,
    run_walk_forward_training,
)
from tests.test_training_env import demo_dataset


def regular_dataset(count: int) -> SnapshotDataset:
    source = demo_dataset().snapshots[0].frame
    snapshots = []
    for index in range(count):
        timestamp = pd.Timestamp("2026-07-22T14:00:00Z") + pd.Timedelta(
            minutes=index
        )
        frame = source.copy()
        frame["collectedAt"] = timestamp.isoformat()
        frame["underlyingPrice"] = 330.0 + index * 0.1
        frame["bid"] = 1.0 + index * 0.01
        frame["ask"] = 1.2 + index * 0.01
        frame["lastPrice"] = 1.1 + index * 0.01
        frame["marketState"] = "REGULAR"
        frame["underlyingPriceSource"] = "regularMarketPrice"
        frame["underlyingQuoteTimeSource"] = "regularMarketTime"
        frame["underlyingQuoteTime"] = timestamp.isoformat()
        snapshots.append(Snapshot(timestamp.isoformat(), frame))
    return SnapshotDataset(tuple(snapshots), "AAPL")


@skipUnless(torch is not None, "install the optional ml extra")
class AgentRuntimeTests(TestCase):
    def test_account_lock_rejects_a_concurrent_cycle(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "agents.db"
            with agent_runtime_lock(database):
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    with agent_runtime_lock(database):
                        self.fail("nested runtime lock should not be acquired")

    def test_selected_checkpoint_advances_once_per_post_evaluation_snapshot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            run_dir = data_dir / "agent_runs" / "arena" / "run-1"
            run_dir.mkdir(parents=True)
            source = regular_dataset(9)
            run_walk_forward_training(
                SnapshotDataset(source.snapshots[:7], source.symbol).engineered(),
                WalkForwardConfig(
                    3,
                    2,
                    2,
                    latest_fold_only=True,
                    training_seed_offsets=(0,),
                    bootstrap_samples=100,
                    bootstrap_min_observations=2,
                    latency_warmup_iterations=1,
                    latency_measured_iterations=2,
                ),
                ModelSpec(hidden_size=4),
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                    seed=11,
                ),
                run_dir,
                env_kwargs={"slot_count": 2, "starting_cash": 10_000},
            )
            pd.concat(
                [snapshot.frame for snapshot in source.snapshots],
                ignore_index=True,
            ).to_csv(data_dir / "AAPL.csv", index=False)
            database = data_dir / "agent_paper.db"

            first = run_paper_agents(
                data_dir=data_dir,
                database=database,
                repo_root=root,
            )
            second = run_paper_agents(
                data_dir=data_dir,
                database=database,
                repo_root=root,
            )
            extended = regular_dataset(10)
            pd.concat(
                [snapshot.frame for snapshot in extended.snapshots],
                ignore_index=True,
            ).to_csv(data_dir / "AAPL.csv", index=False)
            third = run_paper_agents(
                data_dir=data_dir,
                database=database,
                repo_root=root,
            )
            store = AgentPaperStore(database)
            deployments = store.deployments()
            decisions = store.decisions()

        self.assertEqual(first["completed_count"], 1)
        self.assertEqual(first["failure_count"], 0)
        self.assertEqual(first["agents"][0]["new_decisions"], 2)
        self.assertEqual(second["agents"][0]["new_decisions"], 0)
        self.assertEqual(second["agents"][0]["decision_count"], 2)
        self.assertEqual(third["agents"][0]["new_decisions"], 1)
        self.assertEqual(third["agents"][0]["decision_count"], 3)
        self.assertEqual(third["agents"][0]["finalized_decision_count"], 2)
        self.assertEqual(third["agents"][0]["pending_decision_count"], 1)
        self.assertEqual(len(deployments), 1)
        self.assertEqual(len(decisions), 3)
        self.assertEqual(
            len({item["snapshot_timestamp"] for item in decisions}),
            3,
        )
        self.assertEqual(
            {item["reward_horizon"] for item in decisions},
            {
                "through_next_eligible_snapshot",
                "same_snapshot_execution_only",
            },
        )
        self.assertEqual(
            [item["outcome_status"] for item in decisions].count("finalized"),
            2,
        )
        self.assertEqual(
            [item["outcome_status"] for item in decisions].count("pending"),
            1,
        )
        self.assertTrue(
            all(
                item["outcome_return"] is not None
                for item in decisions
                if item["outcome_status"] == "finalized"
            )
        )
        if not deployments[0]["activated"]:
            self.assertTrue(
                all(not any(item["sandbox_orders"]) for item in decisions)
            )
            self.assertTrue(all(not item["executions"] for item in decisions))

    def test_recurrent_state_json_round_trip_restores_exact_cursor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            run_dir = data_dir / "agent_runs" / "run"
            run_dir.mkdir(parents=True)
            source = regular_dataset(7)
            run_walk_forward_training(
                source.engineered(),
                WalkForwardConfig(
                    3,
                    2,
                    2,
                    latest_fold_only=True,
                    training_seed_offsets=(0,),
                    bootstrap_samples=100,
                    bootstrap_min_observations=2,
                    latency_warmup_iterations=1,
                    latency_measured_iterations=2,
                ),
                ModelSpec(hidden_size=4),
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                ),
                run_dir,
                env_kwargs={"slot_count": 2},
            )
            pd.concat(
                [snapshot.frame for snapshot in source.snapshots],
                ignore_index=True,
            ).to_csv(data_dir / "AAPL.csv", index=False)
            spec = discover_selected_deployments(data_dir, repo_root=root)[0]
            from trading_bot.training.trainer import load_checkpoint

            model, manifest = load_checkpoint(spec["checkpoint_path"])
            env_options = {
                key: manifest["environment"][key]
                for key in (
                    "slot_count", "slot_assignment", "max_quantity",
                    "allow_collateralized_option_shorts", "starting_cash",
                    "commission_per_contract", "spread_multiplier",
                    "portfolio_valuation", "underlying_lot_size",
                    "max_abs_underlying_shares", "underlying_commission_per_share",
                    "underlying_slippage_bps", "invalid_action_penalty",
                    "reward_drawdown_penalty", "reward_downside_penalty",
                    "max_abs_delta", "max_abs_gamma", "max_abs_theta",
                    "max_abs_vega", "max_underlying_quote_age_seconds",
                )
            }
            from trading_bot.training.env import OptionsEnv

            env = OptionsEnv(source.engineered(), **env_options)
            observation, _ = env.reset()
            policy = StreamingRecurrentPolicy(model, 2)
            _, diagnostics = policy.act_with_diagnostics(observation)
            payload = json.loads(json.dumps(recurrent_state_to_dict(policy.snapshot())))
            restored = StreamingRecurrentPolicy(model, 2)
            restored.restore(recurrent_state_from_dict(payload, torch))

        self.assertEqual(restored.steps, policy.steps)
        self.assertEqual(restored.last_timestamp, policy.last_timestamp)
        self.assertGreaterEqual(diagnostics["action_confidence"], 0.0)
        self.assertLessEqual(diagnostics["action_confidence"], 1.0)
        self.assertGreaterEqual(diagnostics["normalized_action_entropy"], 0.0)
        self.assertLessEqual(diagnostics["normalized_action_entropy"], 1.0)
        self.assertGreater(diagnostics["decision_factor_count"], 0)
        self.assertLessEqual(
            diagnostics["explorable_action_factor_count"],
            diagnostics["decision_factor_count"],
        )
