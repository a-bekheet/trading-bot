import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.arena import (
    AGENT_ARENA_SCHEMA_VERSION,
    DEFAULT_ARENA_ACTIVATION_MIN_SCORE_ADVANTAGE,
    DEFAULT_ARENA_LATEST_FOLD_ONLY,
    DEFAULT_ARENA_REQUIRE_READY_TAIL,
    DEFAULT_ARENA_SELECTION_SCORE_TOLERANCE,
    DEFAULT_ARENA_TRAINING_SEED_OFFSETS,
    _parser,
    arena_tail_readiness,
    arena_walk_forward_config,
    default_arena_output_dir,
    eligible_arena_dataset,
    recurrent_arena_models,
    run_agent_arena,
)
from trading_bot.training.trainer import TrainingConfig
from trading_bot.training.walk_forward import WalkForwardConfig
from tests.test_walk_forward import walk_forward_dataset


def readiness_dataset(*, last_state: str = "REGULAR") -> SnapshotDataset:
    snapshots = []
    source = walk_forward_dataset()
    for index, snapshot in enumerate(source.snapshots):
        frame = snapshot.frame.copy()
        timestamp = str(frame["collectedAt"].iloc[0])
        frame["marketState"] = (
            last_state if index == len(source.snapshots) - 1 else "REGULAR"
        )
        frame["underlyingQuoteTime"] = timestamp
        snapshots.append(Snapshot(timestamp, frame))
    return SnapshotDataset(tuple(snapshots), "TEST")


class AgentArenaTests(TestCase):
    def test_default_output_directory_is_timestamped_and_timezone_safe(self):
        self.assertEqual(
            default_arena_output_dir(
                datetime(2026, 7, 22, 13, 40, 5, 123456, tzinfo=timezone.utc)
            ),
            Path("data/agent_runs/recurrent-arena/20260722T134005123456Z"),
        )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            default_arena_output_dir(datetime(2026, 7, 22))

    def test_default_arena_uses_three_training_seeds(self):
        self.assertEqual(DEFAULT_ARENA_TRAINING_SEED_OFFSETS, (0, 1, 2))
        self.assertEqual(DEFAULT_ARENA_SELECTION_SCORE_TOLERANCE, 1e-4)
        self.assertEqual(DEFAULT_ARENA_ACTIVATION_MIN_SCORE_ADVANTAGE, 1e-4)
        self.assertTrue(DEFAULT_ARENA_LATEST_FOLD_ONLY)
        self.assertTrue(DEFAULT_ARENA_REQUIRE_READY_TAIL)
        self.assertFalse(_parser().parse_args([]).allow_unready_tail)
        self.assertTrue(
            _parser().parse_args(["--allow-unready-tail"]).allow_unready_tail
        )

    def test_shared_default_arena_config_matches_cli_contract(self):
        config = arena_walk_forward_config()
        args = _parser().parse_args([])

        self.assertEqual(config.min_train_size, args.min_train_size)
        self.assertEqual(config.validation_size, args.validation_size)
        self.assertEqual(config.test_size, args.test_size)
        self.assertEqual(config.embargo, args.embargo)
        self.assertEqual(config.step_size, args.step_size)
        self.assertTrue(config.latest_fold_only)
        self.assertEqual(config.training_seed_offsets, (0, 1, 2))

    def test_tail_readiness_requires_regular_fresh_executable_partitions(self):
        config = WalkForwardConfig(3, 2, 2, latest_fold_only=True)

        ready = arena_tail_readiness(readiness_dataset(), config)
        waiting = arena_tail_readiness(
            readiness_dataset(last_state="PRE"),
            config,
        )
        training_premarket = readiness_dataset()
        snapshots = list(training_premarket.snapshots)
        first = snapshots[0]
        first_frame = first.frame.copy()
        first_frame["marketState"] = "PRE"
        snapshots[0] = Snapshot(first.timestamp, first_frame)
        training_waiting = arena_tail_readiness(
            SnapshotDataset(tuple(snapshots), "TEST"),
            config,
        )

        self.assertTrue(ready["ready"])
        self.assertEqual(ready["training"]["regular_snapshot_count"], 3)
        self.assertEqual(ready["validation"]["regular_snapshot_count"], 2)
        self.assertEqual(ready["test"]["fresh_underlying_quote_count"], 2)
        self.assertEqual(ready["test"]["executable_option_quote_count"], 2)
        self.assertFalse(waiting["ready"])
        self.assertEqual(waiting["test"]["regular_snapshot_count"], 1)
        self.assertFalse(training_waiting["ready"])
        self.assertEqual(training_waiting["training"]["regular_snapshot_count"], 2)

    def test_eligible_arena_dataset_filters_before_all_three_partitions(self):
        source = readiness_dataset()
        snapshots = list(source.snapshots)
        premarket = snapshots[0]
        premarket_frame = premarket.frame.copy()
        premarket_frame["marketState"] = "PRE"
        snapshots.insert(0, Snapshot(premarket.timestamp, premarket_frame))
        material = SnapshotDataset(tuple(snapshots), "TEST")

        eligible, readiness = eligible_arena_dataset(
            material,
            WalkForwardConfig(3, 2, 2, latest_fold_only=True),
        )

        self.assertIsNotNone(eligible)
        self.assertEqual(len(eligible), 7)
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["source_snapshot_count"], 8)
        self.assertEqual(readiness["eligible_snapshot_count"], 7)
        self.assertEqual(readiness["required_eligible_snapshot_count"], 7)
        self.assertEqual(readiness["excluded_non_regular_count"], 1)
        self.assertTrue(
            all(
                snapshot.frame["marketState"].eq("REGULAR").all()
                for snapshot in eligible.snapshots
            )
        )

    def test_unready_tail_skips_training_and_persists_preflight(self):
        with (
            TemporaryDirectory() as directory,
            patch(
                "trading_bot.training.arena.SnapshotDataset.material_from_directory",
                return_value=readiness_dataset(last_state="PRE"),
            ),
            patch(
                "trading_bot.training.arena.run_walk_forward_training",
            ) as runner,
        ):
            output_dir = Path(directory) / "arena"
            result = run_agent_arena(
                data_dir=Path(directory),
                output_dir=output_dir,
                symbols=("TEST",),
                walk_forward_config=WalkForwardConfig(
                    3,
                    2,
                    2,
                    latest_fold_only=True,
                ),
                model_specs=recurrent_arena_models(hidden_size=4),
                training_config=TrainingConfig(episodes=1),
                env_kwargs={"slot_count": 2},
                require_ready_tail=True,
            )

        self.assertEqual(result["completed"], [])
        self.assertEqual(
            result["failures"][0]["error_type"],
            "InsufficientEligibleHistory",
        )
        self.assertFalse(result["preflight"][0]["ready"])
        runner.assert_not_called()

    def test_strict_arena_trains_only_from_eligible_material_states(self):
        source = readiness_dataset()
        premarket = source.snapshots[0]
        premarket_frame = premarket.frame.copy()
        premarket_frame["marketState"] = "PRE"
        material = SnapshotDataset(
            (Snapshot(premarket.timestamp, premarket_frame), *source.snapshots),
            "TEST",
        )
        summary = {
            "folds": [
                {
                    "model_selection": {"selected_model_id": "winner"},
                    "test": [{"total_return": 0.0}],
                }
            ]
        }
        with (
            TemporaryDirectory() as directory,
            patch(
                "trading_bot.training.arena.SnapshotDataset.material_from_directory",
                return_value=material,
            ),
            patch(
                "trading_bot.training.arena.run_walk_forward_training",
                return_value=summary,
            ) as runner,
        ):
            result = run_agent_arena(
                data_dir=Path(directory),
                output_dir=Path(directory) / "arena",
                symbols=("TEST",),
                walk_forward_config=WalkForwardConfig(
                    3,
                    2,
                    2,
                    latest_fold_only=True,
                ),
                model_specs=recurrent_arena_models(hidden_size=4),
                training_config=TrainingConfig(episodes=1),
                env_kwargs={"slot_count": 2},
                require_ready_tail=True,
            )

        trained_dataset = runner.call_args.args[0]
        self.assertEqual(len(trained_dataset), 7)
        self.assertTrue(
            all(
                snapshot.frame["marketState"].eq("REGULAR").all()
                for snapshot in trained_dataset.snapshots
            )
        )
        self.assertEqual(result["preflight"][0]["excluded_non_regular_count"], 1)

    def test_fixed_arena_adds_surface_gnns_and_matched_signal_ablations(self):
        models = recurrent_arena_models(hidden_size=12)

        self.assertEqual(
            [model.kind for model in models[:9]],
            [
                "gru",
                "gru",
                "lstm",
                "lstm",
                "mixture",
                "mixture",
                "gru",
                "lstm",
                "mixture",
            ],
        )
        self.assertEqual(
            [model.action_decoder for model in models[:9]],
            [
                "factorized",
                "single_leg",
                "factorized",
                "single_leg",
                "factorized",
                "single_leg",
                "single_leg",
                "single_leg",
                "single_leg",
            ],
        )
        self.assertEqual(
            [model.encoder for model in models[:9]],
            [
                "flat",
                "flat",
                "flat",
                "flat",
                "flat",
                "flat",
                "surface_graph_set",
                "surface_graph_set",
                "surface_graph_set",
            ],
        )
        self.assertEqual({model.hidden_size for model in models}, {12})
        self.assertEqual(
            {
                (model.graph_hidden_size, model.graph_layers, model.graph_neighbors)
                for model in models[6:9] + models[12:15] + models[18:21]
            },
            {(12, 1, 1)},
        )
        self.assertEqual(
            {model.disabled_feature_groups for model in models[9:15]},
            {("contract_smile_residual",)},
        )
        self.assertEqual(
            {model.disabled_feature_groups for model in models[15:21]},
            {("surface_velocity",)},
        )
        self.assertEqual(
            {
                (model.graph_hidden_size, model.graph_layers, model.graph_neighbors)
                for model in models[9:12] + models[15:18]
            },
            {(32, 2, 3)},
        )
        self.assertEqual(len({model.identifier for model in models}), 21)

    def test_runs_unique_symbols_and_records_per_ticker_failures(self):
        summary = {
            "folds": [
                {
                    "model_selection": {"selected_model_id": "winner"},
                    "test": [{"total_return": 0.01}],
                }
            ],
        }
        with (
            TemporaryDirectory() as directory,
            patch(
                "trading_bot.training.arena.SnapshotDataset.from_directory",
                side_effect=lambda _data, symbol: (
                    (_ for _ in ()).throw(ValueError("too short"))
                    if symbol == "BAD"
                    else object()
                ),
            ) as loader,
            patch(
                "trading_bot.training.arena.run_walk_forward_training",
                return_value=summary,
            ) as runner,
        ):
            output_dir = Path(directory) / "arena"
            result = run_agent_arena(
                data_dir=Path(directory),
                output_dir=output_dir,
                symbols=("aapl", "BAD", "AAPL"),
                walk_forward_config=WalkForwardConfig(3, 2, 2),
                model_specs=recurrent_arena_models(hidden_size=4),
                training_config=TrainingConfig(episodes=1),
                env_kwargs={"slot_count": 2},
            )
            written = json.loads((output_dir / "agent-arena.json").read_text())

        self.assertEqual(result["schema_version"], AGENT_ARENA_SCHEMA_VERSION)
        self.assertEqual(result["symbols"], ["AAPL", "BAD"])
        self.assertEqual(
            result["completed"],
            [
                {
                    "symbol": "AAPL",
                    "summary": str(output_dir / "AAPL-walk-forward.json"),
                    "folds": 1,
                    "selected_model_ids": ["winner"],
                    "heldout_returns": [0.01],
                }
            ],
        )
        self.assertEqual(
            result["failures"],
            [
                {
                    "symbol": "BAD",
                    "error_type": "ValueError",
                    "message": "too short",
                }
            ],
        )
        self.assertEqual(written["completed"], result["completed"])
        self.assertEqual(loader.call_count, 2)
        runner.assert_called_once()

    def test_requires_a_completed_ticker_but_keeps_failure_artifact(self):
        with (
            TemporaryDirectory() as directory,
            patch(
                "trading_bot.training.arena.SnapshotDataset.from_directory",
                side_effect=FileNotFoundError("missing"),
            ),
        ):
            output_dir = Path(directory) / "arena"
            with self.assertRaisesRegex(RuntimeError, "no completed"):
                run_agent_arena(
                    data_dir=Path(directory),
                    output_dir=output_dir,
                    symbols=("MISSING",),
                    walk_forward_config=WalkForwardConfig(3, 2, 2),
                    model_specs=recurrent_arena_models(hidden_size=4),
                    training_config=TrainingConfig(episodes=1),
                    env_kwargs={"slot_count": 2},
                )

            written = json.loads((output_dir / "agent-arena.json").read_text())

        self.assertEqual(written["completed"], [])
        self.assertEqual(written["failures"][0]["symbol"], "MISSING")
