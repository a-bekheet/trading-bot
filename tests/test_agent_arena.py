import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from trading_bot.training.arena import (
    AGENT_ARENA_SCHEMA_VERSION,
    DEFAULT_ARENA_ACTIVATION_MIN_SCORE_ADVANTAGE,
    DEFAULT_ARENA_LATEST_FOLD_ONLY,
    DEFAULT_ARENA_SELECTION_SCORE_TOLERANCE,
    DEFAULT_ARENA_TRAINING_SEED_OFFSETS,
    default_arena_output_dir,
    recurrent_arena_models,
    run_agent_arena,
)
from trading_bot.training.trainer import TrainingConfig
from trading_bot.training.walk_forward import WalkForwardConfig


class AgentArenaTests(TestCase):
    def test_default_output_directory_is_timestamped_and_timezone_safe(self):
        self.assertEqual(
            default_arena_output_dir(
                datetime(2026, 7, 22, 13, 40, 5, 123456, tzinfo=timezone.utc)
            ),
            Path(
                "data/agent_runs/recurrent-arena/"
                "20260722T134005123456Z"
            ),
        )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            default_arena_output_dir(datetime(2026, 7, 22))

    def test_default_arena_uses_three_training_seeds(self):
        self.assertEqual(DEFAULT_ARENA_TRAINING_SEED_OFFSETS, (0, 1, 2))
        self.assertEqual(DEFAULT_ARENA_SELECTION_SCORE_TOLERANCE, 1e-4)
        self.assertEqual(
            DEFAULT_ARENA_ACTIVATION_MIN_SCORE_ADVANTAGE, 1e-4
        )
        self.assertTrue(DEFAULT_ARENA_LATEST_FOLD_ONLY)

    def test_fixed_arena_adds_surface_gnns_and_matched_signal_ablations(self):
        models = recurrent_arena_models(hidden_size=12)

        self.assertEqual(
            [model.kind for model in models],
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
                "gru",
                "lstm",
                "mixture",
                "gru",
                "lstm",
                "mixture",
            ],
        )
        self.assertEqual(
            [model.action_decoder for model in models],
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
                "single_leg",
                "single_leg",
                "single_leg",
                "single_leg",
                "single_leg",
                "single_leg",
            ],
        )
        self.assertEqual(
            [model.encoder for model in models],
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
                for model in models[6:9] + models[12:]
            },
            {(12, 1, 1)},
        )
        self.assertTrue(
            all(
                model.disabled_feature_groups
                == ("contract_smile_residual",)
                for model in models[9:]
            )
        )
        self.assertEqual(
            {
                (model.graph_hidden_size, model.graph_layers, model.graph_neighbors)
                for model in models[9:12]
            },
            {(32, 2, 3)},
        )
        self.assertEqual(len({model.identifier for model in models}), 15)

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
