import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from trading_bot.training.arena import (
    AGENT_ARENA_SCHEMA_VERSION,
    recurrent_arena_models,
    run_agent_arena,
)
from trading_bot.training.trainer import TrainingConfig
from trading_bot.training.walk_forward import WalkForwardConfig


class AgentArenaTests(TestCase):
    def test_fixed_arena_contains_three_distinct_recurrent_agents(self):
        models = recurrent_arena_models(hidden_size=12)

        self.assertEqual([model.kind for model in models], [
            "gru", "lstm", "mixture",
        ])
        self.assertEqual({model.encoder for model in models}, {"flat"})
        self.assertEqual({model.hidden_size for model in models}, {12})
        self.assertEqual(len({model.identifier for model in models}), 3)

    def test_runs_unique_symbols_and_records_per_ticker_failures(self):
        summary = {
            "folds": [{
                "model_selection": {"selected_model_id": "winner"},
                "test": [{"total_return": 0.01}],
            }],
        }
        with TemporaryDirectory() as directory, patch(
            "trading_bot.training.arena.SnapshotDataset.from_directory",
            side_effect=lambda _data, symbol: (
                (_ for _ in ()).throw(ValueError("too short"))
                if symbol == "BAD"
                else object()
            ),
        ) as loader, patch(
            "trading_bot.training.arena.run_walk_forward_training",
            return_value=summary,
        ) as runner:
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
            written = json.loads(
                (output_dir / "agent-arena.json").read_text()
            )

        self.assertEqual(result["schema_version"], AGENT_ARENA_SCHEMA_VERSION)
        self.assertEqual(result["symbols"], ["AAPL", "BAD"])
        self.assertEqual(result["completed"], [{
            "symbol": "AAPL",
            "summary": str(output_dir / "AAPL-walk-forward.json"),
            "folds": 1,
            "selected_model_ids": ["winner"],
            "heldout_returns": [0.01],
        }])
        self.assertEqual(result["failures"], [{
            "symbol": "BAD",
            "error_type": "ValueError",
            "message": "too short",
        }])
        self.assertEqual(written["completed"], result["completed"])
        self.assertEqual(loader.call_count, 2)
        runner.assert_called_once()

    def test_requires_a_completed_ticker_but_keeps_failure_artifact(self):
        with TemporaryDirectory() as directory, patch(
            "trading_bot.training.arena.SnapshotDataset.from_directory",
            side_effect=FileNotFoundError("missing"),
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

            written = json.loads(
                (output_dir / "agent-arena.json").read_text()
            )

        self.assertEqual(written["completed"], [])
        self.assertEqual(written["failures"][0]["symbol"], "MISSING")
