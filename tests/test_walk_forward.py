import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.trainer import TrainingConfig, load_checkpoint
from trading_bot.training.walk_forward import (
    ModelSpec,
    WALK_FORWARD_SCHEMA_VERSION,
    WalkForwardConfig,
    run_walk_forward_training,
)


def walk_forward_dataset() -> SnapshotDataset:
    snapshots = []
    for index in range(7):
        frame = pd.DataFrame([{
            "collectedAt": f"2026-07-21T14:{index:02d}:00Z",
            "contractSymbol": "TEST-C",
            "symbol": "TEST",
            "expiration": "2026-08-21",
            "optionType": "call",
            "strike": 100,
            "bid": 1.0 + index * 0.05,
            "ask": 1.2 + index * 0.05,
            "lastPrice": 1.1 + index * 0.05,
            "impliedVolatility": 0.2,
            "underlyingPrice": 100 + index * 0.1,
            "riskFreeRate": 0.04,
            "delta": 0.5,
            "gamma": 0.01,
            "theta": -0.1,
            "vega": 0.2,
            "volumeLog": 2.0,
            "openInterestLog": 3.0,
        }])
        snapshots.append(Snapshot(str(index), frame))
    return SnapshotDataset(tuple(snapshots), "TEST")


class WalkForwardTrainingTests(TestCase):
    @skipUnless(torch is not None, "install the optional ml extra")
    def test_runner_selects_on_validation_then_reports_held_out_test(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    min_train_size=3,
                    validation_size=2,
                    test_size=2,
                    test_seeds=(31,),
                ),
                ModelSpec(
                    kind="hybrid",
                    encoder="graph",
                    hidden_size=8,
                    graph_hidden_size=4,
                    graph_layers=1,
                    graph_neighbors=1,
                ),
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                    seed=17,
                ),
                output_dir,
                env_kwargs={"slot_count": 1, "starting_cash": 1_000},
            )
            fold = summary["folds"][0]
            checkpoint_path = Path(fold["checkpoint"])
            _, checkpoint_manifest = load_checkpoint(checkpoint_path)
            written_summary = json.loads(
                (output_dir / "TEST-walk-forward.json").read_text()
            )

        self.assertEqual(summary["schema_version"], WALK_FORWARD_SCHEMA_VERSION)
        self.assertEqual(len(summary["folds"]), 1)
        self.assertEqual(fold["selection"]["scope"], "validation_research_demo")
        self.assertEqual(fold["test"][0]["steps"], 1)
        self.assertEqual(
            set(fold["baselines"]),
            {"no_op", "first_feasible", "buy_first_then_delta_hedge"},
        )
        self.assertEqual(set(fold["cost_stress"]), {"base", "double_costs"})
        self.assertEqual(
            set(fold["statistical_comparisons"]),
            {"no_op", "first_feasible", "buy_first_then_delta_hedge"},
        )
        no_op_comparison = fold["statistical_comparisons"]["no_op"][0]
        self.assertEqual(no_op_comparison["status"], "insufficient_history")
        self.assertEqual(no_op_comparison["observations"], 1)
        self.assertIsNone(no_op_comparison["ci_lower"])
        self.assertEqual(no_op_comparison["first_arrival_timestamp"], "6")
        self.assertEqual(no_op_comparison["last_arrival_timestamp"], "6")
        self.assertFalse(no_op_comparison["supports_improvement"])
        self.assertEqual(
            checkpoint_manifest["selection"]["scope"],
            "validation_research_demo",
        )
        provenance = checkpoint_manifest["provenance"]
        fingerprints = provenance["environment_fingerprints"]
        self.assertEqual(len(set(fingerprints.values())), 3)
        self.assertEqual(written_summary["folds"][0]["test"], fold["test"])

    def test_rejects_insufficient_history(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "too short"):
                run_walk_forward_training(
                    walk_forward_dataset(),
                    WalkForwardConfig(5, 2, 2),
                    ModelSpec(hidden_size=8),
                    TrainingConfig(episodes=1),
                    Path(directory),
                )
