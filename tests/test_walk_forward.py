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
from trading_bot.training.sequence import feature_ablation_indices
from trading_bot.training.trainer import TrainingConfig, load_checkpoint
from trading_bot.training.walk_forward import (
    ModelSpec,
    WALK_FORWARD_SCHEMA_VERSION,
    WalkForwardConfig,
    _model_specs_from_args,
    _parser,
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
            {
                "no_op",
                "first_feasible",
                "buy_first_then_delta_hedge",
                "long_volatility_delta_hedge",
            },
        )
        self.assertEqual(set(fold["cost_stress"]), {"base", "double_costs"})
        self.assertEqual(
            set(fold["statistical_comparisons"]),
            {
                "no_op",
                "first_feasible",
                "buy_first_then_delta_hedge",
                "long_volatility_delta_hedge",
            },
        )
        self.assertEqual(
            fold["baseline_configuration"]["long_volatility_delta_hedge"],
            {
                "realized_window": 16,
                "min_coverage": 0.75,
                "min_volatility_edge": 0.02,
                "quantity": 1,
            },
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

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_tournament_selects_one_candidate_before_held_out_evaluation(self):
        candidates = (
            ModelSpec(kind="gru", encoder="flat", hidden_size=8),
            ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=8,
                disabled_feature_groups=("surface_wings",),
            ),
            ModelSpec(kind="lstm", encoder="flat", hidden_size=8),
            ModelSpec(
                kind="gru",
                encoder="flat",
                hidden_size=8,
                algorithm="reinforce",
            ),
            ModelSpec(
                kind="hybrid",
                encoder="graph",
                hidden_size=8,
                graph_hidden_size=4,
                graph_layers=1,
                graph_neighbors=1,
            ),
        )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = run_walk_forward_training(
                walk_forward_dataset(),
                WalkForwardConfig(
                    min_train_size=3,
                    validation_size=2,
                    test_size=2,
                    test_seeds=(41,),
                ),
                candidates,
                TrainingConfig(
                    episodes=1,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                    seed=23,
                ),
                output_dir,
                env_kwargs={"slot_count": 1, "starting_cash": 1_000},
            )
            fold = summary["folds"][0]
            selection = fold["model_selection"]
            candidate_results = selection["candidates"]
            expected = min(
                candidate_results,
                key=lambda result: (
                    -result["selection"]["validation_selection_score"],
                    result["parameter_count"],
                    result["active_input_count"],
                    result["optimizer_updates"],
                    result["model_id"],
                ),
            )
            checkpoint_files = list(output_dir.glob("*.pt"))
            _, manifest = load_checkpoint(checkpoint_files[0])

        self.assertIsNone(summary["model"])
        self.assertEqual(len(summary["candidate_models"]), 5)
        self.assertEqual(len(candidate_results), 5)
        self.assertEqual(len(checkpoint_files), 1)
        self.assertTrue(all(result["episodes_completed"] == 1 for result in candidate_results))
        self.assertTrue(all(not result["stopped_early"] for result in candidate_results))
        self.assertEqual(
            selection["tie_break"],
            [
                "parameter_count",
                "active_input_count",
                "optimizer_updates",
                "model_id",
            ],
        )
        self.assertEqual(selection["criterion"], "validation_selection_score")
        self.assertEqual(
            selection["score_definition"],
            {
                "reward": "validation_total_reward",
                "drawdown_penalty": 0.0,
                "downside_penalty": 0.0,
                "turnover_penalty": 0.0,
            },
        )
        self.assertEqual(selection["selected_model_id"], expected["model_id"])
        self.assertEqual(fold["selection"]["model_id"], expected["model_id"])
        self.assertEqual(
            manifest["provenance"]["model_selection"]["selected_model_id"],
            expected["model_id"],
        )
        self.assertEqual(manifest["model"]["kind"], expected["model"]["kind"])
        self.assertEqual(
            manifest["model"]["encoder"],
            expected["model"]["encoder"],
        )
        self.assertEqual(
            manifest["training"]["algorithm"],
            expected["model"]["algorithm"],
        )
        self.assertEqual(
            manifest["algorithm"],
            (
                "stateful_factorized_ppo"
                if expected["model"]["algorithm"] == "ppo"
                else "stateful_factorized_reinforce_baseline"
            ),
        )
        self.assertEqual(
            tuple(manifest["model"]["masked_input_indices"]),
            feature_ablation_indices(
                tuple(expected["model"]["disabled_feature_groups"]),
                1,
            ),
        )
        ablated = next(
            result
            for result in candidate_results
            if result["model"]["disabled_feature_groups"]
        )
        full = next(
            result
            for result in candidate_results
            if result["model"]["kind"] == "gru"
            and result["model"]["encoder"] == "flat"
            and result["model"]["algorithm"] == ablated["model"]["algorithm"]
            and not result["model"]["disabled_feature_groups"]
        )
        self.assertAlmostEqual(
            ablated["validation_reward_lift_vs_full"],
            ablated["selection"]["validation_total_reward"]
            - full["selection"]["validation_total_reward"],
        )
        self.assertAlmostEqual(
            ablated["validation_score_lift_vs_full"],
            ablated["selection"]["validation_selection_score"]
            - full["selection"]["validation_selection_score"],
        )
        self.assertEqual(
            ablated["active_input_count"] + ablated["masked_input_count"],
            full["active_input_count"],
        )
        self.assertTrue(all("test" not in result for result in candidate_results))

    def test_cli_expands_each_architecture_with_requested_ablation(self):
        args = _parser().parse_args([
            "--candidate",
            "flat:gru",
            "--candidate",
            "graph:hybrid:reinforce",
            "--ablation",
            "surface_wings",
        ])

        specs = _model_specs_from_args(args)

        self.assertEqual(len(specs), 4)
        self.assertEqual(
            sum(bool(spec.disabled_feature_groups) for spec in specs),
            2,
        )
        self.assertEqual(
            {spec.algorithm for spec in specs},
            {"ppo", "reinforce"},
        )

    def test_model_candidates_have_stable_unique_identifiers(self):
        first = ModelSpec(kind="gru", encoder="flat", hidden_size=8)
        same = ModelSpec(kind="gru", encoder="flat", hidden_size=8)
        different = ModelSpec(kind="lstm", encoder="flat", hidden_size=8)

        self.assertEqual(first.identifier, same.identifier)
        self.assertNotEqual(first.identifier, different.identifier)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unique"):
                run_walk_forward_training(
                    walk_forward_dataset(),
                    WalkForwardConfig(3, 2, 2),
                    (first, same),
                    TrainingConfig(episodes=1),
                    Path(directory),
                )

    def test_rejects_invalid_model_candidate(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            ModelSpec(kind="transformer")
        with self.assertRaisesRegex(ValueError, "algorithm"):
            ModelSpec(algorithm="q_learning")

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
