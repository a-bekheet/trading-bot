import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

try:
    import torch
except ImportError:
    torch = None

from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.trainer import TrainingConfig, load_checkpoint
from trading_bot.training.universe_walk_forward import (
    UNIVERSE_WALK_FORWARD_SCHEMA_VERSION,
    _global_chronology,
    _parser,
    _symbols_from_args,
    run_universe_walk_forward_training,
)
from trading_bot.training.walk_forward import (
    ModelSpec,
    WalkForwardConfig,
    _walk_forward_config_from_args,
)
from tests.test_walk_forward import walk_forward_dataset


def renamed_dataset(symbol: str, *, day_offset: int = 0) -> SnapshotDataset:
    base = walk_forward_dataset()
    snapshots = tuple(
        Snapshot(
            timestamp=(
                datetime.fromisoformat(
                    str(snapshot.frame["collectedAt"].iloc[0]).replace(
                        "Z", "+00:00"
                    )
                )
                + timedelta(days=day_offset)
            ).isoformat(),
            frame=snapshot.frame,
        )
        for snapshot in base.snapshots
    )
    return SnapshotDataset(snapshots, symbol)


class UniverseWalkForwardTests(TestCase):
    def test_cli_defaults_to_top50_and_accepts_declared_subset(self):
        default = _parser().parse_args([])
        subset = _parser().parse_args([
            "--universe-symbol",
            "aapl",
            "--universe-symbol",
            "msft",
        ])

        self.assertEqual(len(_symbols_from_args(default)), 50)
        self.assertEqual(default.episodes, 100)
        self.assertEqual(_symbols_from_args(subset), ("AAPL", "MSFT"))
        custom = _parser().parse_args([
            "--short-volatility-min-edge", "0.04",
            "--trend-window", "4",
        ])
        config = _walk_forward_config_from_args(custom)
        self.assertEqual(config.short_volatility_min_edge, 0.04)
        self.assertEqual(config.trend_window, 4)
        with self.assertRaisesRegex(ValueError, "at least two"):
            _symbols_from_args(
                _parser().parse_args(["--universe-symbol", "AAPL"])
            )

    def test_rejects_cross_ticker_temporal_overlap(self):
        first = renamed_dataset("AAA")
        late = renamed_dataset("BBB", day_offset=1)
        split = (
            (first.subset(0, 3), first.subset(3, 5), first.subset(5, 7)),
            (late.subset(0, 3), late.subset(3, 5), late.subset(5, 7)),
        )

        with self.assertRaisesRegex(ValueError, "training arrivals overlap"):
            _global_chronology(split)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_shared_universe_selects_before_per_ticker_heldout_evaluation(self):
        datasets = (renamed_dataset("AAA"), renamed_dataset("BBB"))
        candidates = (
            ModelSpec(
                "gru",
                "flat",
                hidden_size=4,
                auxiliary_horizons=(1, 2),
            ),
            ModelSpec("lstm", "flat", hidden_size=4),
            ModelSpec("gru", "flat", hidden_size=4),
            ModelSpec(
                "gru",
                "flat",
                hidden_size=4,
                burn_in_steps=0,
            ),
            ModelSpec(
                "gru",
                "flat",
                hidden_size=4,
                time_aware_discounting=False,
            ),
            ModelSpec(
                "gru",
                "flat",
                hidden_size=4,
                auxiliary_coefficient=0.0,
                auxiliary_horizons=(1, 2),
            ),
        )
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = run_universe_walk_forward_training(
                datasets,
                WalkForwardConfig(
                    min_train_size=3,
                    validation_size=2,
                    test_size=2,
                    test_seeds=(31,),
                    bootstrap_samples=100,
                    bootstrap_min_observations=2,
                    latency_warmup_iterations=1,
                    latency_measured_iterations=3,
                ),
                candidates,
                TrainingConfig(
                    episodes=2,
                    sequence_length=2,
                    ppo_epochs=1,
                    minibatch_size=4,
                    evaluation_interval=2,
                    selection_patience=None,
                    selection_cross_ticker_std_penalty=0.25,
                    selection_worst_ticker_weight=0.25,
                    auxiliary_coefficient=0.05,
                    auxiliary_horizons=(1, 2),
                    seed=17,
                ),
                output_dir,
                env_kwargs={"slot_count": 1, "starting_cash": 1_000},
            )
            fold = summary["folds"][0]
            checkpoint_files = list(output_dir.glob("*.pt"))
            _, manifest = load_checkpoint(checkpoint_files[0])
            written = json.loads(
                (output_dir / "universe-walk-forward.json").read_text()
            )

        self.assertEqual(
            summary["schema_version"],
            UNIVERSE_WALK_FORWARD_SCHEMA_VERSION,
        )
        self.assertEqual(summary["symbols"], ["AAA", "BBB"])
        self.assertEqual(summary["common_length"], 7)
        self.assertEqual(len(checkpoint_files), 1)
        self.assertEqual(
            fold["selection"]["scope"],
            "validation_universe_research_demo",
        )
        self.assertEqual(set(fold["selection"]["per_symbol"]), {"AAA", "BBB"})
        self.assertEqual(
            set(fold["heldout"]["per_symbol"]),
            {"AAA", "BBB"},
        )
        self.assertEqual(fold["heldout"]["aggregate"]["symbol_count"], 2)
        self.assertEqual(fold["heldout"]["aggregate"]["report_count"], 2)
        self.assertTrue(all(
            candidate["slot_churn_rate"] == 0.0
            for candidate in fold["model_selection"]["candidates"]
        ))
        self.assertTrue(
            all(
                evidence["agent"][0]["steps"] == 1
                for evidence in fold["heldout"]["per_symbol"].values()
            )
        )
        self.assertTrue(all(
            "cash_secured_short_put_delta_hedge" in evidence["baselines"]
            and set(evidence["baseline_cost_stress"][
                "cash_secured_short_put_delta_hedge"
            ]) == {"base", "double_costs"}
            for evidence in fold["heldout"]["per_symbol"].values()
        ))
        self.assertEqual(
            fold["baseline_configuration"][
                "cash_secured_short_put_delta_hedge"
            ]["min_volatility_edge"],
            0.02,
        )
        self.assertTrue(
            all(
                len(set(fingerprints.values())) == 3
                for fingerprints in fold[
                    "environment_fingerprints"
                ].values()
            )
        )
        self.assertTrue(
            all(
                "heldout" not in candidate
                for candidate in fold["model_selection"]["candidates"]
            )
        )
        auxiliary_ablation = next(
            candidate
            for candidate in fold["model_selection"]["candidates"]
            if candidate["model"]["auxiliary_coefficient"] == 0.0
        )
        self.assertEqual(
            auxiliary_ablation["effective_auxiliary_coefficient"],
            0.0,
        )
        self.assertIsNotNone(
            auxiliary_ablation[
                "validation_score_lift_vs_auxiliary_enabled"
            ]
        )
        horizon_ablation = next(
            candidate
            for candidate in fold["model_selection"]["candidates"]
            if (
                candidate["model"]["kind"] == "gru"
                and candidate["model"]["auxiliary_coefficient"] is None
                and candidate["effective_auxiliary_horizons"] == [1]
            )
        )
        self.assertIsNotNone(
            horizon_ablation[
                "validation_score_lift_vs_configured_horizons"
            ]
        )
        burn_in_ablation = next(
            candidate
            for candidate in fold["model_selection"]["candidates"]
            if candidate["model"]["burn_in_steps"] == 0
        )
        self.assertEqual(burn_in_ablation["effective_burn_in_steps"], 0)
        self.assertIsNotNone(
            burn_in_ablation["validation_score_lift_vs_burn_in"]
        )
        discount_ablation = next(
            candidate
            for candidate in fold["model_selection"]["candidates"]
            if candidate["model"]["time_aware_discounting"] is False
        )
        self.assertIsNotNone(
            discount_ablation[
                "validation_score_lift_vs_time_aware_discounting"
            ]
        )
        self.assertEqual(len(manifest["training_environments"]), 2)
        self.assertEqual(
            manifest["provenance"]["universe_walk_forward_schema"],
            UNIVERSE_WALK_FORWARD_SCHEMA_VERSION,
        )
        self.assertEqual(written["folds"][0]["heldout"], fold["heldout"])
