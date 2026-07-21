"""Executable train/validation/test workflow for recurrent option policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from trading_bot.training.baselines import (
    buy_first_then_delta_hedge,
    first_feasible,
    no_op,
)
from trading_bot.training.dataset import SnapshotDataset
from trading_bot.training.env import CONTRACT_FEATURES, OptionsEnv
from trading_bot.training.evaluation import (
    DEFAULT_COST_SCENARIOS,
    cost_stressed_environment,
    run_episode,
)
from trading_bot.training.recurrent import RecurrentConfig
from trading_bot.training.sequence import observation_vector
from trading_bot.training.splits import walk_forward_splits
from trading_bot.training.trainer import (
    TrainingConfig,
    evaluate_recurrent_policy,
    recurrent_policy,
    save_checkpoint,
    train_actor_critic,
)


WALK_FORWARD_SCHEMA_VERSION = "research-demo.walk-forward.v2"


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_size: int
    validation_size: int
    test_size: int
    embargo: int = 0
    step_size: int | None = None
    max_train_size: int | None = None
    test_seeds: tuple[int, ...] = (20_001,)

    def __post_init__(self) -> None:
        if min(self.min_train_size, self.validation_size, self.test_size) < 2:
            raise ValueError("walk-forward partitions require at least two snapshots")
        if not self.test_seeds:
            raise ValueError("at least one held-out test seed is required")


@dataclass(frozen=True)
class ModelSpec:
    kind: str = "gru"
    encoder: str = "flat"
    hidden_size: int = 128
    layers: int = 1
    dropout: float = 0.0
    graph_hidden_size: int = 32
    graph_layers: int = 2
    graph_neighbors: int = 3

    def build(self, env: OptionsEnv) -> RecurrentConfig:
        observation, _ = env.reset(seed=0)
        return RecurrentConfig(
            input_size=observation_vector(observation).shape[0],
            slot_count=env.slot_count,
            action_count=env.action_shape[1],
            action_slot_count=env.action_shape[0],
            hidden_size=self.hidden_size,
            layers=self.layers,
            kind=self.kind,
            dropout=self.dropout,
            encoder=self.encoder,
            contract_feature_count=observation.contracts.shape[1],
            market_feature_count=observation.market.size,
            portfolio_feature_count=observation.portfolio.size,
            graph_hidden_size=self.graph_hidden_size,
            graph_layers=self.graph_layers,
            graph_neighbors=self.graph_neighbors,
            graph_relation_indices=tuple(
                CONTRACT_FEATURES.index(name)
                for name in (
                    "impliedVolatility",
                    "delta",
                    "logMoneyness",
                    "dteDays",
                )
            ),
        )


def _reports_to_dict(reports) -> list[dict[str, Any]]:
    return [report.to_dict() for report in reports]


def run_walk_forward_training(
    dataset: SnapshotDataset,
    walk_forward_config: WalkForwardConfig,
    model_spec: ModelSpec,
    training_config: TrainingConfig,
    output_dir: Path,
    *,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train on each fold, select on validation, then evaluate untouched test."""
    folds = walk_forward_splits(
        len(dataset),
        min_train_size=walk_forward_config.min_train_size,
        validation_size=walk_forward_config.validation_size,
        test_size=walk_forward_config.test_size,
        embargo=walk_forward_config.embargo,
        step_size=walk_forward_config.step_size,
        max_train_size=walk_forward_config.max_train_size,
    )
    if not folds:
        raise ValueError("dataset is too short for the requested walk-forward split")

    output_dir.mkdir(parents=True, exist_ok=True)
    environment_options = dict(env_kwargs or {})
    fold_results = []
    for fold in folds:
        train_data, validation_data, test_data = fold.apply(dataset)
        train_env = OptionsEnv(train_data, **environment_options)
        validation_env = OptionsEnv(validation_data, **environment_options)
        test_env = OptionsEnv(test_data, **environment_options)
        recurrent_config = model_spec.build(train_env)
        fold_training = replace(
            training_config,
            seed=training_config.seed + fold.fold,
        )
        model, metrics = train_actor_critic(
            train_env,
            recurrent_config,
            fold_training,
            selection_env=validation_env,
        )

        test_reports = evaluate_recurrent_policy(
            test_env,
            model,
            fold_training.sequence_length,
            seeds=walk_forward_config.test_seeds,
        )
        baseline_reports = {
            "no_op": [
                run_episode(test_env, no_op, seed)
                for seed in walk_forward_config.test_seeds
            ],
            "first_feasible": [
                run_episode(test_env, first_feasible, seed)
                for seed in walk_forward_config.test_seeds
            ],
            "buy_first_then_delta_hedge": [
                run_episode(test_env, buy_first_then_delta_hedge(), seed)
                for seed in walk_forward_config.test_seeds
            ],
        }
        cost_stress = {}
        for scenario in DEFAULT_COST_SCENARIOS:
            cost_stress[scenario.name] = [
                run_episode(
                    cost_stressed_environment(test_env, scenario),
                    recurrent_policy(model, fold_training.sequence_length),
                    seed,
                )
                for seed in walk_forward_config.test_seeds
            ]

        selected = next(
            item for item in metrics if item["selected_checkpoint"]
        )
        fold_record = {
            "fold": fold.fold,
            "split": fold.to_dict(),
            "environment_fingerprints": {
                "train": train_env.manifest.fingerprint,
                "validation": validation_env.manifest.fingerprint,
                "test": test_env.manifest.fingerprint,
            },
            "selection": {
                "scope": selected["evaluation_scope"],
                "episode": selected["episode"],
                "validation_total_reward": selected["evaluation_total_reward"],
            },
            "test": _reports_to_dict(test_reports),
            "baselines": {
                name: _reports_to_dict(reports)
                for name, reports in baseline_reports.items()
            },
            "cost_stress": {
                name: _reports_to_dict(reports)
                for name, reports in cost_stress.items()
            },
        }
        checkpoint = output_dir / (
            f"{dataset.symbol}-fold-{fold.fold:03d}-{model_spec.encoder}-"
            f"{model_spec.kind}.pt"
        )
        save_checkpoint(
            checkpoint,
            model,
            train_env,
            recurrent_config,
            fold_training,
            metrics,
            provenance={
                "walk_forward_schema": WALK_FORWARD_SCHEMA_VERSION,
                **fold_record,
            },
        )
        fold_record["checkpoint"] = str(checkpoint)
        fold_results.append(fold_record)

    summary = {
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "mode": "research_demo",
        "symbol": dataset.symbol,
        "walk_forward": asdict(walk_forward_config),
        "model": asdict(model_spec),
        "training": asdict(training_config),
        "folds": fold_results,
    }
    summary_path = output_dir / f"{dataset.symbol}-walk-forward.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/walk-forward"))
    parser.add_argument("--min-train-size", type=int, default=500)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--embargo", type=int, default=8)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--max-train-size", type=int)
    parser.add_argument("--kind", choices=("gru", "lstm", "hybrid"), default="gru")
    parser.add_argument("--encoder", choices=("flat", "graph"), default="flat")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--slot-count", type=int, default=32)
    parser.add_argument("--max-quantity", type=int, default=3)
    parser.add_argument("--underlying-lot-size", type=int, default=25)
    parser.add_argument("--max-abs-underlying-shares", type=int, default=500)
    parser.add_argument("--underlying-commission-per-share", type=float, default=0.005)
    parser.add_argument("--underlying-slippage-bps", type=float, default=1.0)
    parser.add_argument("--max-abs-delta", type=float)
    parser.add_argument("--max-abs-gamma", type=float)
    parser.add_argument("--max-abs-theta", type=float)
    parser.add_argument("--max-abs-vega", type=float)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = _parser().parse_args()
    dataset = SnapshotDataset.from_directory(args.data_dir, args.symbol)
    try:
        summary = run_walk_forward_training(
            dataset,
            WalkForwardConfig(
                min_train_size=args.min_train_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                embargo=args.embargo,
                step_size=args.step_size,
                max_train_size=args.max_train_size,
            ),
            ModelSpec(
                kind=args.kind,
                encoder=args.encoder,
                hidden_size=args.hidden_size,
            ),
            TrainingConfig(
                episodes=args.episodes,
                sequence_length=args.sequence_length,
                seed=args.seed,
            ),
            args.output_dir,
            env_kwargs={
                "slot_count": args.slot_count,
                "max_quantity": args.max_quantity,
                "underlying_lot_size": args.underlying_lot_size,
                "max_abs_underlying_shares": args.max_abs_underlying_shares,
                "underlying_commission_per_share": (
                    args.underlying_commission_per_share
                ),
                "underlying_slippage_bps": args.underlying_slippage_bps,
                "max_abs_delta": args.max_abs_delta,
                "max_abs_gamma": args.max_abs_gamma,
                "max_abs_theta": args.max_abs_theta,
                "max_abs_vega": args.max_abs_vega,
            },
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "summary": str(args.output_dir / f"{dataset.symbol}-walk-forward.json"),
        "folds": len(summary["folds"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
