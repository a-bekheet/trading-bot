"""Executable train/validation/test workflow for recurrent option policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from trading_bot.training.baselines import (
    LongVolatilityConfig,
    buy_first_then_delta_hedge,
    first_feasible,
    long_volatility_delta_hedge,
    no_op,
)
from trading_bot.training.dataset import SnapshotDataset
from trading_bot.training.env import CONTRACT_FEATURES, OptionsEnv
from trading_bot.training.evaluation import (
    DEFAULT_COST_SCENARIOS,
    cost_stressed_environment,
    paired_moving_block_bootstrap,
    run_episode,
    run_episode_trace,
)
from trading_bot.training.recurrent import RecurrentConfig
from trading_bot.training.sequence import (
    FEATURE_ABLATION_GROUPS,
    feature_ablation_indices,
    observation_vector,
)
from trading_bot.training.splits import walk_forward_splits
from trading_bot.training.trainer import (
    TrainingConfig,
    recurrent_policy,
    save_checkpoint,
    train_actor_critic,
)


WALK_FORWARD_SCHEMA_VERSION = "research-demo.walk-forward.v12"


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_size: int
    validation_size: int
    test_size: int
    embargo: int = 0
    step_size: int | None = None
    max_train_size: int | None = None
    test_seeds: tuple[int, ...] = (20_001,)
    bootstrap_samples: int = 2_000
    bootstrap_block_length: int | None = None
    bootstrap_confidence: float = 0.95
    bootstrap_min_observations: int = 20
    bootstrap_seed: int = 70_001
    long_volatility_window: int = 16
    long_volatility_min_coverage: float = 0.75
    long_volatility_min_edge: float = 0.02
    long_volatility_quantity: int = 1

    def __post_init__(self) -> None:
        if min(self.min_train_size, self.validation_size, self.test_size) < 2:
            raise ValueError("walk-forward partitions require at least two snapshots")
        if not self.test_seeds:
            raise ValueError("at least one held-out test seed is required")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if (
            self.bootstrap_block_length is not None
            and self.bootstrap_block_length < 1
        ):
            raise ValueError("bootstrap_block_length must be positive")
        if not 0 < self.bootstrap_confidence < 1:
            raise ValueError("bootstrap_confidence must be between zero and one")
        if self.bootstrap_min_observations < 2:
            raise ValueError("bootstrap_min_observations must be at least two")
        LongVolatilityConfig(
            realized_window=self.long_volatility_window,
            min_coverage=self.long_volatility_min_coverage,
            min_volatility_edge=self.long_volatility_min_edge,
            quantity=self.long_volatility_quantity,
        )


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
    initial_hold_bias: float = 5.0
    disabled_feature_groups: tuple[str, ...] = ()
    algorithm: str = "ppo"

    def __post_init__(self) -> None:
        if self.kind not in {"gru", "lstm", "hybrid"}:
            raise ValueError("model kind must be gru, lstm, or hybrid")
        if self.encoder not in {"flat", "graph"}:
            raise ValueError("model encoder must be flat or graph")
        if min(
            self.hidden_size,
            self.layers,
            self.graph_hidden_size,
            self.graph_layers,
            self.graph_neighbors,
        ) < 1:
            raise ValueError(
                "model sizes, layers, and graph neighbors must be positive"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError("model dropout must be in [0, 1)")
        if self.algorithm not in {"ppo", "reinforce"}:
            raise ValueError("model algorithm must be ppo or reinforce")
        feature_ablation_indices(self.disabled_feature_groups, 1)

    @property
    def identifier(self) -> str:
        """Return a stable identifier for artifact joins."""
        canonical = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()[:10]
        return f"{self.encoder}-{self.kind}-{digest}"

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
            initial_hold_bias=self.initial_hold_bias,
            masked_input_indices=feature_ablation_indices(
                self.disabled_feature_groups,
                env.slot_count,
            ),
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


def _normalize_model_specs(
    model_spec: ModelSpec | Sequence[ModelSpec],
) -> tuple[ModelSpec, ...]:
    specs = (
        (model_spec,)
        if isinstance(model_spec, ModelSpec)
        else tuple(model_spec)
    )
    if not specs:
        raise ValueError("at least one model candidate is required")
    if not all(isinstance(spec, ModelSpec) for spec in specs):
        raise TypeError("model candidates must be ModelSpec instances")
    identifiers = [spec.identifier for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("model candidates must be unique")
    return specs


def _selected_metric(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    selected = next(item for item in metrics if item["selected_checkpoint"])
    score = selected["evaluation_total_reward"]
    if score is None or not math.isfinite(float(score)):
        raise ValueError("candidate validation reward must be finite")
    return selected


def run_walk_forward_training(
    dataset: SnapshotDataset,
    walk_forward_config: WalkForwardConfig,
    model_spec: ModelSpec | Sequence[ModelSpec],
    training_config: TrainingConfig,
    output_dir: Path,
    *,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select architectures on validation, then evaluate only the winner on test."""
    model_specs = _normalize_model_specs(model_spec)
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
    long_volatility_config = LongVolatilityConfig(
        realized_window=walk_forward_config.long_volatility_window,
        min_coverage=walk_forward_config.long_volatility_min_coverage,
        min_volatility_edge=walk_forward_config.long_volatility_min_edge,
        quantity=walk_forward_config.long_volatility_quantity,
    )
    fold_results = []
    for fold in folds:
        train_data, validation_data, test_data = fold.apply(dataset)
        train_env = OptionsEnv(train_data, **environment_options)
        validation_env = OptionsEnv(validation_data, **environment_options)
        fold_training = replace(
            training_config,
            seed=training_config.seed + fold.fold,
        )
        candidate_runs = []
        for candidate in model_specs:
            recurrent_config = candidate.build(train_env)
            candidate_training = replace(
                fold_training,
                algorithm=candidate.algorithm,
            )
            model, metrics = train_actor_critic(
                train_env,
                recurrent_config,
                candidate_training,
                selection_env=validation_env,
            )
            selected = _selected_metric(metrics)
            candidate_runs.append({
                "model_id": candidate.identifier,
                "model_spec": candidate,
                "recurrent_config": recurrent_config,
                "model": model,
                "metrics": metrics,
                "training_config": candidate_training,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "masked_input_count": len(
                    recurrent_config.masked_input_indices
                ),
                "active_input_count": (
                    recurrent_config.input_size
                    - len(recurrent_config.masked_input_indices)
                ),
                "selected_episode": selected["episode"],
                "validation_total_reward": float(
                    selected["evaluation_total_reward"]
                ),
                "selection_scope": selected["evaluation_scope"],
                "episodes_completed": len(metrics),
                "stopped_early": bool(metrics[-1]["early_stop_selection"]),
                "optimizer_updates": sum(
                    item["optimizer_updates"] for item in metrics
                ),
                "full_model_id": replace(
                    candidate,
                    disabled_feature_groups=(),
                ).identifier,
            })

        winning_run = min(
            candidate_runs,
            key=lambda run: (
                -run["validation_total_reward"],
                run["parameter_count"],
                run["active_input_count"],
                run["optimizer_updates"],
                run["model_id"],
            ),
        )
        candidate_results = [
            {
                "model_id": run["model_id"],
                "model": asdict(run["model_spec"]),
                "parameter_count": run["parameter_count"],
                "masked_input_count": run["masked_input_count"],
                "active_input_count": run["active_input_count"],
                "episodes_completed": run["episodes_completed"],
                "stopped_early": run["stopped_early"],
                "optimizer_updates": run["optimizer_updates"],
                "validation_reward_lift_vs_full": None,
                "selection": {
                    "scope": run["selection_scope"],
                    "episode": run["selected_episode"],
                    "validation_total_reward": run[
                        "validation_total_reward"
                    ],
                },
            }
            for run in candidate_runs
        ]
        validation_scores = {
            run["model_id"]: run["validation_total_reward"]
            for run in candidate_runs
        }
        for result, run in zip(candidate_results, candidate_runs, strict=True):
            full_score = validation_scores.get(run["full_model_id"])
            if run["model_spec"].disabled_feature_groups and full_score is not None:
                result["validation_reward_lift_vs_full"] = (
                    run["validation_total_reward"] - full_score
                )
        model = winning_run["model"]
        metrics = winning_run["metrics"]
        recurrent_config = winning_run["recurrent_config"]
        selected_training = winning_run["training_config"]
        selected = _selected_metric(metrics)
        candidate_runs.clear()

        # Do not instantiate or evaluate the held-out range until validation
        # has fixed both the architecture and its restored checkpoint.
        test_env = OptionsEnv(test_data, **environment_options)

        test_traces = [
            run_episode_trace(
                test_env,
                recurrent_policy(model, selected_training.sequence_length),
                seed,
            )
            for seed in walk_forward_config.test_seeds
        ]
        test_reports = [trace.report for trace in test_traces]
        baseline_traces = {
            "no_op": [
                run_episode_trace(test_env, no_op, seed)
                for seed in walk_forward_config.test_seeds
            ],
            "first_feasible": [
                run_episode_trace(test_env, first_feasible, seed)
                for seed in walk_forward_config.test_seeds
            ],
            "buy_first_then_delta_hedge": [
                run_episode_trace(test_env, buy_first_then_delta_hedge(), seed)
                for seed in walk_forward_config.test_seeds
            ],
            "long_volatility_delta_hedge": [
                run_episode_trace(
                    test_env,
                    long_volatility_delta_hedge(long_volatility_config),
                    seed,
                )
                for seed in walk_forward_config.test_seeds
            ],
        }
        baseline_reports = {
            name: [trace.report for trace in traces]
            for name, traces in baseline_traces.items()
        }
        statistical_comparisons = {}
        for baseline_index, (name, traces) in enumerate(baseline_traces.items()):
            comparisons = []
            for seed_index, (candidate, baseline) in enumerate(
                zip(test_traces, traces, strict=True)
            ):
                if candidate.timestamps != baseline.timestamps:
                    raise ValueError(
                        "candidate and baseline test paths are not aligned"
                    )
                comparison = paired_moving_block_bootstrap(
                    candidate.step_returns,
                    baseline.step_returns,
                    samples=walk_forward_config.bootstrap_samples,
                    block_length=walk_forward_config.bootstrap_block_length,
                    confidence_level=walk_forward_config.bootstrap_confidence,
                    min_observations=(
                        walk_forward_config.bootstrap_min_observations
                    ),
                    seed=(
                        walk_forward_config.bootstrap_seed
                        + fold.fold * 10_000
                        + baseline_index * 1_000
                        + seed_index
                    ),
                )
                comparisons.append({
                    "test_seed": walk_forward_config.test_seeds[seed_index],
                    "first_arrival_timestamp": (
                        candidate.timestamps[0] if candidate.timestamps else None
                    ),
                    "last_arrival_timestamp": (
                        candidate.timestamps[-1] if candidate.timestamps else None
                    ),
                    **comparison.to_dict(),
                })
            statistical_comparisons[name] = comparisons
        cost_stress = {}
        for scenario in DEFAULT_COST_SCENARIOS:
            cost_stress[scenario.name] = [
                run_episode(
                    cost_stressed_environment(test_env, scenario),
                    recurrent_policy(model, selected_training.sequence_length),
                    seed,
                )
                for seed in walk_forward_config.test_seeds
            ]

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
                "model_id": winning_run["model_id"],
            },
            "model_selection": {
                "criterion": "validation_total_reward",
                "direction": "maximize",
                "tie_break": [
                    "parameter_count",
                    "active_input_count",
                    "optimizer_updates",
                    "model_id",
                ],
                "selected_model_id": winning_run["model_id"],
                "candidates": candidate_results,
            },
            "test": _reports_to_dict(test_reports),
            "baselines": {
                name: _reports_to_dict(reports)
                for name, reports in baseline_reports.items()
            },
            "baseline_configuration": {
                "long_volatility_delta_hedge": asdict(long_volatility_config),
            },
            "statistical_comparisons": statistical_comparisons,
            "cost_stress": {
                name: _reports_to_dict(reports)
                for name, reports in cost_stress.items()
            },
        }
        checkpoint = output_dir / (
            f"{dataset.symbol}-fold-{fold.fold:03d}-"
            f"{winning_run['model_id']}.pt"
        )
        save_checkpoint(
            checkpoint,
            model,
            train_env,
            recurrent_config,
            selected_training,
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
        "model": asdict(model_specs[0]) if len(model_specs) == 1 else None,
        "candidate_models": [
            {"model_id": spec.identifier, "model": asdict(spec)}
            for spec in model_specs
        ],
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/walk-forward"),
    )
    parser.add_argument("--min-train-size", type=int, default=500)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--embargo", type=int, default=8)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--max-train-size", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-block-length", type=int)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-min-observations", type=int, default=20)
    parser.add_argument("--bootstrap-seed", type=int, default=70_001)
    parser.add_argument("--long-volatility-window", type=int, default=16)
    parser.add_argument(
        "--long-volatility-min-coverage", type=float, default=0.75
    )
    parser.add_argument("--long-volatility-min-edge", type=float, default=0.02)
    parser.add_argument("--long-volatility-quantity", type=int, default=1)
    parser.add_argument("--kind", choices=("gru", "lstm", "hybrid"), default="gru")
    parser.add_argument("--encoder", choices=("flat", "graph"), default="flat")
    parser.add_argument("--algorithm", choices=("ppo", "reinforce"), default="ppo")
    parser.add_argument(
        "--candidate",
        action="append",
        metavar="ENCODER:KIND[:ALGORITHM]",
        help=(
            "repeat to select architectures and PPO/REINFORCE using "
            "validation only"
        ),
    )
    parser.add_argument(
        "--ablation",
        action="append",
        choices=tuple(FEATURE_ABLATION_GROUPS),
        help=(
            "repeat to add a validation-only candidate with one named "
            "feature group disabled"
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--initial-hold-bias", type=float, default=5.0)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument(
        "--random-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument(
        "--selection-patience",
        type=int,
        default=3,
        help="evaluations without improvement before stopping; 0 disables",
    )
    parser.add_argument("--selection-min-delta", type=float, default=0.0)
    parser.add_argument("--entropy-coefficient", type=float, default=1e-4)
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


def _model_specs_from_args(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    candidates = args.candidate or [f"{args.encoder}:{args.kind}"]
    specs = []
    for candidate in candidates:
        parts = candidate.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(
                "candidate must use ENCODER:KIND or ENCODER:KIND:ALGORITHM"
            )
        encoder, kind = parts[:2]
        algorithm = parts[2] if len(parts) == 3 else args.algorithm
        specs.append(
            ModelSpec(
                kind=kind,
                encoder=encoder,
                hidden_size=args.hidden_size,
                initial_hold_bias=args.initial_hold_bias,
                algorithm=algorithm,
            )
        )
    full_specs = tuple(specs)
    for group in args.ablation or ():
        specs.extend(
            replace(spec, disabled_feature_groups=(group,))
            for spec in full_specs
        )
    return _normalize_model_specs(specs)


def main() -> None:
    args = _parser().parse_args()
    dataset = SnapshotDataset.from_directory(args.data_dir, args.symbol)
    try:
        model_specs = _model_specs_from_args(args)
        summary = run_walk_forward_training(
            dataset,
            WalkForwardConfig(
                min_train_size=args.min_train_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                embargo=args.embargo,
                step_size=args.step_size,
                max_train_size=args.max_train_size,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_block_length=args.bootstrap_block_length,
                bootstrap_confidence=args.bootstrap_confidence,
                bootstrap_min_observations=args.bootstrap_min_observations,
                bootstrap_seed=args.bootstrap_seed,
                long_volatility_window=args.long_volatility_window,
                long_volatility_min_coverage=args.long_volatility_min_coverage,
                long_volatility_min_edge=args.long_volatility_min_edge,
                long_volatility_quantity=args.long_volatility_quantity,
            ),
            model_specs,
            TrainingConfig(
                episodes=args.episodes,
                sequence_length=args.sequence_length,
                max_steps=args.max_steps,
                random_start=args.random_start,
                evaluation_interval=args.evaluation_interval,
                selection_patience=(
                    None
                    if args.selection_patience == 0
                    else args.selection_patience
                ),
                selection_min_delta=args.selection_min_delta,
                entropy_coefficient=args.entropy_coefficient,
                algorithm=args.algorithm,
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
