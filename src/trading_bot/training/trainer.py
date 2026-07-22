"""Auditable recurrent PPO trainer for the research-demo environment."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from trading_bot.training.env import CONTRACT_FEATURES, OptionsEnv
from trading_bot.training.evaluation import EpisodeReport, run_episode
from trading_bot.training.recurrent import RecurrentConfig, build_recurrent_actor_critic
from trading_bot.training.schemas import FEATURE_VECTOR_SCHEMA_VERSION
from trading_bot.training.sequence import (
    AUXILIARY_TARGET_FEATURES,
    CONTRACT_AUXILIARY_MIN_COVERAGE,
    multi_horizon_auxiliary_targets,
    observation_vector,
)
from trading_bot.market_data.universe import TOP_50_TICKERS


CHECKPOINT_SCHEMA_VERSION = "research-demo.policy.v35"


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 25
    sequence_length: int = 8
    burn_in_steps: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    time_aware_discounting: bool = True
    discount_reference_seconds: float = 900.0
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 64
    target_kl: float = 0.03
    value_coefficient: float = 0.5
    entropy_coefficient: float = 1e-4
    gradient_clip: float = 0.5
    seed: int = 7
    max_steps: int | None = 128
    random_start: bool = True
    evaluation_interval: int = 5
    selection_patience: int | None = 3
    selection_min_delta: float = 0.0
    algorithm: str = "ppo"
    selection_drawdown_penalty: float = 0.0
    selection_downside_penalty: float = 0.0
    selection_turnover_penalty: float = 0.0
    selection_cross_ticker_std_penalty: float = 0.0
    selection_worst_ticker_weight: float = 0.0
    auxiliary_coefficient: float = 0.0
    auxiliary_horizons: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.sequence_length < 1:
            raise ValueError("episodes and sequence_length must be positive")
        if (
            not isinstance(self.burn_in_steps, int)
            or isinstance(self.burn_in_steps, bool)
            or self.burn_in_steps < 0
        ):
            raise ValueError("burn_in_steps must be a non-negative integer")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be between zero and one")
        if not isinstance(self.time_aware_discounting, bool):
            raise ValueError("time_aware_discounting must be a boolean")
        if (
            not math.isfinite(self.discount_reference_seconds)
            or self.discount_reference_seconds <= 0
        ):
            raise ValueError(
                "discount_reference_seconds must be finite and positive"
            )
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.clip_ratio <= 0 or self.value_clip <= 0 or self.target_kl <= 0:
            raise ValueError("PPO clipping and target_kl values must be positive")
        if self.ppo_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("ppo_epochs and minibatch_size must be positive")
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("loss coefficients cannot be negative")
        if (
            not math.isfinite(self.auxiliary_coefficient)
            or self.auxiliary_coefficient < 0
        ):
            raise ValueError("auxiliary_coefficient must be finite and non-negative")
        normalized_horizons = tuple(self.auxiliary_horizons)
        if (
            not normalized_horizons
            or any(
                not isinstance(horizon, int) or isinstance(horizon, bool)
                for horizon in normalized_horizons
            )
            or any(horizon < 1 for horizon in normalized_horizons)
            or tuple(sorted(set(normalized_horizons))) != normalized_horizons
        ):
            raise ValueError(
                "auxiliary_horizons must be unique positive increasing integers"
            )
        object.__setattr__(self, "auxiliary_horizons", normalized_horizons)
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        if (
            self.auxiliary_coefficient > 0
            and self.max_steps is not None
            and self.max_steps < max(self.auxiliary_horizons)
        ):
            raise ValueError(
                "max_steps must reach every enabled auxiliary horizon"
            )
        if not isinstance(self.random_start, bool):
            raise ValueError("random_start must be a boolean")
        if self.evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive")
        if self.selection_patience is not None and self.selection_patience < 1:
            raise ValueError("selection_patience must be positive when provided")
        if (
            not math.isfinite(self.selection_min_delta)
            or self.selection_min_delta < 0
        ):
            raise ValueError("selection_min_delta must be finite and non-negative")
        if self.algorithm not in {"ppo", "reinforce"}:
            raise ValueError("algorithm must be ppo or reinforce")
        selection_penalties = (
            self.selection_drawdown_penalty,
            self.selection_downside_penalty,
            self.selection_turnover_penalty,
            self.selection_cross_ticker_std_penalty,
        )
        if any(
            not math.isfinite(value) or value < 0
            for value in selection_penalties
        ):
            raise ValueError("selection risk penalties must be finite and non-negative")
        if (
            not math.isfinite(self.selection_worst_ticker_weight)
            or not 0 <= self.selection_worst_ticker_weight <= 1
        ):
            raise ValueError("selection_worst_ticker_weight must be in [0, 1]")


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install the ML extra: pip install -e '.[ml]'") from error
    return torch


def _environment_pool(
    env: OptionsEnv | Sequence[OptionsEnv],
    *,
    name: str,
) -> tuple[OptionsEnv, ...]:
    if isinstance(env, OptionsEnv):
        return (env,)
    environments = tuple(env)
    if not environments:
        raise ValueError(f"{name} environment pool cannot be empty")
    if not all(isinstance(item, OptionsEnv) for item in environments):
        raise TypeError(f"{name} environment pool must contain OptionsEnv instances")
    symbols = [item.dataset.symbol for item in environments]
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"{name} environment symbols must be unique")
    return environments


def _detach_hidden(hidden_state):
    """Detach a tensor, LSTM tuple, or dual-recurrent state mapping."""
    if hidden_state is None:
        return None
    if isinstance(hidden_state, dict):
        return {
            name: _detach_hidden(value)
            for name, value in hidden_state.items()
        }
    if isinstance(hidden_state, tuple):
        return tuple(_detach_hidden(value) for value in hidden_state)
    return hidden_state.detach()


def _zero_hidden(config: RecurrentConfig, batch_size: int, torch, device):
    shape = (config.layers, batch_size, config.hidden_size)

    def gru():
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def lstm():
        return gru(), gru()

    if config.kind == "gru":
        return gru()
    if config.kind == "lstm":
        return lstm()
    return {"gru": gru(), "lstm": lstm()}


def _stack_hidden(states, config: RecurrentConfig, torch, device):
    """Stack old-policy chunk states along the recurrent batch dimension."""
    normalized = [
        _zero_hidden(config, 1, torch, device) if state is None else state
        for state in states
    ]

    def stack(items):
        first = items[0]
        if isinstance(first, dict):
            return {name: stack([item[name] for item in items]) for name in first}
        if isinstance(first, tuple):
            return tuple(
                stack([item[index] for item in items])
                for index in range(len(first))
            )
        return torch.cat(items, dim=1).detach().to(device)

    return stack(normalized)


def _one_step_tensor(observation, torch):
    return torch.from_numpy(observation_vector(observation)).view(1, 1, -1)


def recurrent_policy(model, sequence_length: int):
    """Create a deterministic streaming policy for one evaluation episode.

    ``sequence_length`` is retained as the checkpoint/training contract and is
    validated here, while inference carries the recurrent state in constant
    memory rather than rebuilding a padded window on every step.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    torch = _torch()
    device = next(model.parameters()).device
    hidden_state = None

    def policy(observation):
        nonlocal hidden_state
        sequence = _one_step_tensor(observation, torch).to(device)
        action_mask = (
            torch.from_numpy(observation.action_mask).unsqueeze(0).to(device)
        )
        with torch.inference_mode():
            action, _, hidden_state = model.sample_action(
                sequence,
                action_mask,
                deterministic=True,
                hidden_state=hidden_state,
            )
        return action.squeeze(0).cpu().numpy()

    return policy


def benchmark_recurrent_inference(
    model,
    observation,
    sequence_length: int,
    *,
    warmup_iterations: int = 10,
    measured_iterations: int = 100,
) -> dict[str, Any]:
    """Measure streaming batch-one policy latency on one training observation."""
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")
    if measured_iterations < 1:
        raise ValueError("measured_iterations must be positive")
    torch = _torch()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    policy = recurrent_policy(model, sequence_length)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        for _ in range(warmup_iterations):
            policy(observation)
        synchronize()
        durations = []
        for _ in range(measured_iterations):
            started = time.perf_counter_ns()
            policy(observation)
            synchronize()
            durations.append((time.perf_counter_ns() - started) / 1_000.0)
    finally:
        model.train(was_training)

    ordered = sorted(durations)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "schema_version": "research-demo.inference-latency.v1",
        "scope": "streaming_batch_1_training_observation",
        "device": str(device),
        "torch_version": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "median_microseconds": (
            ordered[(len(ordered) - 1) // 2]
            + ordered[len(ordered) // 2]
        ) / 2.0,
        "p95_microseconds": ordered[p95_index],
        "mean_microseconds": sum(ordered) / len(ordered),
    }


def evaluate_recurrent_policy(
    env: OptionsEnv,
    model,
    sequence_length: int,
    seeds: tuple[int, ...] = (101, 102, 103),
) -> list[EpisodeReport]:
    """Evaluate deterministic actions on the explicitly supplied environment."""
    was_training = model.training
    model.eval()
    reports = [
        run_episode(env, recurrent_policy(model, sequence_length), seed)
        for seed in seeds
    ]
    model.train(was_training)
    return reports


def selection_score(
    report: EpisodeReport,
    config: TrainingConfig,
) -> float:
    """Return the declared validation objective from dimensionless metrics."""
    values = (
        float(report.total_reward),
        float(report.max_drawdown),
        float(report.downside_deviation),
        float(report.turnover),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("selection report metrics must be finite")
    reward, drawdown, downside, turnover = values
    if min(drawdown, downside, turnover) < 0:
        raise ValueError("selection risk metrics must be non-negative")
    return float(
        reward
        - config.selection_drawdown_penalty * drawdown
        - config.selection_downside_penalty * downside
        - config.selection_turnover_penalty * turnover
    )


def aggregate_selection_scores(
    scores: Sequence[float],
    config: TrainingConfig,
) -> dict[str, float]:
    """Aggregate ticker scores without allowing mean-only fragility to hide."""
    values = np.asarray(tuple(scores), dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("at least one per-ticker selection score is required")
    if not np.isfinite(values).all():
        raise ValueError("per-ticker selection scores must be finite")
    mean = float(values.mean())
    worst = float(values.min())
    standard_deviation = float(values.std())
    robust_score = (
        (1.0 - config.selection_worst_ticker_weight) * mean
        + config.selection_worst_ticker_weight * worst
        - config.selection_cross_ticker_std_penalty * standard_deviation
    )
    return {
        "score": float(robust_score),
        "mean": mean,
        "worst": worst,
        "standard_deviation": standard_deviation,
    }


def _episode_report_dict(report) -> dict[str, Any]:
    return (
        report.to_dict()
        if hasattr(report, "to_dict")
        else dict(vars(report))
    )


def _generalized_advantages(
    rewards,
    values,
    next_value: float,
    terminal: bool,
    discounts,
    trace_discounts,
    torch,
):
    """GAE with one continuation and trace discount per transition."""
    advantages = torch.zeros_like(values)
    advantage = torch.tensor(0.0)
    following_value = torch.tensor(float(next_value))
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 0.0 if terminal and index == len(rewards) - 1 else 1.0
        delta = (
            rewards[index]
            + discounts[index] * following_value * nonterminal
            - values[index]
        )
        advantage = (
            delta
            + discounts[index]
            * trace_discounts[index]
            * nonterminal
            * advantage
        )
        advantages[index] = advantage
        following_value = values[index]
    return advantages, advantages + values


def _discounted_returns(
    rewards,
    next_value: float,
    terminal: bool,
    discounts,
    torch,
):
    """Causal variable-duration returns with bounded-rollout bootstrap."""
    returns = torch.zeros_like(rewards)
    running = torch.tensor(0.0 if terminal else float(next_value))
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + discounts[index] * running
        returns[index] = running
    return returns


def _elapsed_seconds(start: str, stop: str) -> float:
    """Return a positive duration from ISO-8601 or numeric fixture timestamps."""
    try:
        elapsed = (
            datetime.fromisoformat(stop.replace("Z", "+00:00"))
            - datetime.fromisoformat(start.replace("Z", "+00:00"))
        ).total_seconds()
    except (AttributeError, TypeError, ValueError):
        try:
            elapsed = float(stop) - float(start)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "observation timestamps must be ISO-8601 or numeric"
            ) from error
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("observation timestamps must increase strictly")
    return elapsed


def _duration_adjusted_factors(
    elapsed_seconds: Sequence[float],
    base: float,
    reference_seconds: float,
    time_aware: bool,
) -> np.ndarray:
    """Convert a per-reference-interval factor to each transition duration."""
    elapsed = np.asarray(elapsed_seconds, dtype=np.float64)
    if elapsed.ndim != 1 or not len(elapsed):
        raise ValueError("elapsed_seconds must be a non-empty vector")
    if not np.isfinite(elapsed).all() or np.any(elapsed <= 0):
        raise ValueError("elapsed_seconds must be finite and positive")
    if not math.isfinite(base) or not 0 <= base <= 1:
        raise ValueError("base discount must be finite and in [0, 1]")
    if not math.isfinite(reference_seconds) or reference_seconds <= 0:
        raise ValueError("reference_seconds must be finite and positive")
    exponents = elapsed / reference_seconds if time_aware else np.ones_like(elapsed)
    return np.power(base, exponents)


def _sample_rollout_bounds(
    dataset_length: int,
    max_steps: int | None,
    random_start: bool,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Choose a reproducible causal training segment inside one partition."""
    if dataset_length < 2:
        raise ValueError("recurrent training requires at least two snapshots")
    available_steps = dataset_length - 1
    rollout_steps = (
        available_steps
        if max_steps is None
        else min(max_steps, available_steps)
    )
    latest_start = available_steps - rollout_steps
    start = (
        int(rng.integers(0, latest_start + 1))
        if random_start and latest_start > 0
        else 0
    )
    return start, rollout_steps


def _burn_in_recurrent_context(
    env: OptionsEnv,
    model,
    observation,
    steps: int,
    torch,
):
    """Warm recurrent state on causal no-op context without gradients.

    No-op transitions preserve the random-window zero-position contract while
    advancing market history and stable slot assignment. The complete prefix is
    evaluated in one batched recurrent call to avoid per-step Python overhead.
    """
    if steps < 0:
        raise ValueError("burn-in steps cannot be negative")
    if steps == 0:
        return observation, None

    sequences = []
    hold = np.zeros(env.action_shape[0], dtype=np.int64)
    for _ in range(steps):
        sequences.append(_one_step_tensor(observation, torch))
        observation, _, terminated, truncated, info = env.step(hold)
        if info["invalid_action_count"] or terminated or truncated:
            raise RuntimeError("causal burn-in could not reach the rollout boundary")

    device = next(model.parameters()).device
    sequence = torch.cat(sequences, dim=1).to(device)
    with torch.no_grad():
        _, _, hidden_state = model.forward_sequence(sequence)
    return observation, _detach_hidden(hidden_state)


def train_actor_critic(
    env: OptionsEnv | Sequence[OptionsEnv],
    recurrent_config: RecurrentConfig,
    training_config: TrainingConfig | None = None,
    *,
    selection_env: OptionsEnv | Sequence[OptionsEnv] | None = None,
):
    """Train one recurrent policy over one or more isolated ticker episodes.

    This is an integration trainer for the deterministic research surface. It
    deliberately makes no claim about historical or live trading performance.
    """
    torch = _torch()
    config = training_config or TrainingConfig()
    training_envs = _environment_pool(env, name="training")
    evaluation_envs = (
        training_envs
        if selection_env is None
        else _environment_pool(selection_env, name="selection")
    )
    if len(training_envs) > 1 and config.episodes < len(training_envs):
        raise ValueError(
            "multi-ticker training requires at least one episode per ticker"
        )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rollout_rng = np.random.default_rng(config.seed)
    environment_rng = np.random.default_rng(config.seed + 1_000_003)

    primary_env = training_envs[0]
    observation, _ = primary_env.reset(seed=config.seed)
    actual_input_size = observation_vector(observation).shape[0]
    if recurrent_config.input_size != actual_input_size:
        raise ValueError(
            f"model input_size={recurrent_config.input_size}, environment emits {actual_input_size}"
        )
    model_action_shape = (
        recurrent_config.action_slot_count or recurrent_config.slot_count,
        recurrent_config.action_count,
    )
    if model_action_shape != primary_env.action_shape:
        raise ValueError("model action dimensions do not match the environment")
    if recurrent_config.feature_vector_schema != FEATURE_VECTOR_SCHEMA_VERSION:
        raise ValueError("model feature-vector schema does not match the trainer")
    auxiliary_enabled = config.auxiliary_coefficient > 0
    expected_auxiliary_target_count = (
        len(AUXILIARY_TARGET_FEATURES) * len(config.auxiliary_horizons)
    )
    auxiliary_target_contract = (
        recurrent_config.auxiliary_target_count
        == expected_auxiliary_target_count
    )
    if recurrent_config.auxiliary_target_count not in {
        0,
        expected_auxiliary_target_count,
    }:
        raise ValueError("model auxiliary target layout does not match the trainer")
    if (
        auxiliary_target_contract
        and recurrent_config.auxiliary_horizons != config.auxiliary_horizons
    ):
        raise ValueError("model and trainer auxiliary horizons do not match")
    if auxiliary_enabled and not auxiliary_target_contract:
        raise ValueError(
            "enabled auxiliary loss requires one model output per auxiliary target"
        )
    for pool_name, environments in (
        ("training", training_envs),
        ("selection", evaluation_envs),
    ):
        for environment in environments:
            candidate_observation, _ = environment.reset(seed=config.seed)
            if (
                observation_vector(candidate_observation).shape[0]
                != actual_input_size
            ):
                raise ValueError(
                    f"{pool_name} environment feature layout does not match training"
                )
            if environment.action_shape != primary_env.action_shape:
                raise ValueError(
                    f"{pool_name} environment action dimensions do not match training"
                )
            if len(environment.dataset) < 2:
                raise ValueError(
                    f"{pool_name} environment requires at least two snapshots"
                )
            if (
                pool_name == "training"
                and auxiliary_enabled
                and len(environment.dataset) - 1
                < max(config.auxiliary_horizons)
            ):
                raise ValueError(
                    "training environment is shorter than an enabled "
                    "auxiliary horizon"
                )
    selection_scope = (
        (
            "in_sample_universe_research_demo"
            if len(training_envs) > 1
            else "in_sample_research_demo"
        )
        if selection_env is None
        else (
            "validation_universe_research_demo"
            if len(evaluation_envs) > 1
            else "validation_research_demo"
        )
    )

    model = build_recurrent_actor_critic(recurrent_config)
    # PPO likelihood ratios require the same network behavior during rollout
    # and updates. Eval mode disables optional recurrent dropout but keeps grads.
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metrics: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_episode = 0
    best_state = None
    evaluations_without_improvement = 0

    environment_order: list[int] = []
    for episode in range(config.episodes):
        cycle_position = episode % len(training_envs)
        if cycle_position == 0:
            environment_order = environment_rng.permutation(
                len(training_envs)
            ).tolist()
        environment_index = environment_order[cycle_position]
        episode_env = training_envs[environment_index]
        rollout_start, rollout_step_limit = _sample_rollout_bounds(
            len(episode_env.dataset),
            config.max_steps,
            config.random_start,
            rollout_rng,
        )
        burn_in_steps = min(config.burn_in_steps, rollout_start)
        burn_in_start = rollout_start - burn_in_steps
        observation, _ = episode_env.reset(
            seed=config.seed + episode,
            options={"start_index": burn_in_start},
        )
        observation, hidden_state = _burn_in_recurrent_context(
            episode_env,
            model,
            observation,
            burn_in_steps,
            torch,
        )
        sequences = []
        action_masks = []
        actions = []
        rewards = []
        transition_seconds = []
        old_log_probabilities = []
        old_values = []
        auxiliary_observations = [observation] if auxiliary_target_contract else []
        chunk_hidden_states = []
        invalid_actions = executions = steps = 0
        requested_option_orders = 0
        requested_underlying_orders = 0
        reward_component_totals = {
            "gross_pnl_return": 0.0,
            "fees": 0.0,
            "invalid_action": 0.0,
            "drawdown": 0.0,
            "downside": 0.0,
        }
        slot_changed_count = 0
        slot_comparable_count = 0
        final_terminal = False

        while True:
            if steps % config.sequence_length == 0:
                chunk_hidden_states.append(_detach_hidden(hidden_state))
            sequence = _one_step_tensor(observation, torch)
            action_mask = torch.from_numpy(observation.action_mask).unsqueeze(0)
            with torch.no_grad():
                logits, value, hidden_state = model(
                    sequence,
                    action_mask,
                    hidden_state=hidden_state,
                )
                action = model.actions_from_logits(logits)
            hidden_state = _detach_hidden(hidden_state)
            action_array = action.squeeze(0).detach().cpu().numpy()
            requested_option_orders += int(
                np.count_nonzero(action_array[:episode_env.slot_count])
            )
            requested_underlying_orders += int(action_array[-1] != 0)

            previous_timestamp = observation.timestamp
            observation, reward, terminated, truncated, info = episode_env.step(
                action_array
            )
            transition_seconds.append(
                _elapsed_seconds(previous_timestamp, observation.timestamp)
            )
            if auxiliary_target_contract:
                auxiliary_observations.append(observation)
            sequences.append(sequence.squeeze(0).squeeze(0))
            action_masks.append(action_mask.squeeze(0))
            actions.append(action.squeeze(0))
            rewards.append(float(reward))
            old_log_probabilities.append(
                model.action_log_probabilities(logits, action).squeeze(0)
            )
            old_values.append(value.squeeze(0))
            invalid_actions += int(info["invalid_action_count"])
            executions += len(info["executions"])
            for name, value in info["reward_components"].items():
                reward_component_totals[name] += float(value)
            slot_changed_count += int(info.get("slot_changed_count", 0))
            slot_comparable_count += int(info.get("slot_comparable_count", 0))
            steps += 1
            reached_limit = steps >= rollout_step_limit
            if terminated or truncated or reached_limit:
                final_terminal = terminated or truncated
                break

        next_value = 0.0
        if not final_terminal:
            bootstrap_sequence = _one_step_tensor(observation, torch)
            bootstrap_mask = torch.from_numpy(observation.action_mask).unsqueeze(0)
            with torch.no_grad():
                _, bootstrap_value, _ = model(
                    bootstrap_sequence,
                    bootstrap_mask,
                    hidden_state=hidden_state,
                )
            next_value = float(bootstrap_value.squeeze(0))

        sequences_tensor = torch.stack(sequences)
        masks_tensor = torch.stack(action_masks)
        actions_tensor = torch.stack(actions)
        old_log_probs_tensor = torch.stack(old_log_probabilities)
        old_values_tensor = torch.stack(old_values)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        discounts_tensor = torch.from_numpy(
            _duration_adjusted_factors(
                transition_seconds,
                config.gamma,
                config.discount_reference_seconds,
                config.time_aware_discounting,
            ).astype(np.float32)
        )
        trace_discounts_tensor = torch.from_numpy(
            _duration_adjusted_factors(
                transition_seconds,
                config.gae_lambda,
                config.discount_reference_seconds,
                config.time_aware_discounting,
            ).astype(np.float32)
        )
        if auxiliary_target_contract:
            auxiliary_values, auxiliary_available = (
                multi_horizon_auxiliary_targets(
                    auxiliary_observations,
                    config.auxiliary_horizons,
                )
            )
            auxiliary_targets_tensor = torch.from_numpy(auxiliary_values)
            auxiliary_masks_tensor = torch.from_numpy(auxiliary_available)
        else:
            auxiliary_targets_tensor = None
            auxiliary_masks_tensor = None
        if config.algorithm == "ppo":
            advantages, returns_tensor = _generalized_advantages(
                rewards_tensor,
                old_values_tensor,
                next_value,
                final_terminal,
                discounts_tensor,
                trace_discounts_tensor,
                torch,
            )
        else:
            returns_tensor = _discounted_returns(
                rewards_tensor,
                next_value,
                final_terminal,
                discounts_tensor,
                torch,
            )
            advantages = returns_tensor - old_values_tensor
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        update_metrics = []
        stop_early = False
        chunks = [
            (
                start,
                min(start + config.sequence_length, steps),
                chunk_hidden_states[start // config.sequence_length],
            )
            for start in range(0, steps, config.sequence_length)
        ]
        device = next(model.parameters()).device
        update_epochs = config.ppo_epochs if config.algorithm == "ppo" else 1
        for _ in range(update_epochs):
            order = torch.randperm(len(chunks)).tolist()
            cursor = 0
            while cursor < len(order):
                selected_chunks = []
                selected_steps = 0
                while cursor < len(order):
                    candidate = chunks[order[cursor]]
                    candidate_steps = candidate[1] - candidate[0]
                    if (
                        selected_chunks
                        and selected_steps + candidate_steps > config.minibatch_size
                    ):
                        break
                    selected_chunks.append(candidate)
                    selected_steps += candidate_steps
                    cursor += 1

                batch_size = len(selected_chunks)
                max_length = max(end - start for start, end, _ in selected_chunks)
                batch_sequences = torch.zeros(
                    batch_size,
                    max_length,
                    sequences_tensor.shape[-1],
                    dtype=sequences_tensor.dtype,
                    device=device,
                )
                batch_masks = torch.zeros(
                    batch_size,
                    max_length,
                    *masks_tensor.shape[1:],
                    dtype=torch.bool,
                    device=device,
                )
                batch_masks[..., 0] = True
                valid_steps = torch.zeros(
                    batch_size,
                    max_length,
                    dtype=torch.bool,
                    device=device,
                )
                source_indices = []
                for row, (start, end, _) in enumerate(selected_chunks):
                    length = end - start
                    batch_sequences[row, :length] = sequences_tensor[start:end]
                    batch_masks[row, :length] = masks_tensor[start:end]
                    valid_steps[row, :length] = True
                    source_indices.append(torch.arange(start, end))
                batch = torch.cat(source_indices)
                initial_hidden = _stack_hidden(
                    [state for _, _, state in selected_chunks],
                    recurrent_config,
                    torch,
                    device,
                )
                if auxiliary_enabled:
                    (
                        sequence_logits,
                        sequence_values,
                        sequence_auxiliary,
                        _,
                    ) = model.forward_sequence_with_auxiliary(
                        batch_sequences,
                        batch_masks,
                        hidden_state=initial_hidden,
                    )
                    predicted_auxiliary = sequence_auxiliary[valid_steps]
                else:
                    sequence_logits, sequence_values, _ = model.forward_sequence(
                        batch_sequences,
                        batch_masks,
                        hidden_state=initial_hidden,
                    )
                    predicted_auxiliary = None
                logits = sequence_logits[valid_steps]
                predicted_values = sequence_values[valid_steps]
                new_log_probs = model.action_log_probabilities(
                    logits,
                    actions_tensor[batch],
                )
                log_ratio = new_log_probs - old_log_probs_tensor[batch]
                ratio = log_ratio.exp()
                batch_advantages = advantages[batch].unsqueeze(-1)
                if config.algorithm == "ppo":
                    unclipped = ratio * batch_advantages
                    clipped = ratio.clamp(
                        1 - config.clip_ratio,
                        1 + config.clip_ratio,
                    ) * batch_advantages
                    policy_loss = -torch.minimum(unclipped, clipped).mean()

                    value_delta = predicted_values - old_values_tensor[batch]
                    clipped_values = (
                        old_values_tensor[batch]
                        + value_delta.clamp(
                            -config.value_clip,
                            config.value_clip,
                        )
                    )
                    value_loss = 0.5 * torch.maximum(
                        (predicted_values - returns_tensor[batch]).square(),
                        (clipped_values - returns_tensor[batch]).square(),
                    ).mean()
                else:
                    policy_loss = -(
                        new_log_probs * batch_advantages
                    ).mean()
                    value_loss = 0.5 * (
                        predicted_values - returns_tensor[batch]
                    ).square().mean()
                entropy = model.action_entropies(logits).mean()
                auxiliary_loss = predicted_values.sum() * 0.0
                auxiliary_mae = predicted_values.sum() * 0.0
                if auxiliary_enabled:
                    target_values = auxiliary_targets_tensor[batch].to(device)
                    target_mask = auxiliary_masks_tensor[batch].to(device)
                    available_count = target_mask.sum()
                    if float(available_count) > 0:
                        absolute_error = (
                            predicted_auxiliary - target_values
                        ).abs()
                        element_loss = torch.nn.functional.smooth_l1_loss(
                            predicted_auxiliary,
                            target_values,
                            reduction="none",
                        )
                        auxiliary_loss = (
                            element_loss * target_mask
                        ).sum() / available_count
                        auxiliary_mae = (
                            absolute_error * target_mask
                        ).sum() / available_count
                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
                    + config.auxiliary_coefficient * auxiliary_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                optimizer.step()
                approx_kl = float(((ratio - 1) - log_ratio).mean().detach())
                update_metrics.append(
                    (
                        float(loss.detach()),
                        float(policy_loss.detach()),
                        float(value_loss.detach()),
                        float(entropy.detach()),
                        float(gradient_norm),
                        approx_kl,
                        (
                            float(
                                (ratio.sub(1).abs() > config.clip_ratio)
                                .float()
                                .mean()
                            )
                            if config.algorithm == "ppo"
                            else 0.0
                        ),
                        float(auxiliary_loss.detach()),
                        float(auxiliary_mae.detach()),
                    )
                )
                if config.algorithm == "ppo" and approx_kl > config.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        values_variance = float(returns_tensor.var(unbiased=False))
        explained_variance = (
            1 - float((returns_tensor - old_values_tensor).var(unbiased=False))
            / values_variance
            if values_variance > 1e-12
            else 0.0
        )
        means = np.asarray(update_metrics, dtype=float).mean(axis=0)
        evaluation_reward = None
        evaluation_selection_score = None
        evaluation_max_drawdown = None
        evaluation_downside_deviation = None
        evaluation_turnover = None
        selection_improved = None
        stop_for_selection_patience = False
        should_evaluate = (
            (episode + 1) % config.evaluation_interval == 0
            or episode == config.episodes - 1
        )
        if should_evaluate:
            reports = {
                environment.dataset.symbol: evaluate_recurrent_policy(
                    environment,
                    model,
                    config.sequence_length,
                    seeds=(config.seed + 10_000 + episode + index,),
                )[0]
                for index, environment in enumerate(evaluation_envs)
            }
            ticker_scores = {
                symbol: selection_score(report, config)
                for symbol, report in reports.items()
            }
            aggregate = aggregate_selection_scores(
                ticker_scores.values(),
                config,
            )
            evaluation_reward = float(
                np.mean([report.total_reward for report in reports.values()])
            )
            evaluation_max_drawdown = float(
                np.mean([report.max_drawdown for report in reports.values()])
            )
            evaluation_downside_deviation = float(np.mean([
                report.downside_deviation for report in reports.values()
            ]))
            evaluation_turnover = float(
                np.mean([report.turnover for report in reports.values()])
            )
            evaluation_selection_score = aggregate["score"]
            selection_improved = (
                evaluation_selection_score
                > best_score + config.selection_min_delta
            )
            if selection_improved:
                best_score = evaluation_selection_score
                best_episode = episode
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            stop_for_selection_patience = (
                config.selection_patience is not None
                and evaluations_without_improvement
                >= config.selection_patience
            )
        metrics.append(
            {
                "episode": episode,
                "training_symbol": episode_env.dataset.symbol,
                "training_environment_index": environment_index,
                "steps": steps,
                "rollout_start_index": rollout_start,
                "rollout_end_index": rollout_start + steps,
                "burn_in_start_index": burn_in_start,
                "burn_in_steps": burn_in_steps,
                "total_reward": float(sum(rewards)),
                "reward_components": reward_component_totals.copy(),
                "transition_seconds_mean": float(np.mean(transition_seconds)),
                "transition_seconds_min": float(np.min(transition_seconds)),
                "transition_seconds_max": float(np.max(transition_seconds)),
                "effective_gamma_mean": float(discounts_tensor.mean()),
                "effective_gamma_min": float(discounts_tensor.min()),
                "effective_gamma_max": float(discounts_tensor.max()),
                "effective_lambda_mean": float(trace_discounts_tensor.mean()),
                "effective_lambda_min": float(trace_discounts_tensor.min()),
                "effective_lambda_max": float(trace_discounts_tensor.max()),
                "evaluation_total_reward": evaluation_reward,
                "evaluation_selection_score": evaluation_selection_score,
                "evaluation_selection_score_mean": (
                    aggregate["mean"] if should_evaluate else None
                ),
                "evaluation_worst_ticker_selection_score": (
                    aggregate["worst"] if should_evaluate else None
                ),
                "evaluation_selection_score_std": (
                    aggregate["standard_deviation"]
                    if should_evaluate
                    else None
                ),
                "evaluation_by_symbol": (
                    {
                        symbol: {
                            **_episode_report_dict(report),
                            "selection_score": ticker_scores[symbol],
                        }
                        for symbol, report in reports.items()
                    }
                    if should_evaluate
                    else None
                ),
                "evaluation_max_drawdown": evaluation_max_drawdown,
                "evaluation_downside_deviation": (
                    evaluation_downside_deviation
                ),
                "evaluation_turnover": evaluation_turnover,
                "evaluation_scope": selection_scope,
                "selection_improved": (
                    int(selection_improved)
                    if selection_improved is not None
                    else None
                ),
                "selection_evaluations_without_improvement": (
                    evaluations_without_improvement
                ),
                "early_stop_selection": int(stop_for_selection_patience),
                "loss": float(means[0]),
                "policy_loss": float(means[1]),
                "value_loss": float(means[2]),
                "entropy": float(means[3]),
                "entropy_bonus": float(config.entropy_coefficient * means[3]),
                "gradient_norm": float(means[4]),
                "approx_kl": float(means[5]),
                "clip_fraction": float(means[6]),
                "auxiliary_loss": float(means[7]),
                "auxiliary_weighted_loss": float(
                    config.auxiliary_coefficient * means[7]
                ),
                "auxiliary_mae": float(means[8]),
                "auxiliary_target_coverage": (
                    {
                        f"t+{horizon}": {
                            name: float(coverage)
                            for name, coverage in zip(
                                AUXILIARY_TARGET_FEATURES,
                                auxiliary_masks_tensor.view(
                                    steps,
                                    len(config.auxiliary_horizons),
                                    len(AUXILIARY_TARGET_FEATURES),
                                ).mean(dim=0)[horizon_index].tolist(),
                                strict=True,
                            )
                        }
                        for horizon_index, horizon in enumerate(
                            config.auxiliary_horizons
                        )
                    }
                    if auxiliary_target_contract
                    else {
                        f"t+{horizon}": {
                            name: 0.0 for name in AUXILIARY_TARGET_FEATURES
                        }
                        for horizon in config.auxiliary_horizons
                    }
                ),
                "explained_variance": explained_variance,
                "algorithm": config.algorithm,
                "action_decoder": recurrent_config.action_decoder,
                "action_likelihood_factors": (
                    1
                    if recurrent_config.action_decoder == "single_leg"
                    else episode_env.action_shape[0]
                ),
                "optimizer_updates": len(update_metrics),
                "ppo_updates": (
                    len(update_metrics) if config.algorithm == "ppo" else 0
                ),
                "reinforce_updates": (
                    len(update_metrics)
                    if config.algorithm == "reinforce"
                    else 0
                ),
                "recurrent_chunks": len(chunks),
                "early_stop_kl": int(stop_early),
                "invalid_actions": invalid_actions,
                "executions": executions,
                "requested_option_orders": requested_option_orders,
                "requested_underlying_orders": requested_underlying_orders,
                "mean_requested_orders_per_step": (
                    (requested_option_orders + requested_underlying_orders) / steps
                ),
                "requested_action_rate": (
                    (requested_option_orders + requested_underlying_orders)
                    / (steps * episode_env.action_shape[0])
                ),
                "slot_changed_count": slot_changed_count,
                "slot_comparable_count": slot_comparable_count,
                "slot_churn_rate": (
                    slot_changed_count / slot_comparable_count
                    if slot_comparable_count
                    else 0.0
                ),
            }
        )
        if stop_for_selection_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_episode = len(metrics) - 1
    for index, item in enumerate(metrics):
        item["selected_checkpoint"] = int(index == best_episode)
    return model, metrics


def checkpoint_manifest(
    env: OptionsEnv | Sequence[OptionsEnv],
    recurrent_config: RecurrentConfig,
    training_config: TrainingConfig,
    metrics: list[dict[str, Any]],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    environments = _environment_pool(env, name="checkpoint")
    primary_env = environments[0]
    selected_metric = next(
        (item for item in metrics if item.get("selected_checkpoint")),
        metrics[-1],
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "mode": "research_demo",
        "algorithm": (
            f"stateful_{'single_leg_joint' if recurrent_config.action_decoder == 'single_leg' else 'factorized'}_"
            f"{'ppo' if training_config.algorithm == 'ppo' else 'reinforce_baseline'}"
        ),
        "action_policy": (
            {
                "factorization": "single_leg_joint_categorical",
                "initial_hold_bias": recurrent_config.initial_hold_bias,
                "hard_order_cap": None,
                "maximum_orders_per_step": 1,
                "joint_action_count": (
                    1
                    + primary_env.action_shape[0]
                    * (primary_env.action_shape[1] - 1)
                ),
                "likelihood": "exact_joint_categorical",
            }
            if recurrent_config.action_decoder == "single_leg"
            else {
                "factorization": "independent_masked_rows",
                "initial_hold_bias": recurrent_config.initial_hold_bias,
                "hard_order_cap": None,
            }
        ),
        "temporal_training": {
            "mode": "stateful_tbptt",
            "chunk_length": training_config.sequence_length,
            "padding": "right_only_ignored",
            "burn_in": {
                "maximum_steps": training_config.burn_in_steps,
                "mode": "causal_no_op_context",
                "gradient": "disabled",
                "reward": "excluded",
                "execution": "single_batched_recurrent_call",
            },
            "discounting": {
                "mode": (
                    "elapsed_wall_clock"
                    if training_config.time_aware_discounting
                    else "fixed_transition"
                ),
                "gamma_per_reference_interval": training_config.gamma,
                "gae_lambda_per_reference_interval": (
                    training_config.gae_lambda
                ),
                "reference_seconds": (
                    training_config.discount_reference_seconds
                ),
            },
        },
        "auxiliary_prediction": {
            "enabled": training_config.auxiliary_coefficient > 0,
            "coefficient": training_config.auxiliary_coefficient,
            "targets": list(AUXILIARY_TARGET_FEATURES),
            "horizons": list(training_config.auxiliary_horizons),
            "target_semantics": "cumulative_change_from_policy_state",
            "availability": "endpoint_point_in_time_coverage_mask",
            "contract_matching": "contract_id_at_both_endpoints",
            "contract_aggregation": "cross_sectional_median",
            "contract_minimum_coverage": CONTRACT_AUXILIARY_MIN_COVERAGE,
            "inference_path": "excluded_from_policy_inference",
        },
        "feature_vector_schema": FEATURE_VECTOR_SCHEMA_VERSION,
        "selection": {
            "scope": selected_metric.get(
                "evaluation_scope",
                "in_sample_research_demo",
            ),
            "metric": "evaluation_selection_score",
            "episode": selected_metric["episode"],
            "score": selected_metric["evaluation_selection_score"],
            "score_definition": {
                "reward": "evaluation_total_reward",
                "drawdown_penalty": (
                    training_config.selection_drawdown_penalty
                ),
                "downside_penalty": (
                    training_config.selection_downside_penalty
                ),
                "turnover_penalty": (
                    training_config.selection_turnover_penalty
                ),
                "cross_ticker_std_penalty": (
                    training_config.selection_cross_ticker_std_penalty
                ),
                "worst_ticker_weight": (
                    training_config.selection_worst_ticker_weight
                ),
            },
            "early_stopping": {
                "enabled": training_config.selection_patience is not None,
                "patience": training_config.selection_patience,
                "min_delta": training_config.selection_min_delta,
                "completed_episodes": len(metrics),
                "stopped_early": bool(
                    metrics[-1].get("early_stop_selection", 0)
                ),
            },
        },
        "environment": primary_env.manifest.to_dict(),
        "environment_fingerprint": primary_env.manifest.fingerprint,
        "training_environments": [
            environment.manifest.to_dict() for environment in environments
        ],
        "training_environment_fingerprints": {
            environment.dataset.symbol: environment.manifest.fingerprint
            for environment in environments
        },
        "model": asdict(recurrent_config),
        "training": asdict(training_config),
        "metrics": metrics,
        "provenance": provenance or {},
    }


def save_checkpoint(
    path: Path,
    model,
    env: OptionsEnv | Sequence[OptionsEnv],
    recurrent_config: RecurrentConfig,
    training_config: TrainingConfig,
    metrics: list[dict[str, Any]],
    *,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Save model weights plus a human-readable provenance sidecar."""
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = checkpoint_manifest(
        env,
        recurrent_config,
        training_config,
        metrics,
        provenance,
    )
    torch.save({"state_dict": model.state_dict(), "manifest": manifest}, path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_checkpoint(path: Path):
    """Safely restore a model and its provenance manifest."""
    torch = _torch()
    checkpoint = torch.load(path, weights_only=True)
    manifest = checkpoint.get("manifest", {})
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported or missing checkpoint schema")
    if manifest.get("feature_vector_schema") != FEATURE_VECTOR_SCHEMA_VERSION:
        raise ValueError("checkpoint feature-vector schema is incompatible")
    recurrent_config = RecurrentConfig(**manifest["model"])
    if recurrent_config.feature_vector_schema != FEATURE_VECTOR_SCHEMA_VERSION:
        raise ValueError("model feature-vector schema is incompatible")
    model = build_recurrent_actor_critic(recurrent_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument(
        "--universe",
        choices=("single", "top50"),
        default="single",
        help="train one ticker or one shared policy across the top-50 universe",
    )
    parser.add_argument(
        "--kind",
        choices=("gru", "lstm", "hybrid", "mixture"),
        default="gru",
    )
    parser.add_argument(
        "--encoder",
        choices=("flat", "graph", "graph_set", "attention_set"),
        default="flat",
    )
    parser.add_argument("--algorithm", choices=("ppo", "reinforce"), default="ppo")
    parser.add_argument(
        "--action-decoder",
        choices=("factorized", "single_leg"),
        default="factorized",
    )
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--burn-in-steps", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--graph-hidden-size", type=int, default=32)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-neighbors", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--slot-count", type=int, default=32)
    parser.add_argument(
        "--slot-assignment",
        choices=("stable", "ranked"),
        default="stable",
    )
    parser.add_argument("--max-quantity", type=int, default=3)
    parser.add_argument(
        "--allow-collateralized-option-shorts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "allow covered calls and cash-secured puts; naked shorts stay "
            "forbidden"
        ),
    )
    parser.add_argument("--reward-drawdown-penalty", type=float, default=0.0)
    parser.add_argument("--reward-downside-penalty", type=float, default=0.0)
    parser.add_argument("--underlying-lot-size", type=int, default=25)
    parser.add_argument("--max-abs-underlying-shares", type=int, default=500)
    parser.add_argument("--underlying-commission-per-share", type=float, default=0.005)
    parser.add_argument("--underlying-slippage-bps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument(
        "--random-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--time-aware-discounting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "scale gamma and GAE lambda by wall-clock transition duration"
        ),
    )
    parser.add_argument(
        "--discount-reference-seconds",
        type=float,
        default=900.0,
        help="interval at which configured gamma and GAE lambda apply",
    )
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--entropy-coefficient", type=float, default=1e-4)
    parser.add_argument(
        "--auxiliary-coefficient",
        type=float,
        default=0.0,
        help=(
            "weight for train-only future-market prediction loss; zero disables"
        ),
    )
    parser.add_argument(
        "--auxiliary-horizon",
        action="append",
        type=int,
        help=(
            "repeat for cumulative train-only prediction horizons; defaults "
            "to one step"
        ),
    )
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument(
        "--selection-patience",
        type=int,
        default=3,
        help="evaluations without improvement before stopping; 0 disables",
    )
    parser.add_argument("--selection-min-delta", type=float, default=0.0)
    parser.add_argument("--selection-drawdown-penalty", type=float, default=0.0)
    parser.add_argument("--selection-downside-penalty", type=float, default=0.0)
    parser.add_argument("--selection-turnover-penalty", type=float, default=0.0)
    parser.add_argument(
        "--selection-cross-ticker-std-penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--selection-worst-ticker-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--initial-hold-bias", type=float, default=5.0)
    parser.add_argument("--max-abs-delta", type=float)
    parser.add_argument("--max-abs-gamma", type=float)
    parser.add_argument("--max-abs-theta", type=float)
    parser.add_argument("--max-abs-vega", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def _symbols_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        TOP_50_TICKERS
        if args.universe == "top50"
        else (args.symbol.upper(),)
    )


def _environment_kwargs_from_args(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return the shared environment contract for every training CLI."""
    return {
        "slot_count": args.slot_count,
        "slot_assignment": args.slot_assignment,
        "max_quantity": args.max_quantity,
        "allow_collateralized_option_shorts": (
            args.allow_collateralized_option_shorts
        ),
        "reward_drawdown_penalty": args.reward_drawdown_penalty,
        "reward_downside_penalty": args.reward_downside_penalty,
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
    }


def main() -> None:
    args = _parser().parse_args()
    auxiliary_horizons = tuple(args.auxiliary_horizon or (1,))
    symbols = _symbols_from_args(args)
    environment_kwargs = _environment_kwargs_from_args(args)
    environments = tuple(
        OptionsEnv.from_directory(
            args.data_dir,
            symbol,
            **environment_kwargs,
        )
        for symbol in symbols
    )
    primary_env = environments[0]
    observation, _ = primary_env.reset(seed=args.seed)
    recurrent_config = RecurrentConfig(
        input_size=observation_vector(observation).shape[0],
        slot_count=primary_env.slot_count,
        action_slot_count=primary_env.action_shape[0],
        action_count=primary_env.action_shape[1],
        hidden_size=args.hidden_size,
        kind=args.kind,
        encoder=args.encoder,
        contract_feature_count=observation.contracts.shape[1],
        market_feature_count=observation.market.size,
        portfolio_feature_count=observation.portfolio.size,
        initial_hold_bias=args.initial_hold_bias,
        action_decoder=args.action_decoder,
        graph_relation_indices=tuple(
            CONTRACT_FEATURES.index(name)
            for name in ("impliedVolatility", "delta", "logMoneyness", "dteDays")
        ),
        graph_hidden_size=args.graph_hidden_size,
        graph_layers=args.graph_layers,
        graph_neighbors=args.graph_neighbors,
        attention_heads=args.attention_heads,
        auxiliary_target_count=(
            len(AUXILIARY_TARGET_FEATURES) * len(auxiliary_horizons)
        ),
        auxiliary_horizons=auxiliary_horizons,
    )
    training_config = TrainingConfig(
        episodes=args.episodes,
        sequence_length=args.sequence_length,
        burn_in_steps=args.burn_in_steps,
        seed=args.seed,
        max_steps=args.max_steps,
        random_start=args.random_start,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        time_aware_discounting=args.time_aware_discounting,
        discount_reference_seconds=args.discount_reference_seconds,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        entropy_coefficient=args.entropy_coefficient,
        auxiliary_coefficient=args.auxiliary_coefficient,
        auxiliary_horizons=auxiliary_horizons,
        evaluation_interval=args.evaluation_interval,
        selection_patience=(
            None if args.selection_patience == 0 else args.selection_patience
        ),
        selection_min_delta=args.selection_min_delta,
        selection_drawdown_penalty=args.selection_drawdown_penalty,
        selection_downside_penalty=args.selection_downside_penalty,
        selection_turnover_penalty=args.selection_turnover_penalty,
        selection_cross_ticker_std_penalty=(
            args.selection_cross_ticker_std_penalty
        ),
        selection_worst_ticker_weight=args.selection_worst_ticker_weight,
        algorithm=args.algorithm,
    )
    training_env = environments[0] if len(environments) == 1 else environments
    model, metrics = train_actor_critic(
        training_env,
        recurrent_config,
        training_config,
    )
    output_label = primary_env.dataset.symbol if len(environments) == 1 else "top50"
    output = args.output or Path("data/models") / (
        f"{output_label}-{args.encoder}-{args.kind}-{args.action_decoder}.pt"
    )
    save_checkpoint(
        output,
        model,
        training_env,
        recurrent_config,
        training_config,
        metrics,
    )
    selected = next(item for item in metrics if item["selected_checkpoint"])
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "selected_episode": selected["episode"],
                "selection_scope": selected["evaluation_scope"],
                "selection_score": selected["evaluation_selection_score"],
                "symbols": list(symbols),
                "symbol_count": len(symbols),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
