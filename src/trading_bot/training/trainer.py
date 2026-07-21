"""Auditable recurrent PPO trainer for the research-demo environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trading_bot.training.env import CONTRACT_FEATURES, OptionsEnv
from trading_bot.training.evaluation import EpisodeReport, run_episode
from trading_bot.training.recurrent import RecurrentConfig, build_recurrent_actor_critic
from trading_bot.training.schemas import FEATURE_VECTOR_SCHEMA_VERSION
from trading_bot.training.sequence import observation_vector


CHECKPOINT_SCHEMA_VERSION = "research-demo.ppo.v9"


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 25
    sequence_length: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
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

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.sequence_length < 1:
            raise ValueError("episodes and sequence_length must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be between zero and one")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.clip_ratio <= 0 or self.value_clip <= 0 or self.target_kl <= 0:
            raise ValueError("PPO clipping and target_kl values must be positive")
        if self.ppo_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("ppo_epochs and minibatch_size must be positive")
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("loss coefficients cannot be negative")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        if not isinstance(self.random_start, bool):
            raise ValueError("random_start must be a boolean")
        if self.evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive")


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install the ML extra: pip install -e '.[ml]'") from error
    return torch


def _detach_hidden(hidden_state):
    """Detach a tensor, LSTM tuple, or hybrid recurrent-state mapping."""
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


def _generalized_advantages(
    rewards,
    values,
    next_value: float,
    terminal: bool,
    config: TrainingConfig,
    torch,
):
    advantages = torch.zeros_like(values)
    advantage = torch.tensor(0.0)
    following_value = torch.tensor(float(next_value))
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 0.0 if terminal and index == len(rewards) - 1 else 1.0
        delta = (
            rewards[index]
            + config.gamma * following_value * nonterminal
            - values[index]
        )
        advantage = (
            delta
            + config.gamma * config.gae_lambda * nonterminal * advantage
        )
        advantages[index] = advantage
        following_value = values[index]
    return advantages, advantages + values


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


def train_actor_critic(
    env: OptionsEnv,
    recurrent_config: RecurrentConfig,
    training_config: TrainingConfig | None = None,
    *,
    selection_env: OptionsEnv | None = None,
):
    """Train one recurrent policy with clipped PPO and return model/metrics.

    This is an integration trainer for the deterministic research surface. It
    deliberately makes no claim about historical or live trading performance.
    """
    torch = _torch()
    config = training_config or TrainingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rollout_rng = np.random.default_rng(config.seed)

    observation, _ = env.reset(seed=config.seed)
    actual_input_size = observation_vector(observation).shape[0]
    if recurrent_config.input_size != actual_input_size:
        raise ValueError(
            f"model input_size={recurrent_config.input_size}, environment emits {actual_input_size}"
        )
    model_action_shape = (
        recurrent_config.action_slot_count or recurrent_config.slot_count,
        recurrent_config.action_count,
    )
    if model_action_shape != env.action_shape:
        raise ValueError("model action dimensions do not match the environment")
    if recurrent_config.feature_vector_schema != FEATURE_VECTOR_SCHEMA_VERSION:
        raise ValueError("model feature-vector schema does not match the trainer")
    evaluation_env = env if selection_env is None else selection_env
    selection_observation, _ = evaluation_env.reset(seed=config.seed)
    if observation_vector(selection_observation).shape[0] != actual_input_size:
        raise ValueError("selection environment feature layout does not match training")
    if evaluation_env.action_shape != env.action_shape:
        raise ValueError("selection environment action dimensions do not match training")
    if len(evaluation_env.dataset) < 2:
        raise ValueError("selection environment requires at least two snapshots")
    selection_scope = (
        "in_sample_research_demo"
        if selection_env is None
        else "validation_research_demo"
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

    for episode in range(config.episodes):
        rollout_start, rollout_step_limit = _sample_rollout_bounds(
            len(env.dataset),
            config.max_steps,
            config.random_start,
            rollout_rng,
        )
        observation, _ = env.reset(
            seed=config.seed + episode,
            options={"start_index": rollout_start},
        )
        sequences = []
        action_masks = []
        actions = []
        rewards = []
        old_log_probabilities = []
        old_values = []
        chunk_hidden_states = []
        hidden_state = None
        invalid_actions = executions = steps = 0
        requested_option_orders = 0
        requested_underlying_orders = 0
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
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
            hidden_state = _detach_hidden(hidden_state)
            action_array = action.squeeze(0).detach().cpu().numpy()
            requested_option_orders += int(
                np.count_nonzero(action_array[:env.slot_count])
            )
            requested_underlying_orders += int(action_array[-1] != 0)

            observation, reward, terminated, truncated, info = env.step(
                action_array
            )
            sequences.append(sequence.squeeze(0).squeeze(0))
            action_masks.append(action_mask.squeeze(0))
            actions.append(action.squeeze(0))
            rewards.append(float(reward))
            old_log_probabilities.append(distribution.log_prob(action).squeeze(0))
            old_values.append(value.squeeze(0))
            invalid_actions += int(info["invalid_action_count"])
            executions += len(info["executions"])
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
        advantages, returns_tensor = _generalized_advantages(
            rewards_tensor,
            old_values_tensor,
            next_value,
            final_terminal,
            config,
            torch,
        )
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
        for _ in range(config.ppo_epochs):
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
                sequence_logits, sequence_values, _ = model.forward_sequence(
                    batch_sequences,
                    batch_masks,
                    hidden_state=initial_hidden,
                )
                logits = sequence_logits[valid_steps]
                predicted_values = sequence_values[valid_steps]
                distribution = torch.distributions.Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions_tensor[batch])
                log_ratio = new_log_probs - old_log_probs_tensor[batch]
                ratio = log_ratio.exp()
                batch_advantages = advantages[batch].unsqueeze(-1)
                unclipped = ratio * batch_advantages
                clipped = ratio.clamp(
                    1 - config.clip_ratio,
                    1 + config.clip_ratio,
                ) * batch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                value_delta = predicted_values - old_values_tensor[batch]
                clipped_values = old_values_tensor[batch] + value_delta.clamp(
                    -config.value_clip,
                    config.value_clip,
                )
                value_loss = 0.5 * torch.maximum(
                    (predicted_values - returns_tensor[batch]).square(),
                    (clipped_values - returns_tensor[batch]).square(),
                ).mean()
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
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
                        float((ratio.sub(1).abs() > config.clip_ratio).float().mean()),
                    )
                )
                if approx_kl > config.target_kl:
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
        should_evaluate = (
            (episode + 1) % config.evaluation_interval == 0
            or episode == config.episodes - 1
        )
        if should_evaluate:
            report = evaluate_recurrent_policy(
                evaluation_env,
                model,
                config.sequence_length,
                seeds=(config.seed + 10_000 + episode,),
            )[0]
            evaluation_reward = float(report.total_reward)
            if evaluation_reward > best_score:
                best_score = evaluation_reward
                best_episode = episode
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        metrics.append(
            {
                "episode": episode,
                "steps": steps,
                "rollout_start_index": rollout_start,
                "rollout_end_index": rollout_start + steps,
                "total_reward": float(sum(rewards)),
                "evaluation_total_reward": evaluation_reward,
                "evaluation_scope": selection_scope,
                "loss": float(means[0]),
                "policy_loss": float(means[1]),
                "value_loss": float(means[2]),
                "entropy": float(means[3]),
                "entropy_bonus": float(config.entropy_coefficient * means[3]),
                "gradient_norm": float(means[4]),
                "approx_kl": float(means[5]),
                "clip_fraction": float(means[6]),
                "explained_variance": explained_variance,
                "ppo_updates": len(update_metrics),
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
                    / (steps * env.action_shape[0])
                ),
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_episode = len(metrics) - 1
    for index, item in enumerate(metrics):
        item["selected_checkpoint"] = int(index == best_episode)
    return model, metrics


def checkpoint_manifest(
    env: OptionsEnv,
    recurrent_config: RecurrentConfig,
    training_config: TrainingConfig,
    metrics: list[dict[str, Any]],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_metric = next(
        (item for item in metrics if item.get("selected_checkpoint")),
        metrics[-1],
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "mode": "research_demo",
        "algorithm": "stateful_factorized_ppo",
        "action_policy": {
            "factorization": "independent_masked_rows",
            "initial_hold_bias": recurrent_config.initial_hold_bias,
            "hard_order_cap": None,
        },
        "temporal_training": {
            "mode": "stateful_tbptt",
            "chunk_length": training_config.sequence_length,
            "padding": "right_only_ignored",
        },
        "feature_vector_schema": FEATURE_VECTOR_SCHEMA_VERSION,
        "selection": {
            "scope": selected_metric.get(
                "evaluation_scope",
                "in_sample_research_demo",
            ),
            "metric": "evaluation_total_reward",
            "episode": selected_metric["episode"],
        },
        "environment": env.manifest.to_dict(),
        "environment_fingerprint": env.manifest.fingerprint,
        "model": asdict(recurrent_config),
        "training": asdict(training_config),
        "metrics": metrics,
        "provenance": provenance or {},
    }


def save_checkpoint(
    path: Path,
    model,
    env: OptionsEnv,
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
    parser.add_argument("--kind", choices=("gru", "lstm", "hybrid"), default="gru")
    parser.add_argument("--encoder", choices=("flat", "graph"), default="flat")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--slot-count", type=int, default=32)
    parser.add_argument("--max-quantity", type=int, default=3)
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
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--entropy-coefficient", type=float, default=1e-4)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--initial-hold-bias", type=float, default=5.0)
    parser.add_argument("--max-abs-delta", type=float)
    parser.add_argument("--max-abs-gamma", type=float)
    parser.add_argument("--max-abs-theta", type=float)
    parser.add_argument("--max-abs-vega", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    env = OptionsEnv.from_directory(
        args.data_dir,
        args.symbol,
        slot_count=args.slot_count,
        max_quantity=args.max_quantity,
        underlying_lot_size=args.underlying_lot_size,
        max_abs_underlying_shares=args.max_abs_underlying_shares,
        underlying_commission_per_share=args.underlying_commission_per_share,
        underlying_slippage_bps=args.underlying_slippage_bps,
        max_abs_delta=args.max_abs_delta,
        max_abs_gamma=args.max_abs_gamma,
        max_abs_theta=args.max_abs_theta,
        max_abs_vega=args.max_abs_vega,
    )
    observation, _ = env.reset(seed=args.seed)
    recurrent_config = RecurrentConfig(
        input_size=observation_vector(observation).shape[0],
        slot_count=env.slot_count,
        action_slot_count=env.action_shape[0],
        action_count=env.action_shape[1],
        hidden_size=args.hidden_size,
        kind=args.kind,
        encoder=args.encoder,
        contract_feature_count=observation.contracts.shape[1],
        market_feature_count=observation.market.size,
        portfolio_feature_count=observation.portfolio.size,
        initial_hold_bias=args.initial_hold_bias,
        graph_relation_indices=tuple(
            CONTRACT_FEATURES.index(name)
            for name in ("impliedVolatility", "delta", "logMoneyness", "dteDays")
        ),
    )
    training_config = TrainingConfig(
        episodes=args.episodes,
        sequence_length=args.sequence_length,
        seed=args.seed,
        max_steps=args.max_steps,
        random_start=args.random_start,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        entropy_coefficient=args.entropy_coefficient,
        evaluation_interval=args.evaluation_interval,
    )
    model, metrics = train_actor_critic(env, recurrent_config, training_config)
    output = args.output or Path("data/models") / (
        f"{env.dataset.symbol}-{args.encoder}-{args.kind}.pt"
    )
    save_checkpoint(output, model, env, recurrent_config, training_config, metrics)
    selected = next(item for item in metrics if item["selected_checkpoint"])
    print(
        json.dumps(
            {"checkpoint": str(output), "selected_episode": selected},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
