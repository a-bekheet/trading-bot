"""Auditable recurrent PPO trainer for the research-demo environment."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trading_bot.training.env import CONTRACT_FEATURES, OptionsEnv
from trading_bot.training.evaluation import EpisodeReport, run_episode
from trading_bot.training.recurrent import RecurrentConfig, build_recurrent_actor_critic
from trading_bot.training.sequence import observation_vector


CHECKPOINT_SCHEMA_VERSION = "research-demo.ppo.v1"


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
    entropy_coefficient: float = 0.01
    gradient_clip: float = 0.5
    seed: int = 7
    max_steps: int | None = None
    evaluation_interval: int = 1

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
        if self.evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive")


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install the ML extra: pip install -e '.[ml]'") from error
    return torch


def _sequence_tensor(history: deque[np.ndarray], length: int, torch):
    feature_count = history[-1].shape[0]
    padded = np.zeros((length, feature_count), dtype=np.float32)
    available = list(history)[-length:]
    padded[-len(available):] = np.stack(available)
    return torch.from_numpy(padded).unsqueeze(0)


def recurrent_policy(model, sequence_length: int):
    """Create a deterministic, stateful policy for one evaluation episode."""
    torch = _torch()
    history: deque[np.ndarray] = deque(maxlen=sequence_length)

    def policy(observation):
        history.append(observation_vector(observation))
        sequence = _sequence_tensor(history, sequence_length, torch)
        action_mask = torch.from_numpy(observation.action_mask).unsqueeze(0)
        with torch.inference_mode():
            action, _, _ = model.sample_action(
                sequence,
                action_mask,
                deterministic=True,
            )
        return action.squeeze(0).cpu().numpy()

    return policy


def evaluate_recurrent_policy(
    env: OptionsEnv,
    model,
    sequence_length: int,
    seeds: tuple[int, ...] = (101, 102, 103),
) -> list[EpisodeReport]:
    """Evaluate deterministic actions; results are research-demo/in-sample only."""
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


def train_actor_critic(
    env: OptionsEnv,
    recurrent_config: RecurrentConfig,
    training_config: TrainingConfig | None = None,
):
    """Train one recurrent policy with clipped PPO and return model/metrics.

    This is an integration trainer for the deterministic research surface. It
    deliberately makes no claim about historical or live trading performance.
    """
    torch = _torch()
    config = training_config or TrainingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    observation, _ = env.reset(seed=config.seed)
    actual_input_size = observation_vector(observation).shape[0]
    if recurrent_config.input_size != actual_input_size:
        raise ValueError(
            f"model input_size={recurrent_config.input_size}, environment emits {actual_input_size}"
        )
    if (recurrent_config.slot_count, recurrent_config.action_count) != env.action_shape:
        raise ValueError("model action dimensions do not match the environment")

    model = build_recurrent_actor_critic(recurrent_config)
    # PPO likelihood ratios require the same network behavior during rollout
    # and updates. Eval mode disables optional recurrent dropout but keeps grads.
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metrics: list[dict[str, float | int | None]] = []
    best_score = float("-inf")
    best_episode = 0
    best_state = None

    for episode in range(config.episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        history: deque[np.ndarray] = deque(maxlen=config.sequence_length)
        sequences = []
        action_masks = []
        actions = []
        rewards = []
        old_log_probabilities = []
        old_values = []
        invalid_actions = executions = steps = 0
        final_terminal = False

        while True:
            history.append(observation_vector(observation))
            sequence = _sequence_tensor(history, config.sequence_length, torch)
            action_mask = torch.from_numpy(observation.action_mask).unsqueeze(0)
            with torch.no_grad():
                logits, value, _ = model(sequence, action_mask)
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()

            observation, reward, terminated, truncated, info = env.step(
                action.squeeze(0).detach().cpu().numpy()
            )
            sequences.append(sequence.squeeze(0))
            action_masks.append(action_mask.squeeze(0))
            actions.append(action.squeeze(0))
            rewards.append(float(reward))
            old_log_probabilities.append(distribution.log_prob(action).squeeze(0))
            old_values.append(value.squeeze(0))
            invalid_actions += int(info["invalid_action_count"])
            executions += len(info["executions"])
            steps += 1
            reached_limit = config.max_steps is not None and steps >= config.max_steps
            if terminated or truncated or reached_limit:
                final_terminal = terminated or truncated
                break

        next_value = 0.0
        if not final_terminal:
            bootstrap_history = deque(history, maxlen=config.sequence_length)
            bootstrap_history.append(observation_vector(observation))
            bootstrap_sequence = _sequence_tensor(
                bootstrap_history, config.sequence_length, torch
            )
            bootstrap_mask = torch.from_numpy(observation.action_mask).unsqueeze(0)
            with torch.no_grad():
                _, bootstrap_value, _ = model(bootstrap_sequence, bootstrap_mask)
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
        for _ in range(config.ppo_epochs):
            permutation = torch.randperm(steps)
            for start in range(0, steps, config.minibatch_size):
                batch = permutation[start:start + config.minibatch_size]
                logits, predicted_values, _ = model(
                    sequences_tensor[batch], masks_tensor[batch]
                )
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
                env,
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
                "total_reward": float(sum(rewards)),
                "evaluation_total_reward": evaluation_reward,
                "loss": float(means[0]),
                "policy_loss": float(means[1]),
                "value_loss": float(means[2]),
                "entropy": float(means[3]),
                "gradient_norm": float(means[4]),
                "approx_kl": float(means[5]),
                "clip_fraction": float(means[6]),
                "explained_variance": explained_variance,
                "ppo_updates": len(update_metrics),
                "early_stop_kl": int(stop_early),
                "invalid_actions": invalid_actions,
                "executions": executions,
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
) -> dict[str, Any]:
    selected = next(
        (item["episode"] for item in metrics if item.get("selected_checkpoint")),
        len(metrics) - 1,
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "mode": "research_demo",
        "algorithm": "factorized_ppo",
        "selection": {
            "scope": "in_sample_research_demo",
            "metric": "evaluation_total_reward",
            "episode": selected,
        },
        "environment": env.manifest.to_dict(),
        "environment_fingerprint": env.manifest.fingerprint,
        "model": asdict(recurrent_config),
        "training": asdict(training_config),
        "metrics": metrics,
    }


def save_checkpoint(
    path: Path,
    model,
    env: OptionsEnv,
    recurrent_config: RecurrentConfig,
    training_config: TrainingConfig,
    metrics: list[dict[str, Any]],
) -> Path:
    """Save model weights plus a human-readable provenance sidecar."""
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = checkpoint_manifest(env, recurrent_config, training_config, metrics)
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
    model = build_recurrent_actor_critic(RecurrentConfig(**manifest["model"]))
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    env = OptionsEnv.from_directory(
        args.data_dir,
        args.symbol,
        slot_count=args.slot_count,
        max_quantity=args.max_quantity,
    )
    observation, _ = env.reset(seed=args.seed)
    recurrent_config = RecurrentConfig(
        input_size=observation_vector(observation).shape[0],
        slot_count=env.action_shape[0],
        action_count=env.action_shape[1],
        hidden_size=args.hidden_size,
        kind=args.kind,
        encoder=args.encoder,
        contract_feature_count=observation.contracts.shape[1],
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
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
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
