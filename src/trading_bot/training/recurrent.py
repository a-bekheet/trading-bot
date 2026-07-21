"""Optional PyTorch LSTM/GRU actor-critic for the research environment.

Install the optional dependency with ``pip install -e '.[ml]'``. Keeping this
module out of the default import path keeps the collector lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecurrentConfig:
    input_size: int
    slot_count: int
    action_count: int
    hidden_size: int = 128
    layers: int = 1
    kind: str = "gru"
    dropout: float = 0.0


def build_recurrent_actor_critic(config: RecurrentConfig):
    """Build a GRU or LSTM actor-critic; raises a clear optional-dependency error."""
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - exercised without ML extra
        raise RuntimeError("Install the ML extra: pip install -e '.[ml]'") from error

    if config.kind not in {"gru", "lstm"}:
        raise ValueError("kind must be 'gru' or 'lstm'")
    recurrent = nn.GRU if config.kind == "gru" else nn.LSTM

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.recurrent = recurrent(
                config.input_size,
                config.hidden_size,
                num_layers=config.layers,
                batch_first=True,
                dropout=config.dropout if config.layers > 1 else 0.0,
            )
            self.policy = nn.Linear(config.hidden_size, config.slot_count * config.action_count)
            self.value = nn.Linear(config.hidden_size, 1)

        def forward(self, sequence, action_mask=None):
            encoded, hidden = self.recurrent(sequence)
            final = encoded[:, -1]
            logits = self.policy(final).view(
                sequence.shape[0], config.slot_count, config.action_count
            )
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
            return logits, self.value(final).squeeze(-1), hidden

        def sample_action(self, sequence, action_mask, deterministic=False):
            logits, value, hidden = self(sequence, action_mask)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
            return action, value, hidden

    return ActorCritic()
