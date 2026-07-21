"""Optional PyTorch recurrent actor-critic for the research environment.

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
    """Build a GRU, LSTM, or hybrid actor-critic."""
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - exercised without ML extra
        raise RuntimeError("Install the ML extra: pip install -e '.[ml]'") from error

    if config.kind not in {"gru", "lstm", "hybrid"}:
        raise ValueError("kind must be 'gru', 'lstm', or 'hybrid'")

    def make_recurrent(kind: str):
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        return recurrent(
            config.input_size,
            config.hidden_size,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.input_norm = nn.LayerNorm(config.input_size)
            if config.kind == "hybrid":
                self.recurrent = nn.ModuleDict(
                    {"gru": make_recurrent("gru"), "lstm": make_recurrent("lstm")}
                )
                output_size = 2 * config.hidden_size
            else:
                self.recurrent = make_recurrent(config.kind)
                output_size = config.hidden_size
            self.policy = nn.Linear(output_size, config.slot_count * config.action_count)
            self.value = nn.Linear(output_size, 1)

        def forward(self, sequence, action_mask=None):
            sequence = self.input_norm(sequence)
            if config.kind == "hybrid":
                gru_encoded, gru_hidden = self.recurrent["gru"](sequence)
                lstm_encoded, lstm_hidden = self.recurrent["lstm"](sequence)
                final = torch.cat((gru_encoded[:, -1], lstm_encoded[:, -1]), dim=-1)
                hidden = {"gru": gru_hidden, "lstm": lstm_hidden}
            else:
                encoded, hidden = self.recurrent(sequence)
                final = encoded[:, -1]
            logits = self.policy(final).view(
                sequence.shape[0], config.slot_count, config.action_count
            )
            if action_mask is not None:
                safe_mask = action_mask.bool().clone()
                empty_slots = ~safe_mask.any(dim=-1)
                safe_mask[..., 0] |= empty_slots
                logits = logits.masked_fill(~safe_mask, float("-inf"))
            return logits, self.value(final).squeeze(-1), hidden

        def sample_action(self, sequence, action_mask, deterministic=False):
            logits, value, hidden = self(sequence, action_mask)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
            return action, value, hidden

    return ActorCritic()
