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
    encoder: str = "flat"
    contract_feature_count: int | None = None
    graph_hidden_size: int = 32
    graph_layers: int = 2
    graph_neighbors: int = 3
    graph_relation_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if min(self.input_size, self.slot_count, self.action_count, self.hidden_size) < 1:
            raise ValueError("model dimensions must be positive")
        if self.layers < 1 or self.graph_layers < 1 or self.graph_hidden_size < 1:
            raise ValueError("layer counts and graph_hidden_size must be positive")
        if self.graph_neighbors < 0:
            raise ValueError("graph_neighbors cannot be negative")


def build_recurrent_actor_critic(config: RecurrentConfig):
    """Build a GRU, LSTM, or hybrid actor-critic."""
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - exercised without ML extra
        raise RuntimeError("Install the ML extra: pip install -e '.[ml]'") from error

    if config.kind not in {"gru", "lstm", "hybrid"}:
        raise ValueError("kind must be 'gru', 'lstm', or 'hybrid'")
    if config.encoder not in {"flat", "graph"}:
        raise ValueError("encoder must be 'flat' or 'graph'")
    if config.encoder == "graph" and config.contract_feature_count is None:
        raise ValueError("graph encoder requires contract_feature_count")
    if config.encoder == "graph" and any(
        index < 0 or index >= config.contract_feature_count
        for index in config.graph_relation_indices
    ):
        raise ValueError("graph_relation_indices are outside the contract feature layout")

    temporal_input_size = config.input_size
    if config.encoder == "graph":
        expected = 5 + config.slot_count * (config.contract_feature_count + 1)
        if expected != config.input_size:
            raise ValueError(
                "input_size does not match market, contract, portfolio, and mask dimensions"
            )
        temporal_input_size = 5 + config.slot_count * (config.graph_hidden_size + 1)

    def make_recurrent(kind: str):
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        return recurrent(
            temporal_input_size,
            config.hidden_size,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.input_norm = nn.LayerNorm(temporal_input_size)
            if config.encoder == "graph":
                graph_dimensions = [
                    config.contract_feature_count,
                    *([config.graph_hidden_size] * config.graph_layers),
                ]
                self.contract_norm = nn.LayerNorm(config.contract_feature_count)
                self.graph_message_layers = nn.ModuleList(
                    nn.ModuleDict(
                        {
                            "self": nn.Linear(source, target),
                            "neighbor": nn.Linear(source, target, bias=False),
                        }
                    )
                    for source, target in zip(graph_dimensions, graph_dimensions[1:])
                )
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

        def _graph_encode(self, sequence):
            batch, steps, _ = sequence.shape
            flattened = sequence.reshape(batch * steps, config.input_size)
            contract_start = 2
            contract_end = contract_start + config.slot_count * config.contract_feature_count
            contracts = flattened[:, contract_start:contract_end].view(
                batch * steps, config.slot_count, config.contract_feature_count
            )
            portfolio = flattened[:, contract_end:contract_end + 3]
            valid = flattened[:, contract_end + 3:].bool()
            hidden = self.contract_norm(contracts)

            pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
            adjacency = torch.zeros(
                batch * steps,
                config.slot_count,
                config.slot_count,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            neighbor_count = min(config.graph_neighbors, max(config.slot_count - 1, 0))
            if neighbor_count:
                relation_indices = config.graph_relation_indices or tuple(
                    range(config.contract_feature_count)
                )
                relation = contracts[:, :, relation_indices]
                relation_mask = valid.unsqueeze(-1).to(relation.dtype)
                relation_count = relation_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                relation_mean = (relation * relation_mask).sum(
                    dim=1, keepdim=True
                ) / relation_count
                centered = (relation - relation_mean) * relation_mask
                relation_scale = (
                    centered.square().sum(dim=1, keepdim=True) / relation_count
                ).sqrt().clamp_min(1e-6)
                relation = centered / relation_scale
                distances = torch.cdist(relation, relation)
                diagonal = torch.eye(
                    config.slot_count, dtype=torch.bool, device=hidden.device
                ).unsqueeze(0)
                distances = distances.masked_fill(~pair_valid | diagonal, float("inf"))
                neighbors = distances.topk(neighbor_count, largest=False).indices
                adjacency.scatter_(-1, neighbors, 1.0)
                adjacency *= pair_valid
                adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
            adjacency += torch.diag_embed(valid.to(hidden.dtype))
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)

            for layer in self.graph_message_layers:
                neighbor_mean = adjacency.bmm(hidden) / degree
                hidden = torch.nn.functional.gelu(
                    layer["self"](hidden) + layer["neighbor"](neighbor_mean)
                )
                hidden *= valid.unsqueeze(-1)

            temporal = torch.cat(
                (
                    flattened[:, :2],
                    hidden.flatten(start_dim=1),
                    portfolio,
                    valid.to(hidden.dtype),
                ),
                dim=-1,
            )
            return temporal.view(batch, steps, temporal_input_size)

        def forward(self, sequence, action_mask=None):
            if config.encoder == "graph":
                sequence = self._graph_encode(sequence)
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
