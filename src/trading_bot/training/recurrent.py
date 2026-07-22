"""Optional PyTorch recurrent actor-critic for the research environment.

Install the optional dependency with ``pip install -e '.[ml]'``. Keeping this
module out of the default import path keeps the collector lightweight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trading_bot.training.schemas import FEATURE_VECTOR_SCHEMA_VERSION


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
    market_feature_count: int = 2
    portfolio_feature_count: int = 3
    action_slot_count: int | None = None
    feature_vector_schema: str = FEATURE_VECTOR_SCHEMA_VERSION
    graph_hidden_size: int = 32
    graph_layers: int = 2
    graph_neighbors: int = 3
    graph_relation_indices: tuple[int, ...] = ()
    initial_hold_bias: float = 5.0
    masked_input_indices: tuple[int, ...] = ()
    auxiliary_target_count: int = 0

    def __post_init__(self) -> None:
        if min(self.input_size, self.slot_count, self.action_count, self.hidden_size) < 1:
            raise ValueError("model dimensions must be positive")
        if self.action_slot_count is not None and self.action_slot_count < 1:
            raise ValueError("action_slot_count must be positive")
        if self.market_feature_count < 1 or self.portfolio_feature_count < 1:
            raise ValueError("market and portfolio feature counts must be positive")
        if min(
            self.layers,
            self.graph_layers,
            self.graph_hidden_size,
        ) < 1:
            raise ValueError("layer and graph hidden sizes must be positive")
        if self.graph_neighbors < 0:
            raise ValueError("graph_neighbors cannot be negative")
        if not math.isfinite(self.initial_hold_bias) or self.initial_hold_bias < 0:
            raise ValueError("initial_hold_bias must be finite and nonnegative")
        if len(set(self.masked_input_indices)) != len(self.masked_input_indices):
            raise ValueError("masked_input_indices must be unique")
        if any(
            index < 0 or index >= self.input_size
            for index in self.masked_input_indices
        ):
            raise ValueError("masked_input_indices are outside the input layout")
        if self.auxiliary_target_count < 0:
            raise ValueError("auxiliary_target_count cannot be negative")


def build_recurrent_actor_critic(config: RecurrentConfig):
    """Build a GRU, LSTM, or hybrid actor-critic."""
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - exercised without ML extra
        raise RuntimeError("Install the ML extra: pip install -e '.[ml]'") from error

    if config.kind not in {"gru", "lstm", "hybrid"}:
        raise ValueError("kind must be 'gru', 'lstm', or 'hybrid'")
    graph_encoder = config.encoder in {"graph", "graph_set"}
    if config.encoder not in {"flat", "graph", "graph_set"}:
        raise ValueError("encoder must be 'flat', 'graph', or 'graph_set'")
    if graph_encoder and config.contract_feature_count is None:
        raise ValueError("graph encoder requires contract_feature_count")
    if graph_encoder and any(
        index < 0 or index >= config.contract_feature_count
        for index in config.graph_relation_indices
    ):
        raise ValueError("graph_relation_indices are outside the contract feature layout")

    temporal_input_size = config.input_size
    policy_slot_count = config.action_slot_count or config.slot_count
    if graph_encoder:
        expected = (
            config.market_feature_count
            + config.portfolio_feature_count
            + config.slot_count * (config.contract_feature_count + 1)
        )
        if expected != config.input_size:
            raise ValueError(
                "input_size does not match market, contract, portfolio, and mask dimensions"
            )
        if config.encoder == "graph_set":
            if policy_slot_count != config.slot_count + 1:
                raise ValueError(
                    "graph_set requires one action row per contract plus underlying"
                )
            temporal_input_size = (
                config.market_feature_count
                + config.portfolio_feature_count
                + 2 * config.graph_hidden_size
                + 1
            )
        else:
            temporal_input_size = (
                config.market_feature_count
                + config.portfolio_feature_count
                + config.slot_count * (config.graph_hidden_size + 1)
            )

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
            self.register_buffer(
                "_masked_input_indices",
                torch.tensor(config.masked_input_indices, dtype=torch.long),
                persistent=False,
            )
            self.input_norm = nn.LayerNorm(temporal_input_size)
            if graph_encoder:
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
            if config.encoder == "graph_set":
                self.policy = None
                self.contract_policy = nn.Linear(
                    output_size + config.graph_hidden_size,
                    config.action_count,
                )
                self.underlying_policy = nn.Linear(
                    output_size,
                    config.action_count,
                )
                nn.init.zeros_(self.contract_policy.bias)
                nn.init.zeros_(self.underlying_policy.bias)
                with torch.no_grad():
                    self.contract_policy.bias[0] = config.initial_hold_bias
                    self.underlying_policy.bias[0] = config.initial_hold_bias
            else:
                self.policy = nn.Linear(
                    output_size,
                    policy_slot_count * config.action_count,
                )
                nn.init.zeros_(self.policy.bias)
                with torch.no_grad():
                    self.policy.bias.view(
                        policy_slot_count,
                        config.action_count,
                    )[:, 0] = config.initial_hold_bias
            self.value = nn.Linear(output_size, 1)
            self.auxiliary = (
                nn.Linear(output_size, config.auxiliary_target_count)
                if config.auxiliary_target_count
                else None
            )

        def _graph_encode(self, sequence):
            batch, steps, _ = sequence.shape
            flattened = sequence.reshape(batch * steps, config.input_size)
            contract_start = config.market_feature_count
            contract_end = contract_start + config.slot_count * config.contract_feature_count
            contracts = flattened[:, contract_start:contract_end].view(
                batch * steps, config.slot_count, config.contract_feature_count
            )
            portfolio_end = contract_end + config.portfolio_feature_count
            portfolio = flattened[:, contract_end:portfolio_end]
            valid = flattened[:, portfolio_end:].bool()
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

            valid_values = valid.to(hidden.dtype)
            if config.encoder == "graph_set":
                valid_count = valid_values.sum(dim=1, keepdim=True).clamp_min(1.0)
                pooled_mean = hidden.sum(dim=1) / valid_count
                pooled_max = hidden.masked_fill(
                    ~valid.unsqueeze(-1),
                    float("-inf"),
                ).amax(dim=1)
                pooled_max = torch.where(
                    valid.any(dim=1, keepdim=True),
                    pooled_max,
                    torch.zeros_like(pooled_max),
                )
                valid_fraction = valid_values.mean(dim=1, keepdim=True)
                temporal = torch.cat(
                    (
                        flattened[:, :config.market_feature_count],
                        pooled_mean,
                        pooled_max,
                        portfolio,
                        valid_fraction,
                    ),
                    dim=-1,
                )
            else:
                temporal = torch.cat(
                    (
                        flattened[:, :config.market_feature_count],
                        hidden.flatten(start_dim=1),
                        portfolio,
                        valid_values,
                    ),
                    dim=-1,
                )
            return (
                temporal.view(batch, steps, temporal_input_size),
                hidden.view(
                    batch,
                    steps,
                    config.slot_count,
                    config.graph_hidden_size,
                ),
            )

        def _encode_sequence(self, sequence, hidden_state=None):
            if self._masked_input_indices.numel():
                sequence = sequence.clone()
                sequence.index_fill_(-1, self._masked_input_indices, 0.0)
            node_embeddings = None
            if graph_encoder:
                sequence, node_embeddings = self._graph_encode(sequence)
            sequence = self.input_norm(sequence)
            if config.kind == "hybrid":
                gru_initial = None if hidden_state is None else hidden_state["gru"]
                lstm_initial = None if hidden_state is None else hidden_state["lstm"]
                gru_encoded, gru_hidden = self.recurrent["gru"](
                    sequence, gru_initial
                )
                lstm_encoded, lstm_hidden = self.recurrent["lstm"](
                    sequence, lstm_initial
                )
                encoded = torch.cat((gru_encoded, lstm_encoded), dim=-1)
                hidden = {"gru": gru_hidden, "lstm": lstm_hidden}
            else:
                encoded, hidden = self.recurrent(sequence, hidden_state)
            return encoded, hidden, node_embeddings

        @staticmethod
        def _safe_action_mask(action_mask):
            safe_mask = action_mask.bool().clone()
            empty_slots = ~safe_mask.any(dim=-1)
            safe_mask[..., 0] |= empty_slots
            return safe_mask

        def _actor_critic_outputs(
            self,
            encoded,
            sequence,
            action_mask,
            node_embeddings,
        ):
            if config.encoder == "graph_set":
                if node_embeddings is None:
                    raise RuntimeError("graph_set node embeddings are missing")
                global_context = encoded.unsqueeze(2).expand(
                    -1,
                    -1,
                    config.slot_count,
                    -1,
                )
                option_logits = self.contract_policy(torch.cat(
                    (global_context, node_embeddings),
                    dim=-1,
                ))
                underlying_logits = self.underlying_policy(encoded).unsqueeze(2)
                logits = torch.cat((option_logits, underlying_logits), dim=2)
            else:
                logits = self.policy(encoded).view(
                    sequence.shape[0],
                    sequence.shape[1],
                    policy_slot_count,
                    config.action_count,
                )
            if action_mask is not None:
                if action_mask.ndim == 3:
                    action_mask = action_mask.unsqueeze(1).expand(
                        -1, sequence.shape[1], -1, -1
                    )
                if action_mask.shape != logits.shape:
                    raise ValueError("action_mask does not match recurrent outputs")
                safe_mask = self._safe_action_mask(action_mask)
                logits = logits.masked_fill(~safe_mask, float("-inf"))
            return logits, self.value(encoded).squeeze(-1)

        def forward_sequence(self, sequence, action_mask=None, hidden_state=None):
            """Return actor and critic outputs for every causal time step."""
            encoded, hidden, node_embeddings = self._encode_sequence(
                sequence,
                hidden_state,
            )
            logits, values = self._actor_critic_outputs(
                encoded,
                sequence,
                action_mask,
                node_embeddings,
            )
            return logits, values, hidden

        def forward_sequence_with_auxiliary(
            self,
            sequence,
            action_mask=None,
            hidden_state=None,
        ):
            """Return PPO outputs plus train-only next-market predictions."""
            if self.auxiliary is None:
                raise RuntimeError("model has no auxiliary prediction head")
            encoded, hidden, node_embeddings = self._encode_sequence(
                sequence,
                hidden_state,
            )
            logits, values = self._actor_critic_outputs(
                encoded,
                sequence,
                action_mask,
                node_embeddings,
            )
            return (
                logits,
                values,
                self.auxiliary(encoded),
                hidden,
            )

        def forward(
            self,
            sequence,
            action_mask=None,
            hidden_state=None,
        ):
            """Return the final causal step, preserving the original API."""
            logits, values, hidden = self.forward_sequence(
                sequence,
                action_mask,
                hidden_state,
            )
            return logits[:, -1], values[:, -1], hidden

        def sample_action(
            self,
            sequence,
            action_mask,
            deterministic=False,
            hidden_state=None,
        ):
            logits, value, hidden = self(
                sequence,
                action_mask,
                hidden_state,
            )
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
            return action, value, hidden

    return ActorCritic()
