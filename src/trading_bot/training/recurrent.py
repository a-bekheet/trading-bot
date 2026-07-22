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
    attention_heads: int = 4
    graph_relation_indices: tuple[int, ...] = ()
    initial_hold_bias: float = 5.0
    action_decoder: str = "factorized"
    masked_input_indices: tuple[int, ...] = ()
    auxiliary_target_count: int = 0
    auxiliary_horizons: tuple[int, ...] = (1,)

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
        if self.encoder == "attention_set":
            object.__setattr__(self, "graph_neighbors", 0)
        if self.attention_heads < 1:
            raise ValueError("attention_heads must be positive")
        if not math.isfinite(self.initial_hold_bias) or self.initial_hold_bias < 0:
            raise ValueError("initial_hold_bias must be finite and nonnegative")
        if self.action_decoder not in {"factorized", "single_leg"}:
            raise ValueError("action_decoder must be factorized or single_leg")
        if self.action_decoder == "single_leg" and self.action_count < 2:
            raise ValueError("single_leg action decoder requires a non-hold action")
        if len(set(self.masked_input_indices)) != len(self.masked_input_indices):
            raise ValueError("masked_input_indices must be unique")
        if any(
            index < 0 or index >= self.input_size
            for index in self.masked_input_indices
        ):
            raise ValueError("masked_input_indices are outside the input layout")
        if len(self.masked_input_indices) == self.input_size:
            raise ValueError("masked_input_indices cannot disable every input")
        if self.auxiliary_target_count < 0:
            raise ValueError("auxiliary_target_count cannot be negative")
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


def build_recurrent_actor_critic(config: RecurrentConfig):
    """Build a GRU, LSTM, concatenated hybrid, or gated mixture."""
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - exercised without ML extra
        raise RuntimeError("Install the ML extra: pip install -e '.[ml]'") from error

    if config.kind not in {"gru", "lstm", "hybrid", "mixture"}:
        raise ValueError(
            "kind must be 'gru', 'lstm', 'hybrid', or 'mixture'"
        )
    graph_encoder = config.encoder in {"graph", "graph_set", "attention_set"}
    set_encoder = config.encoder in {"graph_set", "attention_set"}
    fixed_graph_encoder = config.encoder in {"graph", "graph_set"}
    single_leg = config.action_decoder == "single_leg"
    if config.encoder not in {"flat", "graph", "graph_set", "attention_set"}:
        raise ValueError(
            "encoder must be 'flat', 'graph', 'graph_set', or 'attention_set'"
        )
    if graph_encoder and config.contract_feature_count is None:
        raise ValueError("graph encoder requires contract_feature_count")
    if graph_encoder and any(
        index < 0 or index >= config.contract_feature_count
        for index in config.graph_relation_indices
    ):
        raise ValueError("graph_relation_indices are outside the contract feature layout")
    if (
        config.encoder == "attention_set"
        and config.graph_hidden_size % config.attention_heads
    ):
        raise ValueError(
            "attention_set graph_hidden_size must be divisible by attention_heads"
        )

    compact_flat_mask = bool(
        not graph_encoder and config.masked_input_indices
    )
    temporal_input_size = (
        config.input_size - len(config.masked_input_indices)
        if compact_flat_mask
        else config.input_size
    )
    masked_input_index_set = set(config.masked_input_indices)
    active_input_indices = tuple(
        index
        for index in range(config.input_size)
        if index not in masked_input_index_set
    )
    policy_slot_count = config.action_slot_count or config.slot_count
    joint_action_count = 1 + policy_slot_count * (config.action_count - 1)
    effective_graph_neighbors = min(
        config.graph_neighbors,
        max(config.slot_count - 1, 0),
    )
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
        if set_encoder:
            if policy_slot_count != config.slot_count + 1:
                raise ValueError(
                    "set encoders require one action row per contract plus underlying"
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
            self.register_buffer(
                "_active_input_indices",
                torch.tensor(
                    active_input_indices,
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.input_norm = nn.LayerNorm(temporal_input_size)
            if graph_encoder:
                self.contract_norm = nn.LayerNorm(config.contract_feature_count)
                if config.encoder == "attention_set":
                    self.contract_projection = nn.Linear(
                        config.contract_feature_count,
                        config.graph_hidden_size,
                    )
                    self.attention_layers = nn.ModuleList(
                        nn.ModuleDict({
                            "attention": nn.MultiheadAttention(
                                config.graph_hidden_size,
                                config.attention_heads,
                                batch_first=True,
                            ),
                            "attention_norm": nn.LayerNorm(
                                config.graph_hidden_size
                            ),
                            "feedforward": nn.Sequential(
                                nn.Linear(
                                    config.graph_hidden_size,
                                    2 * config.graph_hidden_size,
                                ),
                                nn.GELU(),
                                nn.Linear(
                                    2 * config.graph_hidden_size,
                                    config.graph_hidden_size,
                                ),
                            ),
                            "feedforward_norm": nn.LayerNorm(
                                config.graph_hidden_size
                            ),
                        })
                        for _ in range(config.graph_layers)
                    )
                else:
                    graph_dimensions = [
                        config.contract_feature_count,
                        *([config.graph_hidden_size] * config.graph_layers),
                    ]
                    self.graph_message_layers = nn.ModuleList(
                        nn.ModuleDict(
                            {
                                "self": nn.Linear(source, target),
                                **(
                                    {
                                        "neighbor": nn.Linear(
                                            source,
                                            target,
                                            bias=False,
                                        )
                                    }
                                    if effective_graph_neighbors
                                    else {}
                                ),
                            }
                        )
                        for source, target in zip(
                            graph_dimensions,
                            graph_dimensions[1:],
                        )
                    )
            if config.kind in {"hybrid", "mixture"}:
                self.recurrent = nn.ModuleDict(
                    {"gru": make_recurrent("gru"), "lstm": make_recurrent("lstm")}
                )
                if config.kind == "mixture":
                    self.mixture_gate = nn.Linear(2 * config.hidden_size, 1)
                    nn.init.zeros_(self.mixture_gate.weight)
                    nn.init.zeros_(self.mixture_gate.bias)
                    output_size = config.hidden_size
                else:
                    self.mixture_gate = None
                    output_size = 2 * config.hidden_size
            else:
                self.recurrent = make_recurrent(config.kind)
                self.mixture_gate = None
                output_size = config.hidden_size
            if set_encoder:
                self.policy = None
                self.contract_policy = nn.Linear(
                    output_size + config.graph_hidden_size,
                    config.action_count - 1 if single_leg else config.action_count,
                )
                self.underlying_policy = nn.Linear(
                    output_size,
                    config.action_count - 1 if single_leg else config.action_count,
                )
                nn.init.zeros_(self.contract_policy.bias)
                nn.init.zeros_(self.underlying_policy.bias)
                self.joint_hold = (
                    nn.Linear(output_size, 1) if single_leg else None
                )
                if single_leg:
                    nn.init.zeros_(self.joint_hold.weight)
                    nn.init.constant_(
                        self.joint_hold.bias,
                        config.initial_hold_bias,
                    )
                else:
                    with torch.no_grad():
                        self.contract_policy.bias[0] = config.initial_hold_bias
                        self.underlying_policy.bias[0] = config.initial_hold_bias
            else:
                self.policy = nn.Linear(
                    output_size,
                    (
                        joint_action_count
                        if single_leg
                        else policy_slot_count * config.action_count
                    ),
                )
                nn.init.zeros_(self.policy.bias)
                with torch.no_grad():
                    if single_leg:
                        self.policy.bias[0] = config.initial_hold_bias
                    else:
                        self.policy.bias.view(
                            policy_slot_count,
                            config.action_count,
                        )[:, 0] = config.initial_hold_bias
                self.joint_hold = None
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

            neighbor_count = (
                effective_graph_neighbors if fixed_graph_encoder else 0
            )
            if config.encoder == "attention_set":
                hidden = self.contract_projection(hidden)
                hidden *= valid.unsqueeze(-1)
                key_padding_mask = ~valid
                # MultiheadAttention needs at least one unmasked key. A fully
                # empty surface uses the already-zero first node as a sentinel;
                # every output is masked back to zero below.
                key_padding_mask[:, 0] &= valid.any(dim=1)
                for layer in self.attention_layers:
                    attended, _ = layer["attention"](
                        hidden,
                        hidden,
                        hidden,
                        key_padding_mask=key_padding_mask,
                        need_weights=False,
                    )
                    hidden = layer["attention_norm"](hidden + attended)
                    hidden *= valid.unsqueeze(-1)
                    transformed = layer["feedforward"](hidden)
                    hidden = layer["feedforward_norm"](
                        hidden + transformed
                    )
                    hidden *= valid.unsqueeze(-1)
            elif neighbor_count:
                pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
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
                adjacency = torch.zeros(
                    batch * steps,
                    config.slot_count,
                    config.slot_count,
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                adjacency.scatter_(-1, neighbors, 1.0)
                adjacency *= pair_valid
                adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
                adjacency += torch.diag_embed(valid.to(hidden.dtype))
                degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)

            if fixed_graph_encoder:
                for layer in self.graph_message_layers:
                    transformed = layer["self"](hidden)
                    if neighbor_count:
                        neighbor_mean = adjacency.bmm(hidden) / degree
                        transformed = transformed + layer["neighbor"](neighbor_mean)
                    hidden = torch.nn.functional.gelu(transformed)
                    hidden *= valid.unsqueeze(-1)

            valid_values = valid.to(hidden.dtype)
            if set_encoder:
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
                if compact_flat_mask:
                    sequence = sequence.index_select(
                        -1,
                        self._active_input_indices,
                    )
                else:
                    sequence = sequence.clone()
                    sequence.index_fill_(-1, self._masked_input_indices, 0.0)
            node_embeddings = None
            if graph_encoder:
                sequence, node_embeddings = self._graph_encode(sequence)
            sequence = self.input_norm(sequence)
            if config.kind in {"hybrid", "mixture"}:
                gru_initial = None if hidden_state is None else hidden_state["gru"]
                lstm_initial = None if hidden_state is None else hidden_state["lstm"]
                gru_encoded, gru_hidden = self.recurrent["gru"](
                    sequence, gru_initial
                )
                lstm_encoded, lstm_hidden = self.recurrent["lstm"](
                    sequence, lstm_initial
                )
                combined = torch.cat((gru_encoded, lstm_encoded), dim=-1)
                if config.kind == "mixture":
                    gru_weight = torch.sigmoid(self.mixture_gate(combined))
                    encoded = (
                        gru_weight * gru_encoded
                        + (1.0 - gru_weight) * lstm_encoded
                    )
                else:
                    encoded = combined
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
            if set_encoder:
                if node_embeddings is None:
                    raise RuntimeError("set-encoder node embeddings are missing")
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
                row_logits = torch.cat((option_logits, underlying_logits), dim=2)
                logits = (
                    torch.cat(
                        (
                            self.joint_hold(encoded),
                            row_logits.flatten(start_dim=2),
                        ),
                        dim=-1,
                    )
                    if single_leg
                    else row_logits
                )
            else:
                policy_logits = self.policy(encoded)
                logits = (
                    policy_logits
                    if single_leg
                    else policy_logits.view(
                        sequence.shape[0],
                        sequence.shape[1],
                        policy_slot_count,
                        config.action_count,
                    )
                )
            if action_mask is not None:
                if action_mask.ndim == 3:
                    action_mask = action_mask.unsqueeze(1).expand(
                        -1, sequence.shape[1], -1, -1
                    )
                expected_mask_shape = (
                    *sequence.shape[:2],
                    policy_slot_count,
                    config.action_count,
                )
                if action_mask.shape != expected_mask_shape:
                    raise ValueError("action_mask does not match recurrent outputs")
                safe_mask = self._safe_action_mask(action_mask)
                if single_leg:
                    hold_mask = torch.ones(
                        *safe_mask.shape[:-2],
                        1,
                        dtype=torch.bool,
                        device=safe_mask.device,
                    )
                    joint_mask = torch.cat(
                        (hold_mask, safe_mask[..., 1:].flatten(start_dim=-2)),
                        dim=-1,
                    )
                    logits = logits.masked_fill(~joint_mask, float("-inf"))
                else:
                    logits = logits.masked_fill(~safe_mask, float("-inf"))
            return logits, self.value(encoded).squeeze(-1)

        @staticmethod
        def _categorical(logits):
            return torch.distributions.Categorical(logits=logits)

        def _joint_action_indices(self, actions):
            if actions.shape[-1] != policy_slot_count:
                raise ValueError("actions do not match policy slot count")
            if ((actions < 0) | (actions >= config.action_count)).any():
                raise ValueError("actions are outside the encoded action range")
            active = actions.ne(0)
            if (active.sum(dim=-1) > 1).any():
                raise ValueError("single_leg actions may contain at most one order")
            row = active.to(torch.int64).argmax(dim=-1)
            encoded = actions.gather(-1, row.unsqueeze(-1)).squeeze(-1)
            return torch.where(
                active.any(dim=-1),
                1 + row * (config.action_count - 1) + encoded - 1,
                torch.zeros_like(row),
            )

        def _decode_joint_actions(self, indices):
            offset = (indices - 1).clamp_min(0)
            row = offset // (config.action_count - 1)
            encoded = offset % (config.action_count - 1) + 1
            encoded = encoded * indices.ne(0)
            actions = torch.zeros(
                *indices.shape,
                policy_slot_count,
                dtype=torch.long,
                device=indices.device,
            )
            return actions.scatter(-1, row.unsqueeze(-1), encoded.unsqueeze(-1))

        def actions_from_logits(self, logits, *, deterministic=False):
            """Sample encoded environment actions from policy logits."""
            sampled = (
                logits.argmax(dim=-1)
                if deterministic
                else self._categorical(logits).sample()
            )
            return self._decode_joint_actions(sampled) if single_leg else sampled

        def action_log_probabilities(
            self,
            logits,
            actions,
            *,
            aggregation="components",
        ):
            """Return exact component or joint decoder log probabilities."""
            if aggregation not in {"components", "joint"}:
                raise ValueError(
                    "action likelihood aggregation must be components or joint"
                )
            distribution = self._categorical(logits)
            if single_leg:
                indices = self._joint_action_indices(actions)
                components = distribution.log_prob(indices).unsqueeze(-1)
            else:
                components = distribution.log_prob(actions)
            return (
                components.sum(dim=-1, keepdim=True)
                if aggregation == "joint"
                else components
            )

        def action_entropies(self, logits):
            """Return decoder entropy with one entry per likelihood factor."""
            entropy = self._categorical(logits).entropy()
            return entropy.unsqueeze(-1) if single_leg else entropy

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
            """Return PPO outputs plus train-only future-market predictions."""
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
            action = self.actions_from_logits(
                logits,
                deterministic=deterministic,
            )
            return action, value, hidden

    return ActorCritic()
