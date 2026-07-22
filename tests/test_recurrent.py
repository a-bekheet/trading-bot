from unittest import TestCase, skipUnless

try:
    import torch
except ImportError:
    torch = None

from trading_bot.training.recurrent import RecurrentConfig, build_recurrent_actor_critic


class RecurrentTests(TestCase):
    @staticmethod
    def graph_set_config(
        *,
        kind: str = "gru",
        graph_neighbors: int = 2,
        action_decoder: str = "factorized",
    ) -> RecurrentConfig:
        # market(2), three contracts(4 each), portfolio(3), valid mask(3)
        return RecurrentConfig(
            20,
            3,
            3,
            action_slot_count=4,
            hidden_size=8,
            kind=kind,
            encoder="graph_set",
            contract_feature_count=4,
            market_feature_count=2,
            portfolio_feature_count=3,
            graph_hidden_size=6,
            graph_layers=2,
            graph_neighbors=graph_neighbors,
            graph_relation_indices=(0, 1),
            auxiliary_target_count=5,
            action_decoder=action_decoder,
        )

    @classmethod
    def attention_set_config(
        cls,
        *,
        kind: str = "gru",
        action_decoder: str = "factorized",
    ) -> RecurrentConfig:
        return RecurrentConfig(**{
            **cls.graph_set_config(
                kind=kind,
                action_decoder=action_decoder,
            ).__dict__,
            "encoder": "attention_set",
            "attention_heads": 2,
        })

    def test_positional_hidden_size_remains_backward_compatible(self):
        config = RecurrentConfig(5, 2, 3, 8)
        self.assertEqual(config.hidden_size, 8)
        self.assertIsNone(config.action_slot_count)
        self.assertEqual(config.initial_hold_bias, 5.0)
        attention = RecurrentConfig(
            5,
            2,
            3,
            encoder="attention_set",
            graph_neighbors=3,
        )
        self.assertEqual(attention.graph_neighbors, 0)

    def test_rejects_invalid_sparse_policy_prior(self):
        with self.assertRaisesRegex(ValueError, "initial_hold_bias"):
            RecurrentConfig(5, 2, 3, initial_hold_bias=-1)
        with self.assertRaisesRegex(ValueError, "action_decoder"):
            RecurrentConfig(5, 2, 3, action_decoder="autoregressive")
        with self.assertRaisesRegex(ValueError, "non-hold"):
            RecurrentConfig(5, 2, 1, action_decoder="single_leg")

    def test_rejects_invalid_masked_input_indices(self):
        with self.assertRaisesRegex(ValueError, "masked_input_indices"):
            RecurrentConfig(5, 2, 3, masked_input_indices=(5,))
        with self.assertRaisesRegex(ValueError, "masked_input_indices"):
            RecurrentConfig(5, 2, 3, masked_input_indices=(1, 1))
        with self.assertRaisesRegex(ValueError, "disable every input"):
            RecurrentConfig(2, 1, 3, masked_input_indices=(0, 1))
        with self.assertRaisesRegex(ValueError, "auxiliary_target_count"):
            RecurrentConfig(5, 2, 3, auxiliary_target_count=-1)
        with self.assertRaisesRegex(ValueError, "auxiliary_horizons"):
            RecurrentConfig(5, 2, 3, auxiliary_horizons=(2, 1))
        with self.assertRaisesRegex(ValueError, "attention_heads"):
            RecurrentConfig(5, 2, 3, attention_heads=0)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_auxiliary_head_is_train_only_and_preserves_policy_outputs(self):
        torch.manual_seed(17)
        model = build_recurrent_actor_critic(RecurrentConfig(
            5,
            2,
            3,
            hidden_size=8,
            auxiliary_target_count=5,
        ))
        model.eval()
        sequence = torch.randn(2, 4, 5)
        mask = torch.ones(2, 4, 2, 3, dtype=torch.bool)
        auxiliary_calls = []
        handle = model.auxiliary.register_forward_hook(
            lambda *args: auxiliary_calls.append(1)
        )

        inference_logits, inference_values, _ = model.forward_sequence(
            sequence,
            mask,
        )
        self.assertEqual(auxiliary_calls, [])
        train_logits, train_values, predictions, _ = (
            model.forward_sequence_with_auxiliary(sequence, mask)
        )
        handle.remove()

        self.assertEqual(auxiliary_calls, [1])
        self.assertEqual(tuple(predictions.shape), (2, 4, 5))
        torch.testing.assert_close(inference_logits, train_logits)
        torch.testing.assert_close(inference_values, train_values)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_checkpointed_feature_mask_makes_disabled_inputs_invariant(self):
        torch.manual_seed(13)
        model = build_recurrent_actor_critic(RecurrentConfig(
            5,
            2,
            3,
            hidden_size=8,
            masked_input_indices=(1, 3),
        ))
        model.eval()
        sequence = torch.randn(2, 4, 5)
        changed = sequence.clone()
        changed[..., 1] = 1_000_000
        changed[..., 3] = -1_000_000
        mask = torch.ones(2, 4, 2, 3, dtype=torch.bool)

        first_logits, first_values, _ = model.forward_sequence(sequence, mask)
        second_logits, second_values, _ = model.forward_sequence(changed, mask)

        torch.testing.assert_close(first_logits, second_logits)
        torch.testing.assert_close(first_values, second_values)
        self.assertEqual(model.recurrent.input_size, 3)
        full = build_recurrent_actor_critic(RecurrentConfig(
            5,
            2,
            3,
            hidden_size=8,
        ))
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in full.parameters()),
        )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_recurrent_variants_have_safe_masked_action_shapes(self):
        for kind in ("gru", "lstm", "hybrid", "mixture"):
            model = build_recurrent_actor_critic(RecurrentConfig(
                5,
                2,
                3,
                action_slot_count=3,
                hidden_size=8,
                kind=kind,
            ))
            sequence = torch.zeros(4, 6, 5)
            mask = torch.ones(4, 3, 3, dtype=torch.bool)
            mask[:, 0, 0] = False
            mask[:, 1, :] = False
            logits, value, _ = model(sequence, mask)
            self.assertEqual(tuple(logits.shape), (4, 3, 3))
            self.assertEqual(tuple(value.shape), (4,))
            self.assertTrue(torch.isneginf(logits[:, 0, 0]).all())
            self.assertTrue(torch.isfinite(logits[:, 1, 0]).all())
            self.assertTrue(torch.isneginf(logits[:, 1, 1:]).all())
            action, _, _ = model.sample_action(sequence, mask)
            self.assertTrue((action[:, 1] == 0).all())

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_streaming_state_matches_causal_sequence_for_every_variant(self):
        torch.manual_seed(19)
        for kind in ("gru", "lstm", "hybrid", "mixture"):
            model = build_recurrent_actor_critic(RecurrentConfig(
                5,
                2,
                3,
                hidden_size=8,
                kind=kind,
            ))
            model.eval()
            sequence = torch.randn(2, 6, 5)
            masks = torch.ones(2, 6, 2, 3, dtype=torch.bool)
            masks[:, :, 0, 2] = False

            full_logits, full_values, _ = model.forward_sequence(sequence, masks)
            hidden = None
            streamed_logits = []
            streamed_values = []
            for step in range(sequence.shape[1]):
                logits, values, hidden = model(
                    sequence[:, step:step + 1],
                    masks[:, step],
                    hidden_state=hidden,
                )
                streamed_logits.append(logits)
                streamed_values.append(values)

            torch.testing.assert_close(
                torch.stack(streamed_logits, dim=1),
                full_logits,
            )
            torch.testing.assert_close(
                torch.stack(streamed_values, dim=1),
                full_values,
            )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_mixture_starts_balanced_and_trains_both_recurrent_experts(self):
        torch.manual_seed(23)
        model = build_recurrent_actor_critic(RecurrentConfig(
            5,
            2,
            3,
            hidden_size=8,
            kind="mixture",
        ))
        sequence = torch.randn(2, 4, 5)
        mask = torch.ones(2, 4, 2, 3, dtype=torch.bool)
        gate_logits = []
        handle = model.mixture_gate.register_forward_hook(
            lambda _module, _inputs, output: gate_logits.append(output.detach())
        )

        logits, values, _ = model.forward_sequence(sequence, mask)
        handle.remove()
        loss = logits.square().mean() + values.square().mean()
        loss.backward()

        self.assertEqual(len(gate_logits), 1)
        torch.testing.assert_close(
            torch.sigmoid(gate_logits[0]),
            torch.full((2, 4, 1), 0.5),
        )
        self.assertIsNotNone(model.mixture_gate.weight.grad)
        self.assertIsNotNone(model.recurrent["gru"].weight_ih_l0.grad)
        self.assertIsNotNone(model.recurrent["lstm"].weight_ih_l0.grad)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_mixture_has_fewer_parameters_than_concatenated_hybrid(self):
        for encoder in ("flat", "graph_set"):
            kwargs = (
                {}
                if encoder == "flat"
                else {
                    "encoder": "graph_set",
                    "contract_feature_count": 4,
                    "market_feature_count": 2,
                    "portfolio_feature_count": 3,
                    "action_slot_count": 3,
                    "graph_hidden_size": 4,
                    "graph_layers": 1,
                    "graph_neighbors": 0,
                }
            )
            input_size = 5 if encoder == "flat" else 15
            hybrid = build_recurrent_actor_critic(RecurrentConfig(
                input_size,
                2,
                3,
                hidden_size=8,
                kind="hybrid",
                **kwargs,
            ))
            mixture = build_recurrent_actor_critic(RecurrentConfig(
                input_size,
                2,
                3,
                hidden_size=8,
                kind="mixture",
                **kwargs,
            ))
            self.assertLess(
                sum(parameter.numel() for parameter in mixture.parameters()),
                sum(parameter.numel() for parameter in hybrid.parameters()),
            )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_hold_bias_makes_untrained_large_policy_sparse(self):
        torch.manual_seed(23)
        sparse = build_recurrent_actor_critic(RecurrentConfig(
            5,
            32,
            7,
            action_slot_count=33,
            hidden_size=8,
            initial_hold_bias=5.0,
        ))
        dense = build_recurrent_actor_critic(RecurrentConfig(
            5,
            32,
            7,
            action_slot_count=33,
            hidden_size=8,
            initial_hold_bias=0.0,
        ))
        sequence = torch.zeros(1_024, 1, 5)
        mask = torch.ones(1_024, 33, 7, dtype=torch.bool)

        sparse_actions, _, _ = sparse.sample_action(sequence, mask)
        dense_actions, _, _ = dense.sample_action(sequence, mask)
        sparse_orders = (sparse_actions != 0).sum(dim=1).float().mean()
        dense_orders = (dense_actions != 0).sum(dim=1).float().mean()
        bias = sparse.policy.bias.view(33, 7)

        self.assertLess(float(sparse_orders), 2.0)
        self.assertGreater(float(dense_orders), 20.0)
        torch.testing.assert_close(bias[:, 0], torch.full((33,), 5.0))
        torch.testing.assert_close(bias[:, 1:], torch.zeros(33, 6))

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_single_leg_decoder_has_exact_joint_likelihood_and_masks(self):
        torch.manual_seed(29)
        model = build_recurrent_actor_critic(RecurrentConfig(
            5,
            2,
            3,
            action_slot_count=3,
            hidden_size=8,
            action_decoder="single_leg",
        ))
        sequence = torch.zeros(256, 1, 5)
        mask = torch.ones(256, 3, 3, dtype=torch.bool)
        mask[:, 0, 1] = False

        logits, _, _ = model(sequence, mask)
        actions = model.actions_from_logits(logits)
        log_probabilities = model.action_log_probabilities(logits, actions)
        entropies = model.action_entropies(logits)

        self.assertEqual(tuple(logits.shape), (256, 7))
        self.assertTrue(torch.isneginf(logits[:, 1]).all())
        self.assertEqual(tuple(actions.shape), (256, 3))
        self.assertTrue(((actions != 0).sum(dim=-1) <= 1).all())
        self.assertEqual(tuple(log_probabilities.shape), (256, 1))
        self.assertEqual(tuple(entropies.shape), (256, 1))

        selected = torch.tensor([[0, 2, 0]])
        selected_logits = logits[:1]
        expected = torch.log_softmax(selected_logits, dim=-1)[:, 4]
        torch.testing.assert_close(
            model.action_log_probabilities(selected_logits, selected)[:, 0],
            expected,
        )
        with self.assertRaisesRegex(ValueError, "at most one"):
            model.action_log_probabilities(
                selected_logits,
                torch.tensor([[1, 0, 2]]),
            )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_graph_encoder_masks_padded_contracts(self):
        # Layout: market(2), two contracts with three features, portfolio(3), valid(2).
        config = RecurrentConfig(
            13,
            2,
            3,
            hidden_size=8,
            encoder="graph",
            contract_feature_count=3,
            graph_hidden_size=4,
        )
        model = build_recurrent_actor_critic(config)
        sequence = torch.zeros(2, 3, 13)
        sequence[:, :, -2] = 1  # only the first contract is valid
        changed_padding = sequence.clone()
        changed_padding[:, :, 5:8] = 1_000_000
        mask = torch.ones(2, 2, 3, dtype=torch.bool)

        first_logits, first_value, _ = model(sequence, mask)
        second_logits, second_value, _ = model(changed_padding, mask)

        torch.testing.assert_close(first_logits, second_logits)
        torch.testing.assert_close(first_value, second_value)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_graph_encoder_validates_observation_layout(self):
        config = RecurrentConfig(
            12,
            2,
            3,
            encoder="graph",
            contract_feature_count=3,
        )
        with self.assertRaisesRegex(ValueError, "input_size does not match"):
            build_recurrent_actor_critic(config)
        with self.assertRaisesRegex(ValueError, "one action row"):
            build_recurrent_actor_critic(RecurrentConfig(
                20,
                3,
                3,
                action_slot_count=3,
                encoder="graph_set",
                contract_feature_count=4,
                market_feature_count=2,
                portfolio_feature_count=3,
            ))
        with self.assertRaisesRegex(ValueError, "divisible"):
            build_recurrent_actor_critic(RecurrentConfig(
                20,
                3,
                3,
                action_slot_count=4,
                encoder="attention_set",
                contract_feature_count=4,
                market_feature_count=2,
                portfolio_feature_count=3,
                graph_hidden_size=6,
                attention_heads=4,
            ))

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_set_encoders_are_permutation_equivariant_with_invariant_value(self):
        sequence = torch.randn(2, 5, 20)
        sequence[..., -3:] = 1
        action_mask = torch.ones(2, 5, 4, 3, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 1])
        changed = sequence.clone()
        contracts = sequence[..., 2:14].view(2, 5, 3, 4)
        changed[..., 2:14] = contracts[:, :, permutation].reshape(2, 5, 12)
        changed[..., -3:] = sequence[..., -3:][:, :, permutation]
        changed_mask = torch.cat(
            (action_mask[:, :, permutation], action_mask[:, :, 3:4]),
            dim=2,
        )

        for config in (
            self.graph_set_config(),
            self.attention_set_config(),
        ):
            with self.subTest(encoder=config.encoder):
                torch.manual_seed(31)
                model = build_recurrent_actor_critic(config)
                model.eval()
                logits, values, auxiliary, _ = (
                    model.forward_sequence_with_auxiliary(
                        sequence,
                        action_mask,
                    )
                )
                changed_logits, changed_values, changed_auxiliary, _ = (
                    model.forward_sequence_with_auxiliary(
                        changed,
                        changed_mask,
                    )
                )

                torch.testing.assert_close(
                    changed_logits[:, :, :3],
                    logits[:, :, permutation],
                    rtol=1e-5,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    changed_logits[:, :, 3],
                    logits[:, :, 3],
                    rtol=1e-5,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    changed_values,
                    values,
                    rtol=1e-5,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    changed_auxiliary,
                    auxiliary,
                    rtol=1e-5,
                    atol=1e-6,
                )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_single_leg_set_encoders_preserve_permutation_symmetry(self):
        sequence = torch.randn(2, 4, 20)
        sequence[..., -3:] = 1
        mask = torch.ones(2, 4, 4, 3, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 1])
        changed = sequence.clone()
        contracts = sequence[..., 2:14].view(2, 4, 3, 4)
        changed[..., 2:14] = contracts[:, :, permutation].reshape(2, 4, 12)
        changed[..., -3:] = sequence[..., -3:][:, :, permutation]
        changed_mask = torch.cat((mask[:, :, permutation], mask[:, :, 3:4]), dim=2)

        for config in (
            self.graph_set_config(action_decoder="single_leg"),
            self.attention_set_config(action_decoder="single_leg"),
        ):
            with self.subTest(encoder=config.encoder):
                torch.manual_seed(35)
                model = build_recurrent_actor_critic(config)
                model.eval()
                logits, values, _ = model.forward_sequence(sequence, mask)
                changed_logits, changed_values, _ = model.forward_sequence(
                    changed,
                    changed_mask,
                )
                rows = logits[..., 1:].view(2, 4, 4, 2)
                changed_rows = changed_logits[..., 1:].view(2, 4, 4, 2)

                torch.testing.assert_close(
                    changed_logits[..., 0],
                    logits[..., 0],
                )
                torch.testing.assert_close(
                    changed_rows[..., :3, :],
                    rows[..., permutation, :],
                )
                torch.testing.assert_close(
                    changed_rows[..., 3, :],
                    rows[..., 3, :],
                )
                torch.testing.assert_close(changed_values, values)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_set_encoders_ignore_padded_nodes_and_backpropagate_shared_head(self):
        sequence = torch.randn(2, 4, 20)
        sequence[..., -3:] = torch.tensor([1.0, 1.0, 0.0])
        changed = sequence.clone()
        changed[..., 10:14] = 1_000_000
        mask = torch.ones(2, 4, 4, 3, dtype=torch.bool)
        mask[:, :, 2, 1:] = False

        for config in (
            self.graph_set_config(),
            self.attention_set_config(),
        ):
            with self.subTest(encoder=config.encoder):
                torch.manual_seed(37)
                model = build_recurrent_actor_critic(config)
                logits, values, _ = model.forward_sequence(sequence, mask)
                changed_logits, changed_values, _ = model.forward_sequence(
                    changed,
                    mask,
                )

                torch.testing.assert_close(logits, changed_logits)
                torch.testing.assert_close(values, changed_values)
                loss = logits[..., :2, :].square().mean() + values.square().mean()
                loss.backward()
                self.assertIsNotNone(model.contract_policy.weight.grad)
                self.assertIsNotNone(model.recurrent.weight_ih_l0.grad)
                if config.encoder == "attention_set":
                    attention = model.attention_layers[0]["attention"]
                    self.assertIsNotNone(attention.in_proj_weight.grad)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_set_encoders_streaming_matches_full_sequence_for_every_variant(self):
        for encoder in ("graph_set", "attention_set"):
            for kind in ("gru", "lstm", "hybrid", "mixture"):
                with self.subTest(encoder=encoder, kind=kind):
                    torch.manual_seed(41)
                    config = (
                        self.graph_set_config(kind=kind)
                        if encoder == "graph_set"
                        else self.attention_set_config(kind=kind)
                    )
                    model = build_recurrent_actor_critic(config)
                    model.eval()
                    sequence = torch.randn(1, 5, 20)
                    sequence[..., -3:] = 1
                    masks = torch.ones(1, 5, 4, 3, dtype=torch.bool)
                    full_logits, full_values, _ = model.forward_sequence(
                        sequence,
                        masks,
                    )
                    hidden = None
                    streamed_logits = []
                    streamed_values = []
                    for step in range(sequence.shape[1]):
                        logits, values, hidden = model(
                            sequence[:, step:step + 1],
                            masks[:, step],
                            hidden_state=hidden,
                        )
                        streamed_logits.append(logits)
                        streamed_values.append(values)
                    torch.testing.assert_close(
                        torch.stack(streamed_logits, dim=1),
                        full_logits,
                    )
                    torch.testing.assert_close(
                        torch.stack(streamed_values, dim=1),
                        full_values,
                    )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_attention_set_is_finite_for_an_empty_surface(self):
        torch.manual_seed(42)
        model = build_recurrent_actor_critic(self.attention_set_config())
        sequence = torch.randn(2, 3, 20)
        sequence[..., -3:] = 0
        action_mask = torch.zeros(2, 3, 4, 3, dtype=torch.bool)
        action_mask[..., 0] = True

        logits, values, auxiliary, _ = (
            model.forward_sequence_with_auxiliary(sequence, action_mask)
        )

        self.assertTrue(torch.isfinite(logits[..., 0]).all())
        self.assertTrue(torch.isneginf(logits[..., 1:]).all())
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(auxiliary).all())

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_graph_set_uses_fewer_parameters_than_flattened_graph(self):
        graph_set_config = RecurrentConfig(
            165,
            32,
            7,
            action_slot_count=33,
            hidden_size=16,
            encoder="graph_set",
            contract_feature_count=4,
            market_feature_count=2,
            portfolio_feature_count=3,
            graph_hidden_size=8,
        )
        graph_set = build_recurrent_actor_critic(graph_set_config)
        flattened = build_recurrent_actor_critic(
            RecurrentConfig(**{
                **graph_set_config.__dict__,
                "encoder": "graph",
            })
        )

        self.assertLess(
            sum(parameter.numel() for parameter in graph_set.parameters()),
            sum(parameter.numel() for parameter in flattened.parameters()),
        )

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_zero_neighbor_graph_set_is_finite_and_removes_neighbor_path(self):
        torch.manual_seed(43)
        zero_neighbor = build_recurrent_actor_critic(
            self.graph_set_config(graph_neighbors=0)
        )
        full_graph = build_recurrent_actor_critic(
            self.graph_set_config(graph_neighbors=2)
        )
        self.assertFalse(any(
            ".neighbor." in name
            for name, _ in zero_neighbor.named_parameters()
        ))
        self.assertTrue(any(
            ".neighbor." in name
            for name, _ in full_graph.named_parameters()
        ))
        self.assertLess(
            sum(parameter.numel() for parameter in zero_neighbor.parameters()),
            sum(parameter.numel() for parameter in full_graph.parameters()),
        )

        sequence = torch.randn(2, 3, 20)
        sequence[..., -3:] = 0
        action_mask = torch.zeros(2, 3, 4, 3, dtype=torch.bool)
        action_mask[..., 0] = True
        logits, values, auxiliary, _ = (
            zero_neighbor.forward_sequence_with_auxiliary(
                sequence,
                action_mask,
            )
        )
        self.assertTrue(torch.isfinite(logits[..., 0]).all())
        self.assertTrue(torch.isneginf(logits[..., 1:]).all())
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(auxiliary).all())
