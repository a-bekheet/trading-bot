from unittest import TestCase, skipUnless

try:
    import torch
except ImportError:
    torch = None

from trading_bot.training.recurrent import RecurrentConfig, build_recurrent_actor_critic


class RecurrentTests(TestCase):
    def test_positional_hidden_size_remains_backward_compatible(self):
        config = RecurrentConfig(5, 2, 3, 8)
        self.assertEqual(config.hidden_size, 8)
        self.assertIsNone(config.action_slot_count)

    @skipUnless(torch is not None, "install the optional ml extra")
    def test_recurrent_variants_have_safe_masked_action_shapes(self):
        for kind in ("gru", "lstm", "hybrid"):
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
