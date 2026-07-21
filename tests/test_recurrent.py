from unittest import TestCase, skipUnless

try:
    import torch
except ImportError:
    torch = None

from trading_bot.training.recurrent import RecurrentConfig, build_recurrent_actor_critic


class RecurrentTests(TestCase):
    @skipUnless(torch is not None, "install the optional ml extra")
    def test_recurrent_variants_have_safe_masked_action_shapes(self):
        for kind in ("gru", "lstm", "hybrid"):
            model = build_recurrent_actor_critic(RecurrentConfig(5, 2, 3, hidden_size=8, kind=kind))
            sequence = torch.zeros(4, 6, 5)
            mask = torch.ones(4, 2, 3, dtype=torch.bool)
            mask[:, 0, 0] = False
            mask[:, 1, :] = False
            logits, value, _ = model(sequence, mask)
            self.assertEqual(tuple(logits.shape), (4, 2, 3))
            self.assertEqual(tuple(value.shape), (4,))
            self.assertTrue(torch.isneginf(logits[:, 0, 0]).all())
            self.assertTrue(torch.isfinite(logits[:, 1, 0]).all())
            self.assertTrue(torch.isneginf(logits[:, 1, 1:]).all())
            action, _, _ = model.sample_action(sequence, mask)
            self.assertTrue((action[:, 1] == 0).all())
