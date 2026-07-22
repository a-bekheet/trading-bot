from unittest import TestCase

import pandas as pd

from trading_bot.training.dataset import Snapshot, SnapshotDataset
from trading_bot.training.splits import walk_forward_splits


class WalkForwardSplitTests(TestCase):
    def test_builds_embargoed_expanding_folds(self):
        folds = walk_forward_splits(
            30,
            min_train_size=10,
            validation_size=4,
            test_size=3,
            embargo=2,
            step_size=3,
        )

        self.assertEqual(len(folds), 4)
        self.assertEqual(
            folds[0].to_dict(),
            {
                "fold": 0,
                "train_start": 0,
                "train_end": 10,
                "validation_start": 12,
                "validation_end": 16,
                "test_start": 18,
                "test_end": 21,
                "embargo": 2,
            },
        )
        self.assertEqual(folds[-1].train_end, 19)
        for fold in folds:
            self.assertLessEqual(fold.train_end + fold.embargo, fold.validation_start)
            self.assertLessEqual(
                fold.validation_end + fold.embargo,
                fold.test_start,
            )

        dataset = SnapshotDataset(
            tuple(Snapshot(str(index), pd.DataFrame()) for index in range(30)),
            "TEST",
        )
        train, validation, test = folds[0].apply(dataset)
        self.assertEqual(len(train), 10)
        self.assertEqual(validation.snapshots[0].timestamp, "12")
        self.assertEqual(test.snapshots[0].timestamp, "18")
        self.assertNotEqual(train.fingerprint, validation.fingerprint)
        self.assertNotEqual(validation.fingerprint, test.fingerprint)

    def test_supports_bounded_rolling_training_window(self):
        folds = walk_forward_splits(
            40,
            min_train_size=10,
            max_train_size=12,
            validation_size=2,
            test_size=2,
            step_size=5,
        )

        self.assertEqual(folds[1].train_start, 3)
        self.assertEqual(folds[1].train_end, 15)

    def test_latest_only_uses_all_available_pre_validation_history(self):
        folds = walk_forward_splits(
            53,
            min_train_size=10,
            max_train_size=20,
            validation_size=4,
            test_size=3,
            embargo=2,
            step_size=100,
            latest_only=True,
        )

        self.assertEqual(
            [fold.to_dict() for fold in folds],
            [{
                "fold": 0,
                "train_start": 22,
                "train_end": 42,
                "validation_start": 44,
                "validation_end": 48,
                "test_start": 50,
                "test_end": 53,
                "embargo": 2,
            }],
        )

    def test_latest_only_returns_no_fold_when_tail_leaves_too_little_training(self):
        self.assertEqual(
            walk_forward_splits(
                16,
                min_train_size=10,
                validation_size=3,
                test_size=2,
                embargo=1,
                latest_only=True,
            ),
            (),
        )

    def test_rejects_invalid_sizes(self):
        with self.assertRaisesRegex(ValueError, "partition sizes"):
            walk_forward_splits(
                10,
                min_train_size=0,
                validation_size=2,
                test_size=2,
            )
