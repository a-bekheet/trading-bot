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

    def test_rejects_invalid_sizes(self):
        with self.assertRaisesRegex(ValueError, "partition sizes"):
            walk_forward_splits(
                10,
                min_train_size=0,
                validation_size=2,
                test_size=2,
            )
