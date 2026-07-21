"""Leakage-resistant chronological dataset splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from trading_bot.training.dataset import SnapshotDataset


@dataclass(frozen=True)
class WalkForwardSplit:
    """Index boundaries for one expanding-window research fold."""

    fold: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int
    embargo: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def apply(
        self,
        dataset: SnapshotDataset,
    ) -> tuple[SnapshotDataset, SnapshotDataset, SnapshotDataset]:
        return (
            dataset.subset(self.train_start, self.train_end),
            dataset.subset(self.validation_start, self.validation_end),
            dataset.subset(self.test_start, self.test_end),
        )


def walk_forward_splits(
    length: int,
    *,
    min_train_size: int,
    validation_size: int,
    test_size: int,
    embargo: int = 0,
    step_size: int | None = None,
    max_train_size: int | None = None,
) -> tuple[WalkForwardSplit, ...]:
    """Build expanding or rolling chronological folds with partition embargoes."""
    sizes = (length, min_train_size, validation_size, test_size)
    if length < 1 or any(size < 1 for size in sizes[1:]):
        raise ValueError("dataset and partition sizes must be positive")
    if embargo < 0:
        raise ValueError("embargo cannot be negative")
    step = test_size if step_size is None else step_size
    if step < 1:
        raise ValueError("step_size must be positive")
    if max_train_size is not None and max_train_size < min_train_size:
        raise ValueError("max_train_size cannot be smaller than min_train_size")

    folds = []
    train_end = min_train_size
    while True:
        validation_start = train_end + embargo
        validation_end = validation_start + validation_size
        test_start = validation_end + embargo
        test_end = test_start + test_size
        if test_end > length:
            break
        train_start = (
            max(0, train_end - max_train_size)
            if max_train_size is not None
            else 0
        )
        folds.append(WalkForwardSplit(
            fold=len(folds),
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end,
            embargo=embargo,
        ))
        train_end += step
    return tuple(folds)
