"""XGBoost training evaluation utilities: data filtering, splitting, metric extraction."""

from __future__ import annotations

import logging
from typing import Any, Literal

import ray.data

logger = logging.getLogger(__name__)


def filter_invalid_labels(
    ds: ray.data.Dataset,
    label_col: str = "label",
    invalid_value: int = -1,
) -> ray.data.Dataset:
    """Filter samples with invalid labels.

    Args:
        ds: Input dataset.
        label_col: Label column name.
        invalid_value: Label value to filter out.

    Returns:
        Filtered dataset.
    """
    import pyarrow as pa

    def _filter(batch: pa.Table) -> pa.Table:
        mask = batch.column(label_col).to_numpy() != invalid_value
        return batch.filter(mask)

    before = ds.count()
    ds = ds.map_batches(_filter, batch_format="pyarrow")
    after = ds.count()
    logger.info(
        "Label filter applied (label==%s): %d rows before, %d rows after",
        invalid_value,
        before,
        after,
    )
    return ds


def split_dataset(
    ds: ray.data.Dataset,
    val_size: float = 0.2,
    test_size: float = 0.0,
    seed: int = 42,
    strategy: Literal["RANDOM", "TIME_ORDERED"] = "RANDOM",
    stratify: bool = False,
    label_col: str = "label",
    order_column: str | None = None,
) -> tuple[ray.data.Dataset, ray.data.Dataset | None, ray.data.Dataset | None]:
    """Three-way split: training / validation / test sets.

    Args:
        ds: Input dataset.
        val_size: Validation set proportion, 0 means no split.
        test_size: Test set proportion, 0 means no split.
        seed: Random seed.
        strategy: Row-randomized or explicitly time/order-sorted splitting.
        stratify: Preserve classification label proportions across splits.
        label_col: Classification label used when ``stratify`` is enabled.
        order_column: Required ascending sort key for ``TIME_ORDERED``.

    Returns:
        (train_ds, val_ds, test_ds), with None for unused splits.
    """
    if val_size <= 0 and test_size <= 0:
        return ds, None, None

    if strategy == "TIME_ORDERED":
        if not order_column:
            raise ValueError("TIME_ORDERED split requires order_column")
        if stratify:
            raise ValueError("stratify is incompatible with TIME_ORDERED splitting")
        ds = ds.sort(order_column)
    elif stratify:
        # Ray 2.56+ rejects ``shuffle=True`` together with ``stratify``.
        # Shuffle records explicitly, then ask Ray only for stratification.
        ds = ds.random_shuffle(seed=seed)
        if test_size > 0 and val_size > 0:
            train_val, test_ds = ds.train_test_split(
                test_size=test_size,
                stratify=label_col,
            )
            adjusted_val_size = val_size / (1.0 - test_size)
            randomized_train_val = train_val.random_shuffle(seed=seed + 1)
            train_ds, val_ds = randomized_train_val.train_test_split(
                test_size=adjusted_val_size,
                stratify=label_col,
            )
            return train_ds, val_ds, test_ds

        split_size = val_size if val_size > 0 else test_size
        train_ds, held_out = ds.train_test_split(
            test_size=split_size,
            stratify=label_col,
        )
        if val_size > 0:
            return train_ds, held_out, None
        return train_ds, None, held_out
    else:
        # ``random_shuffle`` shuffles records globally; randomizing block order
        # alone preserves row ordering within every block and biases the split.
        ds = ds.random_shuffle(seed=seed)

    train_frac = 1.0 - val_size - test_size

    if test_size > 0 and val_size > 0:
        train_ds, val_ds, test_ds = ds.split_proportionately([train_frac, val_size])
        return train_ds, val_ds, test_ds
    elif val_size > 0:
        train_ds, val_ds = ds.split_proportionately([train_frac])
        return train_ds, val_ds, None
    else:
        train_ds, test_ds = ds.split_proportionately([train_frac])
        return train_ds, None, test_ds


def compute_metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert raw metrics reported by Ray Train into a serializable summary.

    Args:
        metrics: Metrics dictionary from ray.train.report.

    Returns:
        Processed metrics dictionary (lists kept as-is, others converted to float).
    """

    def _serialize(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, list):
            return [_serialize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _serialize(item) for key, item in value.items()}
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    return {key: _serialize(value) for key, value in metrics.items()}
