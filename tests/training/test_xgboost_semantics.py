"""Training semantics that are independent of Provider wire protocols."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from tributo.exceptions import ModelExportError, TrainingDataError
from tributo.training.progress import TrainingPhase
from tributo.training.xgboost_evaluator import split_dataset
from tributo.training.xgboost_trainer import (
    DataConfig,
    EvaluationConfig,
    OutputConfig,
    TrainingParams,
    XGBoostTrainerImpl,
    XGBoostTrainingConfig,
    _batch_evaluation_partial,
    _checkpoint_model_artifact,
    _compute_evaluation_metrics,
    _exact_rank_metrics_from_group_rows,
    _finalize_evaluation_partial,
    _merge_evaluation_partials,
    _merge_rank_partition_contributions,
    _rank_partition_contribution_batch,
    _rank_partition_prefixes,
    _should_extract_feature_importance,
    run_training_with_config,
)


def test_checkpoint_model_is_published_through_ray_object_store() -> None:
    model = MagicMock()
    model.save_raw.return_value = bytearray(b"portable-model")
    filesystem = object()
    checkpoint = SimpleNamespace(path="remote/checkpoint", filesystem=filesystem)
    typed_checkpoint = MagicMock()
    typed_checkpoint.get_model.return_value = model
    checkpoint_type = MagicMock(return_value=typed_checkpoint)
    xgboost_module = ModuleType("ray.train.xgboost")
    xgboost_module.XGBoostCheckpoint = checkpoint_type  # type: ignore[attr-defined]
    object_ref = object()

    with (
        patch.dict(sys.modules, {"ray.train.xgboost": xgboost_module}),
        patch("ray.put", return_value=object_ref) as put,
    ):
        assert _checkpoint_model_artifact(checkpoint) is object_ref

    checkpoint_type.assert_called_once_with("remote/checkpoint", filesystem=filesystem)
    put.assert_called_once_with(b"portable-model")


def test_random_split_uses_row_shuffle_not_block_order_only() -> None:
    shuffled = MagicMock()
    shuffled.split_proportionately.return_value = ["train", "val"]
    ds = MagicMock()
    ds.random_shuffle.return_value = shuffled

    assert split_dataset(ds, val_size=0.2, seed=7) == ("train", "val", None)
    ds.random_shuffle.assert_called_once_with(seed=7)
    ds.randomize_block_order.assert_not_called()


def test_time_ordered_split_sorts_by_required_column() -> None:
    ordered = MagicMock()
    ordered.split_proportionately.return_value = ["train", "val", "test"]
    ds = MagicMock()
    ds.sort.return_value = ordered

    result = split_dataset(
        ds,
        val_size=0.2,
        test_size=0.2,
        strategy="TIME_ORDERED",
        order_column="event_time",
    )

    assert result == ("train", "val", "test")
    ds.sort.assert_called_once_with("event_time")


def test_stratified_three_way_split_keeps_stratification_for_both_cuts() -> None:
    train_val = MagicMock()
    shuffled = MagicMock()
    reshuffled_train_val = MagicMock()
    ds = MagicMock()
    ds.random_shuffle.return_value = shuffled
    shuffled.train_test_split.return_value = (train_val, "test")
    train_val.random_shuffle.return_value = reshuffled_train_val
    reshuffled_train_val.train_test_split.return_value = ("train", "val")

    result = split_dataset(
        ds,
        val_size=0.2,
        test_size=0.2,
        seed=11,
        stratify=True,
        label_col="label",
    )

    assert result == ("train", "val", "test")
    ds.random_shuffle.assert_called_once_with(seed=11)
    shuffled.train_test_split.assert_called_once_with(
        test_size=0.2,
        stratify="label",
    )
    train_val.random_shuffle.assert_called_once_with(seed=12)
    reshuffled_train_val.train_test_split.assert_called_once_with(
        test_size=0.25,
        stratify="label",
    )


def test_stratified_two_way_never_passes_ray_incompatible_shuffle_flag() -> None:
    shuffled = MagicMock()
    shuffled.train_test_split.return_value = ("train", "val")
    ds = MagicMock()
    ds.random_shuffle.return_value = shuffled

    result = split_dataset(ds, val_size=0.2, seed=5, stratify=True)

    assert result == ("train", "val", None)
    shuffled.train_test_split.assert_called_once_with(
        test_size=0.2,
        stratify="label",
    )


def test_time_ordered_requires_order_column_and_rejects_stratify() -> None:
    with pytest.raises(ValidationError, match="order_column"):
        TrainingParams(split_strategy="TIME_ORDERED")
    with pytest.raises(ValidationError, match="stratify"):
        TrainingParams(
            split_strategy="TIME_ORDERED",
            order_column="event_time",
            stratify=True,
        )


def test_binary_multiclass_and_regression_metrics_are_task_correct() -> None:
    binary = _compute_evaluation_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.2, 0.7]),
        "binary:logistic",
        EvaluationConfig(),
    )
    assert binary["eval_test_rows"] == 4
    assert "eval_auc" in binary and "eval_rmse" not in binary

    multiclass = _compute_evaluation_metrics(
        np.array([0, 1, 2]),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        "multi:softprob",
        EvaluationConfig(),
    )
    assert multiclass["eval_test_rows"] == 3
    assert multiclass["eval_cm_labels"] == ["0", "1", "2"]

    regression = _compute_evaluation_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 1.9, 3.2]),
        "reg:squarederror",
        EvaluationConfig(),
    )
    assert "eval_rmse" in regression and "eval_r2" in regression
    assert "eval_auc" not in regression and "eval_cm" not in regression


def test_multiclass_softmax_and_missing_true_class_use_union_labels() -> None:
    softmax = _compute_evaluation_metrics(
        np.array([0, 0, 1]),
        np.array([0, 2, 2]),
        "multi:softmax",
        EvaluationConfig(),
        num_class=3,
    )
    assert softmax["eval_cm_labels"] == ["0", "1", "2"]
    assert softmax["eval_cm"] == [[1, 0, 1], [0, 0, 1], [0, 0, 0]]
    assert "eval_auc" not in softmax

    softprob_missing_class = _compute_evaluation_metrics(
        np.array([0, 0, 1]),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.2, 0.7]]),
        "multi:softprob",
        EvaluationConfig(),
        num_class=3,
    )
    assert softprob_missing_class["eval_cm_labels"] == ["0", "1", "2"]
    assert len(softprob_missing_class["eval_cm"]) == 3


def test_bounded_partials_merge_binary_and_softmax_without_row_data() -> None:
    evaluation = EvaluationConfig(roc_curve=False, threshold_analysis=False)
    combined = None
    for truth, prediction in (
        (np.array([0, 1]), np.array([0.1, 0.8])),
        (np.array([0, 1]), np.array([0.2, 0.7])),
    ):
        combined = _merge_evaluation_partials(
            combined,
            _batch_evaluation_partial(
                truth,
                prediction,
                "binary:logistic",
                num_class=None,
                config=evaluation,
            ),
        )
    assert combined is not None
    binary = _finalize_evaluation_partial(combined, evaluation)
    assert binary["eval_test_rows"] == 4
    assert binary["eval_f1"] == 1.0

    softmax = _batch_evaluation_partial(
        np.array([0, 1]),
        np.array([2, 2]),
        "multi:softmax",
        num_class=3,
        config=evaluation,
    )
    assert set(softmax) == {"task", "rows", "num_class", "confusion"}
    finalized = _finalize_evaluation_partial(softmax, evaluation)
    assert finalized["eval_cm_labels"] == ["0", "1", "2"]
    assert "eval_auc" not in finalized


def test_exact_distributed_auc_does_not_collapse_close_scores_into_bins() -> None:
    metrics = _exact_rank_metrics_from_group_rows(
        iter(
            [
                {"class_index": 0, "score": 0.501, "positive": 1, "negative": 0},
                {"class_index": 0, "score": 0.500, "positive": 0, "negative": 1},
            ]
        ),
        {0: (1, 1)},
        multiclass=False,
        include_curve=True,
    )

    assert metrics["eval_auc"] == 1.0
    assert metrics["eval_avg_precision"] == 1.0


def test_partition_prefix_reduction_preserves_exact_close_score_order() -> None:
    import pandas as pd

    summaries = [
        {
            "partition_id": "high",
            "class_index": 0,
            "first_score": 0.501,
            "positive_total": 1,
            "negative_total": 0,
        },
        {
            "partition_id": "low",
            "class_index": 0,
            "first_score": 0.500,
            "positive_total": 0,
            "negative_total": 1,
        },
    ]
    packed = pd.DataFrame(
        [
            {
                **summaries[0],
                "scores": [0.501],
                "positives": [1],
                "negatives": [0],
            },
            {
                **summaries[1],
                "scores": [0.500],
                "positives": [0],
                "negatives": [1],
            },
        ]
    )
    totals = {0: (1, 1)}

    contributions = _rank_partition_contribution_batch(
        packed,
        prefixes=_rank_partition_prefixes(summaries),
        totals=totals,
        include_curve=True,
    )
    metrics = _merge_rank_partition_contributions(
        contributions.to_dict("records"),
        totals,
        multiclass=False,
        include_curve=True,
    )

    assert metrics["eval_auc"] == 1.0
    assert metrics["eval_avg_precision"] == 1.0


def test_disabled_evaluation_skips_all_metric_objects() -> None:
    assert (
        _compute_evaluation_metrics(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            "binary:logistic",
            EvaluationConfig(enabled=False),
        )
        == {}
    )
    assert not _should_extract_feature_importance(
        {"evaluation": {"enabled": False, "feature_importance": True}}
    )


def test_artifact_flags_omit_large_evaluation_facts() -> None:
    metrics = _compute_evaluation_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.2, 0.7]),
        "binary:logistic",
        EvaluationConfig(
            roc_curve=False,
            threshold_analysis=False,
            confusion_matrix=False,
        ),
    )
    assert not any(key.startswith("eval_roc_") for key in metrics)
    assert not any(key.startswith("eval_thr_") for key in metrics)
    assert not any(key.startswith("eval_cm_") for key in metrics)


def test_split_smaller_than_worker_count_fails_before_training_submission() -> None:
    trainer = XGBoostTrainerImpl(
        datasets={},
        config={"ray": {"num_workers": 2}},
    )
    trainer.datasets = {"train": MagicMock(), "val": MagicMock()}
    trainer.datasets["train"].count.return_value = 18
    trainer.datasets["val"].count.return_value = 1

    with pytest.raises(TrainingDataError, match="val.*2 workers"):
        trainer._validate_and_partition_splits()


def test_duplicate_entity_and_custom_amount_are_warning_facts_not_mutations() -> None:
    trainer = XGBoostTrainerImpl(
        datasets={},
        config={
            "data": {
                "entity_key_column": "account_id",
                "class_balance_strategy": "CUSTOM_AMOUNT",
                "class_amounts": {"0": 10, "1": 10},
            }
        },
    )
    ds = MagicMock()
    with (
        patch.object(trainer, "_count_duplicates", return_value=2),
        patch.object(trainer, "_count_by_label", return_value={"0": 10, "1": 7}),
    ):
        trainer._collect_data_warnings(ds, {"account_id", "label"})

    assert trainer._warnings == [
        {
            "code": "DUPLICATE_ENTITY_KEYS",
            "column": "account_id",
            "duplicate_rows": 2,
        },
        {
            "code": "CLASS_AMOUNT_MISMATCH",
            "label": "1",
            "expected": 10,
            "actual": 7,
            "deviation": pytest.approx(0.3),
        },
    ]
    ds.drop_duplicates.assert_not_called()


def test_duplicate_count_uses_distributed_scalar_counts_only() -> None:
    ds = MagicMock()
    selected = MagicMock()
    grouped = MagicMock()
    distinct = MagicMock()
    ds.count.return_value = 10
    ds.select_columns.return_value = selected
    selected.groupby.return_value = grouped
    grouped.count.return_value = distinct
    distinct.count.return_value = 7

    assert XGBoostTrainerImpl._count_duplicates(ds, "account_id") == 3
    selected.groupby.assert_called_once_with("account_id")
    distinct.take_all.assert_not_called()


def test_export_is_strict_and_wraps_failure() -> None:
    ds = MagicMock()
    ds.schema.return_value.names = ["x", "label"]
    trainer = XGBoostTrainerImpl(
        datasets={"train": ds},
        config={"output": {"onnx_path": "model.onnx"}},
    )
    trainer._summary = {"status": "succeeded"}
    result = SimpleNamespace(metrics={"n_features": 1}, checkpoint=MagicMock())

    with (
        patch(
            "tributo.training.xgboost_exporter.export_onnx",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(ModelExportError, match="model.onnx"),
    ):
        trainer.export_model(result, "model.onnx")


def test_neutral_summary_has_artifact_and_global_facts() -> None:
    ds = MagicMock()
    ds.schema.return_value.names = ["x", "label"]
    trainer = XGBoostTrainerImpl(
        datasets={"train": ds},
        config={
            "data": {"feature_id_map": {"x": "feature-1"}},
            "output": OutputConfig(onnx_path="model.onnx").model_dump(),
        },
    )
    trainer._summary = {"status": "succeeded"}
    trainer._warnings = [{"code": "EXAMPLE"}]
    result = SimpleNamespace(
        metrics={
            "train-logloss": 0.2,
            "train-logloss_history": [0.5, 0.2],
            "row_count_train": 16,
            "eval_auc": 0.9,
            "n_features": 1,
        },
        checkpoint=object(),
    )
    with patch(
        "tributo.training.xgboost_exporter.export_onnx",
        return_value=("model.onnx", "abc123", 456),
    ):
        trainer.export_model(result, "model.onnx")

    assert trainer._summary["final_metrics"]["train-logloss"] == 0.2
    assert trainer._summary["histories"]["train-logloss"] == [0.5, 0.2]
    assert trainer._summary["row_counts"] == {"train": 16}
    assert trainer._summary["evaluation"]["eval_auc"] == 0.9
    assert trainer._summary["artifact_refs"] == [
        {
            "kind": "onnx",
            "uri": "model.onnx",
            "sha256": "abc123",
            "size_bytes": 456,
        }
    ]
    assert trainer._summary["warnings"] == [{"code": "EXAMPLE"}]


def test_data_semantic_fields_survive_round_trip() -> None:
    raw = {
        "entity_key_column": "account_id",
        "class_balance_strategy": "CUSTOM_AMOUNT",
        "class_amounts": {"0": 10, "1": 12},
    }
    dumped = DataConfig.model_validate(raw).model_dump()
    for key, value in raw.items():
        assert dumped[key] == value


def test_existing_callbacks_are_preserved_when_progress_is_enabled() -> None:
    callback = MagicMock()
    progress = MagicMock()
    trainer = XGBoostTrainerImpl(
        datasets={},
        config=XGBoostTrainingConfig().model_dump(),
        callbacks=[callback],
        _progress=progress,
    )

    assert trainer._callbacks == [callback]
    assert trainer._progress is progress


def test_real_driver_boundaries_emit_the_five_phases_in_order() -> None:
    progress = MagicMock()
    dataset = MagicMock()
    dataset.schema.return_value.names = ["x", "label"]
    dataset.count.return_value = 8
    dataset.repartition.return_value = dataset
    dataset.select_columns.return_value = dataset
    result = SimpleNamespace(metrics={"n_features": 1}, checkpoint=object())
    ray_trainer = MagicMock()
    ray_trainer.fit.return_value = result

    with (
        patch(
            "tributo.training.xgboost_trainer.TrainingProgress.from_environment",
            return_value=progress,
        ),
        patch(
            "tributo.training.data_loader.load_ray_dataset_from_config",
            return_value=dataset,
        ),
        patch(
            "tributo.training.xgboost_evaluator.filter_invalid_labels",
            side_effect=lambda ds, label_col: ds,
        ),
        patch(
            "tributo.training.xgboost_evaluator.split_dataset",
            return_value=(dataset, None, dataset),
        ),
        patch(
            "tributo.training.xgboost_trainer.build_trainer",
            return_value=ray_trainer,
        ),
        patch.object(
            XGBoostTrainerImpl,
            "_evaluate_full_test",
            return_value={"eval_test_rows": 8},
        ),
    ):
        run_training_with_config(
            {
                "data": {"feature_columns": ["x"]},
                "training": {"val_size": 0, "test_size": 0.25},
                "ray": {"num_workers": 2},
            }
        )

    assert [call.args[0] for call in progress.report_phase.call_args_list] == [
        TrainingPhase.LOADING_DATA,
        TrainingPhase.FEATURE_ENGINEERING,
        TrainingPhase.DATA_SPLITTING,
        TrainingPhase.TRAINING,
        TrainingPhase.EVALUATING,
    ]


def test_evaluating_phase_is_omitted_when_evaluation_is_disabled() -> None:
    progress = MagicMock()
    train = MagicMock()
    train.count.return_value = 10
    result = SimpleNamespace(metrics={}, checkpoint=MagicMock())
    ray_trainer = MagicMock()
    ray_trainer.fit.return_value = result
    trainer = XGBoostTrainerImpl(
        datasets={"train": train, "test": MagicMock()},
        config={"evaluation": {"enabled": False}},
        _progress=progress,
    )

    with patch(
        "tributo.training.xgboost_trainer.build_trainer", return_value=ray_trainer
    ):
        trainer.training_loop()

    assert [call.args[0] for call in progress.report_phase.call_args_list] == [
        TrainingPhase.TRAINING
    ]


def test_training_loop_overrides_worker_shard_values_with_global_driver_facts() -> None:
    progress = MagicMock()
    splits = {}
    for name, count in {"train": 160, "val": 20, "test": 20}.items():
        dataset = MagicMock(name=name)
        dataset.count.return_value = count
        splits[name] = dataset
    result = SimpleNamespace(
        metrics={
            "eval_test_rows": 3,
            "row_count_train": 80,
        },
        checkpoint=MagicMock(name="checkpoint"),
    )
    ray_trainer = MagicMock()
    ray_trainer.fit.return_value = result
    trainer = XGBoostTrainerImpl(
        datasets=splits,
        config={"ray": {"num_workers": 2}},
        _progress=progress,
    )

    with (
        patch(
            "tributo.training.xgboost_trainer.build_trainer",
            return_value=ray_trainer,
        ),
        patch.object(
            trainer,
            "_evaluate_full_test",
            return_value={"eval_test_rows": 20, "eval_auc": 0.8},
        ) as evaluate,
    ):
        returned = trainer.training_loop()

    assert returned.metrics["eval_test_rows"] == 20
    assert returned.metrics["row_count_train"] == 160
    assert returned.metrics["row_count_val"] == 20
    assert returned.metrics["row_count_test"] == 20
    evaluate.assert_called_once_with(result.checkpoint)


def test_driver_evaluation_streams_bounded_partials_without_to_pandas() -> None:
    partials = MagicMock()
    partials.iter_rows.return_value = iter(
        [
            {
                "partial": {
                    "task": "regression",
                    "rows": 2,
                    "sum_abs_error": 0.3,
                    "sum_squared_error": 0.05,
                    "sum_y": 3.0,
                    "sum_y_squared": 5.0,
                }
            },
            {
                "partial": {
                    "task": "regression",
                    "rows": 1,
                    "sum_abs_error": 0.2,
                    "sum_squared_error": 0.04,
                    "sum_y": 3.0,
                    "sum_y_squared": 9.0,
                }
            },
        ]
    )
    test_dataset = MagicMock()
    lazy_predictions = MagicMock()
    predictions = MagicMock()
    test_dataset.map_batches.return_value = lazy_predictions
    lazy_predictions.materialize.return_value = predictions
    predictions.map_batches.return_value = partials
    trainer = XGBoostTrainerImpl(
        datasets={"test": test_dataset},
        config={"model": {"objective": "reg:squarederror"}},
        _progress=MagicMock(),
    )
    checkpoint = MagicMock()

    model_artifact = object()
    with patch(
        "tributo.training.xgboost_trainer._checkpoint_model_artifact",
        return_value=model_artifact,
    ):
        metrics = trainer._evaluate_full_test(checkpoint)

    assert metrics["eval_test_rows"] == 3
    assert metrics["eval_mae"] == pytest.approx(0.5 / 3)
    partials.to_pandas.assert_not_called()
    lazy_predictions.materialize.assert_called_once_with()
    kwargs = test_dataset.map_batches.call_args.kwargs
    assert kwargs["fn_kwargs"]["model_artifact"] is model_artifact


def test_production_driver_path_computes_exact_auc_from_sorted_score_groups() -> None:
    partials = MagicMock()
    partials.iter_rows.return_value = iter(
        [
            {
                "partial": {
                    "task": "binary",
                    "rows": 2,
                    "tn": 1,
                    "fp": 0,
                    "fn": 0,
                    "tp": 1,
                    "threshold_counts": [[1, 0, 0, 1]] * 19,
                }
            }
        ]
    )
    contribution_rows = MagicMock()
    contribution_rows.iter_rows.return_value = iter(
        [
            {
                "class_index": 0,
                "first_score": 0.501,
                "auc": 1.0,
                "ap": 1.0,
                "curve": [[1.0, 1.0]],
            }
        ]
    )
    packed = MagicMock()
    packed.select_columns.return_value.take_all.return_value = [
        {
            "partition_id": "high",
            "class_index": 0,
            "first_score": 0.501,
            "positive_total": 1,
            "negative_total": 0,
        },
        {
            "partition_id": "low",
            "class_index": 0,
            "first_score": 0.500,
            "positive_total": 0,
            "negative_total": 1,
        },
    ]
    packed.map_batches.return_value = contribution_rows
    packed_lazy = MagicMock()
    packed_lazy.materialize.return_value = packed
    sorted_groups = MagicMock()
    sorted_groups.map_batches.return_value = packed_lazy
    aggregated = MagicMock()
    aggregated.sort.return_value = sorted_groups
    grouped = MagicMock()
    grouped.aggregate.return_value = aggregated
    score_rows = MagicMock()
    score_rows.groupby.return_value = grouped
    lazy_predictions = MagicMock()
    predictions = MagicMock()
    predictions.map_batches.side_effect = [partials, score_rows]
    lazy_predictions.materialize.return_value = predictions
    test_dataset = MagicMock()
    test_dataset.map_batches.return_value = lazy_predictions
    trainer = XGBoostTrainerImpl(
        datasets={"test": test_dataset},
        config={
            "model": {"objective": "binary:logistic"},
            "evaluation": {"threshold_analysis": True},
        },
        _progress=MagicMock(),
    )

    model_artifact = object()
    with patch(
        "tributo.training.xgboost_trainer._checkpoint_model_artifact",
        return_value=model_artifact,
    ):
        metrics = trainer._evaluate_full_test(MagicMock())

    assert metrics["eval_auc"] == 1.0
    assert metrics["eval_avg_precision"] == 1.0
    lazy_predictions.materialize.assert_called_once_with()
    sorted_groups.iter_rows.assert_not_called()
    packed.select_columns.assert_called_once_with(
        [
            "partition_id",
            "class_index",
            "first_score",
            "positive_total",
            "negative_total",
        ]
    )
