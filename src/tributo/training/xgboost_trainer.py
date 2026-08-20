"""XGBoost distributed training entry point, based on Ray Train XGBoostTrainer."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tributo.data.base import S3Config
from tributo.exceptions import ModelExportError, TrainingDataError
from tributo.integrations.broker import CancellationChecker, EventReporter
from tributo.training.base import BaseTrainer, TrainerSpec
from tributo.training.execution_context import (
    ExecutionContext,
    TrainingCancelledError,
)
from tributo.training.progress import TrainingPhase, TrainingProgress
from tributo.training.registry import register
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data
    from ray.train.xgboost import XGBoostTrainer

logger = logging.getLogger(__name__)


# ── Pydantic config models ──


class DataConfig(BaseModel):
    """Data source configuration."""

    type: str = Field(
        default="csv", description="Data source type: csv | s3 | clickhouse | hive"
    )
    # Canonical database fields are interpreted according to ``type``. The
    # prefixed ch_*/hive_* forms remain supported for backwards compatibility.
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None
    auth: Literal["NONE", "NOSASL", "LDAP", "CUSTOM"] | None = None
    sql: str | None = None
    params: dict[str, Any] | None = None
    sort_key: str | None = None
    parallelism: int | None = None
    batch_size: int | None = None
    shard_mode: str | None = None
    hash_column: str | None = None
    hash_shards: int | None = None
    path: Optional[str] = Field(default=None, description="Local data path")
    uri: Optional[str] = Field(default=None, description="S3 URI")
    format: str = Field(default="parquet", description="Data format: csv | parquet")
    label_col: str = Field(default="label", description="Label column name")
    feature_columns: list[str] = Field(
        default_factory=list, description="Feature columns for training"
    )
    s3: S3Config = Field(default_factory=S3Config)
    feature_id_map: dict[str, str] = Field(default_factory=dict)
    # ── ClickHouse ──
    ch_host: str = ""
    ch_port: int = 9000
    ch_database: str = ""
    ch_user: str = "default"
    ch_password: str = ""
    ch_sql: str = ""
    ch_sql_params: dict[str, Any] = Field(default_factory=dict)
    ch_sort_key: str | None = None
    ch_parallelism: int | None = None
    # ── HiveServer2 ──
    hive_host: str = ""
    hive_port: int = 10000
    hive_database: str = "default"
    hive_user: str = "default"
    hive_password: str | None = None
    hive_auth: Literal["NONE", "NOSASL", "LDAP", "CUSTOM"] | None = None
    hive_sql: str = ""
    hive_sql_params: dict[str, Any] = Field(default_factory=dict)
    hive_batch_size: int = 10_000
    hive_shard_mode: str = "auto"
    hive_hash_column: str | None = None
    hive_hash_shards: int = 64
    hive_parallelism: int | None = None
    # ── Training semantics (broker-neutral protocol inputs) ──
    entity_key_column: str = ""
    class_balance_strategy: Literal["NONE", "CUSTOM_AMOUNT"] = "NONE"
    class_amounts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_class_amounts(self) -> DataConfig:
        if self.class_balance_strategy == "CUSTOM_AMOUNT" and not self.class_amounts:
            raise ValueError("CUSTOM_AMOUNT requires class_amounts")
        return self

    @field_validator("auth", "hive_auth", mode="before")
    @classmethod
    def normalize_hive_auth(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


class ModelConfig(BaseModel):
    """XGBoost model parameters."""

    model_config = ConfigDict(extra="allow")

    objective: str = Field(default="binary:logistic", description="Objective function")
    num_class: int | None = Field(
        default=None,
        ge=2,
        description="Number of multi-class categories, only needed for multi:softprob/multi:softmax",
    )

    @model_validator(mode="after")
    def validate_multiclass_shape(self) -> ModelConfig:
        if self.objective.startswith("multi:") and self.num_class is None:
            raise ValueError("multi-class objectives require num_class")
        return self


class TrainingParams(BaseModel):
    """Training hyperparameters."""

    num_rounds: int = Field(default=100, ge=1, description="Number of boosting rounds")
    early_stopping_rounds: Optional[int] = Field(default=None, ge=1)
    max_rows_per_worker: Optional[int] = Field(default=None, ge=1)
    val_size: float = Field(
        default=0.2, ge=0.0, lt=1.0, description="Validation set proportion"
    )
    test_size: float = Field(
        default=0.0, ge=0.0, lt=1.0, description="Test set proportion"
    )
    seed: int = Field(default=42, description="Random seed")
    split_strategy: Literal["RANDOM", "TIME_ORDERED"] = "RANDOM"
    stratify: bool = False
    order_column: str | None = None

    @model_validator(mode="after")
    def validate_split_semantics(self) -> TrainingParams:
        if self.val_size + self.test_size >= 1.0:
            raise ValueError("val_size + test_size must be less than 1")
        if self.split_strategy == "TIME_ORDERED":
            if not self.order_column:
                raise ValueError("TIME_ORDERED split requires order_column")
            if self.stratify:
                raise ValueError("stratify is incompatible with TIME_ORDERED splitting")
        return self


class RayConfig(BaseModel):
    """Ray cluster configuration."""

    num_workers: int = Field(default=4, ge=1)
    use_gpu: bool = Field(default=False)
    storage_path: Optional[str] = None
    max_failures: int = Field(
        default=0,
        ge=-1,
        description=(
            "Number of automatic retries on worker failure. 0=no retry, -1=infinite retries. "
            "Note: checkpoint resumption is not currently supported; retries restart training from scratch."
        ),
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    onnx_path: Optional[str] = None
    onnx_opset: int = Field(default=12, ge=1)
    metrics_path: Optional[str] = None
    onnx_optional: bool = False


class EvaluationConfig(BaseModel):
    """Controls final evaluation work and potentially large raw facts."""

    enabled: bool = True
    roc_curve: bool = True
    threshold_analysis: bool = True
    confusion_matrix: bool = True
    feature_importance: bool = True


class XGBoostTrainingConfig(BaseModel):
    """Complete configuration for XGBoost distributed training."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingParams = Field(default_factory=TrainingParams)
    ray: RayConfig = Field(default_factory=RayConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


class RoundMetricsReporter:
    """Publish sampled, real-time XGBoost metrics through Broker API v1."""

    def __init__(
        self,
        job_id: str,
        reporter: EventReporter,
        total_rounds: int,
    ) -> None:
        self._job_id = job_id
        self._reporter = reporter
        self._total_rounds = total_rounds
        self._interval = max(1, math.ceil(total_rounds / 100))
        self._published: set[int] = set()
        self._last_round = 0
        self._last_log: dict[str, dict[str, list[float]]] = {}

    def after_iteration(
        self,
        epoch: int,
        evals_log: dict[str, dict[str, list[float]]],
    ) -> None:
        round_number = epoch + 1
        self._last_round = round_number
        if evals_log:
            self._last_log = evals_log
        if (
            round_number == 1
            or round_number == self._total_rounds
            or round_number % self._interval == 0
        ):
            self._publish(round_number, evals_log)

    def after_training(
        self,
        final_round: int | None = None,
        evals_log: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Always report the actual final round, including early stopping."""
        if final_round is not None:
            self._last_round = final_round
        if evals_log:
            self._last_log = evals_log
        if self._last_round and self._last_round not in self._published:
            self._publish(self._last_round, self._last_log)

    def _publish(
        self,
        round_number: int,
        evals_log: dict[str, dict[str, list[float]]],
    ) -> None:
        metrics: dict[str, float] = {"round": float(round_number)}
        for eval_name, values_by_metric in evals_log.items():
            for metric_name, values in values_by_metric.items():
                if values:
                    metrics[f"{eval_name}-{metric_name}"] = float(values[-1])
        try:
            self._reporter.report_metrics(
                self._job_id,
                metrics,
                min(round_number / self._total_rounds, 1.0),
            )
        except Exception:
            logger.warning(
                "Failed to report round %d metrics for job %s",
                round_number,
                self._job_id,
                exc_info=True,
            )
        finally:
            self._published.add(round_number)


def _binary_confusion(y_true: Any, y_pred: Any) -> tuple[int, int, int, int]:
    import numpy as np

    truth = np.asarray(y_true).astype(int)
    predicted = np.asarray(y_pred).astype(int)
    tp = int(((truth == 1) & (predicted == 1)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    tn = int(((truth == 0) & (predicted == 0)).sum())
    return tn, fp, fn, tp


def _precision_recall_f1(
    y_true: Any,
    y_pred: Any,
) -> tuple[float, float, float]:
    _tn, fp, fn, tp = _binary_confusion(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _roc_points(y_true: Any, scores: Any) -> tuple[Any, Any]:
    import numpy as np

    truth = np.asarray(y_true).astype(int)
    values = np.asarray(scores, dtype=float)
    positives = int((truth == 1).sum())
    negatives = int((truth == 0).sum())
    if not positives or not negatives:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(-values, kind="stable")
    sorted_truth = truth[order]
    sorted_scores = values[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, len(sorted_truth) - 1]
    true_positives = np.cumsum(sorted_truth)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    return fpr, tpr


def _binary_auc(y_true: Any, scores: Any) -> float:
    import numpy as np

    fpr, tpr = _roc_points(y_true, scores)
    if not len(fpr):
        return 0.0
    # Explicit trapezoids work across both NumPy 1.x (``trapz``) and newer
    # releases where that alias was removed.
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def _average_precision(y_true: Any, scores: Any) -> float:
    import numpy as np

    truth = np.asarray(y_true).astype(int)
    positives = int((truth == 1).sum())
    if not positives:
        return 0.0
    ordered = truth[np.argsort(-np.asarray(scores, dtype=float), kind="stable")]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / positives)


def _compute_evaluation_metrics(
    y_true: Any,
    y_prediction: Any,
    objective: str,
    config: EvaluationConfig,
    *,
    num_class: int | None = None,
) -> dict[str, Any]:
    """Compute task-correct final metrics from the complete test split."""
    if not config.enabled:
        return {}

    import numpy as np

    y_true = np.asarray(y_true)
    y_prediction = np.asarray(y_prediction)
    report: dict[str, Any] = {"eval_test_rows": int(len(y_true))}

    if objective.startswith("reg:"):
        residual = y_true - y_prediction
        denominator = float(((y_true - y_true.mean()) ** 2).sum())
        report.update(
            {
                "eval_rmse": float(np.sqrt(np.mean(residual**2))),
                "eval_mae": float(np.mean(np.abs(residual))),
                "eval_r2": (
                    1.0 - float((residual**2).sum()) / denominator
                    if denominator
                    else 0.0
                ),
            }
        )
        return report

    if objective.startswith("multi:"):
        probabilities = y_prediction if y_prediction.ndim == 2 else None
        y_pred = (
            probabilities.argmax(axis=1)
            if probabilities is not None
            else y_prediction.astype(int)
        )
        configured_classes = range(
            num_class
            if num_class is not None
            else probabilities.shape[1] if probabilities is not None else 0
        )
        classes = sorted(
            set(configured_classes)
            | set(np.asarray(y_true).astype(int).tolist())
            | set(np.asarray(y_pred).astype(int).tolist())
        )
        class_metrics = [
            _precision_recall_f1(y_true == value, y_pred == value) for value in classes
        ]
        report.update(
            {
                "eval_f1_macro": float(
                    np.mean([metrics[2] for metrics in class_metrics])
                ),
                "eval_precision_macro": float(
                    np.mean([metrics[0] for metrics in class_metrics])
                ),
                "eval_recall_macro": float(
                    np.mean([metrics[1] for metrics in class_metrics])
                ),
            }
        )
        if probabilities is not None:
            auc_values = [
                _binary_auc(y_true == value, probabilities[:, value])
                for value in range(probabilities.shape[1])
            ]
            report["eval_auc"] = float(np.mean(auc_values))
        if config.confusion_matrix:
            label_indices = {value: index for index, value in enumerate(classes)}
            matrix = np.zeros((len(classes), len(classes)), dtype=int)
            for truth, predicted in zip(y_true, y_pred, strict=True):
                matrix[label_indices[truth], label_indices[predicted]] += 1
            report["eval_cm"] = matrix.tolist()
            report["eval_cm_labels"] = [str(value) for value in classes]
        return report

    y_prob = y_prediction
    y_pred = (y_prob >= 0.5).astype(int)
    precision, recall, f1 = _precision_recall_f1(y_true, y_pred)
    report.update(
        {
            "eval_auc": _binary_auc(y_true, y_prob),
            "eval_f1": f1,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_avg_precision": _average_precision(y_true, y_prob),
        }
    )

    if config.confusion_matrix:
        tn, fp, fn, tp = _binary_confusion(y_true, y_pred)
        report.update(
            {
                "eval_cm_tp": tp,
                "eval_cm_fp": fp,
                "eval_cm_fn": fn,
                "eval_cm_tn": tn,
            }
        )
    if config.roc_curve:
        fpr, tpr = _roc_points(y_true, y_prob)
        step = max(1, len(fpr) // 100)
        report["eval_roc_fpr"] = [float(value) for value in fpr[::step]]
        report["eval_roc_tpr"] = [float(value) for value in tpr[::step]]
    if config.threshold_analysis:
        thresholds = np.linspace(0.05, 0.95, 19)
        report["eval_thr_thresholds"] = [round(float(x), 2) for x in thresholds]
        report["eval_thr_precision"] = []
        report["eval_thr_recall"] = []
        report["eval_thr_f1"] = []
        report["eval_thr_predicted_positive"] = []
        for threshold in thresholds:
            threshold_prediction = (y_prob >= threshold).astype(int)
            threshold_precision, threshold_recall, threshold_f1 = _precision_recall_f1(
                y_true, threshold_prediction
            )
            report["eval_thr_precision"].append(threshold_precision)
            report["eval_thr_recall"].append(threshold_recall)
            report["eval_thr_f1"].append(threshold_f1)
            report["eval_thr_predicted_positive"].append(
                int(threshold_prediction.sum())
            )
    return report


_EVALUATION_THRESHOLDS = tuple(index / 20 for index in range(1, 20))


def _batch_evaluation_partial(
    y_true: Any,
    y_prediction: Any,
    objective: str,
    *,
    num_class: int | None,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Build bounded, mergeable statistics for one prediction batch."""
    import numpy as np

    truth = np.asarray(y_true)
    prediction = np.asarray(y_prediction)
    rows = int(len(truth))
    if objective.startswith("reg:"):
        residual = truth - prediction
        return {
            "task": "regression",
            "rows": rows,
            "sum_abs_error": float(np.abs(residual).sum()),
            "sum_squared_error": float((residual**2).sum()),
            "sum_y": float(truth.sum()),
            "sum_y_squared": float((truth**2).sum()),
        }

    if objective.startswith("multi:"):
        probabilities = prediction if prediction.ndim == 2 else None
        predicted = (
            probabilities.argmax(axis=1)
            if probabilities is not None
            else prediction.astype(int)
        )
        classes = num_class or (
            probabilities.shape[1]
            if probabilities is not None
            else int(max(truth.max(initial=0), predicted.max(initial=0))) + 1
        )
        matrix = np.zeros((classes, classes), dtype=int)
        for actual, inferred in zip(
            truth.astype(int), predicted.astype(int), strict=True
        ):
            if actual >= classes or inferred >= classes or actual < 0 or inferred < 0:
                raise TrainingDataError(
                    "Multiclass label/prediction falls outside configured num_class"
                )
            matrix[actual, inferred] += 1
        return {
            "task": "multiclass",
            "rows": rows,
            "num_class": classes,
            "confusion": matrix.ravel().tolist(),
        }

    probability = prediction.astype(float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = _binary_confusion(truth, predicted)
    partial = {
        "task": "binary",
        "rows": rows,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if config.threshold_analysis:
        threshold_counts = []
        for threshold in _EVALUATION_THRESHOLDS:
            threshold_counts.append(
                list(_binary_confusion(truth, probability >= threshold))
            )
        partial["threshold_counts"] = threshold_counts
    return partial


def _sum_nested(left: Any, right: Any) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise TrainingDataError("Incompatible distributed evaluation statistics")
        return [_sum_nested(a, b) for a, b in zip(left, right, strict=True)]
    return left + right


def _merge_evaluation_partials(
    combined: dict[str, Any] | None,
    partial: dict[str, Any],
) -> dict[str, Any]:
    """Merge one bounded partial without retaining any row-level prediction."""
    if combined is None:
        return dict(partial)
    if combined["task"] != partial["task"]:
        raise TrainingDataError("Mixed task types in distributed evaluation")
    if combined.get("num_class") != partial.get("num_class"):
        raise TrainingDataError("Inconsistent num_class in distributed evaluation")
    for key, value in partial.items():
        if key in {"task", "num_class"}:
            continue
        combined[key] = _sum_nested(combined[key], value)
    return combined


def _finalize_evaluation_partial(
    partial: dict[str, Any],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Convert merged sufficient statistics into neutral evaluation facts."""
    import numpy as np

    rows = int(partial["rows"])
    if not rows:
        raise TrainingDataError("Test split produced no evaluation rows")
    report: dict[str, Any] = {"eval_test_rows": rows}
    if partial["task"] == "regression":
        sum_y = float(partial["sum_y"])
        denominator = float(partial["sum_y_squared"]) - sum_y * sum_y / rows
        report.update(
            {
                "eval_rmse": math.sqrt(float(partial["sum_squared_error"]) / rows),
                "eval_mae": float(partial["sum_abs_error"]) / rows,
                "eval_r2": (
                    1.0 - float(partial["sum_squared_error"]) / denominator
                    if denominator
                    else 0.0
                ),
            }
        )
        return report

    if partial["task"] == "multiclass":
        classes = int(partial["num_class"])
        matrix = np.asarray(partial["confusion"], dtype=int).reshape(classes, classes)
        class_metrics = []
        for index in range(classes):
            tp = int(matrix[index, index])
            fp = int(matrix[:, index].sum() - tp)
            fn = int(matrix[index, :].sum() - tp)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            class_metrics.append((precision, recall, f1))
        report.update(
            {
                "eval_precision_macro": float(np.mean([x[0] for x in class_metrics])),
                "eval_recall_macro": float(np.mean([x[1] for x in class_metrics])),
                "eval_f1_macro": float(np.mean([x[2] for x in class_metrics])),
            }
        )
        if config.confusion_matrix:
            report["eval_cm"] = matrix.tolist()
            report["eval_cm_labels"] = [str(index) for index in range(classes)]
        return report

    tn, fp, fn, tp = (int(partial[key]) for key in ("tn", "fp", "fn", "tp"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report.update(
        {
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1,
        }
    )
    if config.confusion_matrix:
        report.update(
            {
                "eval_cm_tp": tp,
                "eval_cm_fp": fp,
                "eval_cm_fn": fn,
                "eval_cm_tn": tn,
            }
        )
    if config.threshold_analysis:
        report["eval_thr_thresholds"] = list(_EVALUATION_THRESHOLDS)
        report["eval_thr_precision"] = []
        report["eval_thr_recall"] = []
        report["eval_thr_f1"] = []
        report["eval_thr_predicted_positive"] = []
        for threshold_counts in partial["threshold_counts"]:
            _threshold_tn, threshold_fp, threshold_fn, threshold_tp = (
                int(value) for value in threshold_counts
            )
            threshold_precision = (
                threshold_tp / (threshold_tp + threshold_fp)
                if threshold_tp + threshold_fp
                else 0.0
            )
            threshold_recall = (
                threshold_tp / (threshold_tp + threshold_fn)
                if threshold_tp + threshold_fn
                else 0.0
            )
            threshold_f1 = (
                2
                * threshold_precision
                * threshold_recall
                / (threshold_precision + threshold_recall)
                if threshold_precision + threshold_recall
                else 0.0
            )
            report["eval_thr_precision"].append(threshold_precision)
            report["eval_thr_recall"].append(threshold_recall)
            report["eval_thr_f1"].append(threshold_f1)
            report["eval_thr_predicted_positive"].append(threshold_tp + threshold_fp)
    return report


def _evaluate_checkpoint_batch(
    batch: Any,
    *,
    model_artifact: Any,
    label_col: str,
    objective: str,
    num_class: int | None,
) -> Any:
    """Ray task: load the object-store model and keep predictions distributed."""
    import numpy as np
    import pandas as pd
    import ray
    import xgboost

    if isinstance(model_artifact, ray.ObjectRef):
        model_artifact = ray.get(model_artifact)
    booster = xgboost.Booster()
    booster.load_model(bytearray(model_artifact))
    labels = batch[label_col].to_numpy()
    predictions = np.asarray(
        _predict_xgboost_batch(booster, batch.drop(columns=[label_col]))
    )
    output: dict[str, Any] = {"y_true": labels}
    if predictions.ndim == 1:
        output["prediction"] = predictions
    else:
        for index in range(predictions.shape[1]):
            output[f"prediction_{index}"] = predictions[:, index]
    return pd.DataFrame(output)


def _checkpoint_model_artifact(checkpoint: Any) -> Any:
    """Publish checkpoint model bytes once through Ray's distributed object store."""
    import ray
    from ray.train.xgboost import XGBoostCheckpoint

    typed_checkpoint = XGBoostCheckpoint(
        checkpoint.path,
        filesystem=checkpoint.filesystem,
    )
    return ray.put(bytes(typed_checkpoint.get_model().save_raw()))


def _prediction_partial_batch(
    batch: Any,
    *,
    objective: str,
    num_class: int | None,
    evaluation: dict[str, Any],
) -> Any:
    """Reduce one distributed prediction batch to fixed-size sufficient stats."""
    import json

    import numpy as np

    probability_columns = sorted(
        (column for column in batch if column.startswith("prediction_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    prediction = (
        batch[probability_columns].to_numpy()
        if probability_columns
        else batch["prediction"].to_numpy()
    )
    partial = _batch_evaluation_partial(
        np.asarray(batch["y_true"]),
        prediction,
        objective,
        num_class=num_class,
        config=EvaluationConfig.model_validate(evaluation),
    )
    return {"partial": [json.dumps(partial, separators=(",", ":"))]}


def _score_rows_batch(
    batch: Any,
    *,
    objective: str,
    num_class: int | None,
) -> Any:
    """Expand predictions into exact score groups without Driver collection."""
    import numpy as np
    import pandas as pd

    truth = np.asarray(batch["y_true"]).astype(int)
    if objective.startswith("multi:"):
        probability_columns = sorted(
            (column for column in batch if column.startswith("prediction_")),
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        if not probability_columns:
            return pd.DataFrame(
                {
                    "class_index": np.array([], dtype=int),
                    "score": np.array([], dtype=float),
                    "positive": np.array([], dtype=int),
                    "negative": np.array([], dtype=int),
                }
            )
        classes = num_class or len(probability_columns)
        rows = []
        for index in range(classes):
            positive = (truth == index).astype(int)
            rows.append(
                pd.DataFrame(
                    {
                        "class_index": index,
                        "score": batch[f"prediction_{index}"].to_numpy(),
                        "positive": positive,
                        "negative": 1 - positive,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    positive = (truth == 1).astype(int)
    return pd.DataFrame(
        {
            "class_index": 0,
            "score": batch["prediction"].to_numpy(),
            "positive": positive,
            "negative": 1 - positive,
        }
    )


def _bounded_curve_append(
    points: list[tuple[float, float]],
    point: tuple[float, float],
    maximum: int = 256,
) -> None:
    points.append(point)
    if len(points) > maximum * 2:
        points[:] = [points[0], *points[2::2]]


def _exact_rank_metrics_from_group_rows(
    rows: Any,
    totals: dict[int, tuple[int, int]],
    *,
    multiclass: bool,
    include_curve: bool,
) -> dict[str, Any]:
    """Compute exact AUC/AP from globally sorted, tie-grouped score counts."""
    per_class: dict[int, dict[str, Any]] = {}
    for class_index, (positives, negatives) in totals.items():
        per_class[class_index] = {
            "positives": positives,
            "negatives": negatives,
            "tp": 0,
            "fp": 0,
            "auc": 0.0,
            "ap": 0.0,
            "curve": [(0.0, 0.0)],
        }

    for row in rows:
        class_index = int(row["class_index"])
        state = per_class[class_index]
        positive_count = int(row["positive"] or 0)
        negative_count = int(row["negative"] or 0)
        total_positive = int(state["positives"])
        total_negative = int(state["negatives"])
        previous_tpr = state["tp"] / total_positive if total_positive else 0.0
        previous_fpr = state["fp"] / total_negative if total_negative else 0.0
        state["tp"] += positive_count
        state["fp"] += negative_count
        tpr = state["tp"] / total_positive if total_positive else 0.0
        fpr = state["fp"] / total_negative if total_negative else 0.0
        state["auc"] += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
        if total_positive and state["tp"] + state["fp"]:
            state["ap"] += (
                positive_count
                / total_positive
                * state["tp"]
                / (state["tp"] + state["fp"])
            )
        if include_curve:
            _bounded_curve_append(state["curve"], (fpr, tpr))

    auc_values = [float(state["auc"]) for state in per_class.values()]
    if multiclass:
        return {"eval_auc": sum(auc_values) / len(auc_values) if auc_values else 0.0}
    state = per_class[0]
    result: dict[str, Any] = {
        "eval_auc": float(state["auc"]),
        "eval_avg_precision": float(state["ap"]),
    }
    if include_curve:
        curve = state["curve"]
        final_point = (
            state["fp"] / state["negatives"] if state["negatives"] else 0.0,
            state["tp"] / state["positives"] if state["positives"] else 0.0,
        )
        if curve[-1] != final_point:
            _bounded_curve_append(curve, final_point)
        result["eval_roc_fpr"] = [point[0] for point in curve]
        result["eval_roc_tpr"] = [point[1] for point in curve]
    return result


def _pack_score_groups_batch(batch: Any) -> Any:
    """Pack one globally sorted Ray block into per-class opaque payload rows."""
    from uuid import uuid4

    import pandas as pd

    columns = [
        "partition_id",
        "class_index",
        "first_score",
        "positive_total",
        "negative_total",
        "scores",
        "positives",
        "negatives",
    ]
    if batch.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for class_index, group in batch.groupby("class_index", sort=False):
        rows.append(
            {
                "partition_id": uuid4().hex,
                "class_index": int(class_index),
                "first_score": float(group["score"].iloc[0]),
                "positive_total": int(group["positive"].sum()),
                "negative_total": int(group["negative"].sum()),
                # Payload remains in Ray's object store. The Driver only reads
                # the scalar columns above to construct partition prefixes.
                "scores": group["score"].astype(float).tolist(),
                "positives": group["positive"].astype(int).tolist(),
                "negatives": group["negative"].astype(int).tolist(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _rank_partition_prefixes(
    summaries: list[dict[str, Any]],
) -> dict[str, tuple[int, int]]:
    """Build global TP/FP prefixes from O(partitions * classes) summaries."""
    prefixes: dict[str, tuple[int, int]] = {}
    running: dict[int, tuple[int, int]] = {}
    ordered = sorted(
        summaries,
        key=lambda row: (int(row["class_index"]), -float(row["first_score"])),
    )
    for row in ordered:
        class_index = int(row["class_index"])
        positive, negative = running.get(class_index, (0, 0))
        prefixes[str(row["partition_id"])] = (positive, negative)
        running[class_index] = (
            positive + int(row["positive_total"]),
            negative + int(row["negative_total"]),
        )
    return prefixes


def _rank_partition_contribution_batch(
    batch: Any,
    *,
    prefixes: dict[str, tuple[int, int]],
    totals: dict[int, tuple[int, int]],
    include_curve: bool,
) -> Any:
    """Compute exact local ranking contributions after Driver prefix planning."""
    import json

    import pandas as pd

    rows: list[dict[str, Any]] = []
    for _, partition in batch.iterrows():
        partition_id = str(partition["partition_id"])
        class_index = int(partition["class_index"])
        tp, fp = prefixes[partition_id]
        total_positive, total_negative = totals[class_index]
        auc = 0.0
        ap = 0.0
        curve: list[tuple[float, float]] = []
        for positive_count, negative_count in zip(
            partition["positives"], partition["negatives"], strict=True
        ):
            positive_count = int(positive_count)
            negative_count = int(negative_count)
            previous_tpr = tp / total_positive if total_positive else 0.0
            previous_fpr = fp / total_negative if total_negative else 0.0
            tp += positive_count
            fp += negative_count
            tpr = tp / total_positive if total_positive else 0.0
            fpr = fp / total_negative if total_negative else 0.0
            auc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
            if total_positive and tp + fp:
                ap += positive_count / total_positive * tp / (tp + fp)
            if include_curve:
                _bounded_curve_append(curve, (fpr, tpr))
        rows.append(
            {
                "partition_id": partition_id,
                "class_index": class_index,
                "first_score": float(partition["first_score"]),
                "auc": auc,
                "ap": ap,
                "curve": json.dumps(curve, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def _merge_rank_partition_contributions(
    rows: Any,
    totals: dict[int, tuple[int, int]],
    *,
    multiclass: bool,
    include_curve: bool,
) -> dict[str, Any]:
    """Merge only O(partitions * classes) worker contribution rows."""
    import json

    contributions = list(rows)
    auc_by_class = dict.fromkeys(totals, 0.0)
    ap_by_class = dict.fromkeys(totals, 0.0)
    for row in contributions:
        class_index = int(row["class_index"])
        auc_by_class[class_index] += float(row["auc"])
        ap_by_class[class_index] += float(row["ap"])

    if multiclass:
        auc_values = list(auc_by_class.values())
        return {"eval_auc": sum(auc_values) / len(auc_values) if auc_values else 0.0}

    result: dict[str, Any] = {
        "eval_auc": auc_by_class.get(0, 0.0),
        "eval_avg_precision": ap_by_class.get(0, 0.0),
    }
    if include_curve:
        curve: list[tuple[float, float]] = [(0.0, 0.0)]
        ordered = sorted(
            contributions,
            key=lambda row: -float(row["first_score"]),
        )
        for row in ordered:
            raw_curve = row["curve"]
            points = json.loads(raw_curve) if isinstance(raw_curve, str) else raw_curve
            for fpr, tpr in points:
                _bounded_curve_append(curve, (float(fpr), float(tpr)))
        positives, negatives = totals.get(0, (0, 0))
        final_point = (1.0 if negatives else 0.0, 1.0 if positives else 0.0)
        if curve[-1] != final_point:
            _bounded_curve_append(curve, final_point)
        result["eval_roc_fpr"] = [point[0] for point in curve]
        result["eval_roc_tpr"] = [point[1] for point in curve]
    return result


def _exact_distributed_rank_metrics(
    predictions: Any,
    *,
    objective: str,
    num_class: int | None,
    partial: dict[str, Any],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Use Ray group-by/sort for exact ranking metrics with bounded Driver state."""
    from ray.data.aggregate import Sum

    if objective.startswith("reg:") or objective == "multi:softmax":
        return {}
    score_rows = predictions.map_batches(
        cast(Any, _score_rows_batch),
        batch_format="pandas",
        batch_size=4096,
        fn_kwargs={"objective": objective, "num_class": num_class},
    )
    grouped = (
        score_rows.groupby(["class_index", "score"])
        .aggregate(
            Sum("positive", alias_name="positive"),
            Sum("negative", alias_name="negative"),
        )
        .sort(["class_index", "score"], descending=[False, True])
    )
    if objective.startswith("multi:"):
        matrix = partial["confusion"]
        classes = int(partial["num_class"])
        totals = {
            index: (
                sum(matrix[index * classes : (index + 1) * classes]),
                int(partial["rows"])
                - sum(matrix[index * classes : (index + 1) * classes]),
            )
            for index in range(classes)
        }
    else:
        totals = {
            0: (
                int(partial["tp"]) + int(partial["fn"]),
                int(partial["tn"]) + int(partial["fp"]),
            )
        }
    packed = grouped.map_batches(
        cast(Any, _pack_score_groups_batch),
        batch_format="pandas",
        batch_size=None,
    ).materialize()
    summary_columns = [
        "partition_id",
        "class_index",
        "first_score",
        "positive_total",
        "negative_total",
    ]
    summaries = packed.select_columns(summary_columns).take_all()
    prefixes = _rank_partition_prefixes(summaries)
    contributions = packed.map_batches(
        cast(Any, _rank_partition_contribution_batch),
        batch_format="pandas",
        batch_size=None,
        fn_kwargs={
            "prefixes": prefixes,
            "totals": totals,
            "include_curve": config.roc_curve and not objective.startswith("multi:"),
        },
    )
    return _merge_rank_partition_contributions(
        contributions.iter_rows(),
        totals,
        multiclass=objective.startswith("multi:"),
        include_curve=config.roc_curve and not objective.startswith("multi:"),
    )


def _predict_xgboost_batch(booster: Any, features: Any) -> Any:
    import xgboost

    return booster.predict(xgboost.DMatrix(features))


# ── XGBoost BaseTrainer implementation ──


class XGBoostTrainerImpl(BaseTrainer):
    """XGBoost distributed trainer, the first implementation of BaseTrainer.

    Encapsulates XGBoostTrainingConfig validation, data preprocessing, distributed training and ONNX export.
    """

    def __init__(
        self,
        datasets: dict[str, ray.data.Dataset],
        config: dict[str, Any],
        run_config: dict[str, Any] | None = None,
        *,
        _validated_config: XGBoostTrainingConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self._progress = cast(
            TrainingProgress,
            kwargs.pop("_progress", None) or TrainingProgress.from_environment(),
        )
        super().__init__(datasets, config, run_config, **kwargs)
        self._train_config = _validated_config or XGBoostTrainingConfig.model_validate(
            config
        )
        self._warnings: list[dict[str, Any]] = []
        self._row_counts: dict[str, int] = {}

    def setup(self) -> None:
        """Data preprocessing: filter invalid labels, split train/val/test."""
        from tributo.training.xgboost_evaluator import (
            filter_invalid_labels,
            split_dataset,
        )

        ds: Any = self.datasets.get("train")
        if ds is None:
            raise ValueError("datasets must contain 'train' key")

        self._progress.report_phase(TrainingPhase.FEATURE_ENGINEERING)
        label_col = self._train_config.data.label_col
        schema_names = set(ds.schema().names)
        required_columns = [label_col, *self._train_config.data.feature_columns]
        order_column = self._train_config.training.order_column
        if order_column:
            required_columns.append(order_column)
        missing = [column for column in required_columns if column not in schema_names]
        if missing:
            raise TrainingDataError(
                f"Missing training columns: {missing}; "
                f"available columns: {sorted(schema_names)}"
            )

        ds = filter_invalid_labels(ds, label_col=label_col)
        self._collect_data_warnings(ds, schema_names)

        if self._train_config.training.stratify and not (
            self._train_config.model.objective.startswith("binary:")
            or self._train_config.model.objective.startswith("multi:")
        ):
            raise TrainingDataError(
                "stratify is only supported for classification objectives"
            )

        self._progress.report_phase(TrainingPhase.DATA_SPLITTING)
        train_ds, val_ds, test_ds = split_dataset(
            ds,
            val_size=self._train_config.training.val_size,
            test_size=self._train_config.training.test_size,
            seed=self._train_config.training.seed,
            strategy=self._train_config.training.split_strategy,
            stratify=self._train_config.training.stratify,
            label_col=label_col,
            order_column=order_column,
        )

        self.datasets = {"train": train_ds}
        if val_ds is not None:
            self.datasets["val"] = val_ds
        if test_ds is not None:
            self.datasets["test"] = test_ds

        feature_columns = self._train_config.data.feature_columns
        if feature_columns:
            selected = feature_columns + [label_col]
            self.datasets = {
                key: split.select_columns(selected)
                for key, split in self.datasets.items()
            }
        self._validate_and_partition_splits()

    @staticmethod
    def _count_duplicates(ds: "ray.data.Dataset", column: str) -> int:
        """Return excess rows for repeated entity keys without mutating data."""
        try:
            total_rows = int(ds.count())
            # Grouping and the outer count both execute distributedly; only the
            # scalar number of groups reaches the Driver. Null keys form one
            # group, so repeated nulls contribute to the warning consistently.
            distinct_rows = int(
                ds.select_columns([column]).groupby(column).count().count()
            )
            return max(0, total_rows - distinct_rows)
        except Exception:
            logger.warning(
                "Failed to inspect duplicate entity keys for %s", column, exc_info=True
            )
            return 0

    @staticmethod
    def _count_by_label(ds: "ray.data.Dataset", label_col: str) -> dict[str, int]:
        """Return actual label counts used only for validation facts."""
        try:
            rows = ds.groupby(label_col).count().take_all()
            return {str(row[label_col]): int(row["count()"]) for row in rows}
        except Exception:
            logger.warning("Failed to inspect class counts", exc_info=True)
            return {}

    def _collect_data_warnings(
        self,
        ds: "ray.data.Dataset",
        schema_names: set[str],
    ) -> None:
        cfg = self._train_config.data
        if cfg.entity_key_column:
            if cfg.entity_key_column not in schema_names:
                self._warnings.append(
                    {
                        "code": "ENTITY_KEY_COLUMN_MISSING",
                        "column": cfg.entity_key_column,
                    }
                )
            else:
                duplicates = self._count_duplicates(ds, cfg.entity_key_column)
                if duplicates:
                    self._warnings.append(
                        {
                            "code": "DUPLICATE_ENTITY_KEYS",
                            "column": cfg.entity_key_column,
                            "duplicate_rows": duplicates,
                        }
                    )

        if cfg.class_balance_strategy == "CUSTOM_AMOUNT":
            actual_counts = self._count_by_label(ds, cfg.label_col)
            for label, expected in cfg.class_amounts.items():
                actual = actual_counts.get(str(label), 0)
                if actual != expected:
                    deviation = (
                        abs(actual - expected) / expected
                        if expected
                        else float(actual != 0)
                    )
                    self._warnings.append(
                        {
                            "code": "CLASS_AMOUNT_MISMATCH",
                            "label": str(label),
                            "expected": expected,
                            "actual": actual,
                            "deviation": deviation,
                        }
                    )

    def _validate_and_partition_splits(self) -> None:
        """Materialize safe split sizes before Ray submits any training workers."""
        workers = self._train_config.ray.num_workers
        for name, dataset in list(self.datasets.items()):
            count = dataset.count()
            # Unit-test fakes and third-party Dataset-compatible objects may not
            # return a concrete count; only enforce the invariant when known.
            if not isinstance(count, int):
                continue
            if count < workers:
                raise TrainingDataError(
                    f"{name} split has {count} rows, insufficient for "
                    f"{workers} workers (each worker requires at least one row)"
                )
            self._row_counts[name] = count
            self.datasets[name] = dataset.repartition(workers, strict=True)

    def training_loop(self) -> Any:
        """Call build_trainer + trainer.fit(), returns Result."""
        cfg = self._train_config
        label_col = cfg.data.label_col

        # Build XGBoost parameters
        xgb_params = {
            k: v
            for k, v in cfg.model.model_dump(exclude={"objective"}).items()
            if not k.startswith("_")
        }
        xgb_params["objective"] = cfg.model.objective

        train_loop_config: dict[str, Any] = {
            "label_col": label_col,
            "xgb_params": xgb_params,
            "num_rounds": cfg.training.num_rounds,
            "max_rows_per_worker": cfg.training.max_rows_per_worker,
            "evaluation": cfg.evaluation.model_dump(),
        }
        val_ds = self.datasets.get("val")
        test_ds = self.datasets.get("test")
        if val_ds is not None and cfg.training.early_stopping_rounds:
            train_loop_config["early_stopping_rounds"] = (
                cfg.training.early_stopping_rounds
            )

        self._progress.report_phase(TrainingPhase.TRAINING)
        trainer = build_trainer(
            ray_dataset=self.datasets["train"],
            train_config=train_loop_config,
            val_dataset=val_ds,
            test_dataset=test_ds,
            num_workers=cfg.ray.num_workers,
            use_gpu=cfg.ray.use_gpu,
            storage_path=cfg.ray.storage_path,
            max_failures=cfg.ray.max_failures,
        )

        logger.info("Starting XGBoost training...")
        result = trainer.fit()
        metrics = result.metrics or {}

        if cfg.evaluation.enabled and test_ds is not None:
            self._progress.report_phase(TrainingPhase.EVALUATING)
            if result.checkpoint is None:
                raise TrainingDataError(
                    "Training result omitted checkpoint required for full test evaluation"
                )
            metrics.update(self._evaluate_full_test(result.checkpoint))

        # Driver-owned Dataset objects are authoritative for global counts.
        for key, dataset in self.datasets.items():
            metrics[f"row_count_{key}"] = int(dataset.count())
        logger.info(
            "Training done: n_features=%s, %s",
            metrics.get("n_features", "?"),
            {
                k: f"{float(v):.4f}"
                for k, v in metrics.items()
                if k != "n_features" and not isinstance(v, list)
            },
        )
        return result

    def _evaluate_full_test(self, checkpoint: Any) -> dict[str, Any]:
        """Merge bounded prediction statistics from the complete test Dataset."""
        import json

        cfg = self._train_config
        model_artifact = _checkpoint_model_artifact(checkpoint)
        predictions = (
            self.datasets["test"]
            .map_batches(
                cast(Any, _evaluate_checkpoint_batch),
                batch_format="pandas",
                batch_size=4096,
                fn_kwargs={
                    "model_artifact": model_artifact,
                    "label_col": cfg.data.label_col,
                    "objective": cfg.model.objective,
                    "num_class": cfg.model.num_class,
                },
            )
            .materialize()
        )
        partials = predictions.map_batches(
            cast(Any, _prediction_partial_batch),
            batch_format="pandas",
            batch_size=4096,
            fn_kwargs={
                "objective": cfg.model.objective,
                "num_class": cfg.model.num_class,
                "evaluation": cfg.evaluation.model_dump(),
            },
        )
        combined: dict[str, Any] | None = None
        for row in partials.iter_rows():
            raw = row["partial"]
            partial = json.loads(raw) if isinstance(raw, str) else raw
            combined = _merge_evaluation_partials(combined, partial)
        if combined is None:
            raise TrainingDataError("Test split produced no evaluation statistics")
        result = _finalize_evaluation_partial(combined, cfg.evaluation)
        result.update(
            _exact_distributed_rank_metrics(
                predictions,
                objective=cfg.model.objective,
                num_class=cfg.model.num_class,
                partial=combined,
                config=cfg.evaluation,
            )
        )
        return result

    def export_model(self, checkpoint: Any, output_path: str) -> None:
        """Export ONNX to local or S3, save metrics. Results written to self._summary.

        Args:
            checkpoint: Ray Train Result object returned by training_loop().
            output_path: ONNX export path, takes precedence over onnx_path in config.
        """
        from tributo.training.xgboost_evaluator import compute_metrics_summary
        from tributo.training.xgboost_exporter import export_onnx, save_metrics

        result = checkpoint
        cfg = self._train_config
        metrics = result.metrics or {}
        s3_cfg_dict = cfg.data.s3.model_dump()

        # Export is strict by default: a successful training result without its
        # requested model artifact is not a successful run.
        onnx_path = output_path or cfg.output.onnx_path
        onnx_hash = ""
        onnx_size = 0
        if onnx_path:
            try:
                n_features = int(metrics.get("n_features", 0))
                if result.checkpoint is None:
                    raise RuntimeError(
                        "Training result has no checkpoint, cannot export ONNX"
                    )
                onnx_path, onnx_hash, onnx_size = export_onnx(
                    checkpoint=result.checkpoint,
                    onnx_path=onnx_path,
                    n_features=n_features,
                    onnx_opset=cfg.output.onnx_opset,
                    s3_cfg=s3_cfg_dict,
                )
            except Exception as exc:
                if not cfg.output.onnx_optional:
                    raise ModelExportError(
                        f"Failed to export ONNX artifact to {onnx_path!r}: {exc}"
                    ) from exc
                logger.warning("Optional ONNX export failed", exc_info=True)
                self._warnings.append(
                    {
                        "code": "OPTIONAL_MODEL_EXPORT_FAILED",
                        "uri": onnx_path,
                        "error": str(exc),
                    }
                )
                onnx_path = None

        # Save metrics
        feature_columns = [
            c for c in self.datasets["train"].schema().names if c != cfg.data.label_col
        ]
        feature_id_map = cfg.data.model_dump().get("feature_id_map", {})
        serializable_metrics = compute_metrics_summary(
            {key: value for key, value in metrics.items() if not key.startswith("_")}
        )
        histories = {
            key.removesuffix("_history"): value
            for key, value in serializable_metrics.items()
            if key.endswith("_history")
        }
        row_counts = {
            key.removeprefix("row_count_"): int(value)
            for key, value in serializable_metrics.items()
            if key.startswith("row_count_")
        }
        evaluation = {
            key: value
            for key, value in serializable_metrics.items()
            if key.startswith("eval_") or key.startswith("feat_imp_")
        }
        final_metrics = {
            key: value
            for key, value in serializable_metrics.items()
            if not key.endswith("_history")
            and not key.startswith("row_count_")
            and not key.startswith("eval_")
            and not key.startswith("feat_imp_")
            and isinstance(value, (int, float))
        }
        artifact_refs = []
        if onnx_path:
            artifact_refs.append(
                {
                    "kind": "onnx",
                    "uri": onnx_path,
                    "sha256": onnx_hash,
                    "size_bytes": onnx_size,
                }
            )
        summary: dict[str, Any] = {
            "onnx_path": onnx_path,
            "onnx_hash": onnx_hash,
            "onnx_size": onnx_size,
            "feature_columns": feature_columns,
            "feature_id_map": feature_id_map,
            "metrics": serializable_metrics,
            "final_metrics": final_metrics,
            "histories": histories,
            "row_counts": row_counts,
            "evaluation": evaluation,
            "artifact_refs": artifact_refs,
        }
        if self._warnings:
            summary["warnings"] = list(self._warnings)

        metrics_path = cfg.output.metrics_path
        if metrics_path:
            save_metrics(summary, metrics_path, s3_cfg=s3_cfg_dict)

        # Write to self._summary for run() return
        self._summary.update(summary)


# ── Training loop (Ray worker level) ──


def _resolve_worker_cancellation() -> tuple[str | None, CancellationChecker | None]:
    """Rebuild cancellation state inside each Ray worker process."""
    context = ExecutionContext.from_environment()
    if context.cancellation is None:
        return None, None
    try:
        return context.cancellation.job_id, context.build_cancellation_checker()
    except Exception:
        logger.warning("Failed to create worker cancellation checker", exc_info=True)
        return None, None


def _resolve_worker_event_reporting(
    world_rank: int,
) -> tuple[str | None, EventReporter | None]:
    """Build a reporter only on rank 0; reporting is an optional side effect."""
    if world_rank != 0:
        return None, None
    context = ExecutionContext.from_environment()
    if context.event_reporter is None:
        return None, None
    try:
        return context.event_reporter.job_id, context.build_event_reporter()
    except Exception:
        logger.warning("Failed to create worker event reporter", exc_info=True)
        return None, None


def _should_extract_feature_importance(config: dict[str, Any]) -> bool:
    evaluation = config.get("evaluation", {})
    return bool(
        evaluation.get("enabled", True) and evaluation.get("feature_importance", True)
    )


def _raise_if_training_cancelled(
    job_id: str | None,
    checker: CancellationChecker | None,
) -> None:
    """Raise a dedicated failure instead of reporting early-stop success."""
    if job_id is None or checker is None:
        return
    try:
        cancelled = checker.is_cancelled(job_id)
    except Exception:
        logger.warning("Cancellation check failed for job %s", job_id, exc_info=True)
        return
    if cancelled:
        raise TrainingCancelledError(f"Training job {job_id!r} was cancelled")


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """XGBoost training main loop executed by each Ray worker.

    Data flow:
    1. Pull the current worker's data partition from the Ray Dataset shard;
    2. Build QuantileDMatrix incrementally via iter_batches (no need to hold all data);
    3. Distributed training with Rabit Allreduce for gradient histogram synchronization across workers;
    4. Rank 0 exports ONNX and reports the path via ray.train.report.

    Config keys:
        label_col (str): Label column name, default "label".
        xgb_params (dict): XGBoost training parameters.
        num_rounds (int): Number of boosting rounds, default 100.
        max_rows_per_worker (int | None): Maximum rows per worker, safety valve.
    """
    import ray.train
    import xgboost

    label_col = config.get("label_col", "label")
    max_rows = config.get("max_rows_per_worker")
    _feature_names: dict[str, list[str]] = {}
    world_rank = ray.train.get_context().get_world_rank()

    # Rebuild on each worker from the opaque environment context. Never send a
    # live Provider client/checker through Ray's Driver-to-worker serialization.
    _cancel_key, _cancel_checker = _resolve_worker_cancellation()
    _raise_if_training_cancelled(_cancel_key, _cancel_checker)
    _report_job_id, _event_reporter = _resolve_worker_event_reporting(world_rank)
    _round_reporter = (
        RoundMetricsReporter(
            _report_job_id,
            _event_reporter,
            total_rounds=int(config.get("num_rounds", 100)),
        )
        if _report_job_id is not None and _event_reporter is not None
        else None
    )

    class _ControlCallback(xgboost.callback.TrainingCallback):
        """Apply cancellation before publishing metrics in deterministic order."""

        def after_iteration(
            self, model: xgboost.Booster, epoch: int, evts_log: dict
        ) -> bool:
            _raise_if_training_cancelled(_cancel_key, _cancel_checker)
            if _round_reporter is not None:
                _round_reporter.after_iteration(epoch, evts_log)
            return False

    def _make_quantile_dmatrix(
        dataset_key: str,
        ref: xgboost.QuantileDMatrix | None = None,
    ) -> xgboost.QuantileDMatrix | None:
        """Build QuantileDMatrix streamingly from a Ray Dataset shard."""
        try:
            shard = ray.train.get_dataset_shard(dataset_key)
        except KeyError:
            return None
        if shard is None:
            return None

        batches = []
        rows_seen = 0
        for batch in shard.iter_batches(batch_format="pyarrow", batch_size=4096):
            _raise_if_training_cancelled(_cancel_key, _cancel_checker)
            if max_rows is not None and rows_seen >= max_rows:
                break
            if max_rows is not None:
                remaining = max_rows - rows_seen
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
            batches.append(batch)
            rows_seen += batch.num_rows

        import pyarrow as pa

        table = pa.concat_tables(batches)
        del batches

        label = table.column(label_col).to_numpy()
        features = table.drop(label_col)

        # Capture feature column names for later importance extraction
        _feature_names[dataset_key] = list(features.schema.names)

        del table

        return xgboost.QuantileDMatrix(features, label=label, ref=ref)

    dtrain = _make_quantile_dmatrix("train")
    if dtrain is None:
        raise RuntimeError("Failed to build training QuantileDMatrix")
    n_features = dtrain.num_col()

    evals = [(dtrain, "train")]
    dval = _make_quantile_dmatrix("val", ref=dtrain)
    if dval is not None:
        evals.append((dval, "val"))

    evals_result: dict[str, Any] = {}
    booster = xgboost.train(
        config.get("xgb_params", {}),
        dtrain,
        num_boost_round=int(config.get("num_rounds", 100)),
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=config.get("early_stopping_rounds"),
        callbacks=[_ControlCallback()],
    )
    _raise_if_training_cancelled(_cancel_key, _cancel_checker)
    if _round_reporter is not None:
        actual_rounds = max(
            (
                len(values)
                for metrics_by_name in evals_result.values()
                for values in metrics_by_name.values()
            ),
            default=int(booster.num_boosted_rounds()),
        )
        _round_reporter.after_training(actual_rounds, evals_result)

    # Row counts per split (all ranks participate)
    row_info = {}
    for key in ("train", "val", "test"):
        try:
            shard = ray.train.get_dataset_shard(key)
            total = 0
            for batch in shard.iter_batches(batch_format="pyarrow", batch_size=4096):
                total += batch.num_rows
            row_info[f"row_count_{key}"] = total
        except Exception:
            pass  # split not available; will be filled by rank 0 below

    if world_rank == 0:
        from ray.train.xgboost import XGBoostCheckpoint

        checkpoint = XGBoostCheckpoint.from_model(booster)

        report_metrics: dict[str, Any] = {"n_features": n_features}
        for eval_name, eval_scores in evals_result.items():
            for metric_name, values in eval_scores.items():
                key = f"{eval_name}-{metric_name}"
                report_metrics[key] = values[-1]
                report_metrics[f"{key}_history"] = values

        # Final test evaluation is deliberately driver-owned: only the Driver
        # has the complete test Dataset, while each worker sees one shard.
        if _should_extract_feature_importance(config):
            try:
                score = booster.get_score(importance_type="gain")
                rank = 1
                for feat, gain in sorted(score.items(), key=lambda x: -x[1]):
                    report_metrics.setdefault("feat_imp_rank", []).append(rank)
                    report_metrics.setdefault("feat_imp_name", []).append(feat)
                    report_metrics.setdefault("feat_imp_score", []).append(
                        round(gain, 6)
                    )
                    rank += 1
            except Exception:
                logger.exception("Feature importance extraction failed")

        report_metrics.update(row_info)

        ray.train.report(report_metrics, checkpoint=checkpoint)
    else:
        ray.train.report({})


# ── Trainer construction ──


@PublicAPI(stability="beta")
def build_trainer(
    ray_dataset: "ray.data.Dataset",
    train_config: dict[str, Any],
    *,
    val_dataset: "ray.data.Dataset | None" = None,
    test_dataset: "ray.data.Dataset | None" = None,
    num_workers: int = 4,
    use_gpu: bool = False,
    storage_path: str | None = None,
    max_failures: int = 0,
) -> "XGBoostTrainer":
    """Build a Ray XGBoostTrainer instance.

    Args:
        ray_dataset: Training dataset.
        train_config: Config dictionary passed to ``train_loop_per_worker``.
        val_dataset: Validation dataset, enables early stopping when provided.
        test_dataset: Test dataset, enables sklearn evaluation on the worker side.
        num_workers: Number of training workers.
        use_gpu: Whether to use GPU.
        storage_path: Ray Train persistent storage path.
        max_failures: Number of automatic retries on worker failure. 0=no retry, -1=infinite retries.
            Checkpoint resumption is not currently supported; retries restart training from scratch.

    Returns:
        An unstarted XGBoostTrainer; call ``.fit()`` to begin training.
    """
    import os
    import tempfile

    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.xgboost import XGBoostTrainer

    datasets: dict[str, "ray.data.Dataset"] = {"train": ray_dataset}
    if val_dataset is not None:
        datasets["val"] = val_dataset
    if test_dataset is not None:
        datasets["test"] = test_dataset

    if storage_path is not None:
        storage = storage_path
    else:
        # Auto-detect: prefer Docker shared volume, then /workspace, then local temp
        if os.path.isdir("/app/.ray_results"):
            storage = "/app/.ray_results"
        elif os.path.isdir("/workspace"):
            storage = "/workspace/ray_results"
        else:
            storage = os.path.join(tempfile.gettempdir(), "ray_results")

    return XGBoostTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            placement_strategy="SPREAD",
        ),
        datasets=datasets,  # type: ignore[arg-type]
        run_config=RunConfig(
            name="tributo-xgboost",
            storage_path=storage,
            # resilience-only: retries restart from scratch, not from checkpoint
            failure_config=FailureConfig(max_failures=max_failures),
        ),
    )


# ── Orchestration entry points ──


def run_training_with_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run XGBoost distributed training from a YAML config dictionary.

    YAML structure::

        data:
          type: csv | s3 | clickhouse | hive
          path: /data/train.parquet       # csv type
          uri: s3://bucket/data.parquet   # s3 type
          format: csv | parquet
          s3: {region, access_key_id, secret_access_key, endpoint}
          label_col: label

        model:
          objective: binary:logistic      # binary classification
          # objective: multi:softprob     # multi-class (num_class required)
          # num_class: 3
          max_depth: 6
          eta: 0.3
          subsample: 1.0
          colsample_bytree: 1.0

        training:
          num_rounds: 100
          early_stopping_rounds: 10
          max_rows_per_worker: null
          val_size: 0.2
          seed: 42

        ray:
          num_workers: 4
          use_gpu: false

        output:
          onnx_path: model.onnx
          onnx_opset: 12
          metrics_path: metrics.json

    Returns:
        ``{"onnx_path": ..., "metrics": ..., "feature_columns": [...]}``
    """
    from tributo.training.data_loader import load_ray_dataset_from_config

    # Pydantic validation
    cfg = XGBoostTrainingConfig.model_validate(config)

    # Load data
    progress = TrainingProgress.from_environment()
    progress.report_phase(TrainingPhase.LOADING_DATA)
    logger.info("Loading data (type=%s)...", cfg.data.type)
    # Preserve unset credential fields so the loader can merge worker-local
    # environment secrets after Pydantic validation. Explicit empty/configured
    # values remain present and therefore take precedence over the environment.
    ds = load_ray_dataset_from_config(cfg.data.model_dump(exclude_unset=True))

    # Execute training using XGBoostTrainerImpl
    trainer = XGBoostTrainerImpl(
        datasets={"train": ds},
        config=config,
        _validated_config=cfg,
        _progress=progress,
    )
    output_path = cfg.output.onnx_path or ""
    return cast(dict[str, Any], trainer.run(output_path=output_path))


# Built-in registration

_trainer_spec = TrainerSpec(name="xgboost", trainer_cls=XGBoostTrainerImpl)
register(_trainer_spec)

# Exported for entry_points discovery (see tributo.plugin)
trainer_spec = _trainer_spec


@PublicAPI(stability="beta")
def run_training_from_json(config_path: str) -> dict[str, Any]:
    """Run XGBoost distributed training from a JSON configuration file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Summary dictionary.
    """
    import json
    from pathlib import Path

    path = Path(config_path)
    if path.suffix in {".yaml", ".yml"}:
        raise ValueError(
            "YAML training config is no longer supported; please use JSON."
        )
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")
    return run_training_with_config(loaded)
