"""Broker-neutral training phase and worker metric reporting semantics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tributo.training.execution_context import ExecutionContext, TrainingCancelledError
from tributo.training.progress import TrainingPhase, TrainingProgress
from tributo.training.xgboost_trainer import (
    RoundMetricsReporter,
    _resolve_worker_event_reporting,
)


def test_phase_checks_cancellation_before_reporting() -> None:
    calls: list[str] = []
    checker = MagicMock()
    checker.is_cancelled.side_effect = lambda job_id: calls.append("cancel") or False
    reporter = MagicMock()
    reporter.report_phase.side_effect = lambda job_id, phase: calls.append(phase)
    progress = TrainingProgress("job-1", reporter=reporter, checker=checker)

    progress.report_phase(TrainingPhase.LOADING_DATA)

    assert calls == ["cancel", "LOADING_DATA"]


def test_cancelled_phase_boundary_raises_without_publishing() -> None:
    checker = MagicMock()
    checker.is_cancelled.return_value = True
    reporter = MagicMock()
    progress = TrainingProgress("job-1", reporter=reporter, checker=checker)

    with pytest.raises(TrainingCancelledError, match="job-1"):
        progress.report_phase(TrainingPhase.TRAINING)

    reporter.report_phase.assert_not_called()


def test_nonterminal_reporter_failure_is_fail_open() -> None:
    reporter = MagicMock()
    reporter.report_phase.side_effect = OSError("broker unavailable")
    progress = TrainingProgress("job-1", reporter=reporter)

    progress.report_phase(TrainingPhase.DATA_SPLITTING)


def test_control_construction_failures_are_fail_open() -> None:
    context = MagicMock(spec=ExecutionContext)
    context.event_reporter = MagicMock(job_id="job-1")
    context.cancellation = MagicMock(job_id="job-1")
    context.build_event_reporter.side_effect = RuntimeError("bad reporter")
    context.build_cancellation_checker.side_effect = RuntimeError("bad checker")

    with patch.object(ExecutionContext, "from_environment", return_value=context):
        progress = TrainingProgress.from_environment()

    assert progress.job_id == "job-1"
    assert progress.reporter is None
    assert progress.checker is None


def test_mismatched_control_job_ids_are_used_independently() -> None:
    context = MagicMock(spec=ExecutionContext)
    context.event_reporter = MagicMock(job_id="job-reporter")
    context.cancellation = MagicMock(job_id="job-cancel")
    reporter = MagicMock()
    checker = MagicMock()
    checker.is_cancelled.return_value = False
    context.build_event_reporter.return_value = reporter
    context.build_cancellation_checker.return_value = checker

    with patch.object(ExecutionContext, "from_environment", return_value=context):
        progress = TrainingProgress.from_environment()

    progress.report_phase(TrainingPhase.TRAINING)

    checker.is_cancelled.assert_called_once_with("job-cancel")
    reporter.report_phase.assert_called_once_with("job-reporter", "TRAINING")
    assert progress.reporter_job_id == "job-reporter"
    assert progress.cancellation_job_id == "job-cancel"


def test_legacy_job_id_initializes_both_control_ids() -> None:
    checker = MagicMock()
    checker.is_cancelled.return_value = False
    progress = TrainingProgress(
        "legacy-job",
        reporter=MagicMock(),
        checker=checker,
    )

    progress.report_phase(TrainingPhase.DATA_SPLITTING)

    progress.checker.is_cancelled.assert_called_once_with("legacy-job")
    progress.reporter.report_phase.assert_called_once_with(
        "legacy-job", "DATA_SPLITTING"
    )


def test_round_metrics_reporter_emits_round_one_interval_and_early_stop_final() -> None:
    reporter = MagicMock()
    rounds = RoundMetricsReporter(
        job_id="job-1",
        reporter=reporter,
        total_rounds=250,
    )

    for epoch in range(7):
        rounds.after_iteration(
            epoch,
            {"train": {"logloss": [0.9 - 0.1 * i for i in range(epoch + 1)]}},
        )
    rounds.after_training()

    emitted_rounds = [
        int(call.args[1]["round"]) for call in reporter.report_metrics.call_args_list
    ]
    assert emitted_rounds == [1, 3, 6, 7]
    assert reporter.report_metrics.call_args_list[-1].args[2] == pytest.approx(7 / 250)


def test_round_metrics_reporter_is_fail_open() -> None:
    reporter = MagicMock()
    reporter.report_metrics.side_effect = OSError("broker unavailable")
    rounds = RoundMetricsReporter("job-1", reporter, total_rounds=1)

    rounds.after_iteration(0, {"train": {"rmse": [0.4]}})
    rounds.after_training()

    reporter.report_metrics.assert_called_once()


def test_external_final_history_covers_early_stop_callback_short_circuit() -> None:
    reporter = MagicMock()
    rounds = RoundMetricsReporter("job-1", reporter, total_rounds=250)
    rounds.after_iteration(0, {"train": {"loss": [0.9]}})

    rounds.after_training(7, {"train": {"loss": [0.9, 0.8, 0.7, 0.6]}})

    assert [
        int(call.args[1]["round"]) for call in reporter.report_metrics.call_args_list
    ] == [1, 7]


def test_nonzero_worker_never_rebuilds_event_reporter() -> None:
    with patch(
        "tributo.training.xgboost_trainer.ExecutionContext.from_environment"
    ) as from_environment:
        assert _resolve_worker_event_reporting(1) == (None, None)
    from_environment.assert_not_called()
