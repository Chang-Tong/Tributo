"""Worker-side broker cancellation wiring tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tributo.integrations.broker import CancellationSpec
from tributo.training.execution_context import (
    ExecutionContext,
    TrainingCancelledError,
)
from tributo.training.xgboost_trainer import (
    _raise_if_training_cancelled,
    _resolve_worker_cancellation,
)


def test_worker_rebuilds_checker_locally_from_environment() -> None:
    checker = MagicMock()
    context = ExecutionContext(
        cancellation=CancellationSpec(
            broker_id="knova-redis",
            job_id="job-1",
            options={"config_env": "TRIBUTO_BROKER_CONFIG_JSON"},
        )
    )
    with (
        patch.object(ExecutionContext, "from_environment", return_value=context),
        patch.object(
            ExecutionContext,
            "build_cancellation_checker",
            return_value=checker,
        ) as build_checker,
    ):
        job_id, resolved = _resolve_worker_cancellation()

    assert job_id == "job-1"
    assert resolved is checker
    build_checker.assert_called_once_with()


def test_cancel_raises_dedicated_error_instead_of_successful_early_stop() -> None:
    checker = MagicMock()
    checker.is_cancelled.return_value = True

    with pytest.raises(TrainingCancelledError, match="job-1"):
        _raise_if_training_cancelled("job-1", checker)


def test_no_context_keeps_ordinary_training_unchanged() -> None:
    with patch.object(
        ExecutionContext, "from_environment", return_value=ExecutionContext()
    ):
        assert _resolve_worker_cancellation() == (None, None)
