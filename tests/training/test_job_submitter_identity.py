"""Stable business and Ray identity tests for training submission."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from ray.job_submission import JobStatus

from tributo.training.job_submitter import (
    SubmissionIdentity,
    submit_training_job_with_identity,
)


def test_submit_with_identity_injects_all_identity_layers(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "src" / "tributo").mkdir(parents=True)
    provider = tmp_path / "provider"
    provider.mkdir()
    wheel = "/provider/tributo_broker_redis-0.1.0-py3-none-any.whl"
    client = MagicMock()
    client.submit_job.return_value = "ray-execution-1"

    with patch(
        "tributo.training.job_submitter._get_submission_client",
        return_value=client,
    ):
        identity = submit_training_job_with_identity(
            "python -m provider.run_training",
            project_root=tmp_path,
            run_id="business-run-1",
            attempt_id="attempt-1",
            env_vars={"OPAQUE_SETTING": "value"},
            extra_py_modules=[provider],
            runtime_pip_packages=[wheel],
            execution_context={
                "cancellation": {
                    "broker_id": "knova-redis",
                    "job_id": "business-run-1",
                    "options": {"config_env": "TRIBUTO_BROKER_CONFIG_JSON"},
                }
            },
        )

    assert isinstance(identity, SubmissionIdentity)
    assert identity.run_id == "business-run-1"
    assert identity.attempt_id == "attempt-1"
    assert identity.job_id == "ray-execution-1"
    assert identity.execution_id == "ray-execution-1"
    call = client.submit_job.call_args.kwargs
    assert call["submission_id"] == identity.submission_id
    runtime_env = call["runtime_env"]
    assert str(provider) in runtime_env["py_modules"]
    assert runtime_env["pip"] == [wheel]
    assert runtime_env["env_vars"]["TRIBUTO_RUN_ID"] == "business-run-1"
    assert runtime_env["env_vars"]["TRIBUTO_ATTEMPT_ID"] == "attempt-1"
    assert runtime_env["env_vars"]["TRIBUTO_SUBMISSION_ID"] == identity.submission_id
    assert "TRIBUTO_EXECUTION_CONTEXT" in runtime_env["env_vars"]
    assert (
        "redis-password-value"
        not in runtime_env["env_vars"]["TRIBUTO_EXECUTION_CONTEXT"]
    )


def test_terminal_duplicate_is_reconciled_without_timestamp_resubmission(
    tmp_path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "src" / "tributo").mkdir(parents=True)
    client = MagicMock()
    client.submit_job.side_effect = RuntimeError("submission_id already exists")
    client.get_job_status.return_value = JobStatus.SUCCEEDED

    with patch(
        "tributo.training.job_submitter._get_submission_client",
        return_value=client,
    ):
        first = submit_training_job_with_identity(
            "python train.py",
            project_root=tmp_path,
            run_id="run-1",
            attempt_id="attempt-1",
        )
        second = submit_training_job_with_identity(
            "python train.py",
            project_root=tmp_path,
            run_id="run-1",
            attempt_id="attempt-1",
        )

    assert first == second
    assert first.job_id == first.submission_id
    assert client.submit_job.call_count == 2
    assert all(
        call.kwargs["submission_id"] == first.submission_id
        for call in client.submit_job.call_args_list
    )


def test_new_attempt_changes_submission_identity(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "src" / "tributo").mkdir(parents=True)
    client = MagicMock()
    client.submit_job.side_effect = ["ray-1", "ray-2"]

    with patch(
        "tributo.training.job_submitter._get_submission_client",
        return_value=client,
    ):
        first = submit_training_job_with_identity(
            "python train.py",
            project_root=tmp_path,
            run_id="run-1",
            attempt_id="attempt-1",
        )
        second = submit_training_job_with_identity(
            "python train.py",
            project_root=tmp_path,
            run_id="run-1",
            attempt_id="attempt-2",
        )

    assert first.submission_id != second.submission_id


@pytest.mark.parametrize("status", [None, "UNKNOWN"])
def test_duplicate_with_unconfirmed_status_is_retryable(tmp_path, status) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "src" / "tributo").mkdir(parents=True)
    client = MagicMock()
    client.submit_job.side_effect = RuntimeError("submission_id already exists")
    client.get_job_status.return_value = status

    with (
        patch(
            "tributo.training.job_submitter._get_submission_client",
            return_value=client,
        ),
        pytest.raises(RuntimeError, match="status could not be confirmed"),
    ):
        submit_training_job_with_identity(
            "python train.py",
            project_root=tmp_path,
            run_id="run-1",
            attempt_id="attempt-1",
        )


def test_duplicate_status_query_failure_is_retryable(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "src" / "tributo").mkdir(parents=True)
    client = MagicMock()
    client.submit_job.side_effect = RuntimeError("submission_id already exists")
    client.get_job_status.side_effect = ConnectionError("dashboard unavailable")

    with (
        patch(
            "tributo.training.job_submitter._get_submission_client",
            return_value=client,
        ),
        pytest.raises(RuntimeError, match="status query failed"),
    ):
        submit_training_job_with_identity(
            "python train.py",
            project_root=tmp_path,
            run_id="run-1",
            attempt_id="attempt-1",
        )
