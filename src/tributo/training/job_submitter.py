"""Training job submission wrapper via Ray Jobs API.

Automatically handles runtime_env configuration (py_modules + working_dir + PYTHONPATH),
abstracting away the underlying differences between /venv and anaconda environments in the official Ray image.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ray.job_submission import JobStatus, JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo._common.retry import retry_with_exponential_backoff
from tributo._common.submission_id import (
    generate_submission_id,
    generate_training_submission_id,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 180


@dataclass(frozen=True)
class SubmissionIdentity:
    """Stable business identity and its reconciled Ray execution."""

    run_id: str
    attempt_id: str
    submission_id: str
    job_id: str

    @property
    def execution_id(self) -> str:
        """Alias for the actual Ray Job identity."""
        return self.job_id


@PublicAPI(stability="beta")
def submit_training_job_with_identity(
    entrypoint: str,
    *,
    run_id: str,
    attempt_id: str,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
    extra_excludes: list[str] | None = None,
    extra_py_modules: list[str | Path] | None = None,
    runtime_pip_packages: list[str] | None = None,
    execution_context: dict[str, Any] | Any | None = None,
) -> SubmissionIdentity:
    """Submit or reconcile exactly one business training attempt.

    Redis redelivery and other transport retries reuse the deterministic
    ``submission_id``. A terminal Ray Job is still the same execution; a new
    execution requires the caller to provide a different ``attempt_id``.
    """
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if not attempt_id:
        raise ValueError("attempt_id must be non-empty")

    submission_id = generate_training_submission_id(run_id, attempt_id, entrypoint)
    identity_env = dict(env_vars or {})
    identity_env.update(
        {
            "TRIBUTO_RUN_ID": run_id,
            "TRIBUTO_ATTEMPT_ID": attempt_id,
            "TRIBUTO_SUBMISSION_ID": submission_id,
        }
    )
    runtime_env = build_runtime_env(
        project_root=project_root,
        env_vars=identity_env,
        extra_excludes=extra_excludes,
        extra_py_modules=extra_py_modules,
        runtime_pip_packages=runtime_pip_packages,
        execution_context=execution_context,
    )

    client = _get_submission_client(dashboard_url)
    try:
        job_id = client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            submission_id=submission_id,
        )
    except RuntimeError as exc:
        if "already exists" not in str(exc).lower():
            raise
        try:
            status = client.get_job_status(submission_id)
        except Exception as status_exc:
            raise RuntimeError(
                f"Existing Ray job {submission_id!r} status query failed"
            ) from status_exc
        if not isinstance(status, JobStatus):
            raise RuntimeError(
                f"Existing Ray job {submission_id!r} status could not be confirmed: "
                f"{status!r}"
            ) from exc
        logger.info(
            "Reconciled existing training job %s with status %s",
            submission_id,
            status,
        )
        job_id = submission_id

    return SubmissionIdentity(
        run_id=run_id,
        attempt_id=attempt_id,
        submission_id=submission_id,
        job_id=cast(str, job_id),
    )


@PublicAPI(stability="beta")
def submit_training_job(
    entrypoint: str,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    project_root: Path | None = None,
    extra_excludes: list[str] | None = None,
) -> str:
    """Submit a training job via the Ray Jobs API.

    Automatically builds the correct ``runtime_env``, ensuring:
    - The latest ``tributo`` code is uploaded to the cluster via ``py_modules``;
    - The entrypoint script is provided via ``working_dir``;
    - ``ray`` and other anaconda-exclusive packages are visible to ``/venv`` via ``PYTHONPATH``.

    Args:
        entrypoint: Entrypoint command, e.g. ``"python examples/xgboost_s3_training.py"``.
        dashboard_url: Ray Dashboard address.
        env_vars: Additional environment variables (training params passed this way).
        project_root: Project root directory, auto-finds pyproject.toml if None.
        extra_excludes: Additional directories/files to exclude from working_dir.

    Returns:
        Submitted job ID on success.

    Raises:
        RuntimeError: Submission failed.
    """
    runtime_env = build_runtime_env(
        project_root=project_root,
        env_vars=env_vars,
        extra_excludes=extra_excludes,
    )

    submission_id = generate_submission_id(
        "train", entrypoint, str(sorted((env_vars or {}).items()))
    )

    client = _get_submission_client(dashboard_url)
    try:
        job_id = client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            submission_id=submission_id,
        )
    except RuntimeError as exc:
        if "already exists" not in str(exc):
            raise
        # Submission ID already exists — check status and either reuse or retry
        logger.warning("Job %s already exists, checking status...", submission_id)
        try:
            status = client.get_job_status(submission_id)
        except Exception:
            status = None
        if status in {JobStatus.PENDING, JobStatus.RUNNING}:
            logger.info("Reusing running job %s", submission_id)
            return submission_id
        # Job finished (SUCCEEDED/FAILED/STOPPED) — retry with timestamped ID
        submission_id = f"{submission_id}-{int(time.time())}"
        job_id = client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            submission_id=submission_id,
        )
    logger.info("Submitted training job %s: %s", job_id, entrypoint)
    return cast(str, job_id)


@retry_with_exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    exceptions=(ConnectionError, TimeoutError, OSError),
)
def _get_submission_client(dashboard_url: str) -> JobSubmissionClient:
    """Create a JobSubmissionClient with retry on connection failures.

    Retries are limited to the connection establishment phase to avoid
    duplicate job submissions when the request has already reached the
    Ray server.
    """
    return JobSubmissionClient(dashboard_url)


@PublicAPI(stability="beta")
def wait_for_job(
    client: JobSubmissionClient,
    job_id: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    poll_interval: int = 2,
) -> dict[str, Any]:
    """Poll and wait for a job to complete.

    Args:
        client: Ray Jobs API client.
        job_id: Job ID to wait for.
        timeout: Maximum wait time in seconds.
        poll_interval: Polling interval in seconds.

    Returns:
        ``{"status": JobStatus, "logs": str}``

    Raises:
        TimeoutError: Timed out without completion.
    """
    start = time.time()
    while time.time() - start < timeout:
        status = client.get_job_status(job_id)
        if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
            logs = client.get_job_logs(job_id)
            return {"status": status, "logs": logs}
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def submit_and_export_onnx(
    entrypoint: str,
    onnx_output_path: str,
    n_features: int,
    *,
    dashboard_url: str = DEFAULT_DASHBOARD_URL,
    env_vars: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, str]:
    """Submit a training job and export ONNX upon completion (Driver side).

    TODO: Implement checkpoint extraction from /tmp/ray_results and ONNX export.
    For now, use Ray Train API (trainer.fit()) for direct handling.

    Args:
        entrypoint: Training script entrypoint command.
        onnx_output_path: ONNX file save path.
        n_features: Number of features.
        dashboard_url: Ray Dashboard address.
        env_vars: Environment variables.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (job_id, onnx_path).

    Raises:
        RuntimeError: Training failed or ONNX export failed.
    """

    # TODO: Implement checkpoint extraction from /tmp/ray_results and ONNX export.
    # Requires submit_training_job + wait_for_job + checkpoint extraction logic.
    # For now, use Ray Train API (trainer.fit()) for direct handling.
    raise NotImplementedError(
        "Checkpoint extraction from Jobs API is not yet implemented. "
        "Use Ray Train API (trainer.fit()) for automatic checkpoint handling."
    )
