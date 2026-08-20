"""Message broker abstraction for ML job lifecycle integration.

Provides abstract interfaces for task consumption, event reporting,
and cancellation checking. Third-party implementations (Redis, Kafka,
Pulsar, etc.) will register via the ``tributo.brokers`` entry point group.

See Also:
    :mod:`tributo.plugin` — plugin discovery infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

BROKER_API_VERSION = 1

# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class Message:
    """A job request consumed from a message queue.

    Attributes:
        job_id: Unique identifier for the job.
        payload: The deserialized request body (e.g. training config dict).
        metadata: Optional routing headers or trace context.
    """

    job_id: str | None
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    delivery_id: str | None = None
    delivery_attempt: int = 1


@dataclass
class JobResult:
    """Outcome of a completed ML training job.

    Attributes:
        job_id: Unique identifier for the job.
        status: Terminal status — ``"success"``, ``"failed"``, or ``"cancelled"``.
        metrics: Final evaluation metrics (e.g. accuracy, loss).
        artifacts: Paths or URIs of produced artifacts (model files, reports).
        error: Human-readable error message when status is ``"failed"``.
    """

    job_id: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    execution_id: str | None = None
    submission_id: str | None = None
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    bundle_id: str | None = None
    bundle_uri: str | None = None
    manifest_uri: str | None = None


class TaskDisposition(str, Enum):
    """Transport action after a runtime handles a task."""

    ACK = "ack"
    RETRY = "retry"


@dataclass(frozen=True)
class TaskOutcome:
    """Result returned by :meth:`BrokerRuntime.handle`."""

    disposition: TaskDisposition
    result: JobResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class CancellationSpec:
    """Serializable description for rebuilding a worker-side checker."""

    broker_id: str
    job_id: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "broker_id": self.broker_id,
            "job_id": self.job_id,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class EventReporterSpec:
    """Serializable description for rebuilding a worker-side event reporter."""

    broker_id: str
    job_id: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "broker_id": self.broker_id,
            "job_id": self.job_id,
            "options": dict(self.options),
        }


# ── Abstract interfaces ───────────────────────────────────────────────────────


class TaskConsumer(ABC):
    """Consume ML job requests from a message queue.

    Implementations wrap a specific broker client (Redis Streams,
    Kafka consumer group, etc.) and yield :class:`Message` objects.
    """

    @abstractmethod
    def poll(self, timeout_ms: int = 5000) -> Message | None:
        """Block until a job request arrives or *timeout_ms* expires.

        Args:
            timeout_ms: Maximum time to wait in milliseconds.

        Returns:
            A :class:`Message` if one is available, or ``None`` on timeout.
        """
        ...

    @abstractmethod
    def ack(self, message: Message) -> None:
        """Acknowledge successful processing of *message*.

        After ``ack`` the broker guarantees the message will not be
        redelivered.
        """
        ...

    def retry(self, message: Message, error: str | None = None) -> None:
        """Leave *message* available for provider-defined redelivery.

        This default is intentionally a no-op so consumers written against
        the original two-method ABC remain source compatible.
        """
        return None

    def recover_pending(self) -> int:
        """Recover abandoned deliveries and return the number queued locally."""
        return 0

    def close(self) -> None:
        """Release transport resources."""
        return None


class EventReporter(ABC):
    """Publish ML job lifecycle events.

    Every event is keyed by *job_id* so downstream systems can
    reconstruct the full timeline of a job.
    """

    @abstractmethod
    def report_phase(self, job_id: str, phase: str) -> None:
        """Report a lifecycle phase transition.

        Typical phases: ``"initializing"``, ``"training"``,
        ``"exporting"``, ``"completed"``.
        """
        ...

    @abstractmethod
    def report_metrics(
        self, job_id: str, metrics: dict[str, float], progress: float
    ) -> None:
        """Report intermediate training metrics.

        Args:
            job_id: The job identifier.
            metrics: Current metric values (e.g. ``{"loss": 0.35}``).
            progress: Progress fraction in ``[0.0, 1.0]``.
        """
        ...

    @abstractmethod
    def report_completed(self, job_id: str, result: JobResult) -> None:
        """Report job completion with final results."""
        ...

    @abstractmethod
    def report_failed(self, job_id: str, error: str) -> None:
        """Report job failure with error details."""
        ...


class CancellationChecker(ABC):
    """Check whether a running job has been requested to cancel.

    The training loop calls :meth:`is_cancelled` periodically and
    stops early when it returns ``True``.
    """

    @abstractmethod
    def is_cancelled(self, job_id: str) -> bool:
        """Return ``True`` if *job_id* should stop early."""
        ...


class BrokerRuntime(ABC):
    """Provider runtime joining a task consumer to job submission logic."""

    @property
    @abstractmethod
    def consumer(self) -> TaskConsumer:
        """Return the transport consumer owned by this runtime."""
        ...

    @abstractmethod
    def handle(self, message: Message) -> TaskOutcome:
        """Validate and submit one broker message."""
        ...

    def close(self) -> None:
        """Release runtime resources."""
        self.consumer.close()


class BrokerPlugin(ABC):
    """Broker API v1 provider loaded through ``tributo.brokers``."""

    api_version: int = BROKER_API_VERSION
    broker_id: str
    capabilities: frozenset[str] = frozenset()

    @abstractmethod
    def validate_config(
        self,
        config: Mapping[str, Any],
        *,
        check_connectivity: bool = False,
    ) -> None:
        """Validate provider configuration, optionally probing connectivity."""
        ...

    @abstractmethod
    def create_runtime(self, config: Mapping[str, Any]) -> BrokerRuntime:
        """Build a fresh runtime for a consumer process."""
        ...

    def create_cancellation_checker(
        self, spec: CancellationSpec
    ) -> CancellationChecker:
        """Rebuild a checker in a worker process when supported."""
        raise NotImplementedError(
            f"Broker provider {self.broker_id!r} does not support cancellation"
        )

    def create_event_reporter(self, spec: EventReporterSpec) -> EventReporter:
        """Rebuild an event reporter in a worker process when supported."""
        raise NotImplementedError(
            f"Broker provider {self.broker_id!r} does not support event reporting"
        )
