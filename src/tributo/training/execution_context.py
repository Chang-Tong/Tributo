"""Broker-neutral worker execution context reconstruction."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from tributo.integrations.broker import (
    CancellationChecker,
    CancellationSpec,
    EventReporter,
    EventReporterSpec,
)
from tributo.integrations.broker_registry import get_broker_plugin

EXECUTION_CONTEXT_ENV = "TRIBUTO_EXECUTION_CONTEXT"
EXECUTION_CONTEXT_SCHEMA = "tributo.execution-context"
EXECUTION_CONTEXT_VERSION = 1


class TrainingCancelledError(RuntimeError):
    """Raised inside a training worker when cooperative cancellation is set."""


@dataclass(frozen=True)
class ExecutionContext:
    """Serializable factories for optional worker-side broker components."""

    schema: str = EXECUTION_CONTEXT_SCHEMA
    version: int = EXECUTION_CONTEXT_VERSION
    cancellation: CancellationSpec | None = None
    event_reporter: EventReporterSpec | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the context as JSON-compatible data."""
        result: dict[str, Any] = {
            "schema": self.schema,
            "version": self.version,
        }
        if self.cancellation is not None:
            _validate_options(self.cancellation.options, "cancellation.options")
            result["cancellation"] = self.cancellation.as_dict()
        if self.event_reporter is not None:
            _validate_options(self.event_reporter.options, "event_reporter.options")
            result["event_reporter"] = self.event_reporter.as_dict()
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ExecutionContext:
        """Parse context data without discovering providers."""
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("execution context must be an object")
        allowed = {"schema", "version", "cancellation", "event_reporter"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown execution context fields: {sorted(map(str, unknown))}"
            )
        schema = value.get("schema", EXECUTION_CONTEXT_SCHEMA)
        version = value.get("version", EXECUTION_CONTEXT_VERSION)
        if schema != EXECUTION_CONTEXT_SCHEMA:
            raise ValueError(f"unsupported execution context schema: {schema!r}")
        if version != EXECUTION_CONTEXT_VERSION:
            raise ValueError(f"unsupported execution context version: {version!r}")
        cancellation = _parse_spec(
            value.get("cancellation"), CancellationSpec, "cancellation"
        )
        reporter = _parse_spec(
            value.get("event_reporter"), EventReporterSpec, "event_reporter"
        )
        return cls(
            schema=EXECUTION_CONTEXT_SCHEMA,
            version=EXECUTION_CONTEXT_VERSION,
            cancellation=cancellation,
            event_reporter=reporter,
        )

    @classmethod
    def from_environment(cls) -> ExecutionContext:
        """Parse context from the Ray worker environment."""
        raw = os.environ.get(EXECUTION_CONTEXT_ENV)
        if not raw:
            return cls()
        value = json.loads(raw)
        return cls.from_mapping(value)

    def build_cancellation_checker(self) -> CancellationChecker | None:
        """Rebuild the explicitly configured cancellation checker."""
        if self.cancellation is None:
            return None
        plugin = get_broker_plugin(self.cancellation.broker_id)
        return cast(
            CancellationChecker,
            plugin.create_cancellation_checker(self.cancellation),
        )

    def build_event_reporter(self) -> EventReporter | None:
        """Rebuild the explicitly configured event reporter."""
        if self.event_reporter is None:
            return None
        plugin = get_broker_plugin(self.event_reporter.broker_id)
        return cast(EventReporter, plugin.create_event_reporter(self.event_reporter))


def _parse_spec(value: Any, spec_type: type[Any], section: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("execution context component must be an object")
    allowed = {"broker_id", "job_id", "options"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {section} fields: {sorted(map(str, unknown))}")
    broker_id = value.get("broker_id")
    job_id = value.get("job_id")
    options = value.get("options", {})
    if not isinstance(broker_id, str) or not broker_id:
        raise ValueError("execution context broker_id must be non-empty")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("execution context job_id must be non-empty")
    if not isinstance(options, Mapping):
        raise TypeError("execution context options must be an object")
    _validate_options(options, f"{section}.options")
    return spec_type(broker_id=broker_id, job_id=job_id, options=dict(options))


def _normalize_key(key: str) -> str:
    snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake)
    return snake.replace("-", "_").lower()


def _is_reference_key(key: str) -> bool:
    return key.endswith(("_env", "_env_var", "_ref", "_reference"))


def _is_sensitive_key(key: str) -> bool:
    exact = {
        "api_key",
        "authorization",
        "credential",
        "dsn",
        "password",
        "passwd",
        "private_key",
        "secret",
        "secret_access_key",
        "token",
        "access_token",
    }
    if key in exact:
        return True
    endings = (
        "_api_key",
        "_credential",
        "_password",
        "_passwd",
        "_private_key",
        "_secret",
        "_secret_access_key",
        "_token",
    )
    prefixes = (
        "access_token_",
        "authorization_",
        "password_",
        "passwd_",
        "private_key_",
        "secret_access_key_",
    )
    return key.endswith(endings) or key.startswith(prefixes)


def _validate_options(value: Any, path: str) -> None:
    """Validate Provider options as JSON-safe metadata and secret references."""
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(
                    f"execution context option key at {path} must be a string"
                )
            key = _normalize_key(raw_key)
            if _is_sensitive_key(key) and not _is_reference_key(key):
                if child not in (None, ""):
                    raise ValueError(
                        f"inline secret or credential is forbidden at {path}.{raw_key}"
                    )
            _validate_options(child, f"{path}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_options(child, f"{path}[{index}]")
        return
    if isinstance(value, str):
        if "-----BEGIN" in value.upper() and "PRIVATE KEY-----" in value.upper():
            raise ValueError(f"inline private-key secret is forbidden at {path}")
        if "://" in value:
            from urllib.parse import urlsplit

            parsed = urlsplit(value)
            if parsed.username is not None or parsed.password is not None:
                raise ValueError(f"URI userinfo credential is forbidden at {path}")
        return
    if value is not None and not isinstance(value, (bool, int, float)):
        raise TypeError(f"execution context option at {path} is not JSON-safe")
