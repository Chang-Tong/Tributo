"""Worker-side reconstruction tests for broker execution context."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tributo.integrations.broker import CancellationSpec, EventReporterSpec
from tributo.training.execution_context import (
    EXECUTION_CONTEXT_SCHEMA,
    ExecutionContext,
)


def test_empty_execution_context_does_not_discover_provider() -> None:
    with patch("tributo.training.execution_context.get_broker_plugin") as get_plugin:
        context = ExecutionContext.from_mapping(None)
        assert context.build_cancellation_checker() is None
        assert context.build_event_reporter() is None
    get_plugin.assert_not_called()


def test_explicit_context_rebuilds_components_by_broker_id() -> None:
    checker = MagicMock()
    reporter = MagicMock()
    plugin = MagicMock()
    plugin.create_cancellation_checker.return_value = checker
    plugin.create_event_reporter.return_value = reporter
    context = ExecutionContext(
        cancellation=CancellationSpec(
            broker_id="fake",
            job_id="job-1",
            options={"config_env": "BROKER_CONFIG"},
        ),
        event_reporter=EventReporterSpec(
            broker_id="fake",
            job_id="job-1",
            options={"config_env": "BROKER_CONFIG"},
        ),
    )

    with patch(
        "tributo.training.execution_context.get_broker_plugin",
        return_value=plugin,
    ) as get_plugin:
        assert context.build_cancellation_checker() is checker
        assert context.build_event_reporter() is reporter

    assert get_plugin.call_args_list[0].args == ("fake",)
    assert get_plugin.call_args_list[1].args == ("fake",)


def test_explicit_missing_provider_fails_closed() -> None:
    context = ExecutionContext.from_mapping(
        {
            "cancellation": {
                "broker_id": "missing",
                "job_id": "job-1",
                "options": {},
            }
        }
    )
    with patch(
        "tributo.training.execution_context.get_broker_plugin",
        side_effect=LookupError("missing provider"),
    ):
        with pytest.raises(LookupError, match="missing provider"):
            context.build_cancellation_checker()


def test_legacy_provider_context_defaults_to_v1_and_canonical_schema() -> None:
    context = ExecutionContext.from_mapping(
        {
            "cancellation": {
                "broker_id": "knova-redis",
                "job_id": "job-1",
                "options": {"config_env": "TRIBUTO_BROKER_CONFIG_JSON"},
            }
        }
    )
    serialized = context.as_dict()
    assert serialized["schema"] == EXECUTION_CONTEXT_SCHEMA
    assert serialized["version"] == 1


@pytest.mark.parametrize(
    "value,match",
    [
        ({"cancelation": {}}, "unknown execution context fields"),
        (
            {
                "cancellation": {
                    "broker_id": "fake",
                    "job_id": "job-1",
                    "options": {},
                    "password": "value",
                }
            },
            "unknown cancellation fields",
        ),
        ({"schema": "wrong", "version": 1}, "unsupported execution context schema"),
        (
            {"schema": EXECUTION_CONTEXT_SCHEMA, "version": 2},
            "unsupported execution context version",
        ),
    ],
)
def test_execution_context_rejects_unknown_or_unsupported_schema(value, match) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        ExecutionContext.from_mapping(value)


@pytest.mark.parametrize(
    "options",
    [
        {"authorization": "Bearer abc"},
        {"accessToken": "abc"},
        {"privateKey": "-----BEGIN PRIVATE KEY-----abc"},
        {"dsn": "postgres://user:password@example/db"},
        {"endpoint": "redis://user:password@example:6379/0"},
        {"nested": {"secretAccessKey": "abc"}},
    ],
)
def test_execution_context_rejects_inline_secrets(options) -> None:
    with pytest.raises(ValueError, match="secret|credential|userinfo"):
        ExecutionContext.from_mapping(
            {
                "cancellation": {
                    "broker_id": "fake",
                    "job_id": "job-1",
                    "options": options,
                }
            }
        )


def test_execution_context_allows_references_and_non_secret_token_metadata() -> None:
    context = ExecutionContext.from_mapping(
        {
            "cancellation": {
                "broker_id": "fake",
                "job_id": "job-1",
                "options": {
                    "config_env": "BROKER_CONFIG",
                    "api_key_env_var": "API_KEY",
                    "token_count": 512,
                },
            }
        }
    )
    assert context.cancellation is not None
    assert context.cancellation.options["token_count"] == 512
