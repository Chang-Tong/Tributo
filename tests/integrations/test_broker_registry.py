"""Tests for isolated third-party broker discovery."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from tributo.integrations.broker import BROKER_API_VERSION, BrokerPlugin
from tributo.integrations.broker_registry import (
    BrokerProviderNotFoundError,
    discover_broker_plugins,
    get_broker_plugin,
    list_broker_provider_statuses,
)


class _Plugin(BrokerPlugin):
    api_version = BROKER_API_VERSION
    broker_id = "fake"
    capabilities = frozenset({"task-consumer"})

    def validate_config(self, config, *, check_connectivity=False):
        pass

    def create_runtime(self, config):
        raise NotImplementedError


@dataclass
class _EntryPoint:
    name: str
    value: str
    loaded: object = _Plugin
    error: Exception | None = None

    def load(self):
        if self.error is not None:
            raise self.error
        return self.loaded


def test_discovery_skips_broken_provider(monkeypatch) -> None:
    entry_points = MagicMock(
        return_value=[
            _EntryPoint("broken", "broken:Plugin", error=ImportError("redis missing")),
            _EntryPoint("fake", "fake:Plugin"),
        ]
    )
    monkeypatch.setattr(
        "tributo.integrations.broker_registry.entry_points", entry_points
    )

    plugins = discover_broker_plugins()

    assert list(plugins) == ["fake"]
    assert isinstance(plugins["fake"], _Plugin)


def test_explicit_missing_or_broken_provider_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "tributo.integrations.broker_registry.entry_points",
        MagicMock(
            return_value=[
                _EntryPoint(
                    "broken", "broken:Plugin", error=ImportError("redis missing")
                )
            ]
        ),
    )

    with pytest.raises(BrokerProviderNotFoundError, match="missing"):
        get_broker_plugin("missing")
    with pytest.raises(BrokerProviderNotFoundError, match="broken"):
        get_broker_plugin("broken")


def test_api_version_and_entry_point_identity_are_validated(monkeypatch) -> None:
    plugin = _Plugin()
    plugin.api_version = 2
    monkeypatch.setattr(
        "tributo.integrations.broker_registry.entry_points",
        MagicMock(return_value=[_EntryPoint("alias", "fake:Plugin", loaded=plugin)]),
    )

    assert discover_broker_plugins() == {}
    with pytest.raises(BrokerProviderNotFoundError, match="alias"):
        get_broker_plugin("alias")


def test_duplicate_provider_names_are_ambiguous_and_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "tributo.integrations.broker_registry.entry_points",
        MagicMock(
            return_value=[
                _EntryPoint("fake", "first:Plugin"),
                _EntryPoint("fake", "second:Plugin"),
            ]
        ),
    )

    assert discover_broker_plugins() == {}
    statuses = list_broker_provider_statuses()
    assert len(statuses) == 1
    assert statuses[0].broker_id == "fake"
    assert statuses[0].available is False
    assert "ambiguous" in (statuses[0].error or "")
    assert "first:Plugin" in (statuses[0].error or "")
    assert "second:Plugin" in (statuses[0].error or "")
    with pytest.raises(BrokerProviderNotFoundError, match="ambiguous"):
        get_broker_plugin("fake")
