"""Discovery and validation for Broker API v1 providers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import entry_points

from tributo.integrations.broker import BROKER_API_VERSION, BrokerPlugin
from tributo.plugin import _get_enabled_plugins

logger = logging.getLogger(__name__)

BROKER_ENTRY_POINT_GROUP = "tributo.brokers"


class BrokerProviderNotFoundError(LookupError):
    """Raised when an explicitly selected provider cannot be used."""


@dataclass(frozen=True)
class BrokerProviderStatus:
    """Discovery result for one unique provider name."""

    broker_id: str
    plugin: BrokerPlugin | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        """Return whether the provider can be selected."""
        return self.plugin is not None and self.error is None


def _load_entry_point(entry_point: object) -> BrokerPlugin:
    name = str(getattr(entry_point, "name", ""))
    loaded = entry_point.load()  # type: ignore[attr-defined]
    plugin = loaded() if isinstance(loaded, type) else loaded
    if not isinstance(plugin, BrokerPlugin):
        raise TypeError(f"entry point {name!r} is not a BrokerPlugin")
    if plugin.broker_id != name:
        raise ValueError(
            f"entry point {name!r} declares broker_id {plugin.broker_id!r}"
        )
    if plugin.api_version != BROKER_API_VERSION:
        raise ValueError(
            f"provider {name!r} uses Broker API {plugin.api_version}; "
            f"Core supports {BROKER_API_VERSION}"
        )
    return plugin


def _broker_entry_points() -> list[object]:
    return list(entry_points(group=BROKER_ENTRY_POINT_GROUP))


def _enabled_entry_points_by_name() -> dict[str, list[object]]:
    enabled = _get_enabled_plugins()
    grouped: dict[str, list[object]] = {}
    for entry_point in _broker_entry_points():
        name = str(getattr(entry_point, "name", ""))
        if enabled is not None and name not in enabled:
            continue
        grouped.setdefault(name, []).append(entry_point)
    return grouped


def _ambiguity_error(name: str, matches: list[object]) -> str:
    values = sorted(str(getattr(match, "value", "unknown")) for match in matches)
    return (
        f"Broker provider {name!r} is ambiguous: multiple entry points are "
        f"installed ({', '.join(values)})"
    )


def list_broker_provider_statuses() -> list[BrokerProviderStatus]:
    """Inspect providers, retaining unavailable and ambiguous entries."""
    statuses: list[BrokerProviderStatus] = []
    for name, matches in sorted(_enabled_entry_points_by_name().items()):
        if len(matches) != 1:
            error = _ambiguity_error(name, matches)
            logger.warning(error)
            statuses.append(BrokerProviderStatus(broker_id=name, error=error))
            continue
        entry_point = matches[0]
        try:
            plugin = _load_entry_point(entry_point)
        except Exception as exc:
            error = f"Broker provider {name!r} could not be loaded: {exc}"
            logger.warning(
                "%s (%s)",
                error,
                getattr(entry_point, "value", "unknown"),
                exc_info=True,
            )
            statuses.append(BrokerProviderStatus(broker_id=name, error=error))
        else:
            statuses.append(BrokerProviderStatus(broker_id=name, plugin=plugin))
    return statuses


def discover_broker_plugins() -> dict[str, BrokerPlugin]:
    """Return valid providers, isolating imports and broken installations."""
    return {
        status.broker_id: status.plugin
        for status in list_broker_provider_statuses()
        if status.available and status.plugin is not None
    }


def get_broker_plugin(broker_id: str) -> BrokerPlugin:
    """Load an explicitly selected provider or fail closed."""
    enabled = _get_enabled_plugins()
    if enabled is not None and broker_id not in enabled:
        raise BrokerProviderNotFoundError(
            f"Broker provider {broker_id!r} is disabled by TRIBUTO_PLUGINS"
        )
    matches = _enabled_entry_points_by_name().get(broker_id, [])
    if not matches:
        raise BrokerProviderNotFoundError(
            f"Broker provider {broker_id!r} is not installed"
        )
    if len(matches) != 1:
        raise BrokerProviderNotFoundError(_ambiguity_error(broker_id, matches))
    try:
        return _load_entry_point(matches[0])
    except Exception as exc:
        raise BrokerProviderNotFoundError(
            f"Broker provider {broker_id!r} could not be loaded: {exc}"
        ) from exc
