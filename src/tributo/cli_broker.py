"""CLI commands for optional Broker API v1 providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from tributo.integrations.broker_runner import BrokerRunner


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Invalid broker config: {exc}") from exc
    if not isinstance(value, dict):
        raise click.ClickException("Broker config root must be a JSON object")
    return value


@click.group()
def broker() -> None:
    """Manage optional message broker providers."""


@broker.command("list")
def list_providers() -> None:
    """List installed and valid broker providers."""
    from tributo.integrations.broker_registry import list_broker_provider_statuses

    statuses = list_broker_provider_statuses()
    if not statuses:
        click.echo("No broker providers installed.")
        return
    for status in statuses:
        if not status.available or status.plugin is None:
            click.echo(f"{status.broker_id}\tunavailable\terror={status.error}")
            continue
        capabilities = ",".join(sorted(status.plugin.capabilities)) or "-"
        click.echo(
            f"{status.broker_id}\tapi={status.plugin.api_version}"
            f"\tcapabilities={capabilities}"
        )


@broker.command()
@click.option("--broker", "broker_id", required=True, help="Broker provider ID")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--check-connectivity", is_flag=True)
def validate(broker_id: str, config_path: Path, check_connectivity: bool) -> None:
    """Validate a provider JSON configuration."""
    from tributo.integrations.broker_registry import get_broker_plugin

    try:
        plugin = get_broker_plugin(broker_id)
        plugin.validate_config(
            _load_config(config_path), check_connectivity=check_connectivity
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Broker provider {broker_id!r} configuration is valid.")


@broker.command()
@click.option("--broker", "broker_id", required=True, help="Broker provider ID")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--once", is_flag=True, help="Poll at most one message and exit")
@click.option(
    "--poll-timeout-ms",
    default=5000,
    show_default=True,
    type=click.IntRange(min=0),
)
def consume(
    broker_id: str, config_path: Path, once: bool, poll_timeout_ms: int
) -> None:
    """Consume training tasks through a broker provider."""
    from tributo.integrations.broker_registry import get_broker_plugin

    try:
        plugin = get_broker_plugin(broker_id)
        config = _load_config(config_path)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    runner = BrokerRunner(plugin, config, poll_timeout_ms=poll_timeout_ms)
    try:
        if once:
            runner.run_once()
        else:
            runner.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()
