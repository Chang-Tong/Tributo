"""Tests for CLI commands using Click's CliRunner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tributo.cli import main


@pytest.fixture
def runner():
    """Provide a Click CliRunner."""
    return CliRunner()


# --- submit ---


class TestSubmitCommand:
    """Tests for the 'submit' CLI command."""

    def test_submit_success(self, runner):
        """submit should print the job ID on success."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.submit.return_value = "job-abc123"
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "submit",
                    "--address",
                    "http://127.0.0.1:8265",
                    "--entrypoint",
                    "python script.py",
                ],
            )
            assert result.exit_code == 0
            assert "job-abc123" in result.output

    def test_submit_with_config_file(self, runner, tmp_path):
        """submit should load config from JSON file."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"num_cpus": 4.0, "num_gpus": 1.0}')
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.submit.return_value = "job-1"
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "submit",
                    "--address",
                    "http://127.0.0.1:8265",
                    "--entrypoint",
                    "python train.py",
                    "--config",
                    str(config_file),
                ],
            )
            assert result.exit_code == 0

    def test_submit_failure(self, runner):
        """submit should print error and exit 1 on failure."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.submit.side_effect = RuntimeError("connection refused")
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "submit",
                    "--address",
                    "http://127.0.0.1:8265",
                    "--entrypoint",
                    "python script.py",
                ],
            )
            assert result.exit_code == 1
            assert "connection refused" in result.output

    def test_submit_missing_entrypoint(self, runner):
        """submit should fail when --entrypoint is missing."""
        result = runner.invoke(
            main,
            [
                "submit",
                "--address",
                "http://127.0.0.1:8265",
            ],
        )
        assert result.exit_code != 0


# --- status ---


class TestStatusCommand:
    """Tests for the 'status' CLI command."""

    def test_status_success(self, runner):
        """status should print the job status."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.get_status.return_value = "RUNNING"
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "status",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 0
            assert "RUNNING" in result.output

    def test_status_failure(self, runner):
        """status should print error and exit 1 on failure."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.get_status.side_effect = RuntimeError("not found")
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "status",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 1
            assert "not found" in result.output


# --- logs ---


class TestLogsCommand:
    """Tests for the 'logs' CLI command."""

    def test_logs_success(self, runner):
        """logs should print the job logs."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.get_logs.return_value = "hello\nworld\n"
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "logs",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 0
            assert "hello" in result.output

    def test_logs_failure(self, runner):
        """logs should print error and exit 1 on failure."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.get_logs.side_effect = RuntimeError("timeout")
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "logs",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 1
            assert "timeout" in result.output


# --- stop ---


class TestStopCommand:
    """Tests for the 'stop' CLI command."""

    def test_stop_success(self, runner):
        """stop should print success message."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.stop_job.return_value = True
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "stop",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 0
            assert "stopped successfully" in result.output

    def test_stop_failure(self, runner):
        """stop should print error and exit 1 on failure."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.stop_job.side_effect = RuntimeError("forbidden")
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "stop",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 1
            assert "forbidden" in result.output


# --- embed ---


class TestEmbedCommand:
    """Tests for the 'embed' CLI command group."""

    def test_embed_list(self, runner):
        """embed list should print model names."""
        with patch("tributo.embeddings.registry.list_models") as mock_list:
            mock_list.return_value = ["bge-small-zh", "bge-base-zh"]
            result = runner.invoke(main, ["embed", "list"])
            assert result.exit_code == 0
            assert "bge-small-zh" in result.output
            assert "bge-base-zh" in result.output

    def test_embed_batch_success(self, runner):
        """embed batch should print the job ID on success."""
        with patch("tributo.embeddings.job_runner.submit_embedding_job") as mock_submit:
            mock_submit.return_value = "embed-job-1"
            result = runner.invoke(
                main,
                [
                    "embed",
                    "batch",
                    "--input",
                    "s3://bucket/in.parquet",
                    "--output",
                    "s3://bucket/out.lance",
                ],
            )
            assert result.exit_code == 0
            assert "embed-job-1" in result.output

    def test_embed_batch_failure(self, runner):
        """embed batch should print error and exit 1 on failure."""
        with patch("tributo.embeddings.job_runner.submit_embedding_job") as mock_submit:
            mock_submit.side_effect = RuntimeError("S3 error")
            result = runner.invoke(
                main,
                [
                    "embed",
                    "batch",
                    "--input",
                    "s3://bucket/in.parquet",
                    "--output",
                    "s3://bucket/out.lance",
                ],
            )
            assert result.exit_code == 1
            assert "S3 error" in result.output


# --- version ---


def test_version(runner):
    """--version should print the version."""
    with patch("importlib.metadata.version", return_value="1.1.0"):
        result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "1.1.0" in result.output


class TestBrokerCommands:
    """Tests for Broker CLI discovery and lifecycle commands."""

    def test_broker_list_does_not_fail_for_broken_provider(self, runner):
        with patch(
            "tributo.integrations.broker_registry.list_broker_provider_statuses"
        ) as statuses:
            statuses.return_value = []
            result = runner.invoke(main, ["broker", "list"])
        assert result.exit_code == 0
        assert "No broker providers installed" in result.output

    def test_broker_list_marks_ambiguous_provider_unavailable(self, runner):
        status = MagicMock()
        status.broker_id = "duplicate"
        status.available = False
        status.plugin = None
        status.error = "ambiguous: multiple entry points are installed"
        with patch(
            "tributo.integrations.broker_registry.list_broker_provider_statuses",
            return_value=[status],
        ):
            result = runner.invoke(main, ["broker", "list"])
        assert result.exit_code == 0
        assert "duplicate" in result.output
        assert "unavailable" in result.output
        assert "ambiguous" in result.output

    def test_ordinary_cli_does_not_discover_broker_providers(self, runner):
        with patch("tributo.integrations.broker_registry.entry_points") as entry_points:
            result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        entry_points.assert_not_called()

    def test_broker_validate_missing_provider_fails_closed(self, runner, tmp_path):
        config = tmp_path / "broker.json"
        config.write_text("{}")
        result = runner.invoke(
            main,
            [
                "broker",
                "validate",
                "--broker",
                "missing",
                "--config",
                str(config),
            ],
        )
        assert result.exit_code != 0
        assert "missing" in result.output

    def test_broker_consume_once_drives_runner(self, runner, tmp_path):
        config = tmp_path / "broker.json"
        config.write_text("{}")
        plugin = MagicMock()
        plugin.broker_id = "fake"
        with (
            patch(
                "tributo.integrations.broker_registry.get_broker_plugin",
                return_value=plugin,
            ),
            patch("tributo.cli_broker.BrokerRunner") as runner_cls,
        ):
            result = runner.invoke(
                main,
                [
                    "broker",
                    "consume",
                    "--broker",
                    "fake",
                    "--config",
                    str(config),
                    "--once",
                ],
            )
        assert result.exit_code == 0, result.output
        runner_cls.return_value.run_once.assert_called_once_with()
        runner_cls.return_value.close.assert_called_once_with()

    def test_broker_consume_rejects_negative_poll_timeout(self, runner, tmp_path):
        config = tmp_path / "broker.json"
        config.write_text("{}")
        result = runner.invoke(
            main,
            [
                "broker",
                "consume",
                "--broker",
                "fake",
                "--config",
                str(config),
                "--poll-timeout-ms",
                "-1",
            ],
        )
        assert result.exit_code == 2
        assert "range" in result.output
        assert "0" in result.output


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
