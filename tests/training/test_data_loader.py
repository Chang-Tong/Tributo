"""Unit tests for training/data_loader.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tributo.training.data_loader import load_ray_dataset_from_config


def test_local_csv_absolute_path_does_not_require_project_root(tmp_path, monkeypatch):
    data_file = tmp_path / "train.csv"
    data_file.write_text("feature,label\n1,0\n")
    unrelated_directory = tmp_path / "worker"
    unrelated_directory.mkdir()
    monkeypatch.chdir(unrelated_directory)

    with patch("ray.data.read_csv") as read_csv:
        expected = MagicMock()
        read_csv.return_value = expected
        actual = load_ray_dataset_from_config(
            {"type": "csv", "format": "csv", "path": str(data_file)}
        )

    assert actual is expected
    read_csv.assert_called_once_with(str(data_file))


def test_local_csv_relative_path_resolves_from_worker_cwd(tmp_path, monkeypatch):
    data_file = tmp_path / "train.csv"
    data_file.write_text("feature,label\n1,0\n")
    monkeypatch.chdir(tmp_path)

    with patch("ray.data.read_csv") as read_csv:
        expected = MagicMock()
        read_csv.return_value = expected
        actual = load_ray_dataset_from_config(
            {"type": "csv", "format": "csv", "path": "train.csv"}
        )

    assert actual is expected
    read_csv.assert_called_once_with(str(data_file))


def test_s3_parquet_uses_parquet_connector():
    with patch("tributo.data.get_connector") as mock_get_connector:
        mock_connector = MagicMock()
        mock_connector.read.return_value = MagicMock()
        mock_get_connector.return_value = mock_connector
        load_ray_dataset_from_config(
            {
                "type": "s3",
                "uri": "s3://bucket/data.parquet",
                "format": "parquet",
            }
        )
        mock_get_connector.assert_called_once_with("parquet")


def test_s3_csv_uses_csv_connector():
    with patch("tributo.data.get_connector") as mock_get_connector:
        mock_connector = MagicMock()
        mock_connector.read.return_value = MagicMock()
        mock_get_connector.return_value = mock_connector
        load_ray_dataset_from_config(
            {
                "type": "s3",
                "uri": "s3://bucket/data.csv",
                "format": "csv",
                "s3": {"region": "us-east-1"},
            }
        )
        mock_get_connector.assert_called_once_with("csv")


def test_s3_unsupported_format_raises():
    with patch("tributo.data.get_connector"):
        try:
            load_ray_dataset_from_config(
                {
                    "type": "s3",
                    "uri": "s3://bucket/data.orc",
                    "format": "orc",
                }
            )
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "unsupported s3 format" in str(exc)


def test_clickhouse_legacy_config_is_dispatched_to_connector():
    with patch("tributo.data.get_connector") as get_connector:
        connector = MagicMock()
        get_connector.return_value = connector
        expected = MagicMock()
        connector.read.return_value = expected

        result = load_ray_dataset_from_config(
            {
                "type": "clickhouse",
                "ch_host": "clickhouse.internal",
                "ch_port": 8123,
                "ch_database": "analytics",
                "ch_user": "reader",
                "ch_password": "secret",
                "ch_sql": "SELECT * FROM events WHERE tenant={tenant:String}",
                "ch_sql_params": {"tenant": "acme"},
                "ch_sort_key": "id",
            }
        )

    assert result is expected
    get_connector.assert_called_once_with("clickhouse")
    assert connector.read.call_args.kwargs == {
        "host": "clickhouse.internal",
        "port": 8123,
        "database": "analytics",
        "username": "reader",
        "password": "secret",
        "sql": "SELECT * FROM events WHERE tenant={tenant:String}",
        "params": {"tenant": "acme"},
        "sort_key": "id",
    }


def test_hive_legacy_config_is_dispatched_to_connector():
    with patch("tributo.data.get_connector") as get_connector:
        connector = MagicMock()
        get_connector.return_value = connector
        expected = MagicMock()
        connector.read.return_value = expected

        result = load_ray_dataset_from_config(
            {
                "type": "hive",
                "hive_host": "hive.internal",
                "hive_port": 10000,
                "hive_database": "warehouse",
                "hive_user": "reader",
                "hive_password": "secret",
                "hive_auth": "LDAP",
                "hive_sql": "SELECT id FROM events WHERE tenant=%(tenant)s",
                "hive_sql_params": {"tenant": "acme"},
                "hive_hash_column": "id",
                "hive_hash_shards": 8,
            }
        )

    assert result is expected
    get_connector.assert_called_once_with("hive")
    assert connector.read.call_args.kwargs == {
        "host": "hive.internal",
        "port": 10000,
        "database": "warehouse",
        "username": "reader",
        "password": "secret",
        "auth": "LDAP",
        "sql": "SELECT id FROM events WHERE tenant=%(tenant)s",
        "params": {"tenant": "acme"},
        "hash_column": "id",
        "hash_shards": 8,
    }


@pytest.mark.parametrize(
    ("kind", "port"),
    [("clickhouse", 8123), ("hive", 10000)],
)
def test_database_canonical_config_is_dispatched_by_type(kind, port):
    with patch("tributo.data.get_connector") as get_connector:
        connector = MagicMock()
        get_connector.return_value = connector
        connector.read.return_value = MagicMock()

        raw = {
            "type": kind,
            "host": f"{kind}.internal",
            "port": port,
            "database": "analytics",
            "username": "reader",
            "password": "secret",
            "sql": "SELECT id FROM events",
            "params": {"tenant": "acme"},
        }
        if kind == "hive":
            raw["auth"] = "CUSTOM"
        load_ray_dataset_from_config(raw)

    get_connector.assert_called_once_with(kind)
    expected = {
        "host": f"{kind}.internal",
        "port": port,
        "database": "analytics",
        "username": "reader",
        "password": "secret",
        "sql": "SELECT id FROM events",
        "params": {"tenant": "acme"},
    }
    if kind == "hive":
        expected["auth"] = "CUSTOM"
    assert connector.read.call_args.kwargs == expected


def test_xgboost_data_config_round_trip_preserves_hive_loader_fields():
    from tributo.training.xgboost_trainer import DataConfig

    config = DataConfig.model_validate(
        {
            "type": "hive",
            "host": "hive.internal",
            "port": 10000,
            "database": "warehouse",
            "username": "reader",
            "password": "secret",
            "auth": "LDAP",
            "sql": "SELECT id FROM events WHERE tenant=%(tenant)s",
            "params": {"tenant": "acme"},
            "hive_hash_column": "id",
        }
    ).model_dump()

    with patch("tributo.data.get_connector") as get_connector:
        connector = MagicMock()
        get_connector.return_value = connector
        connector.read.return_value = MagicMock()
        load_ray_dataset_from_config(config)

    assert connector.read.call_args.kwargs == {
        "host": "hive.internal",
        "port": 10000,
        "database": "warehouse",
        "username": "reader",
        "password": "secret",
        "auth": "LDAP",
        "sql": "SELECT id FROM events WHERE tenant=%(tenant)s",
        "params": {"tenant": "acme"},
        "batch_size": 10_000,
        "shard_mode": "auto",
        "hash_column": "id",
        "hash_shards": 64,
    }


def test_hive_unset_model_credentials_fall_back_to_worker_environment(
    monkeypatch,
):
    from tributo.training.xgboost_trainer import DataConfig

    monkeypatch.setenv("TRIBUTO_HIVE_AUTH", "LDAP")
    monkeypatch.setenv("TRIBUTO_HIVE_PASSWORD", "env-secret")
    config = DataConfig.model_validate(
        {"type": "hive", "hive_sql": "SELECT id FROM events"}
    ).model_dump()

    with patch("tributo.data.get_connector") as get_connector:
        connector = MagicMock()
        get_connector.return_value = connector
        connector.read.return_value = MagicMock()
        load_ray_dataset_from_config(config)

    assert connector.read.call_args.kwargs["auth"] == "LDAP"
    assert connector.read.call_args.kwargs["password"] == "env-secret"


@pytest.mark.parametrize(
    "credentials",
    [
        {"auth": "CUSTOM", "password": "config-secret"},
        {"hive_auth": "CUSTOM", "hive_password": "config-secret"},
    ],
)
def test_hive_explicit_credentials_override_worker_environment(
    credentials, monkeypatch
):
    monkeypatch.setenv("TRIBUTO_HIVE_AUTH", "LDAP")
    monkeypatch.setenv("TRIBUTO_HIVE_PASSWORD", "env-secret")

    with patch("tributo.data.get_connector") as get_connector:
        connector = MagicMock()
        get_connector.return_value = connector
        connector.read.return_value = MagicMock()
        load_ray_dataset_from_config(
            {
                "type": "hive",
                "hive_sql": "SELECT id FROM events",
                **credentials,
            }
        )

    assert connector.read.call_args.kwargs["auth"] == "CUSTOM"
    assert connector.read.call_args.kwargs["password"] == "config-secret"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            {
                "type": "clickhouse",
                "host": "clickhouse.internal",
                "port": 8123,
                "database": "analytics",
                "username": "reader",
                "password": "secret",
                "sql": "SELECT id FROM events",
                "params": {"tenant": "acme"},
                "sort_key": "id",
                "parallelism": 4,
            },
            {
                "host": "clickhouse.internal",
                "port": 8123,
                "database": "analytics",
                "username": "reader",
                "password": "secret",
                "sql": "SELECT id FROM events",
                "params": {"tenant": "acme"},
                "sort_key": "id",
                "parallelism": 4,
            },
        ),
        (
            {
                "type": "hive",
                "host": "hive.internal",
                "port": 10000,
                "database": "warehouse",
                "username": "reader",
                "password": "secret",
                "auth": "CUSTOM",
                "sql": "SELECT id FROM events",
                "params": {"tenant": "acme"},
                "parallelism": 4,
                "batch_size": 512,
                "shard_mode": "hash",
                "hash_column": "id",
                "hash_shards": 8,
            },
            {
                "host": "hive.internal",
                "port": 10000,
                "database": "warehouse",
                "username": "reader",
                "password": "secret",
                "auth": "CUSTOM",
                "sql": "SELECT id FROM events",
                "params": {"tenant": "acme"},
                "parallelism": 4,
                "batch_size": 512,
                "shard_mode": "hash",
                "hash_column": "id",
                "hash_shards": 8,
            },
        ),
    ],
)
def test_xgboost_canonical_sharding_round_trip_reaches_connector(raw, expected):
    from tributo.training.xgboost_trainer import DataConfig

    config = DataConfig.model_validate(raw).model_dump()
    with patch("tributo.data.get_connector") as get_connector:
        connector = MagicMock()
        get_connector.return_value = connector
        connector.read.return_value = MagicMock()
        load_ray_dataset_from_config(config)

    get_connector.assert_called_once_with(raw["type"])
    assert connector.read.call_args.kwargs == expected
