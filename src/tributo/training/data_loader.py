"""Configuration dispatch for distributed Ray training datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import ray.data


def load_ray_dataset_from_config(
    data_config: dict,
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Load a Ray Dataset from a YAML ``data`` section (lazy, data does not land in Driver memory).

    Supports local CSV/Parquet, S3, ClickHouse, and HiveServer2.

    Args:
        data_config: The ``data`` dictionary from YAML.
        project_root_path: Optional base directory for resolving relative paths. If
            omitted, relative paths are resolved from the worker's current directory.

    Returns:
        Ray Dataset, lazily evaluated.
    """
    import ray.data

    data_type = data_config.get("type", "csv")

    if data_type == "csv":
        file_path = data_config.get("path", "")
        fmt = data_config.get("format", "")
        full_path = Path(file_path)
        if not full_path.is_absolute():
            root = project_root_path or Path.cwd()
            full_path = root / full_path
        if not full_path.exists():
            raise FileNotFoundError(f"Data file not found: {full_path}")
        if fmt == "csv":
            return ray.data.read_csv(str(full_path))
        return ray.data.read_parquet(str(full_path))

    if data_type == "s3":
        from tributo.data import S3Config, get_connector

        uri = data_config.get("uri", "")
        if not uri:
            raise ValueError("missing s3 uri in data config")
        fmt = data_config.get("format", "parquet")
        if fmt not in {"parquet", "csv"}:
            raise ValueError(f"unsupported s3 format: {fmt!r}")
        s3_cfg = data_config.get("s3", {})
        s3_config = S3Config(**s3_cfg) if s3_cfg else None
        columns = data_config.get("columns")
        connector = get_connector(fmt)
        return connector.read(path=uri, s3=s3_config, columns=columns)

    if data_type == "clickhouse":
        return _load_database_dataset("clickhouse", data_config)

    if data_type == "hive":
        return _load_database_dataset("hive", data_config)

    raise ValueError(f"unsupported data type: {data_type}")


def _load_database_dataset(kind: str, data_config: dict) -> "ray.data.Dataset":
    """Normalize legacy training keys and delegate to a database connector."""
    from tributo.data import get_connector

    if kind == "clickhouse":
        prefix = "ch"
        defaults: dict[str, object] = {
            "host": os.getenv("TRIBUTO_CLICKHOUSE_HOST", "localhost"),
            "port": int(os.getenv("TRIBUTO_CLICKHOUSE_PORT", "8123")),
            "database": os.getenv("TRIBUTO_CLICKHOUSE_DB", "default"),
            "username": os.getenv("TRIBUTO_CLICKHOUSE_USER", "default"),
            "password": os.getenv("TRIBUTO_CLICKHOUSE_PASSWORD", ""),
        }
        optional: tuple[str, ...] = ("sort_key", "parallelism")
    else:
        prefix = "hive"
        defaults = {
            "host": os.getenv("TRIBUTO_HIVE_HOST", "localhost"),
            "port": int(os.getenv("TRIBUTO_HIVE_PORT", "10000")),
            "database": os.getenv("TRIBUTO_HIVE_DB", "default"),
            "username": os.getenv("TRIBUTO_HIVE_USER", "default"),
            "password": os.getenv("TRIBUTO_HIVE_PASSWORD", ""),
        }
        optional = (
            "batch_size",
            "shard_mode",
            "hash_column",
            "hash_shards",
            "parallelism",
        )

    legacy_aliases = {"username": "user"}
    kwargs: dict[str, object] = {}
    for canonical, default in defaults.items():
        legacy_name = legacy_aliases.get(canonical, canonical)
        canonical_value = data_config.get(canonical)
        kwargs[canonical] = (
            canonical_value
            if canonical_value is not None
            else data_config.get(f"{prefix}_{legacy_name}", default)
        )
    canonical_sql = data_config.get("sql")
    kwargs["sql"] = (
        canonical_sql
        if canonical_sql is not None
        else data_config.get(f"{prefix}_sql", "")
    )
    canonical_params = data_config.get("params")
    kwargs["params"] = (
        canonical_params
        if canonical_params is not None
        else data_config.get(f"{prefix}_sql_params", {})
    ) or {}
    for name in optional:
        value = data_config.get(name)
        if value is None:
            value = data_config.get(f"{prefix}_{name}")
        if value is not None:
            kwargs[name] = value

    return get_connector(kind).read(**kwargs)


def load_dataframe_from_config(
    data_config: dict,
    project_root_path: Path | None = None,
) -> "pd.DataFrame":
    """Load a DataFrame from a YAML ``data`` section (fully loaded into Driver memory).

    ⚠️ Only suitable for small datasets. Use ``load_ray_dataset_from_config`` for large data.

    Args:
        data_config: The ``data`` dictionary from YAML.
        project_root_path: Optional base directory for resolving relative paths. If
            omitted, relative paths are resolved from the worker's current directory.

    Returns:
        pandas DataFrame.
    """
    ds = load_ray_dataset_from_config(data_config, project_root_path)
    return ds.to_pandas()  # type: ignore[no-any-return]
