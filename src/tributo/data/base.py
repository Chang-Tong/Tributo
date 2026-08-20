"""DataConnector abstract base class and shared configuration types.

DataConnector is a configuration-driven data access facade — a higher-level
abstraction than Ray Datasource (the low-level protocol interface).  It
parses format-specific configs and returns ``ray.data.Dataset`` or writes
data consistently across storage backends.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data


_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_ROW_WIDTH_BYTES = 200
_VARIABLE_FIELD_BYTES = 64


def validate_sql_identifier(name: str) -> str:
    """Validate a simple SQL result-column identifier.

    Shard columns are interpolated into wrapper queries because database
    drivers cannot bind identifiers as parameters. Keeping the accepted
    grammar deliberately small makes that interpolation safe and portable.
    """
    if not isinstance(name, str) or not _SQL_IDENTIFIER.fullmatch(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def quote_sql_identifier(name: str) -> str:
    """Validate and backtick-quote a ClickHouse/Hive identifier."""
    return f"`{validate_sql_identifier(name)}`"


def deterministic_order_by(schema: Any) -> str:
    """Return a deterministic ORDER BY clause for OFFSET sharding."""
    if schema is None:
        raise ValueError("A result schema is required for deterministic sharding")
    names = getattr(schema, "names", None)
    if not names:
        raise ValueError("At least one result column is required for sharding")
    return " ORDER BY " + ", ".join(quote_sql_identifier(name) for name in names)


def schema_row_width(schema: Any) -> int:
    """Conservatively estimate bytes per Arrow row for Ray planning."""
    if schema is None:
        return _DEFAULT_ROW_WIDTH_BYTES
    import pyarrow as pa

    total = 0
    for field in schema:
        field_type = field.type
        if pa.types.is_boolean(field_type):
            total += 1
        elif pa.types.is_integer(field_type) or pa.types.is_floating(field_type):
            total += max(1, field_type.bit_width // 8)
        elif (
            pa.types.is_timestamp(field_type)
            or pa.types.is_date(field_type)
            or pa.types.is_time(field_type)
        ):
            total += 8
        else:
            total += _VARIABLE_FIELD_BYTES
    return max(1, total)


@PublicAPI(stability="alpha")
class WriteMode(Enum):
    """Write mode.

    Note:
        APPEND support varies by format.  Parquet drivers typically
        overwrite, while Lance and Iceberg natively support inserts.
    """

    OVERWRITE = "overwrite"
    APPEND = "append"


@PublicAPI(stability="alpha")
class S3Config(BaseModel):
    """S3 connection configuration, shared across all data modules.

    Environment variable fallbacks:
    ``S3_ENDPOINT`` / ``AWS_ENDPOINT_URL`` → endpoint,
    ``AWS_ACCESS_KEY_ID`` → access_key_id,
    ``AWS_SECRET_ACCESS_KEY`` → secret_access_key,
    ``AWS_REGION`` → region.
    """

    access_key_id: Optional[str] = Field(default=None, description="AWS Access Key ID")
    secret_access_key: Optional[str] = Field(
        default=None, description="AWS Secret Access Key"
    )
    endpoint: Optional[str] = Field(default=None, description="S3 endpoint URL")
    region: Optional[str] = Field(default=None, description="AWS region")


@PublicAPI(stability="alpha")
class DataConnector(ABC):
    """Abstract base class for data connectors.

    Subclasses must implement ``read()`` and ``write()``.  The API uses
    ``**kwargs`` with loose signatures — each subclass defines its own
    Pydantic config model for parameter validation.

    Example::

        connector = get_connector("parquet")
        ds = connector.read(path="s3://bucket/data.parquet", s3=S3Config(...))
        connector.write(ds, path="s3://bucket/output", mode=WriteMode.OVERWRITE)
    """

    @abstractmethod
    def read(self, **kwargs: Any) -> ray.data.Dataset:
        """Read data and return a Ray Dataset.

        Args:
            **kwargs: Format-specific config fields, parsed by the
                subclass Pydantic model.

        Returns:
            A lazy ``ray.data.Dataset``.
        """

    @abstractmethod
    def write(self, dataset: ray.data.Dataset, **kwargs: Any) -> None:
        """Write a Ray Dataset to the target storage.

        Args:
            dataset: The dataset to write.
            **kwargs: Format-specific config fields, parsed by the
                subclass Pydantic model.
        """

    def exists(self, **kwargs: Any) -> bool:
        """Check whether the target path exists.

        Args:
            **kwargs: Format-specific config fields.

        Returns:
            ``True`` if the path exists; ``False`` by default.
        """
        return False
