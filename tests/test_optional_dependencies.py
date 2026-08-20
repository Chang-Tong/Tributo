"""Distribution metadata contracts for platform-safe optional dependencies."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def _optional_dependencies() -> dict[str, list[str]]:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    with pyproject.open("rb") as stream:
        project = tomllib.load(stream)["project"]
    return project["optional-dependencies"]


def test_hive_extra_uses_nosasl_pure_python_dependency_stack() -> None:
    optional = _optional_dependencies()
    requirements = [Requirement(value) for value in optional["hive"]]
    by_name = {requirement.name.lower(): requirement for requirement in requirements}

    assert set(by_name) == {"pyhive", "thrift", "thrift-sasl", "pure-sasl"}
    assert by_name["pyhive"].extras == set()
    assert "sasl" not in by_name
    assert Version("0.7.0") in by_name["pyhive"].specifier
    assert Version("0.8.0") not in by_name["pyhive"].specifier
    assert Version("0.24.0") in by_name["thrift"].specifier
    assert Version("0.23.0") not in by_name["thrift"].specifier
    assert Version("1.0") not in by_name["thrift"].specifier
    assert Version("0.4.3") in by_name["thrift-sasl"].specifier
    assert Version("1.0") not in by_name["thrift-sasl"].specifier
    assert Version("0.6.2") in by_name["pure-sasl"].specifier
    assert Version("1.0") not in by_name["pure-sasl"].specifier


def test_training_extra_includes_platform_safe_hive_extra() -> None:
    training = _optional_dependencies()["training"]
    assert "tributo[clickhouse,data,hive]" in training
