"""Runtime environment tests for optional worker-side providers."""

from __future__ import annotations

import json

import pytest

from tributo._common.runtime_env import build_runtime_env


def test_runtime_env_adds_provider_modules_pip_and_context(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    core = tmp_path / "src" / "tributo"
    core.mkdir(parents=True)
    provider = tmp_path / "provider-src"
    provider.mkdir()
    context = {
        "cancellation": {
            "broker_id": "knova-redis",
            "job_id": "job-1",
            "options": {"config_env": "TRIBUTO_BROKER_CONFIG_JSON"},
        }
    }

    runtime_env = build_runtime_env(
        project_root=tmp_path,
        env_vars={"OPAQUE": "unchanged"},
        extra_py_modules=[provider, str(provider)],
        runtime_pip_packages=["/provider/provider.whl"],
        execution_context=context,
    )

    assert runtime_env["py_modules"] == [str(core), str(provider)]
    assert runtime_env["pip"] == ["/provider/provider.whl"]
    assert runtime_env["env_vars"]["OPAQUE"] == "unchanged"
    serialized = json.loads(runtime_env["env_vars"]["TRIBUTO_EXECUTION_CONTEXT"])
    assert serialized["schema"] == "tributo.execution-context"
    assert serialized["version"] == 1
    assert serialized["cancellation"] == context["cancellation"]


def test_runtime_env_without_provider_options_keeps_legacy_shape(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    core = tmp_path / "src" / "tributo"
    core.mkdir(parents=True)

    runtime_env = build_runtime_env(project_root=tmp_path)

    assert runtime_env["py_modules"] == [str(core)]
    assert "pip" not in runtime_env
    assert "TRIBUTO_EXECUTION_CONTEXT" not in runtime_env.get("env_vars", {})


def test_runtime_env_rejects_inline_secret_in_execution_context(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "src" / "tributo").mkdir(parents=True)

    with pytest.raises(ValueError, match="inline secret"):
        build_runtime_env(
            project_root=tmp_path,
            execution_context={
                "cancellation": {
                    "broker_id": "fake",
                    "job_id": "job-1",
                    "options": {"password": "redis-password-value"},
                }
            },
        )
