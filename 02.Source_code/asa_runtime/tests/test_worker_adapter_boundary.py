from __future__ import annotations

from sim_agent.agent_runtime import WorkerAdapterConfig, default_worker_adapter_config, worker_adapter_status


def test_default_worker_adapter_is_disabled_and_not_attached_to_default_runtime() -> None:
    config = default_worker_adapter_config()
    status = worker_adapter_status(config)

    assert config.kind == "disabled"
    assert status.status == "disabled"
    assert status.enabled is False
    assert status.default_runtime_attached is False
    assert status.blocker == "disabled"


def test_tmux_worker_adapter_requires_explicit_command_and_stays_optional() -> None:
    missing = worker_adapter_status(WorkerAdapterConfig(kind="tmux"))
    ready = worker_adapter_status(WorkerAdapterConfig(kind="tmux", command=("omx", "team")))
    process = worker_adapter_status(WorkerAdapterConfig(kind="process", command=("asa-worker",)))

    assert missing.status == "blocked"
    assert missing.blocker == "missing_worker_command"
    assert missing.default_runtime_attached is False
    assert ready.status == "ready"
    assert ready.enabled is True
    assert ready.default_runtime_attached is False
    assert process.status == "ready"
    assert process.default_runtime_attached is False
