from __future__ import annotations

from sim_agent.agent_runtime.agent_specs import SUBAGENT_PRESETS, resolve_subagent_preset
from sim_agent.agent_harness.tools import default_tool_registry


def test_subagent_presets_are_clean_room_non_persistent_and_bounded() -> None:
    # Given: the C001 global bounded subagent preset contract.
    expected_names = ("planner", "architect", "critic", "executor", "verifier")

    # When: presets are resolved by their model-visible names.
    presets = tuple(resolve_subagent_preset(name) for name in expected_names)

    # Then: only the requested clean-room bounded presets are represented.
    assert tuple(SUBAGENT_PRESETS) == expected_names
    assert tuple(preset.name for preset in presets) == expected_names
    assert all(not preset.persistent for preset in presets)
    assert all(preset.clean_room for preset in presets)
    assert all(preset.max_depth == 1 for preset in presets)
    assert all("simulation_" not in preset.name for preset in presets)
    assert "researcher" not in SUBAGENT_PRESETS


def test_subagent_presets_expose_bounded_report_tool_surface_without_bash() -> None:
    # Given: the default runtime tool registry includes model-visible subagent tools.
    registry = default_tool_registry()

    # When: each preset declares the tools a bounded child may use.
    presets = tuple(resolve_subagent_preset(name) for name in SUBAGENT_PRESETS)

    # Then: every preset can return a report and inspect subagent state without process access.
    assert {"subagent_task", "subagent_inspect"}.issubset(registry.tool_names)
    assert all("subagent_inspect" in preset.tool_names for preset in presets)
    assert all("artifact_write" in preset.tool_names for preset in presets)
    assert all("bash_process" not in preset.tool_names for preset in presets)
    assert "code" in resolve_subagent_preset("critic").scope_notes.casefold()
    assert "design" in resolve_subagent_preset("critic").scope_notes.casefold()
    assert "workflow" in resolve_subagent_preset("critic").scope_notes.casefold()
    assert "tool safety" in resolve_subagent_preset("critic").scope_notes.casefold()
    assert "evidence quality" in resolve_subagent_preset("critic").scope_notes.casefold()
    assert "scientific validity" in resolve_subagent_preset("critic").scope_notes.casefold()
    assert "ledger" in resolve_subagent_preset("verifier").scope_notes.casefold()
    assert "validity" in resolve_subagent_preset("verifier").scope_notes.casefold()
