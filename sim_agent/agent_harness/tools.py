from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    boundary: str


@dataclass(frozen=True, slots=True)
class ToolRegistry:
    tools: tuple[ToolDefinition, ...]

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(tool.name for tool in self.tools)


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        tools=(
            ToolDefinition("validate_simulation_request", "schema"),
            ToolDefinition("geometry_ingestion", "geometry"),
            ToolDefinition("md_campaign_planning", "md"),
            ToolDefinition("surrogate_status", "mdn"),
            ToolDefinition("feature_transport", "transport"),
            ToolDefinition("level_set_evolution", "profile"),
            ToolDefinition("compute_routing", "compute"),
            ToolDefinition("artifact_manifest", "evidence"),
            ToolDefinition("literature_registry", "graphdb"),
            ToolDefinition("research_source_lookup", "graphdb"),
            ToolDefinition("source_graph_import_bundle", "graphdb"),
            ToolDefinition("graphdb_ingest_report", "graphdb"),
            ToolDefinition("ui_run_status", "html_ui"),
        )
    )
