from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from sim_agent.schemas._parse import JsonMap


RUNTIME_SPINE_CONTRACT_VERSION: Final = "asa_runtime_spine_contract_v1"
TASK_ONE_EVIDENCE_PATH: Final = ".omo/evidence/task-1-asa-runtime-spine-gap-closure.json"


class RuntimeSpineStatus(StrEnum):
    GAP_OPEN = "gap_open"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class RuntimeSpine:
    spine_id: str
    name: str
    required_assertion: str
    current_gap: str
    acceptance_probe: str
    doc_anchor: str
    status: RuntimeSpineStatus = RuntimeSpineStatus.GAP_OPEN
    evidence_path: str = TASK_ONE_EVIDENCE_PATH

    def to_json(self) -> JsonMap:
        return {
            "spine_id": self.spine_id,
            "name": self.name,
            "status": self.status.value,
            "required_assertion": self.required_assertion,
            "current_gap": self.current_gap,
            "acceptance_probe": self.acceptance_probe,
            "doc_anchor": self.doc_anchor,
            "evidence_path": self.evidence_path,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSpineContract:
    version: str
    spines: tuple[RuntimeSpine, ...]

    def to_json(self) -> JsonMap:
        return {
            "version": self.version,
            "spines": [spine.to_json() for spine in self.spines],
        }


def runtime_spine_contract() -> RuntimeSpineContract:
    return RuntimeSpineContract(
        version=RUNTIME_SPINE_CONTRACT_VERSION,
        spines=(
            RuntimeSpine(
                spine_id="provider_transport",
                name="Model/Provider/Transport",
                required_assertion="Provider calls are selected by api_protocol and never forced through one gateway path.",
                current_gap="ProviderToolChoiceModel posts live choices to a fixed /v1/responses route.",
                acceptance_probe="provider transport tests reject non-Responses adapters using /v1/responses",
                doc_anchor="#modelprovidertransport",
            ),
            RuntimeSpine(
                spine_id="agent_session",
                name="AgentSession",
                required_assertion="One durable session owns role prompt, provider state, history, events, tools, skills, and resume state.",
                current_gap="AsaAgentSession is a frozen DTO for one run goal, not a durable per-agent session object.",
                acceptance_probe="second turn provider payload includes role prompt plus prior session context",
                doc_anchor="#agentsession",
            ),
            RuntimeSpine(
                spine_id="agent_loop",
                name="AgentLoop",
                required_assertion="The loop streams model events, executes tool calls, appends results, and continues to terminal output.",
                current_gap="AgentLoop is a one-shot choose_tools call followed by direct tool execution.",
                acceptance_probe="model to tool to tool_result to second model call test passes",
                doc_anchor="#agentloop",
            ),
            RuntimeSpine(
                spine_id="assembly",
                name="Prompt/Skill/Workflow Assembly",
                required_assertion="System, role, skill, workflow, compaction, transcript, tools, provider metadata, and ledger facts share one assembler.",
                current_gap="Provider input is assembled ad hoc from only the current user_goal and fixed instructions.",
                acceptance_probe="assembled context snapshots include compact summary, active skill, role prompt, and selected tools",
                doc_anchor="#promptskillworkflow-assembly",
            ),
            RuntimeSpine(
                spine_id="subagent_runtime",
                name="Subagent/Task Runtime",
                required_assertion="Persistent agents can spawn bounded subagent jobs with list, inspect, await, cancel, pause, resume, steer, and output refs.",
                current_gap="Subagent task tools exist, but there is no detached controllable job runtime matching the Gajae-like surface.",
                acceptance_probe="subagent lifecycle tests cover spawn, inspect, await, steer, pause, resume, cancel, and owner filtering",
                doc_anchor="#subagenttask-runtime",
            ),
            RuntimeSpine(
                spine_id="context_resume",
                name="Context/Compaction/Resume",
                required_assertion="Compaction checkpoints and transcript tails are replayed into the next provider call after resume.",
                current_gap="Message and compaction files can be written, but live turn construction ignores history and summaries.",
                acceptance_probe="turn two can only pass if compact summary and transcript tail are present in provider context",
                doc_anchor="#contextcompactionresume",
            ),
            RuntimeSpine(
                spine_id="tool_runtime",
                name="Tool Registry/Tool Runtime",
                required_assertion="Provider-visible tools come from registry metadata with schema, provenance, side effects, approval, and gating.",
                current_gap="Tool schemas expose minimal fields and several domain capabilities are placeholder definitions.",
                acceptance_probe="unsafe tools are hidden before provider calls and unavailable placeholders return typed blockers",
                doc_anchor="#tool-registrytool-runtime",
            ),
            RuntimeSpine(
                spine_id="tui_observability",
                name="TUI/UX/Observability",
                required_assertion="TUI palettes, agent routes, model state, runtime timeline, and HUD are rendered from live runtime state.",
                current_gap="TUI direct agent routing records semantic lines while the underlying runtime can still use static fallback.",
                acceptance_probe="PTY tests reconcile visible transcript, runtime events, provider payload, and ledger output",
                doc_anchor="#tuiuxobservability",
            ),
        ),
    )


def runtime_spine_matrix() -> JsonMap:
    return {spine.spine_id: spine.to_json() for spine in runtime_spine_contract().spines}
