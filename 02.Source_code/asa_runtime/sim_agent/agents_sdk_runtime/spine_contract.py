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
                current_gap="Closed: provider transport selects protocol-specific endpoints and payloads from typed provider metadata.",
                acceptance_probe="provider transport tests cover Responses, Chat Completions, Anthropic Messages, Gemini generateContent, Codex OAuth Responses, malformed protocol rejection, and tool-call parsing",
                doc_anchor="#modelprovidertransport",
                status=RuntimeSpineStatus.COMPLETE,
            ),
            RuntimeSpine(
                spine_id="agent_session",
                name="AgentSession",
                required_assertion="One durable session owns role prompt, provider state, history, events, tools, skills, and resume state.",
                current_gap="Closed: global sessions now keep one orchestrator plus persistent per-domain agent sessions with history, role prompts, skills, compaction, events, and per-agent model config.",
                acceptance_probe="direct @agent and resume tests prove two turns share one agent_session_id and provider context sees prior transcript plus role layers",
                doc_anchor="#agentsession",
                status=RuntimeSpineStatus.COMPLETE,
            ),
            RuntimeSpine(
                spine_id="agent_loop",
                name="AgentLoop",
                required_assertion="The loop streams model events, executes tool calls, appends results, and continues to terminal output.",
                current_gap="Closed: AgentLoop emits model/tool/runtime events, executes selected tools, appends tool results, supports continuation, blockers, cancellation, and final output.",
                acceptance_probe="provider and static model tests cover model_start/delta/end, tool_start/end, blockers, tool_result continuation, cancellation, and TUI activity projection",
                doc_anchor="#agentloop",
                status=RuntimeSpineStatus.COMPLETE,
            ),
            RuntimeSpine(
                spine_id="assembly",
                name="Prompt/Skill/Workflow Assembly",
                required_assertion="System, role, skill, workflow, compaction, transcript, tools, provider metadata, and ledger facts share one assembler.",
                current_gap="Closed: provider prompt context is assembled from distinct system, workflow, role, project guidance, skill, compaction, transcript, ledger, and user-turn layers.",
                acceptance_probe="prompt context tests and prompt manifests show layer_kinds, compact summaries, active skills, role prompts, selected tools, and provider-specific message projections",
                doc_anchor="#promptskillworkflow-assembly",
                status=RuntimeSpineStatus.COMPLETE,
            ),
            RuntimeSpine(
                spine_id="subagent_runtime",
                name="Subagent/Task Runtime",
                required_assertion="Persistent agents can spawn bounded subagent jobs with list, inspect, await, cancel, pause, resume, steer, and output refs.",
                current_gap="Closed: domain agents can spawn bounded planner, architect, critic, and executor subagent jobs with inspect/control receipts and owner/depth/recursion blockers.",
                acceptance_probe="subagent lifecycle tests cover spawn, inspect, await, steer, pause, resume, cancel, owner filtering, unknown presets, depth limits, and recursive blockers",
                doc_anchor="#subagenttask-runtime",
                status=RuntimeSpineStatus.COMPLETE,
            ),
            RuntimeSpine(
                spine_id="context_resume",
                name="Context/Compaction/Resume",
                required_assertion="Compaction checkpoints and transcript tails are replayed into the next provider call after resume.",
                current_gap="Closed: manual and threshold compaction write checkpoints, validate replay cursors, inject compact summaries only after replay, and preserve transcript tails across asa --resume.",
                acceptance_probe="/compact, /resume, corrupt-ledger, replay-mismatch, and provider-context tests prove validated summary injection and poison-blocking",
                doc_anchor="#contextcompactionresume",
                status=RuntimeSpineStatus.COMPLETE,
            ),
            RuntimeSpine(
                spine_id="tool_runtime",
                name="Tool Registry/Tool Runtime",
                required_assertion="Provider-visible tools come from registry metadata with schema, provenance, side effects, approval, and gating.",
                current_gap="Closed: provider-visible tools come from typed registry metadata across file, process, artifact, agent bus, handoff, subagent, workflow, MCP descriptor, and domain dry-run families.",
                acceptance_probe="tool registry tests prove schemas, provenance, side effects, approval flags, hidden unsafe tools, typed blockers, and runtime receipts",
                doc_anchor="#tool-registrytool-runtime",
                status=RuntimeSpineStatus.COMPLETE,
            ),
            RuntimeSpine(
                spine_id="tui_observability",
                name="TUI/UX/Observability",
                required_assertion="TUI palettes, agent routes, model state, runtime timeline, and HUD are rendered from live runtime state.",
                current_gap="Closed: TUI renders control room, workboard, HUD, slash palette, direct @agent routes, live activity rail, tool cards, output blocks, runtime events, model/thinking selectors, and clean login/status messages from runtime state.",
                acceptance_probe="PTY and transcript tests reconcile visible chat, activity rail, runtime events, provider payloads, ledgers, selector redraw, redaction, and Ctrl-C/Esc behavior",
                doc_anchor="#tuiuxobservability",
                status=RuntimeSpineStatus.COMPLETE,
            ),
        ),
    )


def runtime_spine_matrix() -> JsonMap:
    return {spine.spine_id: spine.to_json() for spine in runtime_spine_contract().spines}
