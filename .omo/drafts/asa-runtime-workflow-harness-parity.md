---
slug: asa-runtime-workflow-harness-parity
status: awaiting-approval
intent: clear
pending-action: approve plan or run plan review before implementation
approach: Source-controlled Gajae 1:1 workflow-gate parity fixture first, then shared ASA-native typed workflow runtime with start+response tools for all agents, then deterministic deep-interview/ralplan/ultragoal artifact runtimes, then TUI/tool integration.
---

# Draft: asa-runtime-workflow-harness-parity

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
- C1 | Gajae workflow parity matrix covering gate shape, event frames, responses, task/goal context, and workflow-specific gates | active | `02.Source_code/asa_runtime/tests/fixtures/workflow_parity/gajae-workflow-parity-matrix.{json,md}`
- C2 | Shared ASA typed workflow runtime and compatibility wrapper for current `run_workflow_harness_smoke` callers | active | `.omo/evidence/task-3-asa-runtime-workflow-harness-parity.log`
- C3 | Workflow gate protocol/validation with pending/resolved/idempotent/rejected lifecycle and shared `workflow_gate_response` surface | active | `.omo/evidence/task-4-asa-runtime-workflow-harness-parity.log`
- C4 | Deep-interview runtime artifacts: question gate, transcript, ambiguity gate, handoff spec | active | `.omo/evidence/task-5-asa-runtime-workflow-harness-parity.log`
- C5 | RALPlan runtime artifacts: PRD, test spec, planner/architect/critic records, approval gate | active | `.omo/evidence/task-6-asa-runtime-workflow-harness-parity.log`
- C6 | Ultragoal runtime artifacts: brief, goals, ledger, checkpoint records, goal context summary | active | `.omo/evidence/task-7-asa-runtime-workflow-harness-parity.log`
- C7 | TUI/tool/provider integration plus parity audit coverage | active | `.omo/evidence/task-8-asa-runtime-workflow-harness-parity.log`, `.omo/evidence/task-9-asa-runtime-workflow-harness-parity.log`

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->
- Artifact root | Use existing ASA `session_dir / "workflows" / workflow_id` layout for runtime artifacts | Existing tool/TUI code already uses this surface and it avoids introducing repo-root state | yes, by changing runtime storage adapter before implementation
- Parity matrix storage | Keep canonical matrix fixtures in `02.Source_code/asa_runtime/tests/fixtures/workflow_parity/`; mirror generated evidence under `.omo/evidence` only when useful | Tests need a repo-tracked contract; `.omo` is ignored runtime/planning state | yes, by moving fixture paths before implementation
- Gajae parity level | Behavioral/protocol parity, not source copying | User requested 1:1 parity matrix, not code import; copying would violate repo boundary and licensing hygiene | yes, matrix rows can be refined
- Agent availability | Expose workflow start and gate response through the shared ASA tool/runtime registry for every domain agent unless policy explicitly disables it | Workflows are runtime capabilities, not per-agent prompt features | yes, by changing registry/tool policy before implementation
- Runtime execution model | First pass is deterministic state/artifact/gate runtime, not hidden provider calls or recursive LLM orchestration | Keeps workflow behavior testable and shared; later agent-execution adapters can be planned separately | yes
- Ultragoal snapshots | Treat Codex goal snapshots as runtime inputs/checkpoint evidence, not as direct calls to Codex goal-control APIs | ASA runtime should remain provider/tool surface agnostic and should not rely on app-only goal APIs | yes, if later an explicit ASA goal adapter is approved
- Commit/evidence | Plan artifacts may be edited now; implementation commits/evidence commits require orchestrator/user approval | User asked for planning only, no implementation | yes

## Findings (cited - path:lines)
- ASA current harness is smoke-only: `run_workflow_harness_smoke` checks required evidence keys and writes ready/blocked ledgers (`02.Source_code/asa_runtime/sim_agent/agents_sdk_runtime/workflow_harness.py:112-170`).
- ASA currently defines workflow ids and expected evidence for `deep-interview`, `ralplan`, `ultragoal`, plus `ultrawork` and `ultraqa` (`02.Source_code/asa_runtime/sim_agent/agents_sdk_runtime/workflow_harness.py:54-105`).
- TUI `/workflow` and aliases call the smoke runner and print `workflow_harness_ready=true`, workflow status, gate status, evidence keys, and ledger path (`02.Source_code/asa_runtime/sim_agent/cli/tui_workflow.py:16-47`).
- Model-visible `workflow_start` also calls the smoke runner and returns workflow_id/status/gate/evidence/ledger refs (`02.Source_code/asa_runtime/sim_agent/agent_harness/agent_runtime_tools.py:144-168`; `02.Source_code/asa_runtime/sim_agent/agent_harness/tools.py:380-400`).
- Existing tests lock the smoke behavior and tool exposure (`02.Source_code/asa_runtime/tests/test_workflow_harnesses.py:17-102`; `02.Source_code/asa_runtime/tests/test_skill_workflow_runtime.py:61-100`).
- Gajae workflow gates are concrete protocol frames with `gate_id`, `stage`, `kind`, `schema`, `schema_hash`, `context`, `created_at`, and `options` (`//wsl$/Ubuntu/tmp/gajae-code/python/gjc-rpc/tests/test_protocol.py:200-217,324-342`).
- Gajae event frames distinguish wrapped `event` envelopes from flat non-event frames such as `workflow_gate` (`//wsl$/Ubuntu/tmp/gajae-code/python/gjc-rpc/README.md:260-273`).
- Gajae client gate flow rejects invalid enum responses and accepts valid responses (`//wsl$/Ubuntu/tmp/gajae-code/python/gjc-rpc/tests/test_client.py:872-891`).
- Gajae notification context includes task and goal summaries (`//wsl$/Ubuntu/tmp/gajae-code/crates/gjc-notifications/src/protocol.rs:242-256`).
- Gajae action lifecycle uses pending/resolved state, first-valid-resolution-wins, duplicate/idempotency handling, and rejection reasons (`//wsl$/Ubuntu/tmp/gajae-code/crates/gjc-notifications/src/actions.rs:52-237`).

## Decisions (with rationale)
- Plan first, no implementation in this turn. The user explicitly requested planning before development.
- Make the parity matrix the first executable artifact. It converts "Gajae code 1:1 parity" into reviewable rows and prevents accidental OMO-skill-shaped implementation.
- Implement an ASA-native shared runtime layer instead of extending the smoke function. The current function's core abstraction is evidence-key gating, which is the behavior to retire.
- Add a workflow gate response surface. Starting a workflow without an agent-accessible gate response path would miss Gajae's round-trip contract and would block all-agent workflow use.
- Keep compatibility wrappers. Existing tests, TUI commands, and provider-visible tools already depend on current names and should not be broken abruptly.
- Make all workflow blockers honest/resumable. Missing answers, missing PRD/test spec, or missing goal snapshots must produce pending/blocked states with ledgers, not fake ready states.

## Scope IN
- Gajae 1:1 parity matrix and coverage audit.
- ASA shared typed workflow runtime and gate protocol.
- Shared `workflow_start` and `workflow_gate_response` tools available to every ASA domain agent unless policy disables them.
- Deep-interview, ralplan, and ultragoal workflow artifacts.
- TUI slash command integration and model-visible `workflow_start` integration.
- Focused pytest, compileall, and scripted real-surface CLI/TUI smoke.

## Scope OUT (Must NOT have)
- OMO/omx runtime dependency in ASA production code.
- Gajae source copy/import.
- Main worktree edits or edits to currently dirty domain prompt files.
- Root-level runtime package reintroduction.
- Committing `.omo`, `.omx`, `.asa`, caches, or evidence outputs without explicit approval.
- Implementing `ultrawork` and `ultraqa` beyond ensuring existing catalog/schema compatibility, unless the user expands scope.
- Per-domain-agent workflow forks or prompt-only workflow implementations.
- Hidden provider/LLM calls inside the first workflow runtime pass.

## Open questions
- None blocking for plan approval. The main reversible default is artifact storage under `session_dir / "workflows"`.

## Approval gate
status: awaiting-approval
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
