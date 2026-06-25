# asa-runtime-workflow-harness-parity - Work Plan

## TL;DR (For humans)
**What you'll get:** ASA-native implementations of shared workflow runtime surfaces that every ASA agent can use through the same tool/runtime registry: deep interview, RALPlan, and ultragoal-style durable goal execution. The work starts with a Gajae 1:1 parity matrix so behavior is implemented against concrete protocol and artifact expectations, not against a vague memory of OMO.

**Why this approach:** The current ASA harness proves only that evidence keys were supplied; it does not run workflow gates, response validation, transcripts, consensus planning, goal ledgers, or checkpoint reconciliation. The load-bearing decision is to build a typed ASA workflow runtime and keep the old smoke API as a compatibility wrapper until callers and tests are migrated.

**What it will NOT do:** It will not wrap or depend on the OMO CLI at runtime, copy Gajae source/branding, edit the main worktree's dirty prompt files, or claim readiness from evidence-key presence alone.

**Effort:** Large
**Risk:** Medium - this touches model-visible tools, TUI slash commands, and persisted workflow/session artifacts, but can be made safe with red-first tests and compatibility wrappers.
**Decisions to sanity-check:** runtime workflow artifacts stay under the ASA session/workflow directory; source-controlled parity fixtures live under ASA tests; Gajae parity is behavioral/protocol parity rather than source copying; ultragoal snapshots are recorded as ASA checkpoint inputs and do not call Codex goal-mode control APIs directly.

Your next move: approve this plan for worker execution in the new worktree, or run a high-accuracy review of the plan artifact before implementation. Full execution detail follows below.

---

> TL;DR (machine): Large/medium-risk TDD implementation plan for ASA-native shared workflow runtime parity with Gajae workflow gates, response validation, task/goal context, and durable deep-interview/ralplan/ultragoal artifacts.

## Scope
### Must have
- Worktree: implement only in `02.Source_code/.worktrees/runtime-develop-harness-20260625` on branch `codex/runtime-develop-harness-20260625`, base `6176024 Retire root legacy sources behind the ASA runtime boundary`.
- Produce a Gajae 1:1 parity matrix before product implementation:
  - Canonical source-controlled JSON fixture: `02.Source_code/asa_runtime/tests/fixtures/workflow_parity/gajae-workflow-parity-matrix.json`
  - Canonical source-controlled Markdown fixture: `02.Source_code/asa_runtime/tests/fixtures/workflow_parity/gajae-workflow-parity-matrix.md`
  - Optional generated evidence copies: `.omo/evidence/asa-runtime-workflow-harness-parity/gajae-workflow-parity-matrix.{json,md}`
  - Each row maps: Gajae source reference, behavioral contract, ASA target file/API, tests, implementation status, verification evidence.
- Replace smoke-only behavior with an ASA-native typed workflow runtime:
  - shared agent capability available through the common ASA runtime/tool registry for all domain agents unless an explicit tool policy disables it.
  - `WorkflowRuntime` or equivalent state-machine layer.
  - `WorkflowGate` model with stable id, stage, kind, schema, schema_hash, options, context, created_at, pending/resolved status.
  - response validation for boolean/string/enum/object gates with explicit rejection reasons.
  - append-only workflow ledger/events that are resumable and inspectable from TUI/tool outputs.
  - deterministic state/artifact/gate runtime first; do not add hidden provider/LLM calls inside the workflow runtime unless a later plan explicitly adds an agent-execution adapter.
- Preserve `run_workflow_harness_smoke(...)` as a compatibility wrapper for existing imports/tests while making its "ready" state come from the new runtime, not from evidence-key presence alone.
- Implement real ASA runtime artifacts for:
  - `deep-interview`: one-question-per-round gate, transcript, ambiguity score/state, clarified spec/handoff artifact, blocked state when an answer is required.
  - `ralplan`: requirements/PRD artifact, test-spec artifact, planner/architect/critic consensus review records, approval gate, handoff-ready state only when PRD and test spec exist.
  - `ultragoal`: `brief.md`, `goals.json`, `ledger.jsonl`, active-goal checkpoint records, goal status summary, reconciliation with provided Codex goal snapshot or an honest blocked state.
- Wire the runtime through existing surfaces:
  - model-visible `workflow_start` tool in `sim_agent/agent_harness/tools.py`.
  - model-visible `workflow_gate_response` (or clearly named equivalent) tool so any agent can answer pending workflow gates, with schema validation and idempotency handling.
  - executor in `sim_agent/agent_harness/agent_runtime_tools.py`.
  - `/workflow`, `/deep-interview`, `/ralplan`, `/ultragoal`, plus a TUI gate-response path such as `/workflow-response <gate-id> <json-value>` in `sim_agent/cli/tui_workflow.py` and catalog/help text.
  - provider payload schema/regression tests.
- Use Gajae behavioral references as the contract:
  - `workflow_gate` may be a flat non-event notification with `gate_id`, `stage`, `kind`, `schema`, `schema_hash`, `context`, `created_at`, `options` (`python/gjc-rpc/tests/test_protocol.py:200-217`, `324-342`).
  - wrapped event frames use `type=event`, protocol_version, session_id, seq, frame_id, payload event (`python/gjc-rpc/README.md:260-273`).
  - workflow gate event parsing includes `ralplan:approval`, kind `approval`, schema/options/context (`python/gjc-rpc/tests/test_protocol.py:304-322`).
  - client round trip rejects invalid enum response and accepts valid response (`python/gjc-rpc/tests/test_client.py:872-891`).
  - notifications carry task and goal context summaries (`crates/gjc-notifications/src/protocol.rs:242-256`).
  - action lifecycle is first-valid-resolution-wins with duplicate/idempotency/unknown/replier-unavailable rejection paths (`crates/gjc-notifications/src/actions.rs:52-237`).
### Must NOT have (guardrails, anti-slop, scope boundaries)
- Do not implement in the main worktree; do not touch the main worktree's existing dirty domain prompt files.
- Do not copy Gajae source code or namespacing wholesale; implement ASA-native equivalents against the parity matrix.
- Do not make OMO, omx, or the local Codex skill directory a production runtime dependency.
- Do not keep the current "provided evidence keys == workflow ready" behavior as the new truth; it may exist only as a legacy input path that feeds the new runtime.
- Do not implement workflow behavior separately per domain agent. Domain prompts may describe when to use workflows, but the executable workflow runtime and response path must be shared.
- Do not add hidden provider calls, background LLM orchestration, or agent recursion inside the first workflow runtime pass; this plan is for deterministic workflow state, gates, artifacts, and shared tool surfaces.
- Do not reintroduce root-level runtime packages outside `02.Source_code/asa_runtime`.
- Do not hide durable domain-agent behavior in slash skills; runtime workflow behavior belongs in Python source and file-backed artifacts.
- Do not remove security/path-safety behavior for unknown workflow ids.
- Do not skip tests for gate validation, resume artifacts, tool schema, TUI output, and failure/blocker states.
- Do not commit `.omo`, `.omx`, `.asa`, caches, run outputs, or parity evidence unless the user explicitly authorizes evidence/doc commits. Plan artifacts are allowed for this planning turn.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD with pytest; write/adjust failing tests before product code in each todo.
- Required focused tests from `02.Source_code/asa_runtime`:
  - `python3 -m pytest -q tests/test_workflow_harnesses.py tests/test_skill_workflow_runtime.py tests/test_agents_sdk_tool_gateway_runtime.py`
  - new tests: `tests/test_workflow_runtime_parity.py`, `tests/test_workflow_gate_protocol.py`, `tests/test_deep_interview_runtime.py`, `tests/test_ralplan_runtime.py`, `tests/test_ultragoal_runtime.py`, `tests/test_workflow_parity_matrix.py`
- Required compile check:
  - `python3 -m compileall -q sim_agent scripts`
- Required real-surface smoke:
  - run `python3 -m sim_agent` with scripted `/workflow deep-interview`, `/ralplan`, and `/ultragoal` commands using a temp `ASA_SESSION_DIR`; assert stdout includes gate id/stage, blocked/ready status, ledger path, and artifact paths.
- Failure QA scenarios:
  - invalid enum response to a select gate is rejected with `invalid_workflow_gate_response` or ASA equivalent.
  - any domain agent with the shared `workflow_start` and `workflow_gate_response` tools can start a workflow, receive a pending gate, and submit a response through the same runtime path.
  - duplicate/already-resolved gate response is rejected or idempotently accepted according to idempotency key.
  - missing deep-interview answer blocks with a pending gate, not a fake ready state.
  - missing RALPlan PRD/test-spec blocks handoff readiness.
  - missing/corrupt ultragoal `goals.json` or missing Codex snapshot records a resumable blocked state.
  - missing parity matrix row fails the parity-matrix test.
- Evidence path convention:
  - `.omo/evidence/task-<N>-asa-runtime-workflow-harness-parity.<json|log|md>`
  - `.omo/evidence/asa-runtime-workflow-harness-parity/final-verification.log`
  - source-controlled fixtures that tests depend on must live under `02.Source_code/asa_runtime/tests/fixtures/workflow_parity/`; `.omo/evidence` is for run evidence only.

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.
- Wave 1: Todo 1 and Todo 2 only; no product implementation until the parity matrix exists and tests assert its shape.
- Wave 2: Todo 3 and Todo 4; runtime model plus gate protocol are coupled and must land before workflow-specific runtimes.
- Wave 3: Todo 5, Todo 6, Todo 7; the three workflow runtimes can be implemented in parallel after the shared runtime/gate layer is available.
- Wave 4: Todo 8 and Todo 9; integration plus parity audit after workflow runtimes exist.
- Wave 5: Todo 10 and final verification.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | none | 2, 3, 10 | none |
| 2 | 1 | 3, 4, 5, 6, 7, 9 | none |
| 3 | 2 | 4, 5, 6, 7, 8 | 4 after shared test names are agreed |
| 4 | 2, 3 skeleton | 5, 6, 7, 8, 9 | 3 |
| 5 | 3, 4 | 8, 9, 10 | 6, 7 |
| 6 | 3, 4 | 8, 9, 10 | 5, 7 |
| 7 | 3, 4 | 8, 9, 10 | 5, 6 |
| 8 | 5, 6, 7 | 10 | 9 |
| 9 | 2, 5, 6, 7 | 10 | 8 |
| 10 | 8, 9 | final verification | none |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [ ] 1. Establish worktree hygiene and reference inventory
  What to do / Must NOT do: confirm the worker is operating in the new worktree/branch; record main worktree dirtiness as out-of-scope; inventory the exact ASA and Gajae references used for parity. Do not edit product code yet.
  Parallelization: Wave 1 | Blocked by: none | Blocks: 2, 3, 10
  References (executor has NO interview context - be exhaustive): `AGENTS.md`; `02.Source_code/asa_runtime/sim_agent/agents_sdk_runtime/workflow_harness.py:54-170`; `02.Source_code/asa_runtime/sim_agent/cli/tui_workflow.py:16-47`; `02.Source_code/asa_runtime/sim_agent/agent_harness/agent_runtime_tools.py:144-168`; `02.Source_code/asa_runtime/sim_agent/agent_harness/tools.py:380-400`; `02.Source_code/asa_runtime/tests/test_workflow_harnesses.py:17-102`; `02.Source_code/asa_runtime/tests/test_skill_workflow_runtime.py:61-100`; Gajae refs listed under Scope.
  Acceptance criteria (agent-executable): `git -c safe.directory='//wsl$/Ubuntu/home/swym4/01.Project/01.Research/01.Atomistic_Sim_Agent/02.Source_code/.worktrees/runtime-develop-harness-20260625' status --short --branch` shows branch `codex/runtime-develop-harness-20260625`; `.omo/evidence/task-1-asa-runtime-workflow-harness-parity.md` lists reference files and out-of-scope dirty files.
  QA scenarios (name the exact tool + invocation): Git Bash `git status` plus `rg -n "run_workflow_harness_smoke|workflow_start|workflow_gate|ultragoal|ralplan|deep-interview" ...`; Evidence `.omo/evidence/task-1-asa-runtime-workflow-harness-parity.md`
  Commit: N | planning/evidence only unless user authorizes evidence commit.

- [ ] 2. Build the Gajae 1:1 workflow parity matrix and lock it with tests
  What to do / Must NOT do: create JSON and Markdown parity matrix rows for workflow gate shape, event envelope, response validation, action lifecycle, dynamic task/goal context, deep-interview question gate, ralplan approval gate, ultragoal signoff/checkpoint gate, TUI/tool persistence. The canonical matrix files must be source-controlled under `02.Source_code/asa_runtime/tests/fixtures/workflow_parity/`; `.omo/evidence` may contain generated proof copies only. Do not claim unsupported behavior; mark unknowns explicitly.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 3, 4, 5, 6, 7, 9
  References (executor has NO interview context - be exhaustive): `//wsl$/Ubuntu/tmp/gajae-code/python/gjc-rpc/tests/test_protocol.py:200-217,304-342`; `//wsl$/Ubuntu/tmp/gajae-code/python/gjc-rpc/tests/test_client.py:360-376,872-891`; `//wsl$/Ubuntu/tmp/gajae-code/python/gjc-rpc/README.md:260-273`; `//wsl$/Ubuntu/tmp/gajae-code/crates/gjc-notifications/src/protocol.rs:242-256`; `//wsl$/Ubuntu/tmp/gajae-code/crates/gjc-notifications/src/actions.rs:52-237`.
  Acceptance criteria (agent-executable): `python3 -m pytest -q tests/test_workflow_parity_matrix.py` fails before implementation when a required row is absent and passes after `tests/fixtures/workflow_parity/gajae-workflow-parity-matrix.json` includes all required row ids.
  QA scenarios (name the exact tool + invocation): `python3 -m pytest -q tests/test_workflow_parity_matrix.py`; validate JSON loads from `tests/fixtures/workflow_parity/gajae-workflow-parity-matrix.json` and each row has `gajae_reference`, `asa_target`, `tests`, `status`; Evidence `.omo/evidence/task-2-asa-runtime-workflow-harness-parity.json`
  Commit: Y | `test(workflows): lock gajae workflow parity matrix`

- [ ] 3. Introduce ASA workflow runtime model and compatibility boundary
  What to do / Must NOT do: add a typed runtime module under `sim_agent/agents_sdk_runtime/` for definitions, run state, events, artifacts, result payloads, and persistence. Keep `workflow_harness.py` as the compatibility import/wrapper. Do not break existing `run_workflow_harness_smoke` callers.
  Parallelization: Wave 2 | Blocked by: 2 | Blocks: 4, 5, 6, 7, 8
  References (executor has NO interview context - be exhaustive): `workflow_harness.py:17-52` dataclasses; `workflow_harness.py:112-170` current smoke runner; `workflow_harness.py:173-260` unknown-id/path safety and ledger serialization; `tests/test_workflow_harnesses.py:17-102` current compatibility expectations.
  Acceptance criteria (agent-executable): existing workflow tests still pass, and new runtime tests assert a run record can be persisted/resumed without using evidence-key-only readiness.
  QA scenarios (name the exact tool + invocation): `python3 -m pytest -q tests/test_workflow_harnesses.py tests/test_workflow_runtime_parity.py`; Evidence `.omo/evidence/task-3-asa-runtime-workflow-harness-parity.log`
  Commit: Y | `feat(workflows): add typed ASA workflow runtime boundary`

- [ ] 4. Implement workflow gate protocol and response validation
  What to do / Must NOT do: add `WorkflowGate` and gate response validation with Gajae-like fields and lifecycle semantics: pending, resolved, duplicate/idempotency handling, unknown/already-answered/replier-unavailable rejection. Add a shared `workflow_gate_response` runtime executor/tool contract (or equivalently named response surface) so all agents can answer gates through the same path. Do not couple this to a UI-only prompt path.
  Parallelization: Wave 2 | Blocked by: 2, 3 skeleton | Blocks: 5, 6, 7, 8, 9
  References (executor has NO interview context - be exhaustive): Gajae `test_protocol.py:200-217,304-342`; Gajae `test_client.py:872-891`; Gajae `actions.rs:52-237`; ASA `agent_runtime_tools.py:144-168`; ASA `tools.py:380-400`.
  Acceptance criteria (agent-executable): `python3 -m pytest -q tests/test_workflow_gate_protocol.py` proves flat gate payload shape, schema_hash stability, enum rejection, boolean approval acceptance, duplicate/idempotency behavior, unknown-gate rejection, and shared `workflow_gate_response` executor output.
  QA scenarios (name the exact tool + invocation): pytest plus direct Python smoke constructing a select gate and sending invalid/valid responses through `workflow_gate_response`; Evidence `.omo/evidence/task-4-asa-runtime-workflow-harness-parity.log`
  Commit: Y | `feat(workflows): validate workflow gate responses`

- [ ] 5. Implement deep-interview runtime artifacts
  What to do / Must NOT do: implement ASA deep-interview as a deterministic, resumable workflow: one pending question gate at a time, transcript append, ambiguity gate, clarified spec/handoff artifact. Do not ask multiple interview rounds in one gate; do not treat a supplied evidence key as a real answer unless converted into a gate response artifact; do not add hidden provider calls.
  Parallelization: Wave 3 | Blocked by: 3, 4 | Blocks: 8, 9, 10
  References (executor has NO interview context - be exhaustive): current definition in `workflow_harness.py:55-64`; TUI defaults in `tui_workflow.py:16-47,64-77`; deep-interview skill concept from `C:/Users/swym4/.codex/skills/deep-interview/SKILL.md` already read by orchestrator; Gajae gate question examples in `test_client.py:360-376`.
  Acceptance criteria (agent-executable): `python3 -m pytest -q tests/test_deep_interview_runtime.py` proves initial run blocks with `deep-interview:question`, valid answer creates transcript and next ambiguity/handoff state, missing answer remains resumable.
  QA scenarios (name the exact tool + invocation): scripted `python3 -m sim_agent` `/deep-interview --goal ...` then response simulation; Evidence `.omo/evidence/task-5-asa-runtime-workflow-harness-parity.log`
  Commit: Y | `feat(workflows): persist deep interview gates and handoffs`

- [ ] 6. Implement RALPlan runtime artifacts
  What to do / Must NOT do: implement RALPlan as a deterministic artifact-producing planning workflow: requirements/PRD, test spec, planner/architect/critic review records, consensus status, approval gate. Do not mark `verification_plan_ready` unless PRD and test-spec artifacts exist and pass structural checks; do not add hidden provider calls.
  Parallelization: Wave 3 | Blocked by: 3, 4 | Blocks: 8, 9, 10
  References (executor has NO interview context - be exhaustive): current definition in `workflow_harness.py:65-74`; existing tests `test_skill_workflow_runtime.py:61-85`; Gajae `ralplan:approval` contract in `test_protocol.py:304-322`; ralplan skill concept from `C:/Users/swym4/.codex/skills/ralplan/SKILL.md` already read by orchestrator.
  Acceptance criteria (agent-executable): `python3 -m pytest -q tests/test_ralplan_runtime.py` proves missing PRD/test-spec blocks, generated/imported artifacts pass structural validation, approval gate must be resolved before handoff-ready.
  QA scenarios (name the exact tool + invocation): scripted `/ralplan --goal ... --output-dir ...` plus invalid/missing artifact cases; Evidence `.omo/evidence/task-6-asa-runtime-workflow-harness-parity.log`
  Commit: Y | `feat(workflows): persist ralplan consensus artifacts`

- [ ] 7. Implement ultragoal runtime artifacts
  What to do / Must NOT do: implement ASA ultragoal-style deterministic durable goal state: `brief.md`, `goals.json`, `ledger.jsonl`, active-goal checkpoint, status summary, and blocked reconciliation when `codex_goal_snapshot` is missing or malformed. Do not invoke Codex goal-mode control APIs, OMO ultragoal, or hidden provider calls at runtime.
  Parallelization: Wave 3 | Blocked by: 3, 4 | Blocks: 8, 9, 10
  References (executor has NO interview context - be exhaustive): current definition in `workflow_harness.py:95-104`; existing provider exposure in `test_skill_workflow_runtime.py:88-100`; Gajae `ultragoal:signoff` flat gate in `test_protocol.py:200-217`; Gajae task/goal context update in `protocol.rs:242-256`; ultragoal skill concept from `C:/Users/swym4/.codex/skills/ultragoal/SKILL.md` already read by orchestrator.
  Acceptance criteria (agent-executable): `python3 -m pytest -q tests/test_ultragoal_runtime.py` proves initial goal files are created, ledger appends checkpoints, corrupt goal files block safely, missing Codex snapshot records `codex_snapshot_reconciled_or_blocked` as blocked not ready.
  QA scenarios (name the exact tool + invocation): Python test fixture with valid snapshot, missing snapshot, corrupt `goals.json`, duplicate checkpoint; Evidence `.omo/evidence/task-7-asa-runtime-workflow-harness-parity.log`
  Commit: Y | `feat(workflows): persist ultragoal checkpoints`

- [ ] 8. Wire workflow runtime through tool, TUI, catalog, and provider-visible schema
  What to do / Must NOT do: update `workflow_start` and `workflow_gate_response` executor/schema plus TUI slash handlers to expose gate metadata, artifact paths, blockers, response results, and ledger refs from the new runtime. Preserve old commands and compatibility output where tests rely on it. Do not change unrelated tool definitions. Confirm every domain agent gets the same model-visible workflow tools through the shared registry unless explicitly disabled by policy.
  Parallelization: Wave 4 | Blocked by: 5, 6, 7 | Blocks: 10
  References (executor has NO interview context - be exhaustive): `sim_agent/agent_harness/tools.py:380-400`; `sim_agent/agent_harness/agent_runtime_tools.py:144-168`; `sim_agent/cli/tui_workflow.py:16-77`; `sim_agent/cli/tui_catalog.py:32-38`; `tests/test_agents_sdk_tool_gateway_runtime.py`; `tests/test_skill_workflow_runtime.py:61-100`; `tests/test_workflow_harnesses.py:44-75`.
  Acceptance criteria (agent-executable): `python3 -m pytest -q tests/test_workflow_harnesses.py tests/test_skill_workflow_runtime.py tests/test_agents_sdk_tool_gateway_runtime.py` passes with updated expectations for gate/artifact/response fields while preserving existing public command names.
  QA scenarios (name the exact tool + invocation): scripted TUI session with `/workflow deep-interview`, `/workflow-response <gate-id> <json-value>`, `/ralplan`, `/ultragoal`; provider payload schema inspection proving `workflow_start` and `workflow_gate_response` are model-visible to the shared agent tool registry; Evidence `.omo/evidence/task-8-asa-runtime-workflow-harness-parity.log`
  Commit: Y | `feat(workflows): expose runtime gates through ASA tools`

- [ ] 9. Add parity audit guardrails and matrix-to-runtime coverage check
  What to do / Must NOT do: add a small audit that fails when a parity matrix row has no ASA target/test, or when implemented workflow ids lack parity rows. Do not make this depend on local `/tmp/gajae-code` at runtime; the matrix fixture under `tests/fixtures/workflow_parity/` is the source-controlled contract.
  Parallelization: Wave 4 | Blocked by: 2, 5, 6, 7 | Blocks: 10
  References (executor has NO interview context - be exhaustive): `tests/fixtures/workflow_parity/gajae-workflow-parity-matrix.json`; all new workflow runtime tests; `workflow_harness_catalog()`.
  Acceptance criteria (agent-executable): `python3 -m pytest -q tests/test_workflow_parity_matrix.py tests/test_workflow_runtime_parity.py` fails if any required parity row lacks an ASA test or target.
  QA scenarios (name the exact tool + invocation): intentionally remove one row in a temp copy and assert audit fails; Evidence `.omo/evidence/task-9-asa-runtime-workflow-harness-parity.log`
  Commit: Y | `test(workflows): audit workflow parity coverage`

- [ ] 10. Run real-surface verification and update handoff notes
  What to do / Must NOT do: run compileall, focused pytest, and scripted CLI/TUI smoke from the ASA runtime root; collect concise evidence; update `.omo/plans/asa-runtime-workflow-harness-parity.md` only if actual implementation diverged from plan. Do not mark complete with skipped failing checks.
  Parallelization: Wave 5 | Blocked by: 8, 9 | Blocks: final verification
  References (executor has NO interview context - be exhaustive): verification commands in this plan; ASA workspace contract verification commands in `AGENTS.md`.
  Acceptance criteria (agent-executable): all required verification commands pass or a blocker is recorded with exact failing command/output; `.omo/evidence/asa-runtime-workflow-harness-parity/final-verification.log` exists.
  QA scenarios (name the exact tool + invocation): `python3 -m compileall -q sim_agent scripts`; focused pytest list; scripted `python3 -m sim_agent`; Evidence `.omo/evidence/task-10-asa-runtime-workflow-harness-parity.log`
  Commit: N | final evidence/report only unless user authorizes evidence commit.

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit: verify each Must Have has a passing test/evidence row, each Must NOT has no violations, and no main-worktree dirty files were touched.
- [ ] F2. Code quality review: check runtime modules are typed, small, path-safe, and do not create a second framework for ordinary artifact persistence.
- [ ] F3. Real manual QA: run scripted TUI/tool flows for all three workflows and inspect generated artifacts/ledgers.
- [ ] F4. Scope fidelity: confirm no OMO runtime dependency, no copied Gajae code, no root-level runtime package reintroduction.

## Commit strategy
- No commit in this planning turn.
- During implementation, use small atomic commits only after each todo's tests pass and the user/main orchestrator approves committing.
- Suggested commit sequence:
  - `test(workflows): lock gajae workflow parity matrix`
  - `feat(workflows): add typed ASA workflow runtime boundary`
  - `feat(workflows): validate workflow gate responses`
  - `feat(workflows): persist deep interview gates and handoffs`
  - `feat(workflows): persist ralplan consensus artifacts`
  - `feat(workflows): persist ultragoal checkpoints`
  - `feat(workflows): expose runtime gates through ASA tools`
  - `test(workflows): audit workflow parity coverage`
- Commit messages must follow the repo Lore protocol from `AGENTS.md`.

## Success criteria
- The parity matrix exists and maps every required Gajae workflow-gate/task-goal behavior to an ASA target and test.
- `deep-interview`, `ralplan`, and `ultragoal` are no longer evidence-key-only smoke harnesses; each produces real persisted workflow artifacts and gates.
- Existing public surfaces keep working: `run_workflow_harness_smoke`, `/workflow`, `/deep-interview`, `/ralplan`, `/ultragoal`, and the model-visible `workflow_start` tool.
- Missing input creates a resumable blocked/pending-gate state, not a fake ready state.
- Focused pytest and compileall checks pass from `02.Source_code/asa_runtime`.
- Final report includes changed files, evidence paths, remaining risks, and any unimplemented parity rows.
