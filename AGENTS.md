# ASA Runtime Workspace Contract

This repository root is the WSL development home for the Atomistic Simulation Agent
runtime. The canonical production runtime source lives under
`02.Source_code/asa_runtime`.

## Source Boundary

- Put new runtime code in `02.Source_code/asa_runtime`.
- Treat root-level `sim_agent`, `model_gateway`, `scripts`, `tests`, `ui`,
  `ML_KMC_Model`, and `md_agent_window` history as legacy/prototype material.
- Do not reintroduce root-level runtime packages. If a legacy asset is still
  useful, copy or wrap the specific asset through the canonical runtime boundary.
- Keep local research folders, manuscript material, run outputs, and OMX/Codex
  state out of commits unless a task explicitly asks for them.

## Runtime Expectations

- The user-facing entrypoint is `asa` from `02.Source_code/asa_runtime`.
- Keep AgentSession, AgentLoop, provider transport, tool registry, MCP,
  compaction/resume, subagent/task, workflow gates, and TUI observability in the
  Python-native ASA runtime spine.
- Keep provider/model/auth choices typed and visible. Do not add implicit
  provider fallbacks.
- Domain-agent behavior belongs in file-backed system/role prompt layers.
  Slash skills are reusable command/workflow surfaces, not the durable domain
  knowledge store.
- Neo4j/GraphDB access is MCP-first for agents. Local direct helpers may exist
  only as bounded compatibility or smoke-test surfaces.

## Verification Commands

Run from `02.Source_code/asa_runtime` unless noted otherwise:

```bash
python3 -m compileall -q sim_agent scripts
python3 -m pytest -q
cd model_gateway && npm test
```

Useful focused checks:

```bash
python3 -m pytest -q tests/test_prompt_context_assembler.py tests/test_markdown_skill_registry.py
python3 -m pytest -q tests/test_provider_transport.py tests/test_provider_tool_choice_model.py
python3 -m pytest -q tests/test_tui_selector_screen.py tests/test_tui_timeline_resume.py
```

## Cleanup Rules

- Keep `.omo/`, `.omx/`, `.omc/`, local `.asa/`, caches, and evidence outputs
  ignored at the repository root.
- Do not commit `:Zone.Identifier` files.
- Keep commits atomic: runtime behavior, root legacy cleanup, and generated
  evidence/doc artifacts should be separate decisions.
