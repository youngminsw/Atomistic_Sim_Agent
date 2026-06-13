# MSS AGENT KNOWLEDGE BASE

Generated: 2026-06-10

## OVERVIEW

`mss_agent` holds the current multiscale simulation prototype: MD data generation, MDN training, and fixed-trench KMC visualization.

## STRUCTURE

```
mss_agent/
|-- Orchestrator.py       # Prototype top-level MD -> ML -> KMC runner.
|-- ML_KMC_Model/         # Dump conversion, MDN train/infer, KMC heatmap tool.
|-- md_agent_window/      # MD agent harness, MCP tools, inspection agent, LAMMPS utilities.
|-- outputs/              # Runtime output area.
`-- results/              # Simulation result folders.
```

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Current pipeline flow | `Orchestrator.py` | Reads natural language intent, then shells into MD/ML/KMC stages. |
| LLM routing | `md_agent_window/src/config.py`, `md_agent_window/src/llm_client.py` | Antigravity/opencode and direct/local fallbacks. |
| Tool/harness pattern | `md_agent_window/src/agent_core.py`, `md_agent_window/src/tools_lib.py` | ReAct-style loop, tool schema, inspection integration. |
| Inspection/reviewer | `md_agent_window/src/inspection_agent.py`, `md_agent_window/src/inspection_client.py`, `md_agent_window/src/inspection_server.py` | Validation and expert review patterns to preserve. |
| Structure generation | `md_agent_window/src/structure.py`, `md_agent_window/src/lammps_gen.py` | Current LAMMPS structure/input boundary. |
| ML surrogate | `ML_KMC_Model/02_01_Train_Model.py`, `ML_KMC_Model/mdn_model.py`, `ML_KMC_Model/total_model.py` | MDN training and inference. |
| KMC prototype | `ML_KMC_Model/04_KMC_tool.py` | Fixed trench polyline, ray hits, energy bins, heatmap. |
| MCP wrapper | `ML_KMC_Model/MCP_server.py`, `ML_KMC_Model/Gemini_MCP.py` | Existing tool-server shape for ML/KMC orchestration. |

## CURRENT LIMITS

- `Orchestrator.py` references `self.ml_dir` and `self.llm_client` without initializing them in `__init__`.
- `run_kmc_simulation(energy, angle)` accepts parameters but currently runs `04_KMC_tool.py` with its script defaults.
- `04_KMC_tool.py` can accumulate energy on a fixed trench and plot a heatmap, but it does not mutate the trench through time.
- `md_agent_window/src/server.py` advertises amorphous/crystalline `phase`, but the underlying `StructureBuilder.create_substrate` signature currently expects different arguments.
- The current UI surface is command-line oriented; no HTML controller/chat UI is present here.

## REBUILD BOUNDARIES

- `agent_harness/`: Python OpenAI Agents SDK controller, tool registry, tracing, auth-aware model routing, handoffs, and run sessions.
- `llm_endpoints/`: user-configurable model gateway/provider adapter. Openclaw is allowed but not hard-coded; support project-owned OAuth gateways and gajae-code-style provider surfaces through typed configuration.
- `auth_gateway/` or `controller_gateway/`: TypeScript/Bun may be used for OAuth login, PKCE/callback/device flows, credential storage, refresh, stream normalization, and HTML controller integration.
- `geometry/`: 3D PR-patterned substrates, trench/hole masks, material regions, mesh/grid conversion.
- `md/`: crystalline/amorphous builders, PR and target material handling, LAMMPS input generation, force-field validation, run monitoring, trajectory parsing.
- `ml_surrogate/`: dataset schema, MDN training, inference, uncertainty, model registry.
- `kmc/`: ion launch distributions, transport/ray tracing, wall interaction, energy deposition fields.
- `level_set/`: profile grid, velocity law from energy/yield fields, geometry update over time.
- `graphdb/`: paper ingestion, material/force-field/process ontology, provenance-backed retrieval.
- `ui/`: HTML controller and chat surface, run status, 3D profile timeline, click-to-inspect energy/profile diagnostics.

## PHYSICS CONTRACT

- The first target simulation is 3D etching of a PR-patterned structure where exposed lower regions etch downward over time.
- Geometry must support both trenches and holes through the same 3D mask/profile abstraction.
- PR etches much more slowly than the target material; PR selectivity/etch rate must be configurable.
- MD must record incident state, impact site, local surface context, outgoing state, sputter/removal signal, and deposited energy.
- MD output is usable only after verification confirms successful LAMMPS completion, sane trajectory files, energy/unit consistency, and valid event parsing.
- Amorphous substrates are first-class; do not approximate them as crystal-only CIF replication.
- KMC must output spatial energy transfer fields suitable for Level-Set consumption.
- Level-Set owns trench/profile evolution from an initial flat surface or supplied profile over discrete time steps.
- Visual outputs must show 3D profile versus time and allow clicking any position to inspect energy transfer, local removal law, and profile evolution history.

## DO NOT

- Do not treat a single static heatmap as etch profile evolution.
- Do not hide material, force-field, model, provider, auth, or endpoint choices inside prompts only; they need typed configuration, controller visibility, and run-ledger evidence.
- Do not add implicit Claude/OpenAI/Openclaw fallback endpoints. Any provider, including Openclaw, ChatGPT/Codex-style OAuth, local vLLM/Ollama, or direct official APIs, must be selected explicitly by the user or a saved config.
- Do not build a 2D-only trench simulator that cannot represent holes.
- Do not mix generated result files with source architecture changes.
