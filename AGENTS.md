# ASA Runtime Workspace Contract

This repository root is the published mirror of the new production runtime boundary
for the Atomistic Simulation Agent (ASA). In the private research workspace, the
same runtime lives under `02.Source_code/asa_runtime`.

## Source split

- This root is the new runtime package:
  - Python CLI/TUI: `sim_agent/cli`
  - Python runtime/harness: `sim_agent/agents_sdk_runtime`, `sim_agent/agent_harness`
  - Python simulation boundaries: `sim_agent/md`, `sim_agent/ml_surrogate`,
    `sim_agent/kmc`, `sim_agent/level_set`, `sim_agent/transport`
  - Python controller server: `sim_agent/ui`
  - Static controller UI: `ui`
  - TypeScript OAuth/model gateway prototype: `model_gateway`
  - Runtime tests: `tests`
- `md_agent_window` and `ML_KMC_Model` are legacy/prototype reference code. Do not
  put new runtime code there.
- It is acceptable for ASA runtime to reference selected legacy assets, especially
  force-field and fixture files under `md_agent_window/Reference` and
  `md_agent_window/results`.

## Runtime expectations

- The user-facing entrypoint is `asa` from `pyproject.toml`.
- New provider/model/auth choices must be represented as typed configuration and
  written to ledgers. Do not add implicit Claude/OpenAI/Openclaw fallbacks.
- Keep the production runtime Python-first. Use TypeScript only for gateway/auth
  concerns where it reduces complexity.
- Keep MD, ML surrogate, KMC/ray tracing, Level-Set evolution, GraphDB memory, UI,
  and model gateway as separate module boundaries.
- Do not couple Level-Set mutation into MDN inference.
- Every MD execution path must retain a verification gate before declaring data
  physically usable.

## Verification commands

Run from this directory unless noted otherwise:

```bash
python3 -m pytest -q
python3 -m pytest tests/test_source_payload.py tests/test_agents_sdk_runtime.py -q
python3 scripts/run_agent_cli.py --offline --goal "Smoke ASA runtime" --output-dir /tmp/asa-runtime-smoke
asa --help
printf '/status\n/exit\n' | asa
```

For TypeScript gateway work:

```bash
cd model_gateway
npm run typecheck
```
