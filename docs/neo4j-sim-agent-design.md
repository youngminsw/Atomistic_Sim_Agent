# Neo4j Simulation Agent Design

Generated: 2026-06-10

## Purpose

This document defines the gated Neo4j design for the Atomistic Simulation Agent knowledge layer. The graph is a derived index for provenance-backed simulation knowledge. It is not a source of truth and must not rewrite original files, papers, run artifacts, model checkpoints, or user data.

No code in the current implementation connects to Neo4j, creates a database, runs migrations, or writes data. The only supported mode is dry-run planning until the user explicitly approves DB work with `user_db_approval=true`.

## Database Boundary

- Proposed database name: `atomistic_sim_agent_knowledge`
- Database role: empty demo knowledge database for this project only.
- Empty database requirement: the selected demo database name must not already exist unless the user explicitly chooses a conflict-resolution path later.
- Write gate: `user_db_approval=true` is required before any future create, migration, import, or write command can be considered.
- Current runtime mode: `neo4j_write_enabled=false`
- Smoke query for a future approved connection: `RETURN 1 AS ok`
- Existing DBs are treated as protected. This project must not write into another running DB by default.

## Graph Layers

- Literature facts: source-backed physics claims, papers, documentation, and policy notes.
- MD runs: validated LAMMPS/MD event datasets, run logs, event counts, and verification status.
- Material states: material IDs, phases, damage descriptors, roughness, RDF/order summaries, and active-layer state.
- Surrogate models: MDN or later interaction kernels, coverage, uncertainty, training data provenance, and model manifests.
- Feature simulations: transport, energy deposition, Level-Set timelines, process recipe, and geometry references.
- UI artifacts: manifest, timeline, diagnostics, screenshots, and clickable profile/energy outputs.

## Labels

- `SimAgentSourceItem`
- `DocumentUnderstanding`
- `PhysicsClaim`
- `CanonicalEntity`
- `ReviewCandidate`
- `SyncRun`
- `MDRun`
- `MaterialState`
- `SurrogateModel`
- `FeatureSimulation`
- `SimulationArtifact`
- `UIArtifact`

## Relationships

- `HAS_UNDERSTANDING`
- `SUPPORTS_CLAIM`
- `MENTIONS_ENTITY`
- `USED_BY_MODULE`
- `NEEDS_REVIEW`
- `USES_MATERIAL_STATE`
- `TRAINED_MODEL`
- `DRIVES_SIMULATION`
- `PRODUCED_ARTIFACT`
- `VISUALIZES_ARTIFACT`

## Constraints

- `SimAgentSourceItem.source_url IS UNIQUE`
- `PhysicsClaim.record_id IS UNIQUE`
- `CanonicalEntity.name IS UNIQUE`
- `MDRun.run_id IS UNIQUE`
- `SurrogateModel.kernel_id IS UNIQUE`
- `FeatureSimulation.simulation_id IS UNIQUE`
- `SimulationArtifact.artifact_uri IS UNIQUE`

## Ingestion Flow

1. Scan source metadata from local provenance records, verified MD reports, model manifests, simulation run bundles, and UI artifacts.
2. Compute stable IDs from source URL/path, run ID, kernel ID, simulation ID, or artifact URI.
3. Export replayable JSONL and Cypher plan files before any future import.
4. Stage rows with a `sync_run_id`.
5. Validate counts, required fields, source URLs, and conflict checks.
6. Activate staged source-owned rows only after a future approved import succeeds.
7. Leave all unrelated labels, nodes, relationships, and databases untouched.

## Source-To-Graph Import Bundle

The Research GraphDB Agent must produce a reviewable bundle before any live Neo4j write:

- `sources.jsonl`: stable source identity, URL/path, title, source kind, source-owned label, and source status.
- `understandings.jsonl`: bounded summaries, purpose, important terms, source evidence, and extraction status.
- `claims.jsonl`: physics or harness claims, confidence, tags, module usage, source URL, and review flag.
- `canonical_entities.jsonl`: conservative topic/module entities derived from tags and module usage.
- `import.cypher`: parameterized import plan; it is not executed by the current dry-run implementation.
- `manifest.json`: schema, database boundary, conflict status, artifact names, and ingest report payload.
- `ingest_report.json`: accepted/blocked status, blocker reasons, counts, smoke query, and artifact paths.
- `retrieval_context.md`: agent-facing query contract and graph ownership boundaries.

An ingest report with `accepted=false` is a hard blocker for the production agent. A dry-run report with
`accepted=true` means the source-to-graph import artifacts are complete and ready for a separately approved
Neo4j import, not that a live DB write has occurred.

After explicit approval, `apply_graphdb_import_bundle.py --out graphdb_write_report.json` writes the live
Neo4j import report used by the production-readiness gate. That report must show `applied=true`,
`status=applied`, no blockers, and imported source, understanding, claim, and entity rows.

## Conflict Check

The gate must compare the proposed database name against existing database names before any future write mode.

- If `atomistic_sim_agent_knowledge` already exists, report `database_name_conflict`.
- If any protected DB is present, keep it untouched.
- A conflict never triggers automatic deletion, migration, rename, or overwrite.
- Conflict resolution requires a separate user decision.

## Rollback

- Export JSONL and Cypher plan before any import.
- Tag all staged nodes and relationships with `sync_run_id`.
- If a staged import fails, remove only source-owned staged rows for that `sync_run_id`.
- Do not delete source files, papers, model files, run artifacts, or unrelated graph rows.
- Preserve unrelated databases and labels.

## Read And Write Modes

- Dry run: build and print the schema, constraints, conflict checks, rollback plan, and export artifact names.
- Attempt write without approval: reject with `user_db_approval_required`.
- Attempt write with approval: future work may convert the plan into an import, but the current implementation still keeps `neo4j_write_enabled=false`.

## Agent Retrieval Rules

Agents should query source-backed claims before using physics assumptions. Retrieval must prefer:

- source URL/path and title;
- extracted claim;
- confidence;
- tags;
- module usage;
- review candidates for uncertain merges or low-confidence claims.

Agents must not treat graph facts as validated MD data unless the node is connected to a verified `MDRun`.

Fast source-backed lookup:

```cypher
MATCH (source:SimAgentSourceItem)-[:SUPPORTS_CLAIM]->(claim:PhysicsClaim)
WHERE any(tag IN claim.tags WHERE tag IN $tags)
RETURN claim.record_id, claim.claim, claim.confidence, source.source_url
ORDER BY claim.confidence DESC
LIMIT 10
```

## Compatibility Note

The user requested later reference to `youngminsw/Personal_Knowledge_Agent_Kit`. This design leaves room to borrow source indexing, entity resolution, and MCP query patterns from `youngminsw/Personal_Knowledge_Agent_Kit` after its source content is available and reviewed. No compatibility claim is made yet beyond this future integration boundary.
