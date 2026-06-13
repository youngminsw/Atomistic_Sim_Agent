from __future__ import annotations

from dataclasses import dataclass

from .types import GraphDBDryRunBundle, KnowledgeRegistryError, ProvenanceRecord, SourceKind


@dataclass(frozen=True, slots=True)
class ProvenanceRegistry:
    records: tuple[ProvenanceRecord, ...]

    def list_by_tag(self, tag: str) -> tuple[ProvenanceRecord, ...]:
        return tuple(record for record in self.records if tag in record.tags)

    def with_record(self, record: ProvenanceRecord) -> ProvenanceRegistry:
        _validate_record(record)
        return ProvenanceRegistry(self.records + (record,))

    def export_graphdb_dry_run(self) -> GraphDBDryRunBundle:
        return GraphDBDryRunBundle(
            neo4j_write_enabled=False,
            database_name="atomistic_sim_agent_knowledge",
            database_role="empty_demo_database",
            requires_empty_database=True,
            labels=(
                "SimAgentSourceItem",
                "DocumentUnderstanding",
                "PhysicsClaim",
                "CanonicalEntity",
                "ReviewCandidate",
                "SyncRun",
            ),
            relationships=(
                "HAS_UNDERSTANDING",
                "SUPPORTS_CLAIM",
                "MENTIONS_ENTITY",
                "USED_BY_MODULE",
                "NEEDS_REVIEW",
            ),
            constraints=(
                "SimAgentSourceItem.source_url IS UNIQUE",
                "PhysicsClaim.record_id IS UNIQUE",
                "CanonicalEntity.name IS UNIQUE",
            ),
            records=self.records,
        )


def seeded_provenance_registry() -> ProvenanceRegistry:
    registry = ProvenanceRegistry(())
    for record in _seed_records():
        registry = registry.with_record(record)
    return registry


def _validate_record(record: ProvenanceRecord) -> None:
    if not record.source_url:
        raise KnowledgeRegistryError("source_required")
    if not record.claim:
        raise KnowledgeRegistryError("claim_required")
    if not 0.0 <= record.confidence <= 1.0:
        raise KnowledgeRegistryError("confidence_out_of_range")
    if not record.tags:
        raise KnowledgeRegistryError("tag_required")


def _seed_records() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            record_id="agents-sdk-tools",
            source_url="https://openai.github.io/openai-agents-python/tools/",
            title="OpenAI Agents SDK tools",
            claim="Agent tools should be typed runtime functions with explicit schemas and tool behavior controls.",
            tags=("agents_sdk", "harness"),
            confidence=0.84,
            extracted_on="2026-06-10",
            used_by=("agent_harness", "knowledge"),
            source_kind=SourceKind.DOCS,
        ),
        ProvenanceRecord(
            record_id="model-provider-policy",
            source_url="repo://plans/atomistic-sim-agent-rebuild.md#model-provider-policy",
            title="Atomistic Simulation Agent model provider policy",
            claim="Model calls must route through an explicit user-selected provider or gateway configuration; high-stakes control and physics decisions require high reasoning.",
            tags=("model_provider", "endpoint_policy"),
            confidence=0.95,
            extracted_on="2026-06-10",
            used_by=("llm_endpoints", "agent_harness"),
            source_kind=SourceKind.POLICY,
        ),
        ProvenanceRecord(
            record_id="economou-plasma-etch-review",
            source_url="https://www.chee.uh.edu/sites/chbe/files/faculty/economou/tsf_00_review.pdf",
            title="Plasma etching for microelectronics fabrication",
            claim="Feature-scale etching separates plasma/sheath flux inputs from surface and profile evolution models.",
            tags=("etch_review", "feature_scale"),
            confidence=0.78,
            extracted_on="2026-06-10",
            used_by=("kmc", "level_set", "knowledge"),
            source_kind=SourceKind.PAPER,
        ),
        ProvenanceRecord(
            record_id="md-force-field-sensitivity",
            source_url="https://www.osti.gov/pages/biblio/2326056",
            title="Force-field sensitivity in irradiation and sputtering simulations",
            claim="MD force-field choice changes sputtering and damage predictions, so force-field provenance and validation must be tracked.",
            tags=("force_field", "md", "sputtering"),
            confidence=0.76,
            extracted_on="2026-06-10",
            used_by=("md", "ml_surrogate", "knowledge"),
            source_kind=SourceKind.PAPER,
        ),
        ProvenanceRecord(
            record_id="md-derived-surrogate-reference",
            source_url="https://ouci.dntb.gov.ua/en/works/4rD2aZw9/",
            title="MD-derived machine-learning surrogate for ion-surface interactions",
            claim="A surrogate should preserve output distributions and uncertainty instead of only mean reflected states.",
            tags=("surrogate", "mdn", "uncertainty"),
            confidence=0.72,
            extracted_on="2026-06-10",
            used_by=("ml_surrogate", "kmc", "knowledge"),
            source_kind=SourceKind.PAPER,
        ),
        ProvenanceRecord(
            record_id="sethian-level-set-etch",
            source_url="https://math.berkeley.edu/~sethian/2006/Papers/sethian.etch3.pdf",
            title="Level set methods for etching, deposition, and lithography development",
            claim="Feature profile evolution can be represented by interface motion driven by local velocity fields.",
            tags=("level_set", "profile_evolution"),
            confidence=0.8,
            extracted_on="2026-06-10",
            used_by=("level_set", "knowledge"),
            source_kind=SourceKind.PAPER,
        ),
        ProvenanceRecord(
            record_id="lammps-ar-si-zbl-template",
            source_url="repo://02.Source_code/mss_agent/sim_agent/md/input_script.py#ar-si-zbl",
            title="LAMMPS Ar on Si ZBL input deck pattern",
            claim="LAMMPS Ar-on-Si impact decks should preserve generated code samples and include a ZBL close-range collision term before execution review.",
            tags=("lammps_code_sample", "md", "sputtering", "force_field"),
            confidence=0.82,
            extracted_on="2026-06-12",
            used_by=("md_agent", "qa_agent", "knowledge"),
            source_kind=SourceKind.CODE_SAMPLE,
        ),
        ProvenanceRecord(
            record_id="si-tersoff-potential-library",
            source_url="repo://02.Source_code/mss_agent/md_agent_window/Reference/force_field_library/potentials/Si.tersoff",
            title="Repo-local Si Tersoff potential catalog entry",
            claim="The Si Tersoff potential path must be cataloged as provenance and validated before use in MD sputtering or surrogate data generation.",
            tags=("potential_library", "force_field", "md"),
            confidence=0.8,
            extracted_on="2026-06-12",
            used_by=("md_agent", "research_graphdb_agent", "knowledge"),
            source_kind=SourceKind.POTENTIAL,
        ),
        ProvenanceRecord(
            record_id="slurm-job-script-qa-rule",
            source_url="repo://02.Source_code/mss_agent/sim_agent/agents_sdk_runtime/session_runtime.py#slurm-job-script-gate",
            title="Slurm job script QA gate",
            claim="QA gate review is required before a responsible agent submits any Slurm job script to local or remote compute resources.",
            tags=("qa_rule", "slurm", "job_script", "md"),
            confidence=0.9,
            extracted_on="2026-06-12",
            used_by=("qa_agent", "md_agent", "infra_agent", "knowledge"),
            source_kind=SourceKind.POLICY,
        ),
    )
