from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph_import import GraphImportBundle, build_source_graph_import_bundle, report_payload
from .graphdb_access import (
    AgentGraphContext,
    GraphDBConnectionConfig,
    agent_graph_context_payload,
    build_agent_graph_context,
)
from .graphdb_gate import GraphDBGatePlan
from .provenance_registry import ProvenanceRegistry
from .research_tools import ResearchAnswer, ResearchQuestion, answer_research_question


@dataclass(frozen=True, slots=True)
class ResearchAgentResult:
    status: str
    graphdb_write: bool
    bundle: GraphImportBundle
    agent_context: AgentGraphContext
    answer: ResearchAnswer
    agent_context_path: Path
    answer_path: Path

    def summary_lines(self) -> tuple[str, ...]:
        role_agents = ",".join(query.agent_id for query in self.agent_context.role_queries)
        return (
            f"research_agent_status={self.status}",
            f"graphdb_ingest_accepted={str(self.bundle.report.accepted).lower()}",
            f"graphdb_write={str(self.graphdb_write).lower()}",
            f"database_name={self.bundle.report.database_name}",
            f"source_count={self.bundle.report.source_count}",
            f"claim_count={self.bundle.report.claim_count}",
            f"entity_count={self.bundle.report.entity_count}",
            f"agent_access_enabled={str(self.agent_context.agent_access_enabled).lower()}",
            f"role_query_agents={role_agents}",
            f"answer_status={self.answer.answer_status}",
            f"agent_graph_context_path={self.agent_context_path}",
            f"research_answer_path={self.answer_path}",
            f"retrieval_context_path={self.bundle.retrieval_context_path}",
        )


def build_research_agent_artifacts(
    registry: ProvenanceRegistry,
    gate_plan: GraphDBGatePlan,
    output_dir: Path,
    *,
    sync_run_id: str,
    question: ResearchQuestion,
    connection: GraphDBConnectionConfig | None = None,
) -> ResearchAgentResult:
    bundle = build_source_graph_import_bundle(registry, gate_plan, output_dir, sync_run_id=sync_run_id)
    context = build_agent_graph_context(gate_plan, connection)
    answer = answer_research_question(registry, question)

    agent_context_path = output_dir / "agent_graph_context.json"
    answer_path = output_dir / "research_answer.json"
    agent_context_path.write_text(
        _json(
            {
                "graph": agent_graph_context_payload(context),
                "ingest_report": report_payload(bundle.report),
                "retrieval_context_path": str(bundle.retrieval_context_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answer_path.write_text(_json(_answer_payload(answer)) + "\n", encoding="utf-8")

    return ResearchAgentResult(
        status="ready" if bundle.report.accepted else "blocked",
        graphdb_write=False,
        bundle=bundle,
        agent_context=context,
        answer=answer,
        agent_context_path=agent_context_path,
        answer_path=answer_path,
    )


def research_agent_payload(result: ResearchAgentResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "graphdb_write": result.graphdb_write,
        "bundle": {
            "output_dir": str(result.bundle.output_dir),
            "manifest_path": str(result.bundle.manifest_path),
            "ingest_report_path": str(result.bundle.ingest_report_path),
            "retrieval_context_path": str(result.bundle.retrieval_context_path),
            "report": report_payload(result.bundle.report),
        },
        "agent_context_path": str(result.agent_context_path),
        "answer_path": str(result.answer_path),
        "answer": _answer_payload(result.answer),
    }


def _answer_payload(answer: ResearchAnswer) -> dict[str, Any]:
    return {
        "answer_status": answer.answer_status,
        "summary": answer.summary,
        "citations": [
            {"record_id": citation.record_id, "source_url": citation.source_url, "title": citation.title}
            for citation in answer.citations
        ],
        "evidence_record_ids": list(answer.evidence_record_ids),
        "graph_lookup_query": answer.graph_lookup_query,
    }


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)
