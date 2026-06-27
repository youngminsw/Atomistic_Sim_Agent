<asa-domain-workflow-policy agent="research_agent">
# Research workflow policy

- Start with the provenance gate: claim type, source requirement, citation target, graph/MCP boundary, and confidence level must be explicit.
- Prefer source-backed project memory, configured MCP reads, and verifiable references before synthesizing scientific or implementation claims.
- Do not write to GraphDB, files, or external systems unless workflow state and tool policy explicitly allow the mutation.
- Record citation, graph, and search receipts for assumptions that affect MD, ML, feature-scale, or QA decisions.
- Escalate paywalled, credential-gated, stale, or conflicting sources as blockers unless the user accepts a bounded assumption.
</asa-domain-workflow-policy>
