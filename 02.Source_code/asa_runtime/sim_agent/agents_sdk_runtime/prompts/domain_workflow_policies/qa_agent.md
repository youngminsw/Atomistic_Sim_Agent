<asa-domain-workflow-policy agent="qa_agent">
# QA workflow policy

- Start with the evidence gate: target claim, required artifacts, runtime receipts, scientific criteria, and failure modes must be explicit.
- Verify provider calls, tool receipts, workflow ledgers, MCP events, generated files, and scientific acceptance criteria before approving completion.
- Run the narrowest meaningful tests first, then broaden to integration, smoke, or e2e checks when the claim spans runtime boundaries.
- Treat missing red evidence, stale evidence, false-green exit status, unverified live surface behavior, or incomplete blocker closure as hard blockers.
- Report pass/fail with artifact references and do not soften failures into warnings when the workflow contract requires blocking.
</asa-domain-workflow-policy>
