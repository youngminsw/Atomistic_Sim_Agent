<asa-domain-workflow-policy agent="ml_agent">
# ML workflow policy

- Start with the dataset gate: source trajectories, labels, splits, leakage controls, feature definitions, units, and uncertainty targets must be explicit.
- Train or evaluate surrogates only from receipt-backed MD or approved fixture data.
- Record model manifests, metrics, calibration checks, failure cases, and reproducibility parameters before reporting readiness.
- Treat data leakage, missing provenance, unstable calibration, or unsupported extrapolation as hard blockers.
- Hand off only gated surrogate outputs and uncertainty envelopes to feature-scale and QA agents.
</asa-domain-workflow-policy>
