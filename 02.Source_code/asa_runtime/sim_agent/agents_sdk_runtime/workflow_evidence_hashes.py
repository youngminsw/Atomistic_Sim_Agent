from __future__ import annotations

import hashlib
from pathlib import Path

from sim_agent.schemas._parse import JsonMap


type EvidenceHashValue = str | int | float | bool | None | JsonMap | list["EvidenceHashValue"] | tuple[
    "EvidenceHashValue", ...
]


def artifact_hashes(ledger: JsonMap | None) -> JsonMap:
    raw_hashes = ledger.get("artifact_hashes") if ledger is not None else None
    return raw_hashes if isinstance(raw_hashes, dict) else {}


def verify_artifact_hash(path: Path, artifact_text: str, hashes: JsonMap, blockers: list[str]) -> None:
    expected = _normalized_sha256(hashes.get(artifact_text))
    if expected is None:
        blockers.append("artifact_hash_missing")
        return
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        blockers.append("artifact_missing")
        return
    if actual != expected:
        blockers.append("artifact_hash_mismatch")


def _normalized_sha256(value: EvidenceHashValue) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.removeprefix("sha256:")
