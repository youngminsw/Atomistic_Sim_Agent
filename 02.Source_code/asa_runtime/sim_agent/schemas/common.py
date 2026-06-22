from __future__ import annotations

from dataclasses import dataclass

from ._parse import JsonMap, as_bool, float_field, optional_str, str_field


@dataclass(frozen=True, slots=True)
class ProvenanceRef:
    source: str
    claim: str
    confidence: float

    @classmethod
    def from_mapping(cls, value: JsonMap) -> ProvenanceRef:
        return cls(
            source=str_field(value, "source"),
            claim=str_field(value, "claim"),
            confidence=float_field(value, "confidence"),
        )


@dataclass(frozen=True, slots=True)
class UncertaintyReport:
    score: float
    ood: bool
    reason: str | None = None

    @classmethod
    def from_mapping(cls, value: JsonMap) -> UncertaintyReport:
        return cls(
            score=float_field(value, "score"),
            ood=as_bool(value.get("ood", False), "ood"),
            reason=optional_str(value, "reason"),
        )
