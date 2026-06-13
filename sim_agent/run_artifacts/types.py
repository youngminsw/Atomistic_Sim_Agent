from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class RunArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunBundle:
    run_id: str
    output_dir: Path
    manifest_path: Path
    timeline_path: Path
    diagnostics_path: Path

    @property
    def artifact_count(self) -> int:
        return 3
