from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class RuntimeToolPolicy:
    policy_id: str
    mode: str
    process_allowlist: tuple[tuple[str, ...], ...]

    @property
    def process_policy_summary(self) -> str:
        return f"{self.mode}:exact_argv_allowlist:{len(self.process_allowlist)}"


DEFAULT_RUNTIME_TOOL_POLICY: Final = RuntimeToolPolicy(
    policy_id="safe-smoke-process-v1",
    mode="local_smoke",
    process_allowlist=(
        ("python3", "-c", "print('asa-tool-ok')"),
        ("python3", "-c", "print('gateway-tool-ok')"),
        ("python3", "--version"),
    ),
)


def is_process_allowed(policy: RuntimeToolPolicy, argv: tuple[str, ...]) -> bool:
    return argv in policy.process_allowlist
