from __future__ import annotations

import os
from typing import Final, TextIO


DEBUG_OUTPUT_ENV: Final = "ASA_TUI_DEBUG"
RECOMMENDED_PROFILE: Final = "codex-pro"


def machine_output(output_stream: TextIO) -> bool:
    debug = os.environ.get(DEBUG_OUTPUT_ENV, "").strip().lower()
    if debug in {"1", "true", "yes", "on"}:
        return True
    return not output_stream.isatty()


def write_login_success(
    output_stream: TextIO,
    *,
    provider: str,
    label: str,
) -> None:
    output_stream.write("Login successful.\n")
    output_stream.write(f"Signed in with {label}.\n")
    output_stream.write(f"Credential saved for {provider}.\n")
    output_stream.write(f"Next: /model profile {RECOMMENDED_PROFILE}\n")
