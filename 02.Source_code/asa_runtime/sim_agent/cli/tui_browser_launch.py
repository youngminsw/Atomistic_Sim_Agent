from __future__ import annotations

import os
import subprocess
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO


BROWSER_OPEN_ENV = "ASA_BROWSER_OAUTH_OPEN"


def open_url_in_browser(url: str, flags: Sequence[str]) -> bool:
    disabled = os.environ.get(BROWSER_OPEN_ENV, "").lower() in {"0", "false", "no", "off"}
    if disabled or "no_open" in flags:
        return False
    if _is_wsl():
        return _open_with_windows_default_browser(url)
    try:
        if webbrowser.open(url, new=1, autoraise=True):
            return True
    except webbrowser.Error:
        pass
    return _open_with_windows_default_browser(url)


def write_oauth_browser_block(
    output_stream: TextIO,
    *,
    url: str,
    opened: bool,
    instructions: str | None = None,
    callback_url: str | None = None,
    user_code: str | None = None,
) -> None:
    output_stream.write("\nOAuth Login\n")
    output_stream.write("Open this URL in your browser:\n")
    output_stream.write(f"{url}\n")
    output_stream.write("If the browser is hidden, copy/paste the URL above.\n")
    if user_code:
        output_stream.write(f"Code: {user_code}\n")
    if callback_url:
        output_stream.write(f"Callback: {callback_url}\n")
    if instructions:
        output_stream.write(f"{instructions}\n")
    if not opened:
        output_stream.write("Browser did not report as opened; copy the URL above if needed.\n")
    output_stream.write("\n")


def _open_with_windows_default_browser(url: str) -> bool:
    if not _is_wsl():
        return False
    for command in _windows_open_commands(url):
        try:
            completed = subprocess.run(
                list(command),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            continue
        if completed.returncode == 0:
            return True
    return False


def _windows_open_commands(url: str) -> tuple[tuple[str, ...], ...]:
    return (
        ("rundll32.exe", "url.dll,FileProtocolHandler", url),
        ("/mnt/c/Windows/System32/rundll32.exe", "url.dll,FileProtocolHandler", url),
        ("cmd.exe", "/c", "start", "", url),
        ("/mnt/c/Windows/System32/cmd.exe", "/c", "start", "", url),
    )


def _is_wsl() -> bool:
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version
