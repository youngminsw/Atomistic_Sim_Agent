from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final


SECRET_REDACTION_MARKER: Final = "[REDACTED_SECRET]"
_SECRET_VALUE_CHARS: Final = r"""[^\s"'`,;)}\]]+"""
_AUTHORIZATION_BEARER_PATTERN: Final = re.compile(
    rf"(?i)\b(Authorization\s*:\s*Bearer\s+)({_SECRET_VALUE_CHARS})",
)
_BEARER_PATTERN: Final = re.compile(rf"(?i)\b(Bearer\s+)({_SECRET_VALUE_CHARS})")
_ASSIGNMENT_SECRET_PATTERN: Final = re.compile(
    rf"""(?ix)
    \b(
      [A-Z0-9_.-]*
      (?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|AUTH)
      [A-Z0-9_.-]*
      \s*=\s*
    )
    ({_SECRET_VALUE_CHARS})
    """,
)
_JSON_SECRET_PATTERN: Final = re.compile(
    rf"""(?ix)
    (
      ["']?
      [A-Z0-9_.-]*
      (?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|AUTHORIZATION)
      [A-Z0-9_.-]*
      ["']?
      \s*:\s*
      ["']
    )
    ([^"'\s,}}]+)
    (["'])
    """,
)


def redact_secret_text(value: str) -> str:
    redacted = _AUTHORIZATION_BEARER_PATTERN.sub(rf"\1{SECRET_REDACTION_MARKER}", value)
    redacted = _BEARER_PATTERN.sub(rf"\1{SECRET_REDACTION_MARKER}", redacted)
    redacted = _ASSIGNMENT_SECRET_PATTERN.sub(rf"\1{SECRET_REDACTION_MARKER}", redacted)
    return _JSON_SECRET_PATTERN.sub(rf"\1{SECRET_REDACTION_MARKER}\3", redacted)


def redact_secret_value(value: object) -> object:
    if isinstance(value, str):
        return redact_secret_text(value)
    if isinstance(value, Mapping):
        return redact_secret_map(value)
    if isinstance(value, list | tuple):
        return [redact_secret_value(item) for item in value]
    return value


def redact_secret_map(value: Mapping[str, object]) -> dict[str, object]:
    return {redact_secret_text(str(key)): redact_secret_value(item) for key, item in value.items()}


def redact_secret_maps(values: tuple[Mapping[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(redact_secret_map(value) for value in values)
