from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final


DEFAULT_AUTO_COMPACT_THRESHOLD_PERCENT: Final = 70
DEFAULT_AUTO_COMPACT_THRESHOLD_TOKENS: Final = -1
DEFAULT_CONTEXT_WINDOW_TOKENS: Final = 0
DEFAULT_RESERVE_TOKENS: Final = 16_384
DEFAULT_KEEP_RECENT_TOKENS: Final = 20_000
DEFAULT_CHARS_PER_TOKEN: Final = 4
MIN_THRESHOLD_PERCENT: Final = 1
MAX_THRESHOLD_PERCENT: Final = 99


@dataclass(frozen=True, slots=True)
class CompactionTokenSettings:
    enabled: bool = True
    threshold_percent: int = DEFAULT_AUTO_COMPACT_THRESHOLD_PERCENT
    threshold_tokens: int = DEFAULT_AUTO_COMPACT_THRESHOLD_TOKENS
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS


@dataclass(frozen=True, slots=True)
class CompactionTokenBudget:
    enabled: bool
    context_window_tokens: int
    threshold_tokens: int
    reserve_tokens: int
    keep_recent_tokens: int


def compaction_budget_for_model(
    provider: str,
    model: str,
    settings: CompactionTokenSettings,
) -> CompactionTokenBudget | None:
    if not settings.enabled:
        return CompactionTokenBudget(
            enabled=False,
            context_window_tokens=0,
            threshold_tokens=0,
            reserve_tokens=max(0, settings.reserve_tokens),
            keep_recent_tokens=max(1, settings.keep_recent_tokens),
        )
    context_window = settings.context_window_tokens or known_context_window_tokens(provider, model)
    if context_window <= 0:
        return None
    return CompactionTokenBudget(
        enabled=True,
        context_window_tokens=context_window,
        threshold_tokens=resolve_threshold_tokens(context_window, settings),
        reserve_tokens=effective_reserve_tokens(context_window, settings),
        keep_recent_tokens=max(1, settings.keep_recent_tokens),
    )


def resolve_threshold_tokens(context_window_tokens: int, settings: CompactionTokenSettings) -> int:
    if context_window_tokens <= 1:
        return 1
    if settings.threshold_tokens > 0:
        return _clamp(settings.threshold_tokens, 1, context_window_tokens - 1)
    if settings.threshold_percent > 0:
        percent = _clamp(settings.threshold_percent, MIN_THRESHOLD_PERCENT, MAX_THRESHOLD_PERCENT)
        return _clamp((context_window_tokens * percent) // 100, 1, context_window_tokens - 1)
    reserve = effective_reserve_tokens(context_window_tokens, settings)
    return _clamp(context_window_tokens - reserve, 1, context_window_tokens - 1)


def effective_reserve_tokens(context_window_tokens: int, settings: CompactionTokenSettings) -> int:
    proportional_reserve = max(0, (context_window_tokens * 15) // 100)
    return max(proportional_reserve, max(0, settings.reserve_tokens))


def should_compact_tokens(estimated_context_tokens: int, budget: CompactionTokenBudget) -> bool:
    return budget.enabled and estimated_context_tokens > budget.threshold_tokens


def estimate_messages_tokens(messages: tuple[Mapping[str, object], ...] | list[Mapping[str, object]]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def estimate_message_tokens(message: Mapping[str, object]) -> int:
    role = message.get("role")
    content = message.get("content")
    role_tokens = estimate_text_tokens(role) if isinstance(role, str) else 0
    return 4 + role_tokens + estimate_text_tokens(_content_text(content))


def estimate_text_tokens(value: str) -> int:
    if not value:
        return 0
    return max(1, (len(value) + DEFAULT_CHARS_PER_TOKEN - 1) // DEFAULT_CHARS_PER_TOKEN)


def known_context_window_tokens(provider: str, model: str) -> int:
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip().lower()
    if normalized_provider in {"offline", "static"}:
        return 0
    if normalized_provider in {"openai-codex", "openai", "oauth_gateway"} and normalized_model.startswith("gpt-5"):
        return 272_000
    if normalized_provider in {"openai-codex", "openai", "oauth_gateway"} and normalized_model.startswith("gpt-4.1"):
        return 1_000_000
    if normalized_provider in {"anthropic", "google-antigravity"} and normalized_model.startswith("claude"):
        return 200_000
    if normalized_provider in {"gemini", "google-gemini-cli", "google-antigravity"} and normalized_model.startswith("gemini"):
        return 1_000_000
    if normalized_provider == "google-antigravity" and normalized_model.startswith("gpt-oss"):
        return 128_000
    if normalized_provider == "local_gateway" and normalized_model:
        return 272_000
    return 0


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
