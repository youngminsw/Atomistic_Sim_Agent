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


@dataclass(frozen=True, slots=True)
class ProviderVisibleTokenEstimate:
    total_tokens: int
    breakdown: Mapping[str, int]


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
    if not is_valid_threshold_percent(settings.threshold_percent):
        return None
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
        return _clamp((context_window_tokens * settings.threshold_percent) // 100, 1, context_window_tokens - 1)
    reserve = effective_reserve_tokens(context_window_tokens, settings)
    return _clamp(context_window_tokens - reserve, 1, context_window_tokens - 1)


def effective_reserve_tokens(context_window_tokens: int, settings: CompactionTokenSettings) -> int:
    proportional_reserve = max(0, (context_window_tokens * 15) // 100)
    return max(proportional_reserve, max(0, settings.reserve_tokens))


def should_compact_tokens(estimated_context_tokens: int, budget: CompactionTokenBudget) -> bool:
    return budget.enabled and estimated_context_tokens > budget.threshold_tokens


def estimate_messages_tokens(messages: tuple[Mapping[str, object], ...] | list[Mapping[str, object]]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def estimate_provider_visible_tokens(payload: Mapping[str, object]) -> ProviderVisibleTokenEstimate:
    breakdown = {
        "system_context": 0,
        "compact_summary": 0,
        "role_context": 0,
        "caller_context": 0,
        "workflow_context": 0,
        "project_context": 0,
        "skill_context": 0,
        "messages": 0,
        "tools": 0,
        "other_context": 0,
    }
    _add_prompt_layers(breakdown, payload.get("layers"))
    _add_text_fields(
        breakdown,
        payload,
        (
            ("system_context", ("system_context", "instructions", "system", "systemInstruction")),
            ("compact_summary", ("compact_summary",)),
            ("role_context", ("role_context",)),
            ("caller_context", ("caller_context",)),
            ("workflow_context", ("workflow_context", "workflow_state", "ledger_facts", "tool_history")),
            ("project_context", ("project_context",)),
            ("skill_context", ("skill_context", "skills")),
        ),
    )
    _add_payload_messages(breakdown, payload)
    tools = payload.get("tools")
    if tools is not None:
        breakdown["tools"] += estimate_text_tokens(_content_text(tools))
    breakdown["other_context"] += _estimate_other_payload_tokens(payload)
    return ProviderVisibleTokenEstimate(sum(breakdown.values()), breakdown)


def estimate_message_tokens(message: Mapping[str, object]) -> int:
    role = message.get("role")
    role_tokens = estimate_text_tokens(role) if isinstance(role, str) else 0
    return 4 + role_tokens + estimate_text_tokens(_message_content_text(message))


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


def is_valid_threshold_percent(value: int) -> bool:
    return not isinstance(value, bool) and MIN_THRESHOLD_PERCENT <= value <= MAX_THRESHOLD_PERCENT


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _message_content_text(message: Mapping[str, object]) -> str:
    parts: list[str] = []
    for field in ("content", "parts"):
        value = message.get(field)
        if value is not None:
            parts.append(_content_text(value))
    return "\n".join(part for part in parts if part)


def _add_prompt_layers(breakdown: dict[str, int], raw_layers: object) -> None:
    if not isinstance(raw_layers, list | tuple):
        return
    for raw_layer in raw_layers:
        if not isinstance(raw_layer, Mapping):
            continue
        content = raw_layer.get("content")
        tokens = estimate_text_tokens(_content_text(content))
        kind = raw_layer.get("kind")
        if not isinstance(kind, str):
            breakdown["other_context"] += tokens
            continue
        breakdown[_surface_for_layer_kind(kind)] += tokens


def _surface_for_layer_kind(kind: str) -> str:
    match kind:
        case "system_policy":
            return "system_context"
        case "compact_summary":
            return "compact_summary"
        case "domain_role" | "subagent_role":
            return "role_context"
        case "caller_context":
            return "caller_context"
        case "workflow_policy" | "workflow_state" | "ledger_facts" | "tool_history":
            return "workflow_context"
        case "project_guidance":
            return "project_context"
        case "skills":
            return "skill_context"
        case _:
            return "other_context"


def _add_text_fields(
    breakdown: dict[str, int],
    payload: Mapping[str, object],
    aliases_by_surface: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    for surface, aliases in aliases_by_surface:
        for alias in aliases:
            value = payload.get(alias)
            if value is not None:
                breakdown[surface] += estimate_text_tokens(_content_text(value))


def _add_payload_messages(breakdown: dict[str, int], payload: Mapping[str, object]) -> None:
    for field in ("input", "messages", "contents"):
        value = payload.get(field)
        if isinstance(value, list | tuple):
            _add_message_sequence(breakdown, value)


def _add_message_sequence(breakdown: dict[str, int], messages: list[object] | tuple[object, ...]) -> None:
    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            breakdown["messages"] += estimate_text_tokens(_content_text(raw_message))
            continue
        role = raw_message.get("role")
        if role == "system":
            breakdown["system_context"] += estimate_text_tokens(_message_content_text(raw_message))
        else:
            breakdown["messages"] += estimate_message_tokens(raw_message)


def _estimate_other_payload_tokens(payload: Mapping[str, object]) -> int:
    counted = {
        "layers",
        "system_context",
        "instructions",
        "system",
        "systemInstruction",
        "compact_summary",
        "role_context",
        "caller_context",
        "workflow_context",
        "workflow_state",
        "ledger_facts",
        "tool_history",
        "project_context",
        "skill_context",
        "skills",
        "input",
        "messages",
        "contents",
        "tools",
    }
    other = {key: value for key, value in payload.items() if key not in counted}
    return estimate_text_tokens(_content_text(other)) if other else 0


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
