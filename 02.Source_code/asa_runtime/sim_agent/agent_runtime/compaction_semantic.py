from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Final, Protocol

from .compaction_redaction import redact_secret_maps, redact_secret_text


PROMPT_DIR: Final = Path(__file__).with_name("prompts") / "compaction"
SYSTEM_PROMPT_NAME: Final = "summarization-system"
SUMMARY_PROMPT_NAME: Final = "compaction-summary"
PROMPT_NAMES: Final = (
    SYSTEM_PROMPT_NAME,
    SUMMARY_PROMPT_NAME,
    "compaction-update-summary",
    "compaction-short-summary",
    "compaction-turn-prefix",
    "file-operations",
    "compaction-summary-context",
)
PATH_PATTERN: Final = re.compile(
    r"""(?ix)
    \b(?P<op>read|open|view|cat|write|edit|modify|create|delete)
    \s*\(\s*(?:path|file|target)?\s*=?\s*
    (?P<quote>["'])
    (?P<path>[^"']+)
    (?P=quote)
    """
)


@dataclass(frozen=True, slots=True)
class SemanticSummaryRequest:
    agent_id: str
    compact_id: str
    compact_mode: str
    summary_source: str
    system_prompt: str
    prompt: str
    messages_to_summarize: tuple[dict[str, object], ...]
    turn_prefix_messages: tuple[dict[str, object], ...]
    retained_messages: tuple[dict[str, object], ...]
    previous_summary: str
    additional_focus: str = ""


@dataclass(frozen=True, slots=True)
class SemanticSummaryResult:
    summary: str
    short_summary: str = ""
    preserve_data: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class SemanticSummaryUnavailable(RuntimeError):
    reason: str

    def __str__(self) -> str:
        if self.reason == "semantic_summary_auth_missing" or self.reason.startswith("semantic_summarizer_unavailable"):
            return self.reason
        return f"semantic_summarizer_unavailable:{self.reason}"


class CompactionSemanticSummarizer(Protocol):
    def summarize(self, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        """Return a semantic checkpoint for the compacted conversation span."""


def load_compaction_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def semantic_prompt_contract() -> dict[str, object]:
    hashes = {name: _sha256(load_compaction_prompt(name)) for name in PROMPT_NAMES}
    return {
        "kind": "gajae_compaction",
        "prompt_file_names": tuple(f"{name}.md" for name in PROMPT_NAMES),
        "prompt_sha256": hashes,
        "system_prompt_sha256": hashes[SYSTEM_PROMPT_NAME],
        "summary_prompt_sha256": hashes[SUMMARY_PROMPT_NAME],
    }


def build_semantic_summary_request(
    *,
    agent_id: str,
    compact_id: str,
    compact_mode: str,
    summary_source: str,
    messages: tuple[Mapping[str, object], ...],
    first_kept_sequence: int,
    summary_cutoff_sequence: int,
    previous_summary: str = "",
    additional_focus: str = "",
) -> SemanticSummaryRequest:
    raw_messages_to_summarize = tuple(
        dict(message) for message in messages if _is_summarized(message, first_kept_sequence, summary_cutoff_sequence)
    )
    raw_retained_messages = tuple(dict(message) for message in messages if _is_retained(message, first_kept_sequence))
    messages_to_summarize = redact_secret_maps(raw_messages_to_summarize)
    retained_messages = redact_secret_maps(raw_retained_messages)
    turn_prefix_messages = retained_messages[:2]
    system_prompt = load_compaction_prompt(SYSTEM_PROMPT_NAME)
    prompt = _render_summary_prompt(
        agent_id=agent_id,
        compact_id=compact_id,
        compact_mode=compact_mode,
        summary_source=summary_source,
        previous_summary=redact_secret_text(previous_summary),
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        retained_messages=retained_messages,
        additional_focus=redact_secret_text(additional_focus),
    )
    return SemanticSummaryRequest(
        agent_id=agent_id,
        compact_id=compact_id,
        compact_mode=compact_mode,
        summary_source=summary_source,
        system_prompt=system_prompt,
        prompt=prompt,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        retained_messages=retained_messages,
        previous_summary=redact_secret_text(previous_summary),
        additional_focus=redact_secret_text(additional_focus),
    )


def render_provider_compaction_summary(summary: str) -> str:
    template = load_compaction_prompt("compaction-summary-context")
    return template.replace("{{summary}}", redact_secret_text(summary.strip()))


def extract_semantic_file_operations(
    messages: Iterable[Mapping[str, object]],
    previous_details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    read_files = _strings_from_details(previous_details, "readFiles")
    modified_files = _strings_from_details(previous_details, "modifiedFiles")
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            _collect_file_operations(content, read_files, modified_files)
    modified = tuple(dict.fromkeys(modified_files))
    read = tuple(path for path in dict.fromkeys(read_files) if path not in set(modified))
    return {"readFiles": list(read), "modifiedFiles": list(modified)}


def upsert_file_operations(summary: str, read_files: Iterable[str], modified_files: Iterable[str]) -> str:
    without_read = _replace_or_append_block(summary, "read-files", _xml_block("read-files", tuple(read_files)))
    return _replace_or_append_block(without_read, "modified-files", _xml_block("modified-files", tuple(modified_files)))


def _render_summary_prompt(
    *,
    agent_id: str,
    compact_id: str,
    compact_mode: str,
    summary_source: str,
    previous_summary: str,
    messages_to_summarize: tuple[dict[str, object], ...],
    turn_prefix_messages: tuple[dict[str, object], ...],
    retained_messages: tuple[dict[str, object], ...],
    additional_focus: str,
) -> str:
    base = load_compaction_prompt("compaction-update-summary" if previous_summary else SUMMARY_PROMPT_NAME)
    sections = [
        base,
        load_compaction_prompt("compaction-short-summary"),
        load_compaction_prompt("compaction-turn-prefix"),
        load_compaction_prompt("file-operations"),
    ]
    focus = additional_focus.strip()
    if focus:
        sections.append("Additional focus:\n" + focus)
    sections.extend(
        (
            f"agent_id: {agent_id}",
            f"compact_id: {compact_id}",
            f"compact_mode: {compact_mode}",
            f"summary_source: {summary_source}",
            "<previous-summary>\n" + previous_summary.strip() + "\n</previous-summary>",
            "<conversation>\n" + _render_messages(messages_to_summarize) + "\n</conversation>",
            "<turn-prefix>\n" + _render_messages(turn_prefix_messages) + "\n</turn-prefix>",
            "<retained-tail>\n" + _render_messages(retained_messages) + "\n</retained-tail>",
        )
    )
    return "\n\n".join(sections)


def _render_messages(messages: tuple[dict[str, object], ...]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        sequence = message.get("sequence")
        label = role if isinstance(role, str) else "message"
        prefix = f"[{sequence}] {label}" if isinstance(sequence, int) and not isinstance(sequence, bool) else label
        lines.append(f"{prefix}: {content}" if isinstance(content, str) else f"{prefix}:")
    return "\n".join(lines)


def _is_summarized(message: Mapping[str, object], first_kept_sequence: int, summary_cutoff_sequence: int) -> bool:
    sequence = _sequence(message)
    return sequence < first_kept_sequence if sequence is not None else summary_cutoff_sequence > 0


def _is_retained(message: Mapping[str, object], first_kept_sequence: int) -> bool:
    sequence = _sequence(message)
    return sequence >= first_kept_sequence if sequence is not None else False


def _sequence(message: Mapping[str, object]) -> int | None:
    value = message.get("sequence")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _collect_file_operations(content: str, read_files: list[str], modified_files: list[str]) -> None:
    for match in PATH_PATTERN.finditer(content):
        path = match.group("path")
        op = match.group("op").lower()
        if op in {"write", "edit", "modify", "create", "delete"}:
            modified_files.append(path)
        else:
            read_files.append(path)


def _strings_from_details(details: Mapping[str, object] | None, key: str) -> list[str]:
    if details is None:
        return []
    value = details.get(key)
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]


def _replace_or_append_block(summary: str, tag: str, block: str) -> str:
    pattern = re.compile(rf"<{tag}>.*?</{tag}>", re.DOTALL)
    if pattern.search(summary):
        return pattern.sub(block, summary)
    return summary.rstrip() + "\n\n" + block


def _xml_block(tag: str, paths: tuple[str, ...]) -> str:
    lines = [f"<{tag}>"]
    lines.extend(f"  <file>{escape(path)}</file>" for path in paths)
    lines.append(f"</{tag}>")
    return "\n".join(lines)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
