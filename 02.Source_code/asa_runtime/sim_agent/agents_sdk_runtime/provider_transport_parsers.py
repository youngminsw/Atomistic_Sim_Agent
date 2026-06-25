from __future__ import annotations

from sim_agent.schemas._parse import JsonMap


def openai_responses_tool_calls(response: JsonMap) -> tuple[JsonMap, ...]:
    output = response.get("output")
    if isinstance(output, list):
        calls = [
            item
            for item in output
            if isinstance(item, dict) and item.get("type") in {"tool_call", "function_call"}
        ]
        if calls:
            return tuple(calls)
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list):
        calls = [item for item in tool_calls if isinstance(item, dict)]
        if calls:
            return tuple(calls)
    return ()


def openai_responses_final_text(response: JsonMap) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
    return "\n".join(chunks)


def openai_chat_tool_calls(response: JsonMap) -> tuple[JsonMap, ...]:
    choices = response.get("choices")
    if not isinstance(choices, list):
        return ()
    calls: list[JsonMap] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            calls.extend(item for item in tool_calls if isinstance(item, dict))
    return tuple(calls)


def openai_chat_final_text(response: JsonMap) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list):
        return ""
    chunks: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            chunks.append(content)
    return "\n".join(chunks)


def anthropic_tool_calls(response: JsonMap) -> tuple[JsonMap, ...]:
    content = response.get("content")
    if not isinstance(content, list):
        return ()
    return tuple(
        {"name": item.get("name"), "arguments": item.get("input", {})}
        for item in content
        if isinstance(item, dict) and item.get("type") == "tool_use"
    )


def anthropic_final_text(response: JsonMap) -> str:
    content = response.get("content")
    if not isinstance(content, list):
        return ""
    chunks = [
        item.get("text")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
    ]
    return "\n".join(chunks)


def gemini_tool_calls(response: JsonMap) -> tuple[JsonMap, ...]:
    calls: list[JsonMap] = []
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        return ()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                calls.append({"name": function_call.get("name"), "arguments": function_call.get("args", {})})
    return tuple(calls)


def gemini_final_text(response: JsonMap) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        return ""
    chunks: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
    return "\n".join(chunks)
