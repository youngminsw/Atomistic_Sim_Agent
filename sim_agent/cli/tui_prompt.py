from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol, TextIO

from sim_agent.agents_sdk_runtime.runtime import AGENT_ROLES

from .tui_catalog import SIMULATION_SKILLS, command_names, suggested_commands


@dataclass(frozen=True, slots=True)
class SlashCompletionRow:
    value: str
    insert_text: str
    kind: Literal["command", "skill", "agent"]
    meta: str
    display: str


class PromptSessionProtocol(Protocol):
    def prompt(self, message: str) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class TuiPromptReader:
    input_stream: TextIO
    output_stream: TextIO
    prompt_session: PromptSessionProtocol | None
    echoes_input: bool

    def read_line(self) -> str:
        if self.prompt_session is not None:
            try:
                return self.prompt_session.prompt("asa> ") + "\n"
            except EOFError:
                return ""
            except KeyboardInterrupt:
                self.output_stream.write("^C\n")
                return ""
        if self.echoes_input:
            try:
                return input("asa> ") + "\n"
            except EOFError:
                return ""
            except KeyboardInterrupt:
                self.output_stream.write("^C\n")
                return ""
        self.output_stream.write("asa> ")
        self.output_stream.flush()
        return self.input_stream.readline()


def build_prompt_reader(input_stream: TextIO, output_stream: TextIO) -> TuiPromptReader:
    interactive = _interactive_stdio(input_stream, output_stream)
    if not interactive:
        return TuiPromptReader(
            input_stream=input_stream,
            output_stream=output_stream,
            prompt_session=None,
            echoes_input=False,
        )
    prompt_session = _build_prompt_session()
    if prompt_session is None:
        _configure_readline()
    return TuiPromptReader(
        input_stream=input_stream,
        output_stream=output_stream,
        prompt_session=prompt_session,
        echoes_input=True,
    )


def slash_completion_rows(prefix: str) -> tuple[SlashCompletionRow, ...]:
    token = prefix.strip()
    if not token.startswith("/"):
        return ()
    rows = [
        SlashCompletionRow(
            value=command.name,
            insert_text=command.name,
            kind="command",
            meta=command.summary,
            display=command.usage,
        )
        for command in suggested_commands(token)
    ]
    if token == "/" or "/skills".startswith(token):
        rows.extend(
            SlashCompletionRow(
                value=name,
                insert_text="/skills",
                kind="skill",
                meta=summary,
                display=f"skill: {name}",
            )
            for name, summary in SIMULATION_SKILLS
        )
    return tuple(rows)


def agent_completion_rows(prefix: str) -> tuple[SlashCompletionRow, ...]:
    token = prefix.strip().removeprefix("@")
    rows = [
        SlashCompletionRow(
            value="@orchestrator",
            insert_text="@orchestrator",
            kind="agent",
            meta="default chat agent; routes work, approvals, and final assembly",
            display="@orchestrator",
        )
    ]
    rows.extend(
        SlashCompletionRow(
            value=f"@{role.role_id}",
            insert_text=f"@{role.role_id}",
            kind="agent",
            meta=role.boundary,
            display=f"@{role.role_id}  {role.display_name}",
        )
        for role in AGENT_ROLES
    )
    return tuple(row for row in rows if row.value.removeprefix("@").startswith(token))


def prompt_completion_rows(prefix: str) -> tuple[SlashCompletionRow, ...]:
    token = prefix.strip()
    if token.startswith("/"):
        return slash_completion_rows(token)
    if token.startswith("@"):
        return agent_completion_rows(token)
    return ()


def _build_prompt_session() -> PromptSessionProtocol | None:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import CompleteEvent, Completer, Completion
        from prompt_toolkit.document import Document
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.key_binding.key_processor import KeyPressEvent
        from prompt_toolkit.shortcuts import CompleteStyle
        from prompt_toolkit.styles import Style
    except ImportError:
        return None

    class SlashCompleter(Completer):
        def get_completions(
            self,
            document: Document,
            complete_event: CompleteEvent,
        ) -> Iterable[Completion]:
            del complete_event
            text = _active_token(document.text_before_cursor)
            if not text:
                return
            for row in prompt_completion_rows(text):
                yield Completion(
                    row.insert_text,
                    start_position=-len(text),
                    display=row.display,
                    display_meta=row.meta,
                )

    key_bindings = KeyBindings()

    @key_bindings.add("/")
    def open_slash_menu(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("/")
        event.current_buffer.start_completion(select_first=False)

    @key_bindings.add("@")
    def open_agent_menu(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("@")
        event.current_buffer.start_completion(select_first=False)

    return PromptSession(
        completer=SlashCompleter(),
        complete_while_typing=True,
        complete_style=CompleteStyle.COLUMN,
        key_bindings=key_bindings,
        reserve_space_for_menu=10,
        bottom_toolbar=HTML(
            "<b>/</b> palette  "
            "<style fg='ansicyan'><b>@</b> agents</style>  "
            "<style fg='ansiyellow'>/login</style> "
            "<style fg='ansicyan'>/hud</style> "
            "<style fg='ansicyan'>/model</style> "
            "<style fg='ansicyan'>/team</style> "
            "<style fg='ansicyan'>/compact</style> "
            "<style fg='ansicyan'>/skills</style> "
            "<style fg='ansicyan'>/runtime</style> "
            "<style fg='ansicyan'>/setup</style>"
        ),
        style=Style.from_dict(
            {
                "bottom-toolbar": "bg:#111614 #e5ebe7",
                "completion-menu": "bg:#111614 #d6ddd8",
                "completion-menu.completion.current": "bg:#33413b #ffffff",
                "completion-menu.meta.completion": "bg:#111614 #9aa8a0",
                "completion-menu.meta.completion.current": "bg:#33413b #d8a657",
            }
        ),
    )


def _configure_readline() -> None:
    try:
        import readline
    except ImportError:
        return

    names = command_names()

    def complete(text: str, state: int) -> str | None:
        matches = tuple(name for name in names if name.startswith(text))
        if state < len(matches):
            return matches[state]
        return None

    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")


def _active_token(text: str) -> str:
    parts = text.split()
    if not parts:
        return text
    return parts[-1]


def _interactive_stdio(input_stream: TextIO, output_stream: TextIO) -> bool:
    return (
        input_stream is sys.stdin
        and output_stream is sys.stdout
        and input_stream.isatty()
        and output_stream.isatty()
    )
