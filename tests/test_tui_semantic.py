from __future__ import annotations

from io import StringIO

from sim_agent.cli.tui_semantic import filter_semantic_tty_output


def test_tty_semantic_filter_hides_machine_readable_lines() -> None:
    output = _TtyStringIO()
    filtered = filter_semantic_tty_output(output)

    filtered.write("agent_direct_route=md_agent\n")
    filtered.write("provider=openai model=gpt-5.5 auth_mode=oauth\n")
    filtered.write("Login succeeded.\n")
    filtered.write("asa> ")

    assert output.getvalue() == "Login succeeded.\nasa> "


def test_non_tty_semantic_filter_keeps_machine_readable_lines() -> None:
    output = StringIO()
    filtered = filter_semantic_tty_output(output)

    filtered.write("agent_direct_route=md_agent\n")

    assert output.getvalue() == "agent_direct_route=md_agent\n"


class _TtyStringIO(StringIO):
    @property
    def encoding(self) -> str:
        return "utf-8"

    def isatty(self) -> bool:
        return True
