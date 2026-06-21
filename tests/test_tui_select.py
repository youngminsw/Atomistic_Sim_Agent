from __future__ import annotations

from io import StringIO

from sim_agent.cli.tui_select import CTRL_C_KEY, DOWN_KEY, ESC_KEY, MenuOption, UP_KEY, _decode_key_bytes, _move_up, _render_options


def test_menu_redraw_moves_to_previous_line_beginning() -> None:
    output = StringIO()

    _move_up(3, output)

    assert output.getvalue() == "\x1b[3A\r"


def test_menu_redraw_clears_each_line_from_column_one() -> None:
    output = StringIO()

    _render_options(
        (
            MenuOption("oauth", "OAuth gateway", "browser/OAuth backed model gateway"),
            MenuOption("api_key", "API key", "direct provider token or key"),
        ),
        1,
        output,
    )

    assert output.getvalue().startswith("\r\x1b[2K  OAuth gateway")
    assert "\n\r\x1b[2K❯ API key" in output.getvalue()


def test_raw_key_bytes_decode_arrow_and_interrupt_keys() -> None:
    assert _decode_key_bytes(b"\x1b[A") == UP_KEY
    assert _decode_key_bytes(b"\x1b[B") == DOWN_KEY
    assert _decode_key_bytes(b"\x1b") == ESC_KEY
    assert _decode_key_bytes(b"\x03") == CTRL_C_KEY
