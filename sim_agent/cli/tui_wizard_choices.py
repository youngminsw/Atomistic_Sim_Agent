from __future__ import annotations

from typing import TextIO

from .tui_select import MenuOption, choose_option, prompt_visible


def choice(input_stream: TextIO, output_stream: TextIO, title: str, values: tuple[str, ...]) -> str:
    selected = choose_option(
        title,
        tuple(MenuOption(value, value, f"{title.lower()}={value}") for value in values),
        input_stream,
        output_stream,
    )
    return selected or values[0]


def choice_or_prompt(
    input_stream: TextIO,
    output_stream: TextIO,
    title: str,
    values: tuple[str, ...],
    default: str,
) -> str:
    selected = choice(input_stream, output_stream, title, (*values, "custom"))
    if selected == "custom":
        return prompt_visible(title, default, input_stream, output_stream)
    return selected


def goal_choice(input_stream: TextIO, output_stream: TextIO) -> str:
    selected = choose_option(
        "Simulation Goal",
        (
            MenuOption("ar_si_hole", "Ar / Si hole", "3D amorphous Si hole etch planning"),
            MenuOption("ar_si_trench", "Ar / Si trench", "3D amorphous Si trench etch planning"),
            MenuOption("custom", "Custom", "type a free-form simulation goal"),
        ),
        input_stream,
        output_stream,
    )
    match selected:  # noqa: MATCH_OK - menu values are open terminal strings.
        case "ar_si_hole" | None:
            return "Plan Ar etching on amorphous Si with a 3D hole pattern"
        case "ar_si_trench":
            return "Plan Ar etching on amorphous Si with a 3D trench pattern"
        case _:
            return prompt_visible(
                "Simulation goal",
                "Plan Ar etching on amorphous Si with a 3D hole pattern",
                input_stream,
                output_stream,
            )
