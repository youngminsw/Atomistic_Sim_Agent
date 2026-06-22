from __future__ import annotations

from typing import TextIO

from .tui_run import handle_run
from .tui_state import TuiState, append_event
from .tui_wizard_choices import choice, choice_or_prompt, goal_choice
from .tui_workflow import handle_workflow


def interview_run_wizard(state: TuiState, input_stream: TextIO, output_stream: TextIO) -> TuiState:
    goal = goal_choice(input_stream, output_stream)
    material = choice_or_prompt(input_stream, output_stream, "Material", ("Si", "SiO2", "Al2O3", "C"), "Si")
    ion = choice_or_prompt(input_stream, output_stream, "Ion", ("Ar", "CFx", "O", "Cl", "F"), "Ar")
    phase = choice(input_stream, output_stream, "Phase", ("amorphous", "crystalline"))
    feature = choice(input_stream, output_stream, "Feature", ("hole", "trench"))
    append_event(state, "wizard_deep_interview_prefill", goal)
    output_stream.write("deep_interview_prefill=true\n")
    next_state = handle_workflow(("deep-interview", "--output-dir", str(state.session_dir / "workflows")), state, output_stream)
    return handle_run(
        (
            "--material",
            material,
            "--phase",
            phase,
            "--ion",
            ion,
            "--feature-type",
            feature,
            "--output-dir",
            str(state.session_dir / "wizard-run"),
            goal,
        ),
        next_state,
        output_stream,
    )
