from __future__ import annotations

from typing import TextIO

from .tui_state import TuiState


def handle_guide(state: TuiState, output_stream: TextIO) -> None:
    output_stream.write("초보자 안내\n")
    output_stream.write("friendly_guide=true\n")
    output_stream.write(f"model_visible={state.model.provider}/{state.model.name}/{state.model.reasoning_effort}/{state.model.auth_mode}\n")
    output_stream.write("goal_input_mode=plain_language_or_slash_command\n")
    output_stream.write("plain_goal_hint=그냥 하고 싶은 시뮬레이션을 문장으로 입력해도 됩니다\n")
    output_stream.write("quick_start=1:/model status\n")
    output_stream.write("quick_start=2:/wizard\n")
    output_stream.write("quick_start=3:/login\n")
    output_stream.write("quick_start=4:원하는 시뮬레이션 목표를 문장으로 입력\n")
    output_stream.write("quick_start=5:/runtime tools\n")
    output_stream.write("quick_start=6:/status\n")
    output_stream.write("next_step=/wizard\n")
    output_stream.write("next_step=/model status\n")
    output_stream.write("next_step=/workflow deep-interview\n")
    output_stream.write("next_step=/memory\n")
    output_stream.write("next_step=/tools\n")
    output_stream.write("next_step=/runtime tools\n")
    output_stream.write("friendly_note=긴 --옵션을 외우지 않아도 /wizard 에서 방향키로 설정할 수 있습니다\n")
