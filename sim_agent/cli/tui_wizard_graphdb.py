from __future__ import annotations

from typing import TextIO

from sim_agent.runtime_config import load_runtime_config

from .tui_select import MenuOption, choose_option, prompt_visible
from .tui_setup import handle_setup
from .tui_state import TuiState


def graphdb_wizard(state: TuiState, input_stream: TextIO, output_stream: TextIO) -> TuiState:
    graphdb = load_runtime_config().graphdb
    preset = choose_option(
        "GraphDB Profile",
        (
            MenuOption("current", "Current saved", f"{graphdb.database} · {graphdb.uri_env}"),
            MenuOption("project_default", "Project default", "youngmin-lab Neo4j env vars"),
            MenuOption("local", "Local Neo4j", "localhost bolt with standard env vars"),
            MenuOption("custom", "Custom", "type URI/env/database fields"),
            MenuOption("cancel", "Cancel", "return to shell"),
        ),
        input_stream,
        output_stream,
    )
    match preset:  # noqa: MATCH_OK - menu values are open terminal strings.
        case "current":
            uri, database, uri_env, user_env, password_env = (
                graphdb.uri,
                graphdb.database,
                graphdb.uri_env,
                graphdb.user_env,
                graphdb.password_env,
            )
        case "project_default":
            uri, database, uri_env, user_env, password_env = (
                "bolt://youngmin-lab:7687",
                "atomistic_sim_agent_knowledge",
                "NEO4J_URI",
                "NEO4J_USERNAME",
                "NEO4J_PASSWORD",
            )
        case "local":
            uri, database, uri_env, user_env, password_env = (
                "bolt://localhost:7687",
                "atomistic_sim_agent_knowledge",
                "NEO4J_URI",
                "NEO4J_USERNAME",
                "NEO4J_PASSWORD",
            )
        case "custom":
            uri = prompt_visible("Neo4j URI", graphdb.uri, input_stream, output_stream)
            database = prompt_visible("Project DB name", graphdb.database, input_stream, output_stream)
            uri_env = prompt_visible("URI env", graphdb.uri_env, input_stream, output_stream)
            user_env = prompt_visible("User env", graphdb.user_env, input_stream, output_stream)
            password_env = prompt_visible("Password env", graphdb.password_env, input_stream, output_stream)
        case _:
            output_stream.write("wizard_cancelled=true\n")
            return state
    return handle_setup(
        (
            "graphdb",
            "--uri",
            uri,
            "--uri-env",
            uri_env,
            "--user-env",
            user_env,
            "--password-env",
            password_env,
            "--database",
            database,
        ),
        state,
        output_stream,
    )
