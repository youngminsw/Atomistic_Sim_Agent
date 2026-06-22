from __future__ import annotations

from dataclasses import dataclass

from sim_agent.schemas._parse import JsonMap, as_str


@dataclass(frozen=True, slots=True)
class GraphDBRuntimeConfig:
    uri: str
    uri_env: str
    user_env: str
    password_env: str
    database: str


def default_graphdb_config() -> GraphDBRuntimeConfig:
    return GraphDBRuntimeConfig(
        uri="bolt://youngmin-lab:7687",
        uri_env="NEO4J_URI",
        user_env="NEO4J_USERNAME",
        password_env="NEO4J_PASSWORD",
        database="atomistic_sim_agent_knowledge",
    )


def graphdb_from_payload(payload: JsonMap) -> GraphDBRuntimeConfig:
    return GraphDBRuntimeConfig(
        uri=_optional_text(payload, "uri", "bolt://youngmin-lab:7687"),
        uri_env=_optional_text(payload, "uri_env", "NEO4J_URI"),
        user_env=_optional_text(payload, "user_env", "NEO4J_USERNAME"),
        password_env=_optional_text(payload, "password_env", "NEO4J_PASSWORD"),
        database=_optional_text(payload, "database", "atomistic_sim_agent_knowledge"),
    )


def graphdb_payload(graphdb: GraphDBRuntimeConfig) -> JsonMap:
    return {
        "uri": graphdb.uri,
        "uri_env": graphdb.uri_env,
        "user_env": graphdb.user_env,
        "password_env": graphdb.password_env,
        "database": graphdb.database,
    }


def _optional_text(payload: JsonMap, field: str, default: str) -> str:
    value = payload.get(field, default)
    return as_str(value, field)
