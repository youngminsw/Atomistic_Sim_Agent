from __future__ import annotations

from sim_agent.schemas._parse import JsonMap, as_mapping, require


MODEL_PROVIDER_PAYLOAD_KEY = "model_provider"
LEGACY_LLM_ENDPOINT_PAYLOAD_KEY = "llm_endpoint"


def model_provider_payload(payload: JsonMap, field: str = MODEL_PROVIDER_PAYLOAD_KEY) -> JsonMap:
    if payload.get(MODEL_PROVIDER_PAYLOAD_KEY) is not None:
        return as_mapping(payload[MODEL_PROVIDER_PAYLOAD_KEY], field)
    return as_mapping(
        require(payload, LEGACY_LLM_ENDPOINT_PAYLOAD_KEY, f"{MODEL_PROVIDER_PAYLOAD_KEY} required"),
        LEGACY_LLM_ENDPOINT_PAYLOAD_KEY,
    )
