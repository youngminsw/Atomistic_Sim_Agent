from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
FIXTURE_ROOT = SOURCE_ROOT / "tests" / "fixtures" / "requests"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _schema_api():
    try:
        from sim_agent import schemas
    except ModuleNotFoundError as exc:
        pytest.fail(f"schema API not implemented: {exc}")
    return schemas


def test_valid_physical_bombardment_request_parses() -> None:
    # Given: a complete physical-bombardment request with a sample event bundle.
    schemas = _schema_api()
    payload = _load_fixture("valid_ar_si_pr_hole.json")

    # When: the request crosses the schema boundary.
    request = schemas.SimulationRequest.from_mapping(payload)
    event_bundle = schemas.EventBundle.from_mapping(payload["sample_event_bundle"])

    # Then: typed fields preserve the required physics contracts.
    assert request.request_id == "valid_ar_si_pr_hole"
    assert request.recipe.ion_energy_distribution.kind == "histogram"
    assert request.recipe.ion_angular_distribution.polar_max_deg == 60.0
    assert request.scene.surface_state.amorphous_index == 0.0
    assert event_bundle.sputtering.yield_atoms_per_ion == pytest.approx(1.1)
    assert event_bundle.uncertainty.ood is False


def test_missing_iedf_is_rejected_with_clear_error() -> None:
    # Given: a request missing the ion energy distribution.
    schemas = _schema_api()
    payload = _load_fixture("missing_iedf.json")

    # When / Then: the boundary rejects it with the user-facing field name.
    with pytest.raises(schemas.SchemaValidationError, match="IonEnergyDistribution required"):
        schemas.SimulationRequest.from_mapping(payload)


def test_direct_openai_endpoint_is_accepted_when_explicitly_configured() -> None:
    schemas = _schema_api()
    payload = _load_fixture("direct_openai_valid.json")

    request = schemas.SimulationRequest.from_mapping(payload)

    assert request.model_provider.provider == "openai"
    assert request.model_provider.base_url == "https://api.openai.com/v1"
    assert request.llm_endpoint.provider == "openai"
    assert request.llm_endpoint.base_url == "https://api.openai.com/v1"


def test_model_provider_key_is_the_canonical_request_shape() -> None:
    schemas = _schema_api()
    payload = _load_fixture("direct_openai_valid.json")
    payload["model_provider"] = payload.pop("llm_endpoint")

    request = schemas.SimulationRequest.from_mapping(payload)

    assert request.model_provider.provider == "openai"
    assert request.model_provider.base_url == "https://api.openai.com/v1"


def test_openclaw_provider_with_non_openclaw_base_url_is_rejected() -> None:
    schemas = _schema_api()
    payload = _load_fixture("openclaw_provider_bad_base_url.json")

    with pytest.raises(schemas.ProviderConfigPolicyError, match="ProviderConfigPolicyError"):
        schemas.SimulationRequest.from_mapping(payload)


def test_openclaw_base_url_with_query_is_rejected() -> None:
    schemas = _schema_api()
    payload = _load_fixture("valid_ar_si_pr_hole.json")
    payload["llm_endpoint"]["base_url"] = "https://openclaw.local/v1?upstream=https://example.com"

    with pytest.raises(schemas.ProviderConfigPolicyError, match="ProviderConfigPolicyError"):
        schemas.SimulationRequest.from_mapping(payload)
