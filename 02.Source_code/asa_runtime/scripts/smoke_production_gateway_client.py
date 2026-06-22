from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.agents_sdk_runtime.gateway_client import run_production_gateway_client_smoke
from sim_agent.llm_endpoints import ModelPolicyError, ModelProviderConfig, ProviderConfigPolicyError
from sim_agent.schemas._parse import as_mapping
from sim_agent.schemas.errors import SchemaValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    try:
        payload = as_mapping(json.loads(Path(args.request).read_text(encoding="utf-8")), "simulation_request")
        endpoint = ModelProviderConfig.from_mapping(as_mapping(payload.get("llm_endpoint"), "llm_endpoint"))
        result = run_production_gateway_client_smoke(
            payload,
            endpoint,
            Path(args.output_dir),
            api_key=args.api_key,
            offline=args.offline,
            timeout_s=args.timeout_s,
        )
    except (
        json.JSONDecodeError,
        OSError,
        SchemaValidationError,
        ProviderConfigPolicyError,
        ModelPolicyError,
    ) as exc:
        print(str(exc))
        return 1

    print(f"production_smoke={str(result.production_smoke).lower()}")
    print(f"fake_gateway_model={str(result.fake_gateway_model).lower()}")
    print(f"provider={result.provider}")
    print(f"model={result.model}")
    print(f"auth_mode={result.auth_mode}")
    print(f"gateway_request_id={result.gateway_request_id or ''}")
    for blocker in result.blockers:
        print(f"hard_blocker={blocker}")
    for path in result.session_files:
        print(f"session_file={path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
