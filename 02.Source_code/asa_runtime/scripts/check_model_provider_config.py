from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.llm_endpoints import ModelPolicyError, ModelProviderConfig, ProviderConfigPolicyError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config = ModelProviderConfig.from_mapping(payload)
        spec = config.to_agents_sdk_model_spec()
    except (json.JSONDecodeError, OSError, ModelPolicyError, ProviderConfigPolicyError) as exc:
        print(str(exc))
        return 1

    print("model_provider_config=true")
    print(f"provider={config.provider}")
    print(f"primary_model={spec.model}")
    print(f"reasoning={spec.reasoning_effort}")
    print(f"use_case={config.use_case.value}")
    print(f"base_url={spec.base_url}")
    print(f"structured_outputs={str(spec.structured_outputs).lower()}")
    print(f"streaming={str(spec.streaming).lower()}")
    print(f"api_key_env={spec.api_key_env}")
    print(f"auth_mode={spec.auth_mode}")
    print(f"auth_refresh_configured={str(spec.auth_refresh_command is not None).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
