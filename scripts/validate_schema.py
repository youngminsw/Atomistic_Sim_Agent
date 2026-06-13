from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent import schemas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    args = parser.parse_args()

    try:
        payload = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
        request = schemas.SimulationRequest.from_mapping(payload)
        event_valid = False
        if isinstance(payload, dict) and "sample_event_bundle" in payload:
            schemas.EventBundle.from_mapping(payload["sample_event_bundle"])
            event_valid = True
        print("schema_valid=true")
        print(f"request_id={request.request_id}")
        print(f"event_bundle_valid={str(event_valid).lower()}")
        return 0
    except schemas.SchemaValidationError as exc:
        print(str(exc))
        return 1
    except ValueError as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
