from __future__ import annotations

import argparse
import os

from sim_agent.runtime_config import load_runtime_config
from sim_agent.schemas._parse import as_mapping, as_sequence
from sim_agent.ui.model_auth import (
    CREDENTIAL_STORE_ENV,
    login_model_provider,
    model_auth_status_payload,
    run_model_gateway_smoke_from_controller,
)


def add_auth_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    endpoint = load_runtime_config().model_endpoint
    parser = subparsers.add_parser("auth", help="Login, inspect, and smoke-test model provider credentials.")
    auth_subparsers = parser.add_subparsers(dest="auth_command", required=True)
    login = auth_subparsers.add_parser("login", help="Store a model provider access token.")
    login.add_argument("--provider", default=endpoint.provider)
    login.add_argument("--access-token")
    login.add_argument("--api-key")
    login.add_argument("--api-key-env")
    login.add_argument("--auth-mode", choices=("api_key", "oauth", "gateway"), default=endpoint.auth_mode)
    login.add_argument("--refresh-token")
    login.add_argument("--credential-store")
    status = auth_subparsers.add_parser("status", help="Show redacted credential status.")
    status.add_argument("--credential-store")
    smoke = auth_subparsers.add_parser("smoke", help="Call the configured model gateway API.")
    smoke.add_argument("--provider", default=endpoint.provider)
    smoke.add_argument("--model", default=endpoint.model)
    smoke.add_argument("--base-url", default=endpoint.base_url)
    smoke.add_argument("--auth-mode", choices=("api_key", "oauth", "gateway", "none"), default=endpoint.auth_mode)
    smoke.add_argument("--api-key-env", default=endpoint.api_key_env)
    smoke.add_argument("--credential-store")


def run_auth(args: argparse.Namespace) -> int:
    _apply_store(args)
    match args.auth_command:
        case "login":
            return _login(args)
        case "status":
            return _status()
        case "smoke":
            return _smoke(args)
        case _:
            print("unknown_auth_command")
            return 1


def _login(args: argparse.Namespace) -> int:
    access_token = _login_token(args)
    if access_token is None:
        print("auth_login_error=token_required")
        return 1
    payload = login_model_provider(
        {
            "provider": args.provider,
            "auth_mode": args.auth_mode,
            "access_token": access_token,
            "refresh_token": args.refresh_token,
            "expires_in_s": 3600,
        }
    )
    print("auth_login_ok=true")
    print(f"provider={payload['provider']}")
    print(f"provider_credential_store={payload['provider_credential_store']}")
    return 0


def _status() -> int:
    payload = model_auth_status_payload()
    providers = as_sequence(payload["providers"], "providers")
    print("auth_status_ok=true")
    if not providers:
        print("provider=none")
        return 0
    for provider in providers:
        item = as_mapping(provider, "provider_status")
        print(
            f"provider={item['provider']} logged_in={item['logged_in']} "
            f"auth_mode={item.get('auth_mode', 'oauth')} expires={item['expires']}"
        )
    return 0


def _smoke(args: argparse.Namespace) -> int:
    payload = run_model_gateway_smoke_from_controller(
        {
            "model_provider": {
                "provider": args.provider,
                "model": args.model,
                "reasoning_effort": "high",
                "base_url": args.base_url,
                "auth_mode": args.auth_mode,
                "api_key_env": args.api_key_env,
            },
            "request": {
                "request_id": "asa-auth-smoke",
                "ion_species": "Ar",
                "target_material": "Si",
            },
        }
    )
    print(f"auth_smoke_ok={str(payload['ok']).lower()}")
    print(f"gateway_request_id={payload['gateway_request_id'] or ''}")
    return 0 if payload["ok"] is True else 1


def _apply_store(args: argparse.Namespace) -> None:
    store = getattr(args, "credential_store", None)
    if isinstance(store, str) and store:
        os.environ[CREDENTIAL_STORE_ENV] = store


def _login_token(args: argparse.Namespace) -> str | None:
    if isinstance(args.access_token, str) and args.access_token:
        return args.access_token
    if isinstance(args.api_key, str) and args.api_key:
        return args.api_key
    if isinstance(args.api_key_env, str) and args.api_key_env:
        value = os.environ.get(args.api_key_env)
        if value:
            return value
    return None
