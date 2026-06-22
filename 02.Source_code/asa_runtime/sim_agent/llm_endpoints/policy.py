from __future__ import annotations

from os import environ
from typing import Final
from urllib.parse import urlparse, urlunparse


OPENCLAW_ALLOWED_BASE_URLS_ENV: Final = "OPENCLAW_ALLOWED_BASE_URLS"
DEFAULT_OPENCLAW_BASE_URLS: Final[frozenset[str]] = frozenset(
    {
        "https://openclaw.local/v1",
        "https://api.openclaw.ai/v1",
        "https://openclaw.ai/v1",
    }
)


def normalize_openclaw_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname is None
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = parsed.hostname.lower()
    if port is not None:
        netloc = f"{netloc}:{port}"
    normalized_path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), netloc, normalized_path, "", "", ""))


def allowed_openclaw_base_urls() -> frozenset[str]:
    configured = {
        normalize_openclaw_base_url(candidate.strip())
        for candidate in environ.get(OPENCLAW_ALLOWED_BASE_URLS_ENV, "").split(",")
        if candidate.strip()
    }
    return DEFAULT_OPENCLAW_BASE_URLS | {candidate for candidate in configured if candidate}


def is_allowed_openclaw_base_url(base_url: str) -> bool:
    return normalize_openclaw_base_url(base_url) in allowed_openclaw_base_urls()
