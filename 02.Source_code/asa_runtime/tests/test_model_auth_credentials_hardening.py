from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_oauth_credentials_are_written_by_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a configured credential store and spies for atomic filesystem calls.
    from sim_agent.ui import model_auth

    store = tmp_path / "credentials.json"
    monkeypatch.setenv(model_auth.PROVIDER_CREDENTIAL_STORE_ENV, str(store))
    events: list[tuple[str, Path | int]] = []
    original_replace = model_auth.os.replace
    original_chmod = model_auth.os.chmod

    def spy_chmod(
        path: str | os.PathLike[str],
        mode: int,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        events.append(("chmod", Path(path)))
        original_chmod(path, mode, follow_symlinks=follow_symlinks)

    def spy_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        events.append(("replace", Path(src)))
        assert Path(src).parent == store.parent
        assert Path(src) != store
        assert Path(dst) == store
        assert Path(src).stat().st_mode & 0o777 == 0o600
        original_replace(src, dst)

    monkeypatch.setattr(model_auth.os, "chmod", spy_chmod)
    monkeypatch.setattr(model_auth.os, "replace", spy_replace)

    # When: OAuth credentials are stored.
    model_auth.login_model_provider(
        {
            "provider": "openai-codex",
            "access_token": "atomic-access",
            "refresh_token": "atomic-refresh",
        }
    )

    # Then: a secure temp file is chmodded before atomic replacement and final mode is 0600.
    assert [name for name, _path in events] == ["chmod", "replace"]
    assert store.stat().st_mode & 0o777 == 0o600
    assert json.loads(store.read_text(encoding="utf-8"))["openai-codex"]["credentials"]["access"] == "atomic-access"


def test_partial_credential_write_leaves_previous_store_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an existing credential store and an atomic replace failure.
    from sim_agent.ui import model_auth

    store = tmp_path / "credentials.json"
    store.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "provider": "openai-codex",
                    "credentials": {"access": "old-access", "expires": 4_102_444_800_000},
                    "updatedAtMs": 1,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(store, 0o600)
    monkeypatch.setenv(model_auth.PROVIDER_CREDENTIAL_STORE_ENV, str(store))

    def fail_replace(_src: str | os.PathLike[str], _dst: str | os.PathLike[str]) -> None:
        raise OSError("simulated replace interruption")

    monkeypatch.setattr(model_auth.os, "replace", fail_replace)

    # When: the replace step is interrupted.
    with pytest.raises(OSError, match="simulated replace interruption"):
        model_auth.login_model_provider(
            {
                "provider": "openai-codex",
                "access_token": "new-access",
                "refresh_token": "new-refresh",
            }
        )

    # Then: the previous store remains readable and no temp secret is left behind.
    current = json.loads(store.read_text(encoding="utf-8"))
    assert current["openai-codex"]["credentials"]["access"] == "old-access"
    assert sorted(path.name for path in store.parent.iterdir()) == ["credentials.json"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink unsupported")
def test_credential_store_symlink_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: the configured credential store path is a symlink to another file.
    from sim_agent.ui import model_auth

    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    store = tmp_path / "credentials.json"
    os.symlink(target, store)
    monkeypatch.setenv(model_auth.PROVIDER_CREDENTIAL_STORE_ENV, str(store))

    # When / Then: login refuses to follow the symlink and leaves the target untouched.
    with pytest.raises(model_auth.ModelAuthError) as exc_info:
        model_auth.login_model_provider(
            {
                "provider": "openai-codex",
                "access_token": "symlink-access",
            }
        )
    assert exc_info.value.code == "provider_credential_store_symlink_refused"
    assert target.read_text(encoding="utf-8") == "{}\n"


def test_corrupt_credential_store_raises_typed_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a configured credential store containing invalid JSON.
    from sim_agent.ui import model_auth

    store = tmp_path / "credentials.json"
    store.write_text("{not-json}\n", encoding="utf-8")
    monkeypatch.setenv(model_auth.PROVIDER_CREDENTIAL_STORE_ENV, str(store))

    # When / Then: status refuses the corrupt store with a typed error code.
    with pytest.raises(model_auth.ModelAuthError) as exc_info:
        model_auth.model_auth_status_payload()
    assert exc_info.value.code == "provider_credential_store_corrupt"
