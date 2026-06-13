from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import sys
from pathlib import Path
from threading import Thread


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import JsonMap, as_mapping


@dataclass(frozen=True, slots=True)
class StaticServer:
    base_url: str
    server: ThreadingHTTPServer
    thread: Thread


def test_acquire_potential_candidate_fetches_http_sidecar_and_validates(
    tmp_path: Path,
) -> None:
    from sim_agent.materials import PotentialAcquisitionRequest, acquire_potential_candidate

    # Given
    _write_valid_tersoff_fixture(tmp_path)
    fake_lammps = _write_fake_lammps(tmp_path, success=True)
    with _serve_static(tmp_path) as static:
        source_url = f"{static.base_url}/potentials/Si.tersoff"
        metadata_url = f"{static.base_url}/potentials/Si.tersoff.metadata.json"

        # When
        report = acquire_potential_candidate(
            PotentialAcquisitionRequest(
                source_url=source_url,
                metadata_url=metadata_url,
                material_id="Si",
                ion_species="Ar",
                required_elements=("Si", "Ar"),
                sandbox_command=(sys.executable, str(fake_lammps)),
                sandbox_work_dir=tmp_path / "sandbox-ok",
            ),
            repo_root=SOURCE_ROOT,
        )

    # Then
    assert report.ok is True
    assert report.validation.ok is True
    assert report.candidate_payload["source_url"] == source_url
    assert report.candidate_payload["element_symbols"] == ["Si", "Ar"]
    assert len(str(report.candidate_payload["source_sha256"])) == 64
    assert "potential_file_fetched" in report.payload["acquisition_evidence"]
    assert "format_parse_passed" in report.payload["acquisition_evidence"]
    assert report.candidate_payload["syntax_smoke_passed"] is True
    assert report.payload["gate_status"] == "potential_candidate_accepted"


def test_acquire_potential_candidate_rejects_when_sandbox_smoke_is_missing(
    tmp_path: Path,
) -> None:
    from sim_agent.materials import PotentialAcquisitionRequest, acquire_potential_candidate

    # Given
    _write_valid_tersoff_fixture(tmp_path)

    # When
    report = acquire_potential_candidate(
        PotentialAcquisitionRequest(
            source_url=(tmp_path / "potentials" / "Si.tersoff").as_uri(),
            metadata_url=(tmp_path / "potentials" / "Si.tersoff.metadata.json").as_uri(),
            material_id="Si",
            ion_species="Ar",
            required_elements=("Si", "Ar"),
        ),
        repo_root=SOURCE_ROOT,
    )

    # Then
    assert report.ok is False
    assert report.payload["gate_status"] == "potential_candidate_rejected"
    assert report.candidate_payload["syntax_smoke_passed"] is False
    assert "syntax_smoke_failed" in report.validation.payload["errors"]
    assert report.payload["sandbox_smoke"]["smoke_status"] == "sandbox_smoke_not_run"


def test_acquire_potential_candidate_rejects_unpublished_reaxff_fixture(
    tmp_path: Path,
) -> None:
    from sim_agent.materials import PotentialAcquisitionRequest, acquire_potential_candidate

    # Given
    potential_path = tmp_path / "ffield.reax"
    metadata_path = tmp_path / "ffield.reax.metadata.json"
    potential_path.write_text(
        "# elements: Si O Ar\nReactive force field fixture\n",
        encoding="utf-8",
    )
    metadata_path.write_text(json.dumps(_bad_reaxff_metadata()), encoding="utf-8")

    # When
    report = acquire_potential_candidate(
        PotentialAcquisitionRequest(
            source_url=potential_path.as_uri(),
            metadata_url=metadata_path.as_uri(),
            material_id="SiO2",
            ion_species="Ar",
            required_elements=("Si", "O", "Ar"),
        ),
        repo_root=SOURCE_ROOT,
    )

    # Then
    assert report.ok is False
    assert report.payload["gate_status"] == "potential_candidate_rejected"
    assert "reaxff_real_units_required" in report.validation.payload["errors"]
    assert "reaxff_publication_required" in report.validation.payload["errors"]
    assert "reaxff_fitted_system_required" in report.validation.payload["errors"]


def test_acquire_potential_candidate_cli_writes_validated_report(tmp_path: Path) -> None:
    # Given
    _write_valid_tersoff_fixture(tmp_path)
    fake_lammps = _write_fake_lammps(tmp_path, success=True)
    report_path = tmp_path / "acquisition_report.json"

    # When
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "acquire_potential_candidate.py"),
            "--source-url",
            (tmp_path / "potentials" / "Si.tersoff").as_uri(),
            "--metadata-url",
            (tmp_path / "potentials" / "Si.tersoff.metadata.json").as_uri(),
            "--material",
            "Si",
            "--ion",
            "Ar",
            "--required-elements",
            "Si,Ar",
            "--lammps-command",
            f"{sys.executable} {fake_lammps}",
            "--out",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then
    payload = as_mapping(json.loads(report_path.read_text(encoding="utf-8")), "report")
    candidate = as_mapping(payload["candidate"], "candidate")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "potential_acquisition_ok=true" in result.stdout
    assert "gate_status=potential_candidate_accepted" in result.stdout
    assert len(str(candidate["source_sha256"])) == 64
    assert candidate["syntax_smoke_passed"] is True


def test_run_potential_sandbox_smoke_rejects_failed_lammps(tmp_path: Path) -> None:
    from sim_agent.materials import PotentialSandboxSmokeRequest, run_potential_sandbox_smoke

    # Given
    candidate = _valid_tersoff_metadata() | {
        "source_url": (tmp_path / "Si.tersoff").as_uri(),
        "element_symbols": ["Si", "Ar"],
    }
    potential_text = "# elements: Si Ar\nSi Si Si 3.0\n"
    fake_lammps = _write_fake_lammps(tmp_path, success=False)

    # When
    report = run_potential_sandbox_smoke(
        PotentialSandboxSmokeRequest(
            candidate_payload=candidate,
            potential_text=potential_text,
            work_dir=tmp_path / "sandbox-failed",
            lammps_command=(sys.executable, str(fake_lammps)),
        )
    )

    # Then
    assert report.ok is False
    assert report.payload["smoke_status"] == "sandbox_smoke_failed"
    assert report.payload["syntax_smoke_passed"] is False
    assert "lammps_return_code=7" in report.payload["errors"]


def _write_valid_tersoff_fixture(root: Path) -> None:
    potential_root = root / "potentials"
    potential_root.mkdir()
    (potential_root / "Si.tersoff").write_text(
        "# elements: Si Ar\nSi Si Si 3.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n",
        encoding="utf-8",
    )
    (potential_root / "Si.tersoff.metadata.json").write_text(
        json.dumps(_valid_tersoff_metadata()),
        encoding="utf-8",
    )


def _write_fake_lammps(root: Path, success: bool) -> Path:
    script = root / ("fake_lammps_ok.py" if success else "fake_lammps_fail.py")
    if success:
        script.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "deck = Path(sys.argv[sys.argv.index('-in') + 1])\n"
            "(deck.parent / 'log.lammps').write_text('Loop time of 0.01 on 1 procs\\n')\n"
            "print('Loop time of 0.01 on 1 procs')\n"
            "raise SystemExit(0)\n",
            encoding="utf-8",
        )
    else:
        script.write_text(
            "import sys\n"
            "print('ERROR: Pair coeff failed', file=sys.stderr)\n"
            "raise SystemExit(7)\n",
            encoding="utf-8",
        )
    return script


def _valid_tersoff_metadata() -> JsonMap:
    return {
        "potential_id": "si-tersoff-zbl-acquired-v001",
        "material_id": "Si",
        "ion_species": "Ar",
        "pair_style": "hybrid/overlay tersoff zbl",
        "potential_name": "Downloaded Si.tersoff + ZBL overlay",
        "provenance_url": "https://github.com/lammps/lammps/blob/develop/potentials/README",
        "license": "repo-test",
        "lammps_unit_style": "metal",
        "atom_type_mapping": ["Si", "Ar"],
        "fitted_system": "Si collision cascade fixture with ZBL overlay for Ar projectile",
        "transferability_scope": "Ar physical sputtering of Si with ZBL high-energy term",
    }


def _bad_reaxff_metadata() -> JsonMap:
    return {
        "potential_id": "bad-reaxff-acquired",
        "material_id": "SiO2",
        "ion_species": "Ar",
        "pair_style": "reaxff",
        "potential_name": "Unvetted ffield.reax",
        "source_url": "will-be-overridden",
        "license": "unknown",
        "lammps_unit_style": "metal",
        "atom_type_mapping": ["Si", "O", "Ar"],
        "transferability_scope": "element match only",
    }


@contextmanager
def _serve_static(root: Path) -> Iterator[StaticServer]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    host, port = server.server_address
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield StaticServer(f"http://{host}:{port}", server, thread)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
