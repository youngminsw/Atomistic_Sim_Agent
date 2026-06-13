# MCP_server.py
# FastMCP-based MCP server wrapping your ML pipeline (01 -> 02 -> 03) + KMC (04).
# - Tools:
#   dump2csv, train_random, infer_best, run_pipeline, list_artifacts, run_kmc, run_pipeline_kmc
#
# Run:
#   python MCP_server.py   (stdio transport; used by Gemini_MCP.py)

from __future__ import annotations

import os
import sys
import json
import shlex
import subprocess
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP  # pip install fastmcp

mcp = FastMCP("ML Pipeline MCP (dump→train→infer→kmc)")


# -------------------------
# Helpers
# -------------------------
def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _must_exist(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"{what} is a directory, expected file: {p}")


def _run(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run command, capture output for debugging in MCP clients."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
    )
    return {
        "cmd": " ".join(shlex.quote(x) for x in cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
    }


def _default_python() -> str:
    return sys.executable


def _clamp_int(x: int, lo: int, hi: int, name: str) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _load_module_from_file(module_name: str, file_path: Path):
    """Load a python module from an arbitrary .py path (works even if filename starts with digits)."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# -------------------------
# Implementation functions (CALLABLE)
# -------------------------
def dump2csv_impl(workdir: str = ".") -> Dict[str, Any]:
    """
    Run 01_Dump2csv.py in workdir to generate mdn_input.csv and mdn_output.csv.

    Expects:
      - incident.dump, reflected.dump exist in workdir
      - 01_Dump2csv.py exists in workdir

    Returns paths to generated CSVs and logs.
    """
    wd = _as_path(workdir).resolve()
    _must_exist(wd / "incident.dump", "incident.dump")
    _must_exist(wd / "reflected.dump", "reflected.dump")
    _must_exist(wd / "01_Dump2csv.py", "01_Dump2csv.py")

    logs = _run([_default_python(), "01_Dump2csv.py"], cwd=wd)
    if logs["returncode"] != 0:
        raise RuntimeError(f"dump2csv failed:\n{logs}")

    x_csv = wd / "mdn_input.csv"
    y_csv = wd / "mdn_output.csv"
    _must_exist(x_csv, "mdn_input.csv")
    _must_exist(y_csv, "mdn_output.csv")

    return {
        "workdir": str(wd),
        "mdn_input_csv": str(x_csv),
        "mdn_output_csv": str(y_csv),
        "logs": logs,
    }


def train_random_impl(
    workdir: str = ".",
    n_trials: int = 10,
    seed: Optional[int] = None,
    out_base: str = "rs_runs",
    summary_path: str = "random_search_summary.json",
    space_json: Optional[str] = None,
    space_json_str: Optional[str] = None,
    overwrite: bool = False,
    python_exe: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run 02_02_Random_Train_Model.py in workdir to perform random search training.

    Expects:
      - mdn_input.csv, mdn_output.csv exist in workdir
      - 02_02_Random_Train_Model.py exists in workdir
      - 02_01_Train_Model.py exists in workdir (called by 02_02)

    Produces:
      - <out_base>/ (trial_* dirs)
      - <summary_path>
    """
    wd = _as_path(workdir).resolve()

    _must_exist(wd / "mdn_input.csv", "mdn_input.csv")
    _must_exist(wd / "mdn_output.csv", "mdn_output.csv")
    _must_exist(wd / "02_02_Random_Train_Model.py", "02_02_Random_Train_Model.py")
    _must_exist(wd / "02_01_Train_Model.py", "02_01_Train_Model.py")

    # Guardrails: 너무 큰 trial 방지
    n_trials = int(n_trials)
    n_trials = _clamp_int(n_trials, 1, 200, "n_trials")

    py = python_exe or _default_python()

    cmd = [
        py,
        "02_02_Random_Train_Model.py",
        "--x_csv",
        "mdn_input.csv",
        "--y_csv",
        "mdn_output.csv",
        "--train_script",
        "02_01_Train_Model.py",
        "--n_trials",
        str(n_trials),
        "--out_base",
        out_base,
        "--summary_path",
        summary_path,
        "--python",
        py,
    ]

    if seed is not None:
        cmd += ["--seed", str(int(seed))]

    if space_json is not None:
        cmd += ["--space_json", space_json]

    if space_json_str is not None:
        cmd += ["--space_json_str", space_json_str]

    if overwrite:
        cmd += ["--overwrite"]

    logs = _run(cmd, cwd=wd)
    if logs["returncode"] != 0:
        raise RuntimeError(f"train_random failed:\n{logs}")

    summary = wd / summary_path
    _must_exist(summary, f"summary ({summary_path})")

    best = None
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
        best = data.get("best", None)
    except Exception:
        best = None

    return {
        "workdir": str(wd),
        "summary_path": str(summary),
        "best": best,
        "logs": logs,
    }


def infer_best_impl(
    workdir: str = ".",
    infer_samples: int = 5,
    infer_in: str = "mdn_input.csv",
    infer_out: str = "pred_samples.csv",
    summary_path: str = "random_search_summary.json",
    rs_base: str = "rs_runs",
) -> Dict[str, Any]:
    """
    Run Infer_Model.py with --auto_best using random_search_summary.json.

    Expects:
      - Infer_Model.py exists
      - summary exists (from train_random)
      - input CSV exists (default: mdn_input.csv)
    """
    wd = _as_path(workdir).resolve()
    _must_exist(wd / "Infer_Model.py", "Infer_Model.py")
    _must_exist(wd / summary_path, f"summary ({summary_path})")
    _must_exist(wd / infer_in, f"input CSV ({infer_in})")

    infer_samples = int(infer_samples)
    infer_samples = _clamp_int(infer_samples, 1, 1000, "infer_samples")

    cmd = [
        _default_python(),
        "Infer_Model.py",
        "--input_csv",
        infer_in,
        "--output_csv",
        infer_out,
        "--n_samples",
        str(infer_samples),
        "--auto_best",
        "--summary_path",
        summary_path,
        "--rs_base",
        rs_base,
    ]
    logs = _run(cmd, cwd=wd)
    if logs["returncode"] != 0:
        raise RuntimeError(f"infer_best failed:\n{logs}")

    outp = wd / infer_out
    _must_exist(outp, f"inference output ({infer_out})")

    return {
        "workdir": str(wd),
        "pred_csv": str(outp),
        "logs": logs,
    }


def list_artifacts_impl(workdir: str = ".") -> Dict[str, Any]:
    wd = _as_path(workdir).resolve()
    keys = [
        "mdn_input.csv",
        "mdn_output.csv",
        "random_search_summary.json",
        "pred_samples.csv",
    ]
    present = {k: str((wd / k)) for k in keys if (wd / k).is_file()}

    rs_runs = wd / "rs_runs"
    trials: List[str] = []
    if rs_runs.is_dir():
        trials = sorted([p.name for p in rs_runs.glob("trial_*") if p.is_dir()])[:200]

    return {"workdir": str(wd), "present": present, "rs_trials": trials}


def run_kmc_impl(
    workdir: str = ".",
    energy_ev: float = 100.0,
    angle_deg: float = 0.0,
    num_ions: int = 10000,
    run_id: str = "kmc_run",
    out_png: Optional[str] = None,
    seed: Optional[int] = None,
    # surrogate controls
    use_surrogate: bool = True,
    auto_best: bool = True,
    device: str = "auto",
    mode: str = "sample",  # "sample" or "mean"
    # allow explicit paths
    kmc_script: str = "04_KMC_tool.py",
) -> Dict[str, Any]:
    """
    Run KMC trench simulation using your 04_KMC_tool.py.

    - If use_surrogate=True: loads MDN surrogate (auto_best) and runs KMC+heatmap output.
    - If use_surrogate=False: still runs via 04_KMC_tool.py runner if available (it always builds surrogate),
      so for now we enforce use_surrogate=True unless you later add a pure-KMC runner in 04_KMC_tool.py.

    Returns:
      dict with heatmap path + bin energies.
    """
    wd = _as_path(workdir).resolve()

    # Guardrails: 너무 큰 ion 수 방지
    num_ions = _clamp_int(int(num_ions), 1, 2_000_000, "num_ions")

    kmc_path = (wd / kmc_script).resolve()
    _must_exist(kmc_path, kmc_script)

    if not use_surrogate:
        # 현재 04_KMC_tool.py는 run_kmc_with_mdn()이 기본이라 surrogate를 항상 씀.
        # 나중에 pure KMC runner를 추가하면 여기서 분기 가능.
        raise ValueError(
            "use_surrogate=False는 아직 지원하지 않게 막아뒀어. "
            "현재 04_KMC_tool.py는 run_kmc_with_mdn() 구조라 surrogate가 필수야."
        )

    # dynamic import (works even though filename starts with digits)
    kmc_mod = _load_module_from_file("kmc_tool_dynamic", kmc_path)

    if not hasattr(kmc_mod, "run_kmc_with_mdn"):
        raise AttributeError(f"{kmc_script} must define run_kmc_with_mdn().")

    run_fn = getattr(kmc_mod, "run_kmc_with_mdn")

    result = run_fn(
        workdir=wd,
        energy_ev=float(energy_ev),
        angle_deg=float(angle_deg),
        num_ions=int(num_ions),
        run_id=str(run_id),
        out_png=out_png,
        seed=seed,
        auto_best=bool(auto_best),
        device=str(device),
        mode=str(mode),
    )

    # ensure JSON-serializable
    return json.loads(json.dumps(result))


# -------------------------
# Tools (WRAPPERS ONLY)
# -------------------------
@mcp.tool
def dump2csv(workdir: str = ".") -> Dict[str, Any]:
    return dump2csv_impl(workdir)


@mcp.tool
def train_random(
    workdir: str = ".",
    n_trials: int = 10,
    seed: Optional[int] = None,
    out_base: str = "rs_runs",
    summary_path: str = "random_search_summary.json",
    space_json: Optional[str] = None,
    space_json_str: Optional[str] = None,
    overwrite: bool = False,
    python_exe: Optional[str] = None,
) -> Dict[str, Any]:
    return train_random_impl(
        workdir=workdir,
        n_trials=n_trials,
        seed=seed,
        out_base=out_base,
        summary_path=summary_path,
        space_json=space_json,
        space_json_str=space_json_str,
        overwrite=overwrite,
        python_exe=python_exe,
    )


@mcp.tool
def infer_best(
    workdir: str = ".",
    infer_samples: int = 5,
    infer_in: str = "mdn_input.csv",
    infer_out: str = "pred_samples.csv",
    summary_path: str = "random_search_summary.json",
    rs_base: str = "rs_runs",
) -> Dict[str, Any]:
    return infer_best_impl(
        workdir=workdir,
        infer_samples=infer_samples,
        infer_in=infer_in,
        infer_out=infer_out,
        summary_path=summary_path,
        rs_base=rs_base,
    )


@mcp.tool
def run_pipeline(
    workdir: str = ".",
    # train controls
    n_trials: int = 10,
    seed: Optional[int] = None,
    out_base: str = "rs_runs",
    summary_path: str = "random_search_summary.json",
    space_json: Optional[str] = None,
    space_json_str: Optional[str] = None,
    overwrite: bool = False,
    python_exe: Optional[str] = None,
    # infer controls
    infer_samples: int = 5,
    infer_in: str = "mdn_input.csv",
    infer_out: str = "pred_samples.csv",
    rs_base: str = "rs_runs",
    # skip flags
    skip_train: bool = False,
    skip_infer: bool = False,
) -> Dict[str, Any]:
    """
    All-in-one pipeline:
      1) dump2csv
      2) train_random (optional)
      3) infer_best  (optional)

    NOTE:
      내부에서는 tool 객체를 직접 호출하지 않고, *_impl() callable 함수들을 호출한다.
    """
    wd = _as_path(workdir).resolve()

    step1 = dump2csv_impl(str(wd))

    step2 = None
    if not skip_train:
        step2 = train_random_impl(
            workdir=str(wd),
            n_trials=n_trials,
            seed=seed,
            out_base=out_base,
            summary_path=summary_path,
            space_json=space_json,
            space_json_str=space_json_str,
            overwrite=overwrite,
            python_exe=python_exe,
        )

    step3 = None
    if not skip_infer:
        step3 = infer_best_impl(
            workdir=str(wd),
            infer_samples=infer_samples,
            infer_in=infer_in,
            infer_out=infer_out,
            summary_path=summary_path,
            rs_base=rs_base,
        )

    return {
        "workdir": str(wd),
        "dump2csv": step1,
        "train_random": step2,
        "infer_best": step3,
    }


@mcp.tool
def list_artifacts(workdir: str = ".") -> Dict[str, Any]:
    return list_artifacts_impl(workdir)


# ---- NEW: KMC tool ----
@mcp.tool
def run_kmc(
    workdir: str = ".",
    energy_ev: float = 100.0,
    angle_deg: float = 0.0,
    num_ions: int = 10000,
    run_id: str = "kmc_run",
    out_png: Optional[str] = None,
    seed: Optional[int] = None,
    use_surrogate: bool = True,
    auto_best: bool = True,
    device: str = "auto",
    mode: str = "sample",
    kmc_script: str = "04_KMC_tool.py",
) -> Dict[str, Any]:
    return run_kmc_impl(
        workdir=workdir,
        energy_ev=energy_ev,
        angle_deg=angle_deg,
        num_ions=num_ions,
        run_id=run_id,
        out_png=out_png,
        seed=seed,
        use_surrogate=use_surrogate,
        auto_best=auto_best,
        device=device,
        mode=mode,
        kmc_script=kmc_script,
    )


# ---- Optional: end-to-end including KMC ----
@mcp.tool
def run_pipeline_kmc(
    workdir: str = ".",
    # pipeline (dump/train/infer) controls
    n_trials: int = 10,
    seed: Optional[int] = None,
    out_base: str = "rs_runs",
    summary_path: str = "random_search_summary.json",
    space_json: Optional[str] = None,
    space_json_str: Optional[str] = None,
    overwrite: bool = False,
    python_exe: Optional[str] = None,
    infer_samples: int = 5,
    infer_in: str = "mdn_input.csv",
    infer_out: str = "pred_samples.csv",
    rs_base: str = "rs_runs",
    skip_train: bool = False,
    skip_infer: bool = False,
    # KMC controls
    energy_ev: float = 100.0,
    angle_deg: float = 0.0,
    num_ions: int = 10000,
    run_id: str = "kmc_run",
    out_png: Optional[str] = None,
    kmc_seed: Optional[int] = None,
    auto_best: bool = True,
    device: str = "auto",
    mode: str = "sample",
    kmc_script: str = "04_KMC_tool.py",
) -> Dict[str, Any]:
    """
    Full pipeline:
      1) dump2csv
      2) train_random (optional)
      3) infer_best  (optional; but train summary is needed for auto_best)
      4) run_kmc (uses auto_best model artifacts)
    """
    wd = _as_path(workdir).resolve()

    pipe = run_pipeline(
        workdir=str(wd),
        n_trials=n_trials,
        seed=seed,
        out_base=out_base,
        summary_path=summary_path,
        space_json=space_json,
        space_json_str=space_json_str,
        overwrite=overwrite,
        python_exe=python_exe,
        infer_samples=infer_samples,
        infer_in=infer_in,
        infer_out=infer_out,
        rs_base=rs_base,
        skip_train=skip_train,
        skip_infer=skip_infer,
    )

    kmc = run_kmc_impl(
        workdir=str(wd),
        energy_ev=energy_ev,
        angle_deg=angle_deg,
        num_ions=num_ions,
        run_id=run_id,
        out_png=out_png,
        seed=kmc_seed,
        use_surrogate=True,
        auto_best=auto_best,
        device=device,
        mode=mode,
        kmc_script=kmc_script,
    )

    return {"workdir": str(wd), "pipeline": pipe, "kmc": kmc}


if __name__ == "__main__":
    mcp.run(transport="stdio")
