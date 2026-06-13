#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


# =========================
# Default hyperparameter space (fallback)
# =========================
DEFAULT_SPACE: Dict[str, Any] = {
    "num_gaussians": [2, 3, 5],
    "hidden_dim": [64, 128],
    "dropout": [0.2, 0.3],
    "lr": [1e-3, 5e-4],
    "batch": [256, 512],
}


def _load_space(space_json: Optional[str], space_json_str: Optional[str]) -> Dict[str, Any]:
    """
    Load search space from either:
      - JSON file path (space_json)
      - raw JSON string (space_json_str)
    else fallback to DEFAULT_SPACE
    """
    if space_json_str:
        try:
            obj = json.loads(space_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string for --space_json_str: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError("--space_json_str must decode to a JSON object (dict).")
        return obj

    if space_json:
        p = Path(space_json)
        if not p.is_file():
            raise FileNotFoundError(f"--space_json file not found: {p}")
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("--space_json must contain a JSON object (dict).")
        return obj

    return DEFAULT_SPACE


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)


def _sample_config(space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Randomly sample one configuration from the given space.
    The space format is:
      { "param": [choices...], ... }
    """
    cfg = {}
    for k, v in space.items():
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"SPACE['{k}'] must be a non-empty list. Got: {v}")
        cfg[k] = random.choice(v)
    return cfg


def _run_train(
    python_exe: str,
    train_script: str,
    x_csv: str,
    y_csv: str,
    cfg: Dict[str, Any],
    run_dir: Path,
) -> None:
    cmd = [
        python_exe,
        train_script,
        "--x", x_csv,
        "--y", y_csv,
        "--out_dir", str(run_dir),
        "--num_gaussians", str(cfg["num_gaussians"]),
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--dropout", str(cfg["dropout"]),
        "--lr", str(cfg["lr"]),
        "--batch", str(cfg["batch"]),
    ]
    subprocess.run(cmd, check=True)


def _read_val_nll(run_dir: Path) -> float:
    """
    Read best_val_nll from:
      <run_dir>/runs/*_mdn4d/report.json
    (keeps original behavior)
    """
    runs_root = run_dir / "runs"
    runs = list(runs_root.glob("*_mdn4d"))
    if not runs:
        raise RuntimeError(f"No run directory found under: {runs_root}")

    report_path = runs[0] / "report.json"
    if not report_path.is_file():
        raise RuntimeError(f"report.json not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        return float(report["best"]["best_val_nll"])
    except Exception as e:
        raise RuntimeError(f"Could not parse best_val_nll from {report_path}") from e


def main():
    ap = argparse.ArgumentParser(description="Random search wrapper for 02_01_Train_Model.py (MDN 4D)")
    ap.add_argument("--x_csv", default="mdn_input.csv", help="Path to input CSV (default: mdn_input.csv)")
    ap.add_argument("--y_csv", default="mdn_output.csv", help="Path to output CSV (default: mdn_output.csv)")
    ap.add_argument("--train_script", default="02_01_Train_Model.py", help="Training script (default: 02_01_Train_Model.py)")
    ap.add_argument("--n_trials", type=int, default=10, help="Number of random trials (default: 10)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (default: None)")
    ap.add_argument("--out_base", default="rs_runs", help="Base output directory for trials (default: rs_runs)")
    ap.add_argument("--summary_path", default="random_search_summary.json", help="Summary JSON path (default: random_search_summary.json)")
    ap.add_argument("--space_json", default=None, help="Path to JSON file that defines search space (dict)")
    ap.add_argument("--space_json_str", default=None, help="Raw JSON string that defines search space (dict)")
    ap.add_argument("--python", default=os.environ.get("PYTHON", "python"), help="Python executable to use (default: env PYTHON or 'python')")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite trial directories if they exist (default: False)")
    args = ap.parse_args()

    if args.n_trials <= 0:
        raise ValueError("--n_trials must be >= 1")

    # Resolve inputs
    x_csv = Path(args.x_csv)
    y_csv = Path(args.y_csv)
    train_script = Path(args.train_script)
    if not x_csv.is_file():
        raise FileNotFoundError(f"x_csv not found: {x_csv}")
    if not y_csv.is_file():
        raise FileNotFoundError(f"y_csv not found: {y_csv}")
    if not train_script.is_file():
        raise FileNotFoundError(f"train_script not found: {train_script}")

    space = _load_space(args.space_json, args.space_json_str)

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)

    best = None
    history = []

    for i in range(args.n_trials):
        cfg = _sample_config(space)
        run_dir = out_base / f"trial_{i:02d}"

        if run_dir.exists():
            if args.overwrite:
                # remove contents (simple & safe-ish)
                for p in sorted(run_dir.rglob("*"), reverse=True):
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        try:
                            p.rmdir()
                        except OSError:
                            pass
            else:
                raise FileExistsError(
                    f"Trial directory already exists: {run_dir}\n"
                    f"Use --overwrite to reuse it, or change --out_base."
                )

        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Trial {i+1}/{args.n_trials} | config = {cfg}")
        _run_train(
            python_exe=args.python,
            train_script=str(train_script),
            x_csv=str(x_csv),
            y_csv=str(y_csv),
            cfg=cfg,
            run_dir=run_dir,
        )

        val_nll = _read_val_nll(run_dir)
        history.append({"trial": i, "config": cfg, "val_nll": val_nll})

        print(f"   → val NLL = {val_nll:.6f}")

        if best is None or val_nll < best["val_nll"]:
            best = {"trial": i, "config": cfg, "val_nll": val_nll}
            print("   ⭐ New BEST!")

    summary = {"best": best, "all_trials": history}

    summary_path = Path(args.summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n✅ Random search finished")
    print(f"🧾 Summary saved to: {summary_path}")
    print("🏆 BEST config:")
    print(best)


if __name__ == "__main__":
    main()
