#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_Infer_Model.py

4D MDN inference module:
    x = (v_xin, v_yin, v_zin, E_in)
    y = (v_xout, v_yout, v_zout, E_out)

Loads artifacts produced by 02_01_Train_Model.py (train_mdn_4d.py):
    - x_scaler.pkl
    - y_scaler.pkl
    - checkpoints/best_mdn_model.pt

Examples:
  # single input (stochastic sample)
  python 03_Infer_Model.py --vx 10 --vy 5 --vz -50 --E 1200

  # multiple samples for one input
  python 03_Infer_Model.py --vx 10 --vy 5 --vz -50 --E 1200 --n_samples 5

  # deterministic output (mixture mean)
  python 03_Infer_Model.py --vx 10 --vy 5 --vz -50 --E 1200 --mode mean

  # batch from CSV (expects v_xin,v_yin,v_zin,E_in; saves v_xout,v_yout,v_zout,E_out)
  python 03_Infer_Model.py --input_csv new_inputs.csv --output_csv pred_samples.csv --n_samples 5

Notes:
- mode=sample returns stochastic samples from the mixture distribution.
- mode=mean returns the mixture mean (deterministic).
- hidden_dim/num_gaussians/dropout MUST match training hyperparameters.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import joblib

from mdn_model import MultiOutputMDN


def _load_best_from_summary(summary_path: Path):
    """Return (best_config_dict, best_trial_index) from random_search_summary.json."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    best = data.get("best", None)
    if not best or "config" not in best:
        raise ValueError(f"Invalid summary format: missing best/config in {summary_path}")
    cfg = best["config"]
    trial = best.get("trial", None)
    return cfg, trial


def _maybe_apply_auto_best(args):
    """
    If auto-best is enabled and summary exists, override model hyperparams and (optionally) art_dir
    to point at the best trial directory (rs_runs/trial_XX) when artifacts exist there.
    """
    # explicit disable
    if args.no_auto_best:
        return args

    summary_path = Path(args.summary_path)
    if not summary_path.is_file():
        # If user explicitly asked for auto_best, be loud; else silently skip
        if args.auto_best:
            raise FileNotFoundError(f"auto_best requested but summary not found: {summary_path}")
        return args

    cfg, trial = _load_best_from_summary(summary_path)

    # Override hyperparams from best config
    if "hidden_dim" in cfg:
        args.hidden_dim = int(cfg["hidden_dim"])
    if "num_gaussians" in cfg:
        args.num_gaussians = int(cfg["num_gaussians"])
    if "dropout" in cfg:
        args.dropout = float(cfg["dropout"])

    # If artifacts are in rs_runs/trial_XX and current art_dir doesn't contain expected files,
    # auto-switch art_dir to best trial folder.
    rs_base = Path(args.rs_base)
    if trial is not None:
        best_dir = rs_base / f"trial_{int(trial):02d}"
        # expected files
        need = [best_dir / "x_scaler.pkl", best_dir / "y_scaler.pkl", best_dir / "checkpoints" / "best_mdn_model.pt"]
        if all(p.is_file() for p in need):
            # Only override art_dir if user left default "." OR if current art_dir is missing artifacts
            cur = Path(args.art_dir)
            cur_need = [cur / "x_scaler.pkl", cur / "y_scaler.pkl", cur / "checkpoints" / "best_mdn_model.pt"]
            if str(args.art_dir) == "." or not all(p.is_file() for p in cur_need):
                args.art_dir = str(best_dir)

    return args


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@torch.no_grad()
def mdn_sample(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    """
    Sample y from MDN outputs.

    pi:    [B,K]
    mu:    [B,K,D]
    sigma: [B,K,D]
    returns: [B,n_samples,D]
    """
    B, K = pi.shape
    D = mu.shape[-1]

    cat = torch.distributions.Categorical(probs=pi)
    comps = cat.sample((n_samples,))   # [n_samples,B]
    comps = comps.transpose(0, 1)      # [B,n_samples]

    mu_g = torch.gather(mu, 1, comps.unsqueeze(-1).expand(B, n_samples, D))
    sg_g = torch.gather(sigma, 1, comps.unsqueeze(-1).expand(B, n_samples, D))

    eps = torch.randn_like(mu_g)
    return mu_g + sg_g * eps


@torch.no_grad()
def mdn_mean(pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Mixture mean E[y|x] = sum_k pi_k * mu_k. returns [B,D]"""
    return (pi.unsqueeze(-1) * mu).sum(dim=1)


def load_artifacts(art_dir: Path, hidden_dim: int, num_gaussians: int, dropout: float, device: torch.device):
    x_scaler_path = art_dir / "x_scaler.pkl"
    y_scaler_path = art_dir / "y_scaler.pkl"
    ckpt_path = art_dir / "checkpoints" / "best_mdn_model.pt"

    for p in (x_scaler_path, y_scaler_path, ckpt_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing artifact: {p}")

    x_scaler = joblib.load(str(x_scaler_path))
    y_scaler = joblib.load(str(y_scaler_path))

    model = MultiOutputMDN(
        input_dim=4,
        hidden_dim=hidden_dim,
        output_dim=4,
        num_gaussians=num_gaussians,
        dropout_rate=dropout,
    ).to(device)

    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, x_scaler, y_scaler, ckpt_path


def infer_array(model, x_scaler, y_scaler, X: np.ndarray, device: torch.device, n_samples: int = 1, mode: str = "sample") -> np.ndarray:
    """
    X: [N,4] raw
    returns:
      sample: [N,n_samples,4] raw
      mean:   [N,1,4] raw
    """
    Xs = x_scaler.transform(X).astype(np.float32)
    xb = torch.from_numpy(Xs).to(device)

    pi, mu, sigma = model(xb)

    if mode == "mean":
        ym = mdn_mean(pi, mu).cpu().numpy()
        y_raw = y_scaler.inverse_transform(ym)
        return y_raw[:, None, :]

    ys = mdn_sample(pi, mu, sigma, n_samples=n_samples).cpu().numpy()  # [N,S,4] scaled
    N, S, D = ys.shape
    ys2 = ys.reshape(N * S, D)
    ys_raw = y_scaler.inverse_transform(ys2).reshape(N, S, D)
    return ys_raw


def main():
    ap = argparse.ArgumentParser(description="Infer with trained 4D MDN model (sample or mean).")
    ap.add_argument("--art_dir", default=".", help="Directory containing x_scaler.pkl, y_scaler.pkl, checkpoints/")
    ap.add_argument("--hidden_dim", type=int, default=64, help="Must match training hidden_dim")
    ap.add_argument("--num_gaussians", type=int, default=3, help="Must match training num_gaussians")
    ap.add_argument("--dropout", type=float, default=0.30, help="Must match training dropout_rate")

    ap.add_argument("--auto_best", action="store_true", help="Auto-load best config/artifacts from random_search_summary.json if present (recommended)")
    ap.add_argument("--no_auto_best", action="store_true", help="Disable auto-best behavior")
    ap.add_argument("--summary_path", default="random_search_summary.json", help="Path to random search summary JSON (default: random_search_summary.json)")
    ap.add_argument("--rs_base", default="rs_runs", help="Base directory for random search trials (default: rs_runs)")

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--mode", default="sample", choices=["sample", "mean"], help="sample: stochastic, mean: deterministic")

    ap.add_argument("--vx", type=float, default=None)
    ap.add_argument("--vy", type=float, default=None)
    ap.add_argument("--vz", type=float, default=None)
    ap.add_argument("--E", type=float, default=None)

    ap.add_argument("--input_csv", default=None, help="CSV with columns v_xin,v_yin,v_zin,E_in (or vx,vy,vz,E)")
    ap.add_argument("--output_csv", default=None, help="Where to save predictions (optional)")
    ap.add_argument("--n_samples", type=int, default=1, help="Number of stochastic samples per input (mode=sample)")

    args = ap.parse_args()

    args = _maybe_apply_auto_best(args)

    device = _device_from_arg(args.device)
    art_dir = Path(args.art_dir)

    model, x_scaler, y_scaler, ckpt_path = load_artifacts(
        art_dir=art_dir,
        hidden_dim=args.hidden_dim,
        num_gaussians=args.num_gaussians,
        dropout=args.dropout,
        device=device,
    )

    if args.input_csv is None:
        if None in (args.vx, args.vy, args.vz, args.E):
            raise ValueError("Provide either --input_csv OR all of --vx --vy --vz --E")
        X = np.array([[args.vx, args.vy, args.vz, args.E]], dtype=np.float32)
        Y = infer_array(model, x_scaler, y_scaler, X, device, n_samples=args.n_samples, mode=args.mode)

        print("✅ Loaded:", ckpt_path)
        print("Input x = [vx, vy, vz, E_in] =", X[0].tolist())
        if args.mode == "mean":
            print("Output y(mean) = [vx_out, vy_out, vz_out, E_out] =", Y[0, 0].tolist())
        else:
            for s in range(Y.shape[1]):
                print(f"Output y(sample {s+1}) =", Y[0, s].tolist())
        return

    df = pd.read_csv(args.input_csv)

    col_sets = [
        ["v_xin", "v_yin", "v_zin", "E_in"],
        ["vx", "vy", "vz", "E"],
        ["vx", "vy", "vz", "E_in"],
        ["v_x", "v_y", "v_z", "E_in"],
    ]
    cols = None
    for cs in col_sets:
        if all(c in df.columns for c in cs):
            cols = cs
            break
    if cols is None:
        raise KeyError(f"Input CSV missing required columns. Tried {col_sets}. Available: {list(df.columns)}")

    X = df[cols].to_numpy(dtype=np.float32)
    Y = infer_array(model, x_scaler, y_scaler, X, device, n_samples=args.n_samples, mode=args.mode)

    out_rows = []
    for i in range(Y.shape[0]):
        for s in range(Y.shape[1]):
            out_rows.append({
                "row": i,
                "sample": s,
                "v_xout": float(Y[i, s, 0]),
                "v_yout": float(Y[i, s, 1]),
                "v_zout": float(Y[i, s, 2]),
                "E_out":  float(Y[i, s, 3]),
            })
    df_out = pd.DataFrame(out_rows)

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.output_csv, index=False)
        print(f"✅ Saved predictions: {args.output_csv}")
    else:
        print(df_out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
