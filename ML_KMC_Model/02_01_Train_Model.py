#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_mdn_4d.py

4D MDN training script compatible with:
- mdn_model.py (MultiOutputMDN, mdn_multi_loss)
- mdn_infer.py / total_model.py expecting:
    x_scaler.pkl, y_scaler.pkl, checkpoints/best_mdn_model.pt

Dataset expectation (recommended, produced by mdn_prep.py -> save_mdn_io):
- mdn_input.csv  has columns including: v_xin, v_yin, v_zin, E_in (plus optional extras)
- mdn_output.csv has columns including: v_xout, v_yout, v_zout, E_out (plus optional extras)

This script trains an MDN p(y|x) where:
    x = (vx, vy, vz, E_in)
    y = (vx_out, vy_out, vz_out, E_out)

Usage example:
    python train_mdn_4d.py --x mdn_input.csv --y mdn_output.csv

Outputs:
    checkpoints/best_mdn_model.pt
    x_scaler.pkl
    y_scaler.pkl
    runs/<timestamp>_mdn4d/report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mdn_model import MultiOutputMDN, mdn_multi_loss


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_columns(df: pd.DataFrame, candidates):
    """Return first candidate list fully present in df, else raise."""
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    raise KeyError(
        "Required columns not found. Tried:\n"
        + "\n".join([str(cols) for cols in candidates])
        + f"\nAvailable columns: {list(df.columns)}"
    )


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


@torch.no_grad()
def mdn_predict_mean(model: MultiOutputMDN, xb: torch.Tensor):
    """Mixture mean: E[y|x] = sum_k pi_k * mu_k."""
    pi, mu, sigma = model(xb)
    # pi: [B,K], mu: [B,K,D]
    mean = (pi.unsqueeze(-1) * mu).sum(dim=1)  # [B,D]
    return mean


def evaluate(model, loader, device):
    model.eval()
    nlls = []
    mses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pi, mu, sigma = model(xb)
        nll = mdn_multi_loss(pi, mu, sigma, yb)
        nlls.append(float(nll.detach().cpu().item()))

        y_mean = mdn_predict_mean(model, xb)
        mse = torch.mean((y_mean - yb) ** 2)
        mses.append(float(mse.detach().cpu().item()))
    return {
        "nll": float(np.mean(nlls)) if nlls else float("nan"),
        "mse_mean": float(np.mean(mses)) if mses else float("nan"),
    }


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Train 4D MDN (vx,vy,vz,E_in -> vx,vy,vz,E_out)")
    ap.add_argument("--x", required=True, help="Path to mdn_input.csv")
    ap.add_argument("--y", required=True, help="Path to mdn_output.csv")
    ap.add_argument("--out_dir", default=".", help="Base output directory (default: current)")
    ap.add_argument("--checkpoints", default="checkpoints", help="Checkpoint dir relative to out_dir")
    ap.add_argument("--runs", default="runs", help="Run logs dir relative to out_dir")

    ap.add_argument("--test_size", type=float, default=0.10, help="Fraction for test split")
    ap.add_argument("--val_size", type=float, default=0.10, help="Fraction for val split (from remaining)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_gaussians", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.30)

    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=5.0)

    ap.add_argument("--patience", type=int, default=25, help="Early stopping patience on val NLL")
    ap.add_argument("--min_delta", type=float, default=1e-4, help="Minimum val NLL improvement to reset patience")

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    set_seed(args.seed)

    out_base = Path(args.out_dir)
    ckpt_dir = out_base / args.checkpoints
    runs_dir = out_base / args.runs
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    dfX = pd.read_csv(args.x)
    dfY = pd.read_csv(args.y)

    # Column resolution: prefer renamed columns from mdn_prep.save_mdn_io
    x_cols = resolve_columns(dfX, candidates=[
        ["v_xin", "v_yin", "v_zin", "E_in"],
        ["v_pavx", "v_pavy", "v_pavz", "E_in"],
        ["v_pavx", "v_pavy", "v_pavz", "v_E_in"],  # fallback (if raw dump-like naming sneaks in)
    ])
    y_cols = resolve_columns(dfY, candidates=[
        ["v_xout", "v_yout", "v_zout", "E_out"],
        ["v_vxout", "v_vyout", "v_vzout", "v_ke"],
        ["v_vxout", "v_vyout", "v_vzout", "E_out"],
    ])

    X = dfX[x_cols].to_numpy(dtype=np.float32)
    Y = dfY[y_cols].to_numpy(dtype=np.float32)

    if len(X) != len(Y):
        n = min(len(X), len(Y))
        X, Y = X[:n], Y[:n]

    if X.shape[0] < 100:
        raise ValueError(f"Too few samples ({X.shape[0]}). Need more data to train reliably.")

    # Splits
    idx = np.arange(X.shape[0])
    X_trainval, X_test, Y_trainval, Y_test, idx_trainval, idx_test = train_test_split(
        X, Y, idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    # val split from remaining
    val_frac = args.val_size / max(1e-12, (1.0 - args.test_size))
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=val_frac, random_state=args.seed, shuffle=True
    )

    # Scale
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    Y_train_s = y_scaler.fit_transform(Y_train)
    X_val_s   = x_scaler.transform(X_val)
    Y_val_s   = y_scaler.transform(Y_val)
    X_test_s  = x_scaler.transform(X_test)
    Y_test_s  = y_scaler.transform(Y_test)

    # Persist scalers to out_dir root (to match your existing inference code)
    joblib.dump(x_scaler, str(out_base / "x_scaler.pkl"))
    joblib.dump(y_scaler, str(out_base / "y_scaler.pkl"))

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Model
    model = MultiOutputMDN(
        input_dim=4,
        hidden_dim=args.hidden_dim,
        output_dim=4,
        num_gaussians=args.num_gaussians,
        dropout_rate=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # DataLoaders
    train_ds = NumpyDataset(X_train_s, Y_train_s)
    val_ds   = NumpyDataset(X_val_s, Y_val_s)
    test_ds  = NumpyDataset(X_test_s, Y_test_s)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    # Training loop with early stopping
    best_val = float("inf")
    best_path = ckpt_dir / "best_mdn_model.pt"
    history = []

    patience_left = args.patience

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pi, mu, sigma = model(xb)
            loss = mdn_multi_loss(pi, mu, sigma, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_metrics = evaluate(model, val_loader, device)
        val_nll = val_metrics["nll"]

        row = {
            "epoch": epoch,
            "train_nll": train_loss,
            "val_nll": val_nll,
            "val_mse_mean": val_metrics["mse_mean"],
            "elapsed_sec": float(time.time() - t0),
        }
        history.append(row)

        # Early stopping / best
        improved = (best_val - val_nll) > args.min_delta
        if improved:
            best_val = val_nll
            patience_left = args.patience
            torch.save(model.state_dict(), str(best_path))
        else:
            patience_left -= 1

        if epoch == 1 or epoch % 10 == 0 or improved or patience_left == 0:
            print(
                f"[{epoch:04d}] train_nll={train_loss:.5f}  "
                f"val_nll={val_nll:.5f}  best_val={best_val:.5f}  "
                f"patience_left={patience_left}"
            )

        if patience_left <= 0:
            print("Early stopping triggered.")
            break

    # Load best and evaluate on test
    if best_path.is_file():
        model.load_state_dict(torch.load(str(best_path), map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    # Save report
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_mdn4d"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_id": run_id,
        "args": vars(args),
        "device": str(device),
        "data": {
            "n_total": int(X.shape[0]),
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "x_cols_used": x_cols,
            "y_cols_used": y_cols,
        },
        "best": {
            "best_val_nll": float(best_val),
            "best_checkpoint": str(best_path),
        },
        "test": test_metrics,
        "history": history,
        "artifacts": {
            "x_scaler": str(out_base / "x_scaler.pkl"),
            "y_scaler": str(out_base / "y_scaler.pkl"),
            "checkpoint": str(best_path),
        },
    }

    with open(run_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n✅ Training complete.")
    print(f" - Best checkpoint: {best_path}")
    print(f" - x_scaler.pkl / y_scaler.pkl saved in: {out_base}")
    print(f" - Test metrics: NLL={test_metrics['nll']:.5f}, MSE(mean)={test_metrics['mse_mean']:.5f}")
    print(f" - Run report: {run_dir / 'report.json'}")


if __name__ == "__main__":
    import time
    main()
