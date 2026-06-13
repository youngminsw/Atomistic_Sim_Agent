"""train_mdn.py (comments removed; safe for contexts that dislike '#')"""
import os
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from mdn_model import MultiOutputMDN, mdn_multi_loss

class FullMDNDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

if __name__ == "__main__":
    x_df = pd.read_csv("mdn_input.csv")
    y_df = pd.read_csv("mdn_output.csv")

    input_cols  = ["v_xin","v_yin","v_zin","E_in"]
    target_cols = ["v_xout","v_yout","v_zout","E_out"]

    x = x_df[input_cols].values
    y = y_df[target_cols].values

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y)
    joblib.dump(x_scaler, "x_scaler.pkl")
    joblib.dump(y_scaler, "y_scaler.pkl")

    x_tr, x_val, y_tr, y_val = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)
    tr_ds = FullMDNDataset(x_tr, y_tr)
    val_ds = FullMDNDataset(x_val, y_val)
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    input_dim     = len(input_cols)
    output_dim    = len(target_cols)
    num_gaussians = 3
    hidden_dim    = 64
    dropout_rate  = 0.3
    lr            = 1e-3
    epochs        = 1000
    patience      = 50

    model = MultiOutputMDN(input_dim, hidden_dim, output_dim, num_gaussians, dropout_rate)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    no_improve = 0
    os.makedirs("checkpoints", exist_ok=True)
    train_losses, val_losses = [], []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            pi, mu, sigma = model(xb)
            loss = mdn_multi_loss(pi, mu, sigma, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pi, mu, sigma = model(xb)
                val_loss += mdn_multi_loss(pi, mu, sigma, yb).item()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f"[{ep}] Train {tr_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints/best_mdn_model.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("mdn_loss_plot.png", dpi=150)
