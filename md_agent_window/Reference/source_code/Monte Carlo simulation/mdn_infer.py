# ===== mdn_infer.py =====
import torch
import numpy as np
import joblib
from mdn_model import MultiOutputMDN

# MDN 샘플링 유틸
def sample_from_mdn(pi, mu, sigma):
    cat = torch.distributions.Categorical(pi)
    comp = cat.sample()
    idx = comp.unsqueeze(-1).expand(-1, mu.size(-1)).unsqueeze(1)
    m = mu.gather(1, idx).squeeze(1)
    s = sigma.gather(1, idx).squeeze(1)
    return (m + torch.randn_like(s) * s).cpu().numpy()

# 환경 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 스케일러 및 모델 로드
x_scaler     = joblib.load("x_scaler.pkl")
y_scaler     = joblib.load("y_scaler.pkl")
model        = MultiOutputMDN(4, 64, 4, 3, dropout_rate=0.3).to(device)
model.load_state_dict(torch.load("checkpoints/best_mdn_model.pt", map_location=device))
model.eval()

if __name__ == "__main__":
    # 수동 입력: [vx, vy, vz, E]
    manual = [[58.664,42.372,-125.341, 10473.58765]]
    X_scaled = x_scaler.transform(np.array(manual))
    xb = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        pi, mu, sigma = model(xb)
        Ys = sample_from_mdn(pi, mu, sigma)
    Y = y_scaler.inverse_transform(Ys)
    for i, y in enumerate(Y):
        vx_out, vy_out, vz_out, E_out = y
        print(f"#{i}: vx_out={vx_out:.3f}, vy_out={vy_out:.3f}, vz_out={vz_out:.3f}, E_out={E_out:.3f}")
