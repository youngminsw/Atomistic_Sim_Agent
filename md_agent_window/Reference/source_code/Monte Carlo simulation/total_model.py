
# ===== total_model.py =====
import numpy as np
import torch
import joblib
from mdn_model import MultiOutputMDN
from mdn_infer import sample_from_mdn

# Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASS   = 1  # atomic mass units ###########
# MASS1 = MASS/1000/6.02214076E+23

# Load model & scalers
model   = MultiOutputMDN(4,64,4,3,dropout_rate=0.3).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best_mdn_model.pt", map_location=DEVICE))
    ##checkpoint파일에서 모델 가중치 부름
model.eval()

# 입출력 데이터 스케일 조정
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# 2D→3D 변환 ( 입사에너지, 입사각을 3차원 xyz로 변환 )
def convert_2d_to_3d(E_in, theta_in):
    # E_fin=E_in*1.602176634E-19
    # print(E_fin,E_in)
    while True:
        V   = np.sqrt(2*E_in/MASS)
        phi = np.random.uniform(0, 2 * np.pi)
        vx  = V * np.sin(theta_in) * np.cos(phi)
        vy  = np.abs(V * np.sin(theta_in) * np.sin(phi)) ######################
        vz  = -np.abs(V * np.cos(theta_in)) ############################
        # 범위 검사: vx[0,144], vy[0,144], vz[-144,0]        ####################
        # if 0 <= vx <= 144 and 0 <= vy <= 144 and -144 <= vz <= 0:
        #     print("exit convert_2d_to_3d")
        return vx, vy, vz 
        
# 3D→2D 변환 ( xyz를 입사 에너지와 각으로 )
def convert_3d_to_2d(vx, vy, vz):
    # if vz < 0:
    #     print("Implant")
    #     return None
    V     = np.sqrt(vx**2 + vy**2 + vz**2)
    E_out = MASS * (V)**2 /2
    theta = np.arccos(vz / V)
    return E_out, theta

# MDN surrogate inference 대리추론
## 3d변환한 벡터와 입사 에너지를 스케일링 -> pi, mu, sigma 출력
def surrogate_predict(vx, vy, vz, E_in):
    x_raw    = np.array([[vx, vy, vz, E_in]])
    x_scaled = x_scaler.transform(x_raw)
    xb       = torch.tensor(x_scaled, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pi, mu, sigma = model(xb)
    y_scaled = sample_from_mdn(pi, mu, sigma)
    y_orig   = y_scaler.inverse_transform(y_scaled)
    vx_p, vy_p, vz_p, E_p = y_orig[0]
    # threshold on vz'
    if vz_p < 0:

        return None
    return vx_p, vy_p, vz_p, E_p

# Execution example
if __name__ == "__main__":
    E_in     = 10473.58765
    theta_deg = np.random.uniform(5, 85)
    theta_in = np.deg2rad(theta_deg)

    # 2D→3D
    vx, vy, vz  = convert_2d_to_3d(E_in, theta_in)
    print(f"[입사] E={E_in:.3f}, θ_in={theta_deg:.3f}°, v=({vx:.3f},{vy:.3f},{vz:.3f})")

    # surrogate predict
    pred = surrogate_predict(vx, vy, vz, E_in)
    if pred is None:
        print("vz' threshold not met; no output")
    else:
        vx_p, vy_p, vz_p, E_p = pred
        print(f"[모델출력] v'=({vx_p:.3f},{vy_p:.3f},{vz_p:.3f}), E'={E_p:.3f}")
        res = convert_3d_to_2d(vx_p, vy_p, vz_p)
        E_out, theta_out = res
        print(f"[출력] E'={E_out:.3f}, θ'={np.degrees(theta_out):.1f}°")
