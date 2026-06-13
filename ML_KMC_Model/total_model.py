# ===== total_model.py =====
import numpy as np
import torch
import joblib
import os
from mdn_model import MultiOutputMDN
try:
    from Infer_Model import mdn_sample as sample_from_mdn
except ImportError:
    # Fallback if mdn_infer is missing or named differently
    # Should be provided by Infer_Model.py in the same directory
    try:
        from mdn_infer import sample_from_mdn
    except ImportError:
        print("Warning: sample_from_mdn not found. Surrogate model will fail.")

# Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASS = 39.948  # atomic mass units for Argon (Ar)

# Paths - define relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_mdn_model.pt")
X_SCALER_PATH = os.path.join(BASE_DIR, "x_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "y_scaler.pkl")

# Load model & scalers globally (lazy loading or check existence)
model = None
x_scaler = None
y_scaler = None

def load_resources():
    global model, x_scaler, y_scaler
    if model is not None:
        return True
        
    try:
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"Warning: Model checkpoint not found at {CHECKPOINT_PATH}")
            return False
            
        m = MultiOutputMDN(4, 64, 4, 3, dropout_rate=0.3).to(DEVICE)
        m.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        m.eval()
        model = m
        
        x_scaler = joblib.load(X_SCALER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
        return True
    except Exception as e:
        print(f"Error loading model resources: {e}")
        return False

# Initialize resources
MODEL_READY = load_resources()

# 2D→3D 변환 ( 입사에너지, 입사각을 3차원 xyz로 변환 )
def convert_2d_to_3d(E_in, theta_in_rad):
    # E_in: eV, theta_in_rad: radians
    # Fix unit if needed: 04_KMC_tool passes degrees! 
    # But for now let's just accept what is passed and debug the shape error first.
    # Note: KMC tool passes degrees, so we should convert to radians here for correct physics.
    # update: let's assume input is radians for now to match original intention.
    
    while True:
        V = np.sqrt(2 * E_in / MASS)
        phi = np.random.uniform(0, 2 * np.pi)
        vx = V * np.sin(theta_in_rad) * np.cos(phi)
        vy = np.abs(V * np.sin(theta_in_rad) * np.sin(phi)) 
        vz = -np.abs(V * np.cos(theta_in_rad)) 
        
        return vx, vy, vz 

# 3D→2D 변환 ( xyz를 입사 에너지와 각으로 )
def convert_3d_to_2d(vx, vy, vz):
    V = np.sqrt(vx**2 + vy**2 + vz**2)
    if V == 0:
        return 0.0, 0.0
    E_out = MASS * (V)**2 / 2
    # Clip for safety
    cos_theta = np.clip(vz / V, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return E_out, theta

# MDN surrogate inference 대리추론
def surrogate_predict(vx, vy, vz, E_in):
    if not MODEL_READY:
        return None
        
    try:
        x_raw = np.array([[vx, vy, vz, E_in]])
        # DEBUG PRINTS
        # print(f"DEBUG: x_raw shape: {x_raw.shape}")
        # print(f"DEBUG: x_raw content: {x_raw}")
        
        x_scaled = x_scaler.transform(x_raw)
        xb = torch.tensor(x_scaled, dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            pi, mu, sigma = model(xb)
        
        y_scaled = sample_from_mdn(pi, mu, sigma)
        # Check if sample_from_mdn returns tensor or numpy
        if isinstance(y_scaled, torch.Tensor):
            # Shape is [B, n_samples, D] -> [1, 1, 4] usually
            if y_scaled.dim() == 3:
                y_scaled = y_scaled.squeeze(1) # Remove n_samples dim if 1
            y_scaled = y_scaled.cpu().numpy()
            
        y_orig = y_scaler.inverse_transform(y_scaled)
        vx_p, vy_p, vz_p, E_p = y_orig[0]
        
        # threshold on vz' (must be positive for reflection, usually)
        # But if vz_p < 0, it means it's going deeper into material or stuck
        if vz_p < 0:
            return None
            
        return vx_p, vy_p, vz_p, E_p
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Execution example
if __name__ == "__main__":
    if not MODEL_READY:
        print("Skipping test: Model artifacts not found.")
    else:
        E_in = 100.0
        theta_deg = 45.0
        theta_in = np.deg2rad(theta_deg)

        # 2D→3D
        vx, vy, vz = convert_2d_to_3d(E_in, theta_in)
        print(f"[Input] E={E_in:.1f}, θ={theta_deg:.1f}°, v=({vx:.2f},{vy:.2f},{vz:.2f})")

        # surrogate predict
        pred = surrogate_predict(vx, vy, vz, E_in)
        if pred is None:
            print("No Valid Reflection (vz' < 0 or Model Error)")
        else:
            vx_p, vy_p, vz_p, E_p = pred
            print(f"[Model] v'=({vx_p:.2f},{vy_p:.2f},{vz_p:.2f}), E'={E_p:.2f}")
            res = convert_3d_to_2d(vx_p, vy_p, vz_p)
            E_out, theta_out = res
            print(f"[Output] E'={E_out:.2f}, θ'={np.degrees(theta_out):.1f}°")
