# ===== mdn_prep.py (save시에 컬럼명만 변경) =====
import pandas as pd
import numpy as np
from pathlib import Path

def _parse_incident_records(incident_path):
    """
    incident.dump 파서 (고정 오프셋 가정: 'ITEM: TIMESTEP' 아래 9번째 줄에 원자 1줄)
    덤프 형식(요청 반영):
      ITEM: ATOMS x y z v_pavx v_pavy v_pavz v_E_in v_speed v_theta v_phi
    우리가 가져올 필드(순서 유지):
      [x, y, z, v_pavx, v_pavy, v_pavz, E_in, theta_in]  -> 인덱스 [0,1,2,3,4,5,6,8]
    """
    records = []
    with open(incident_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            try:
                atom_line = lines[i + 9].strip().split()
                idxs = [0, 1, 2, 3, 4, 5, 6, 8]  # x y z v_pavx v_pavy v_pavz v_E_in v_theta
                vals = [atom_line[j] for j in idxs]
                rec = list(map(float, vals))
                if len(rec) == 8:
                    records.append(rec)
                i += 10
            except Exception:
                i += 1
        else:
            i += 1
    # 각 원소: [x, y, z, v_pavx, v_pavy, v_pavz, E_in, theta_in]
    return records

def _parse_reflected_records(reflected_path):
    """
    reflected.dump 파서 (고정 오프셋 가정):
      ITEM: ATOMS id type x y z v_vxout v_vyout v_vzout v_ke
    필요 열:
      [x, y, z, v_vxout, v_vyout, v_vzout, v_ke]  -> 인덱스 [2:9]
    """
    records = []
    with open(reflected_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            try:
                atom_line = lines[i + 9].strip().split()
                vals = atom_line[2:9]  # x, y, z, v_vxout, v_vyout, v_vzout, v_ke
                rec = list(map(float, vals))
                if len(rec) == 7:
                    records.append(rec)
                i += 10
            except Exception:
                i += 1
        else:
            i += 1
    # 각 원소: [x, y, z, v_vxout, v_vyout, v_vzout, v_ke]
    return records

def _safe_theta_from_cos_arg(cos_arg):
    """arccos 입력을 [-1,1]로 클리핑하고 deg로 반환"""
    cos_arg = np.clip(cos_arg, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_arg)))

def parse_dump_for_mdn(incident_path, reflected_path):
    """
    두 파일을 받아 X/Y 구성.
    - X: [x_in,y_in,z_in,v_pavx,v_pavy,v_pavz,E_in,theta_in]  (incident.dump에서 직접 읽음)
    - Y: [x_out,y_out,z_out,v_vxout,v_vyout,v_vzout,v_ke,theta_out]
      * theta_out은 반사 속도로 계산(+z 기준)
    페어링: 순서대로 1:1 (최소 길이 기준)
    """
    inc_records = _parse_incident_records(incident_path)
    ref_records = _parse_reflected_records(reflected_path)

    n_pairs = min(len(inc_records), len(ref_records))
    if n_pairs == 0:
        print("⚠️ 매칭 가능한 (incident, reflected) 페어가 없습니다.")
        df_X = pd.DataFrame(columns=['x_in','y_in','z_in','v_pavx','v_pavy','v_pavz','E_in','theta_in'])
        df_Y = pd.DataFrame(columns=['x_out','y_out','z_out','v_vxout','v_vyout','v_vzout','v_ke','theta_out'])
        return df_X, df_Y

    X_data, Y_data = [], []
    for k in range(n_pairs):
        # incident (입력) — 덤프에서 그대로
        x_in, y_in, z_in, v_pavx, v_pavy, v_pavz, E_in_dump, theta_in_dump = inc_records[k]
        E_in = E_in_dump
        theta_in = theta_in_dump
        X_data.append([x_in, y_in, z_in, v_pavx, v_pavy, v_pavz, E_in, theta_in])

        # reflected (출력) — theta_out 계산(+z 기준)
        x_out, y_out, z_out, v_vxout, v_vyout, v_vzout, v_ke = ref_records[k]
        v_out_mag = np.linalg.norm([v_vxout, v_vyout, v_vzout])
        theta_out = _safe_theta_from_cos_arg(v_vzout / v_out_mag) if v_out_mag > 0 else 0.0
        Y_data.append([x_out, y_out, z_out, v_vxout, v_vyout, v_vzout, v_ke, theta_out])

    # 내부 DataFrame은 기존 이름 유지 (다른 코드 호환성 보존)
    X_cols = ['x_in','y_in','z_in','v_pavx','v_pavy','v_pavz','E_in','theta_in']
    Y_cols = ['x_out','y_out','z_out','v_vxout','v_vyout','v_vzout','v_ke','theta_out']

    df_X = pd.DataFrame(X_data, columns=X_cols)
    df_Y = pd.DataFrame(Y_data, columns=Y_cols)
    return df_X, df_Y

def save_mdn_io(df_X, df_Y, prefix="./"):
    """
    엑셀/CSV 저장 시 컬럼 이름만 변경:
      - 입력: v_pavx→v_xin, v_pavy→v_yin, v_pavz→v_zin
      - 출력: v_vxout→v_xout, v_vyout→v_yout, v_vzout→v_zout, v_ke→E_out
    """
    out_dir = Path(prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_rename = {
        'v_pavx': 'v_xin',
        'v_pavy': 'v_yin',
        'v_pavz': 'v_zin',
    }
    y_rename = {
        'v_vxout': 'v_xout',
        'v_vyout': 'v_yout',
        'v_vzout': 'v_zout',
        'v_ke': 'E_out',
    }

    df_X_out = df_X.rename(columns=x_rename)
    df_Y_out = df_Y.rename(columns=y_rename)

    df_X_out.to_csv(out_dir / "mdn_input.csv", index=False)
    df_Y_out.to_csv(out_dir / "mdn_output.csv", index=False)

    print("✅ 저장 완료:")
    print(f" - {out_dir / 'mdn_input.csv'} (열명 변경 적용)")
    print(f" - {out_dir / 'mdn_output.csv'} (열명 변경 적용)")

if __name__ == "__main__":
    # 스크립트와 같은 폴더의 incident.dump / reflected.dump 사용
    BASE = Path(__file__).parent
    incident_path  = BASE / "incident.dump"
    reflected_path = BASE / "reflected.dump"

    for p in (incident_path, reflected_path):
        if not p.is_file():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

    df_X, df_Y = parse_dump_for_mdn(str(incident_path), str(reflected_path))
    save_mdn_io(df_X, df_Y, prefix=BASE)
