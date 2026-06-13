#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# mdn_sputtered.csv와 sputtered.dump만 사용해서 Etch Rate 계산
# - 두께 환산: Δz = N_sput / (n * A)
#   n: Si 원자수 밀도(기본 0.04995 atoms/Å^3), A: 단면적 = Lx*Ly (Å^2)
# - 시간: 이벤트(=TIMESTEP 블록) 수 * 이벤트당 시간(기본 0.701 ps)
# - 출력: Etch Rate = {값} nm/min

import pandas as pd

CSV_PATH = "mdn_sputtered.csv"
DUMP_PATH = "sputtered.dump"

# ---- 사용자 조정 가능 상수 ----
N_DENSITY_SI = 0.04995   # atoms/Å^3  (Si, ρ≈2.33 g/cm^3 기준)
T_EVENT_PS    = 0.701    # ps per ion event (run 200 + run 500 + deposit 1 step @ 0.001 ps)

def parse_lx_ly_and_event_count(dump_path: str):
    """sputtered.dump에서 Lx, Ly(Å)와 TIMESTEP 블록 개수(이벤트 수)를 추출"""
    lx = ly = None
    n_evt = 0
    with open(dump_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i, L = 0, len(lines)
    while i < L:
        line = lines[i]
        if line.startswith("ITEM: TIMESTEP"):
            n_evt += 1
            i += 1  # timestep 값
        elif lx is None and line.startswith("ITEM: BOX BOUNDS"):
            # 바로 다음 3줄이 x/y/z 경계
            xlo, xhi = map(float, lines[i+1].split()[:2])
            ylo, yhi = map(float, lines[i+2].split()[:2])
            lx = xhi - xlo
            ly = yhi - ylo
            i += 4
        else:
            i += 1

    if lx is None or ly is None:
        raise RuntimeError("BOX BOUNDS를 덤프에서 찾지 못했습니다.")
    return lx, ly, n_evt

def main():
    # 1) 스퍼터된 원자 수 (행 수)
    df = pd.read_csv(CSV_PATH)
    N_sput = len(df)

    # 2) Lx, Ly와 이벤트 수
    lx, ly, n_evt = parse_lx_ly_and_event_count(DUMP_PATH)
    A = lx * ly  # Å^2

    # 3) 총 제거 두께(Å)
    if A <= 0 or N_DENSITY_SI <= 0:
        er_nm_min = 0.0
    else:
        dz_A = N_sput / (N_DENSITY_SI * A)  # Å

        # 4) 총 시간(s)
        t_total_s = (n_evt * T_EVENT_PS) * 1e-12  # ps → s

        # 5) Etch rate (nm/min)
        if t_total_s > 0:
            er_Aps = dz_A / t_total_s      # Å/s
            er_nm_min = er_Aps * 0.1 * 60  # (Å/s → nm/min)
        else:
            er_nm_min = 0.0

    print(f"Etch Rate = {er_nm_min:.6f} nm/min")

if __name__ == "__main__":
    main()
