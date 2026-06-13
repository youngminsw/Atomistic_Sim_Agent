import pandas as pd

Z_CUT = 28.0
path = "mdn_output.csv"

# CSV 읽기
df = pd.read_csv(path)

# 컬럼 이름 공백 제거 및 숫자 변환(문자/NaN 대비)
df.columns = [c.strip() for c in df.columns]
if "z_out" not in df.columns:
    raise KeyError(f"'z_out' 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")

z = pd.to_numeric(df["z_out"], errors="coerce").dropna()

# 규칙: z_out < 28 → injection, z_out > 28 → reflected
injection_cnt = (z < Z_CUT).sum()
reflected_cnt = (z > Z_CUT).sum()

# 만약 z_out == 28도 반사로 포함하려면 아래로 교체:
# reflected_cnt = (z >= Z_CUT).sum()
# injection_cnt = (z <  Z_CUT).sum()

print(f"reflected atom : {reflected_cnt}개")
print(f"injection atom : {injection_cnt}개")
