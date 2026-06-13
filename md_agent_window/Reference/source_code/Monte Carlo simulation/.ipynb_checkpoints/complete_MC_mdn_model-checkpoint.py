import numpy as np
import matplotlib.pyplot as plt

# total_model.py 에 정의된 함수들
from total_model import convert_2d_to_3d, surrogate_predict, convert_3d_to_2d

# ────────────────────────────────────────────────────────────────────────────────
# 트렌치 윤곽 정의
points = np.array([
    (2,   700),
    (10,  700),
    (10,    0),
    (50,    0),
    (50,  700),
    (58,  700)
], dtype=float)

# ────────────────────────────────────────────────────────────────────────────────
# 광선-세그먼트 교차 검사 함수
def ray_segment_intersection(origin, direction, p1, p2):
    seg = p2 - p1
    d = seg[1]*direction[0] - seg[0]*direction[1]
    if abs(d) < 1e-8:
        return None
    diff = origin - p1
    ua = (seg[0]*diff[1] - seg[1]*diff[0]) / d
    ub = (direction[0]*diff[1] - direction[1]*diff[0]) / d
    if ua >= 0 and 0 <= ub <= 1:
        return origin + ua * direction
    return None

# ────────────────────────────────────────────────────────────────────────────────
# 세그먼트 법선 계산 (입사 방향 기준)
def compute_segment_normal(p1, p2, incident_dir):
    edge = p2 - p1
    rawN = np.array([-edge[1], edge[0]], dtype=float)
    if np.dot(rawN, incident_dir) > 0:
        rawN = -rawN
    return -rawN / np.linalg.norm(rawN)

# ────────────────────────────────────────────────────────────────────────────────
# MDN 기반 궤적 시뮬레이션
# - 반사각은 MDN 예측 θ_o를 사용
# - θ_o 부호(sign) 판정: positive→x' 동일(sign(ix)), negative→x' 반대(-sign(ix))
def simulate_ion_trajectory_mdn(origin, direction, points, E_init, max_bounces=10):
    traj = [origin.copy()]
    collisions = []
    pos = origin.copy()
    dir_vec = direction.copy()
    E_in = E_init
    bounces = 0

    while bounces < max_bounces:
        # 1) 교차 검색
        nearest_inter, nearest_idx, min_dist = None, -1, np.inf
        for i in range(len(points)-1):
            p1, p2 = points[i], points[i+1]
            inter = ray_segment_intersection(pos, dir_vec, p1, p2)
            if inter is not None:
                dist = np.linalg.norm(inter - pos)
                if 1e-8 < dist < min_dist:
                    min_dist, nearest_inter, nearest_idx = dist, inter, i
        if nearest_inter is None:
            print(f"No intersections after {bounces} bounces, stopping.")
            break

        # 2) 충돌점 기록
        pos = nearest_inter
        traj.append(pos.copy())
        collisions.append(pos.copy())

        # 3) 법선 및 입사각 계산
        p1, p2 = points[nearest_idx], points[nearest_idx+1]
        normal = compute_segment_normal(p1, p2, -dir_vec)
        cos_i = np.dot(-dir_vec, normal)
        theta_i = np.arccos(np.clip(cos_i, -1, 1))
        deg_i = np.degrees(theta_i)

        # 4) MDN 예측으로 2D 반사각 θ_o (양의 값) 얻기
        vx, vy, vz = convert_2d_to_3d(E_in, theta_i)
        pred = surrogate_predict(vx, vy, vz, E_in)
        if pred is None:
            bounces += 1
            print(f"Bounce {bounces}: ion disappeared.")
            break
        vx_p, vy_p, vz_p, E_out = pred
        proj = convert_3d_to_2d(vx_p, vy_p, vz_p)
        if proj is None:
            print(f"Bounce {bounces+1}: projection failed.")
            break
        _, theta_o = proj
        deg_o_mag = np.degrees(theta_o)

        # 5) 로컬 좌표계 설정 x'=segment dir, y'=-normal
        tangent = p2 - p1
        tangent /= np.linalg.norm(tangent)
        y_axis = -normal
        # 입사선의 로컬 x' 성분
        ix = np.dot(dir_vec, tangent)

        # 6) signed reflection angle
        #    θ_o positive -> keep sign(ix); θ_o negative -> reverse sign(ix)
        sign_ix = np.sign(ix)
        if deg_o_mag >= 0:
            x_sign = sign_ix
        else:
            x_sign = -sign_ix
        deg_o = deg_o_mag * np.sign(deg_o_mag)

        # 7) 진행 벡터 계산: x'=sinθ_o * x_sign, y'=cosθ_o (반사선 y'은 입사선 y' 부호와 반대)
        # 입사선의 로컬 y' 성분
        iy = np.dot(dir_vec, y_axis)
        sign_iy = np.sign(iy) if iy != 0 else 1
        # local_x: x' 성분
        local_x = x_sign * np.sin(theta_o)
        # local_y: y' 성분 (반사선은 입사선 y' 부호 반전)
        local_y = -sign_iy * np.cos(theta_o)
        # 글로벌 방향 벡터 변환
        refl_vec = local_x * tangent + local_y * y_axis
        refl_vec /= np.linalg.norm(refl_vec)
        refl_vec /= np.linalg.norm(refl_vec)

        # 8) 결과 출력 및 업데이트
        bounces += 1
        print(f"Bounce {bounces}: Incidence={deg_i:.2f}°, Reflection={deg_o:.2f}°")
        print(f"  Outgoing vector: [{refl_vec[0]:.4f}, {refl_vec[1]:.4f}]")
        pos, dir_vec, E_in = pos, refl_vec, E_in##########

    return np.array(traj), np.array(collisions), bounces

# ────────────────────────────────────────────────────────────────────────────────
# 실행 및 시각화
min_x, max_x = points[:,0].min(), points[:,0].max()
max_y = points[:,1].max()
origin = np.array([np.random.uniform(min_x+10, max_x-10), max_y+10])

dir_x = np.random.uniform(-0.5, 0.5)
dir_y = -5.0
direction = np.array([dir_x, dir_y])
direction /= np.linalg.norm(direction)

traj, collisions, bnc = simulate_ion_trajectory_mdn(origin, direction, points, E_init=10473.58765)

plt.figure(figsize=(8,8))
for i in range(len(points)-1):
    plt.plot(points[i:i+2,0], points[i:i+2,1], 'k-', lw=2)
plt.plot(traj[:,0], traj[:,1], 'o-', markersize=3)
plt.scatter(*origin, c='green', marker='*', s=50)
if collisions.size:
    plt.scatter(collisions[:,0], collisions[:,1], c='red', s=20)
plt.axis('equal')
plt.show()
