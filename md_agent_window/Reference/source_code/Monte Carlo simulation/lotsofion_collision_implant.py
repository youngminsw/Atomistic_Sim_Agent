import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# total_model.py 에 정의된 함수들
from total_model import convert_2d_to_3d, surrogate_predict, convert_3d_to_2d

# ────────────────────────────────────────────────────────────────────────────────
# 트렌치 윤곽 정의
points = np.array([
    (2, 700),
    (10, 700),
    (10,   0),
    (50,   0),
    (50, 700),
    (58, 700)
], dtype=float)
# points = np.array([
#     (2, 700), (10, 700), (7, 600), (18, 220), (0.8, 150), (6, 150), (7.5, 165), 
#           (14, 160), (8, 40), (30, 0), (52, 40), (46, 160), (52.5, 165), (54, 150), (59.2, 150), 
#           (42, 220), (53, 600), (50, 700), (58, 700)
# ], dtype=float)
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

def compute_segment_normal(p1, p2, incident_dir):
    edge = p2 - p1
    rawN = np.array([-edge[1], edge[0]], dtype=float)
    if np.dot(rawN, incident_dir) > 0:
        rawN = -rawN
    return -rawN / np.linalg.norm(rawN)

def simulate_ion_trajectory_mdn(origin, direction, points, E_init, max_bounces=10):
    traj = [origin.copy()]
    collisions = []
    pos = origin.copy()
    dir_vec = direction.copy()
    E_in = E_init
    bounces = 0

    while bounces < max_bounces:
        nearest_inter, nearest_idx, min_dist = None, -1, np.inf
        for i in range(len(points)-1):
            p1, p2 = points[i], points[i+1]
            inter = ray_segment_intersection(pos, dir_vec, p1, p2)
            if inter is not None:
                dist = np.linalg.norm(inter - pos)
                if 1e-8 < dist < min_dist:
                    min_dist, nearest_inter, nearest_idx = dist, inter, i
        if nearest_inter is None:
            break

        pos = nearest_inter
        traj.append(pos.copy())
        collisions.append(pos.copy())

        p1, p2 = points[nearest_idx], points[nearest_idx+1]
        normal = compute_segment_normal(p1, p2, -dir_vec)
        cos_i = np.dot(-dir_vec, normal)
        theta_i = np.arccos(np.clip(cos_i, -1, 1))

        vx, vy, vz = convert_2d_to_3d(E_in, theta_i)
        pred = surrogate_predict(vx, vy, vz, E_in)
        if pred is None:
            return np.array(traj), np.array(collisions), pos.copy(), bounces, True

        vx_p, vy_p, vz_p, E_out = pred
        if np.isnan(E_out) or E_out <= 0:
            break

        proj = convert_3d_to_2d(vx_p, vy_p, vz_p)
        if proj is None:
            break
        _, theta_o = proj

        tangent = (p2 - p1)
        tangent /= np.linalg.norm(tangent)
        y_axis = -normal
        ix = np.dot(dir_vec, tangent)
        iy = np.dot(dir_vec, y_axis)
        sign_ix = np.sign(ix) if ix != 0 else 1
        sign_iy = np.sign(iy) if iy != 0 else 1

        local_x = sign_ix * np.sin(theta_o)
        local_y = -sign_iy * np.cos(theta_o)
        refl_vec = local_x * tangent + local_y * y_axis
        refl_vec /= np.linalg.norm(refl_vec)

        bounces += 1
        dir_vec = refl_vec
        E_in = E_out

    return np.array(traj), np.array(collisions), None, bounces, False

# ────────────────────────────────────────────────────────────────────────────────
num_ions = 10
records = []
all_trajectories = []
injection_points = []

min_x, max_x = points[:,0].min(), points[:,0].max()
max_y = points[:,1].max()

for n in range(num_ions):
    origin    = np.array([np.random.uniform(min_x+10, max_x-10), max_y+3])
    direction = np.random.uniform([-0.5, -3], [0.5, -0.5])
    direction /= np.linalg.norm(direction)

    traj, collisions, inj_pt, bounces, injected = simulate_ion_trajectory_mdn(
        origin, direction, points, E_init=10473.58765)

    # 기록용
    collision_str = ";".join(f"{x:.3f},{y:.3f}" for x, y in collisions) if collisions.size else ""
    inj_x, inj_y = (inj_pt[0], inj_pt[1]) if injected else (np.nan, np.nan)
    records.append({
        "ion_index": n,
        "injected": injected,
        "injection_x": inj_x,
        "injection_y": inj_y,
        "collision_points": collision_str
    })

    # 시각화용
    if injected or collisions.size:
        all_trajectories.append(traj)
        if injected:
            injection_points.append(inj_pt)

# DataFrame 생성 및 엑셀로 저장
df = pd.DataFrame(records)
excel_path = "./ion_results.xlsx"
df.to_excel(excel_path, index=False)

# 시각화
# plt.figure(figsize=(8,8))
# for traj in all_trajectories:
#     plt.plot(traj[:,0], traj[:,1], linewidth=1, alpha=0.7)
# if injection_points:
#     inj = np.vstack(injection_points)
#     plt.scatter(inj[:,0], inj[:,1], marker='x', s=50, label='Injection Point')
# for i in range(len(points)-1):
#     plt.plot(points[i:i+2,0], points[i:i+2,1], 'k-', lw=2)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title(f'{len(all_trajectories)} Trajectories (× = injected)')
# plt.legend()
# plt.axis('equal')
# plt.grid(True)
# plt.show()

# 시각화
fig, ax = plt.subplots(figsize=(3, 8))
for traj in all_trajectories:
    ax.plot(traj[:,0], traj[:,1], linewidth=1, alpha=0.7)
if injection_points:
    inj = np.vstack(injection_points)
    ax.scatter(inj[:,0], inj[:,1], marker='x', s=50, c='red')  # label 제거
for i in range(len(points)-1):
    ax.plot(points[i:i+2,0], points[i:i+2,1], 'k-', lw=2)

# 축, 격자, 라벨, 틱 제거
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.set_title(f'{len(all_trajectories)} Ar Ion Trajectories')
ax.axis('equal')
ax.grid(False)
plt.show()
