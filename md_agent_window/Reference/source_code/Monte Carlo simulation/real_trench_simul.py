import numpy as np
import matplotlib.pyplot as plt

# MDN 서러게이트 관련 함수들 import
from total_model import convert_2d_to_3d, surrogate_predict, convert_3d_to_2d

def random_knudsen_cosine_angle():
    mean_angle = np.pi
    stddev_angle = np.pi / 6
    angle = np.random.normal(mean_angle, stddev_angle)
    return np.clip(angle, 0, np.pi)

def reflect_knudsen_cosine(direction, normal):
    theta = random_knudsen_cosine_angle()
    refl = np.array([np.sin(theta), np.cos(theta)], dtype=float)
    refl = refl - 2 * np.dot(refl, normal) * normal
    return refl / np.linalg.norm(refl)

def reflect_specular(direction, normal):
    # 진짜 specular 반사
    return direction - 2 * np.dot(direction, normal) * normal

def reflect_mdn(direction, normal, E_in):
    """
    MDN surrogate를 이용해 반사 방향을 계산.
    surrogate_predict가 None을 반환하면 실패로 간주.
    """
    theta_in = np.arccos(abs(np.dot(-direction, normal)))
    vx, vy, vz = convert_2d_to_3d(E_in, theta_in)
    pred = surrogate_predict(vx, vy, vz, E_in)
    if pred is None:
        return None, None
    vx_p, vy_p, vz_p, _ = pred
    result = convert_3d_to_2d(vx_p, vy_p, vz_p)
    if result is None:
        return None, None
    _, theta_out = result

    tangent = np.array([normal[1], -normal[0]], dtype=float)
    sign = np.random.choice([-1, 1])
    new_dir = -np.cos(theta_out) * normal + sign * np.sin(theta_out) * tangent
    return new_dir / np.linalg.norm(new_dir), E_in

def ray_segment_intersection(origin, direction, p3, p4):
    """
    origin + ua*direction  (ua >= 0)
    segment p3->p4           (0 <= ub <= 1)
    """
    seg = p4 - p3
    d = seg[1] * direction[0] - seg[0] * direction[1]
    if abs(d) < 1e-8:
        return None
    diff = origin - p3
    ua = ( seg[0]*diff[1] - seg[1]*diff[0] ) / d
    ub = ( direction[0]*diff[1] - direction[1]*diff[0] ) / d
    if ua >= 0 and 0 <= ub <= 1:
        return origin + ua * direction
    return None

def compute_segment_normal(p1, p2, incident_dir):
    """
    세그먼트(p1->p2)의 법선을 선호 알고리즘에 맞춰 계산해 반환.
    1) rawN = [-edge[1], edge[0]]
    2) rawN·incident_dir > 0 → rawN = -rawN
    3) return -rawN / ||rawN||
    """
    edge = p2 - p1
    rawN = np.array([-edge[1], edge[0]], dtype=float)
    if np.dot(rawN, incident_dir) > 0:
        rawN = -rawN
    return -rawN / np.linalg.norm(rawN)

def simulate_ion_trajectory(initial_position, initial_direction,
                            points, nodes,
                            max_bounces=20,
                            reflection_type='spec',
                            E_init=None):
    pos = initial_position.astype(float)
    dir = initial_direction.astype(float)
    dir /= np.linalg.norm(dir)

    if reflection_type == 'mdn' and E_init is None:
        raise ValueError("reflection_type='mdn'일 때는 E_init을 지정해야 합니다.")
    E = E_init

    trajectory      = [pos.copy()]
    collision_pts   = []
    hit_normals     = []
    hit_seg_indices = []
    bounces         = 0
    epsilon         = 1e-2   # 충돌 후 벽에서 벗어나도록 보정값

    while bounces < max_bounces and pos[1] <= 705:
        closest_pt   = None
        min_dist     = np.inf
        best_normal  = None
        best_seg_idx = None

        # 각 세그먼트에 대해 교차점 검사
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            inter = ray_segment_intersection(pos, dir, p1, p2)
            if inter is not None:
                dist = np.linalg.norm(inter - pos)
                if dist < min_dist:
                    min_dist     = dist
                    closest_pt   = inter
                    best_seg_idx = i
                    # 선호 알고리즘으로 법선 계산 (incident_dir는 -dir)
                    best_normal = compute_segment_normal(p1, p2, -dir)

        if closest_pt is None:
            break  # 더 이상 충돌이 없으면 종료

        # 충돌 정보 저장
        pos = closest_pt
        trajectory.append(pos.copy())
        collision_pts.append(pos.copy())
        hit_normals.append(best_normal)
        hit_seg_indices.append(best_seg_idx)

        # 반사 처리
        if reflection_type == 'cos':
            dir = reflect_knudsen_cosine(dir, best_normal)
        elif reflection_type == 'spec':
            dir = reflect_specular(dir, best_normal)
        elif reflection_type == 'mdn':
            new_dir, new_E = reflect_mdn(dir, best_normal, E)
            if new_dir is None:
                break
            dir, E = new_dir, new_E
        else:
            raise ValueError(f"Unknown reflection_type: {reflection_type}")

        bounces += 1
        # 벽에서 소량 이동해 재교차 방지
        pos = pos + best_normal * epsilon

    return (
        np.array(trajectory),
        np.array(collision_pts),
        bounces,
        hit_seg_indices,
        np.array(hit_normals)
    )

# ────────────────────────────────────────────────────────────────────────────────
# 메인 실행 예시

points = np.array([
    (2,   700),
    (10,  700),
    (10,    0),
    (50,    0),
    (50,  700),
    (58,  700)
], dtype=float)

# 노드 생성 (unchanged)
total_len = sum(np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1))
seg_lens  = [np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1)]
num_nodes = 10
d_node    = total_len / (num_nodes - 1)
nodes, cur, rem = [points[0].copy()], points[0].copy(), d_node
for i, L in enumerate(seg_lens):
    unit = (points[i+1] - points[i]) / L
    while rem < L:
        cur = cur + unit * rem
        nodes.append(cur.copy())
        L  -= rem
        rem = d_node
    rem -= L
    cur = points[i+1].copy()
nodes = np.array(nodes, dtype=float)

# 시뮬레이션 파라미터 및 실행
init_pos   = np.array([32, 702], dtype=float)
init_dir   = np.array([1, -10], dtype=float)
reflection = 'mdn'
E_initial  = 10473.58765

traj, cps, bnc, seg_idxs, normals = simulate_ion_trajectory(
    init_pos, init_dir, points, nodes,
    max_bounces=20,
    reflection_type=reflection,
    E_init=E_initial
)

# 결과 시각화 및 출력
plt.figure(figsize=(8,8))
for i in range(len(points)-1):
    plt.plot(points[i:i+2,0], points[i:i+2,1], 'b-')
plt.scatter(nodes[:,0], nodes[:,1], s=5, label='Nodes')
plt.plot(traj[:,0], traj[:,1], 'o-', markersize=2, label='Trajectory')
if cps.size:
    plt.scatter(cps[:,0], cps[:,1], color='red', s=10, label='Collisions')
plt.title(f'Bounces: {bnc}')
plt.xlabel('X'); plt.ylabel('Y'); plt.legend(); plt.grid(True)
plt.show()

print("Hit segment indices:", seg_idxs)
print("Hit normals:\n", normals)
