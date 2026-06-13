import numpy as np
import matplotlib.pyplot as plt

# total_model.py 에 정의된 함수들
from total_model import convert_2d_to_3d, surrogate_predict, convert_3d_to_2d

# ────────────────────────────────────────────────────────────────────────────────
# 간단한 직사각형 트렌치 정의 (좌측 상단부터 시계방향)
points = np.array([
    (0, 700),   # 왼쪽 위
    (0,   0),   # 왼쪽 아래
    (50,  0),   # 오른쪽 아래
    (50, 700)  # 오른쪽 위
], dtype=float)
# ────────────────────────────────────────────────────────────────────────────────
def generate_trench_nodes(points, num_nodes=500):
    segment_lengths = []
    total_length = 0

    # 각 선분 길이 계산
    for i in range(len(points) - 1):
        segment_length = np.linalg.norm(points[i+1] - points[i])
        segment_lengths.append(segment_length)
        total_length += segment_length

    # 전체 노드 간 간격
    distance_between_nodes = total_length / (num_nodes - 1)

    # 노드 생성
    nodes = [points[0].copy()]
    current_pos = points[0].copy()
    remaining_distance = distance_between_nodes
    i = 0

    while len(nodes) < num_nodes and i < len(points) - 1:
        p1, p2 = points[i], points[i+1]
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        direction = seg_vec / seg_len

        while remaining_distance < seg_len:
            current_pos += direction * remaining_distance
            nodes.append(current_pos.copy())
            seg_len -= remaining_distance
            remaining_distance = distance_between_nodes
        remaining_distance -= seg_len
        i += 1

    return np.array(nodes)

# ────────────────────────────────────────────────────────────────────────────────
def ray_segment_intersection(origin, direction, p1, p2):
    seg = p2 - p1
    d = seg[1]*direction[0] - seg[0]*direction[1]
    
    if abs(d) < 1e-8:
        return None
        
    diff = origin - p1
    ua = (seg[0]*diff[1] - seg[1]*diff[0]) / d
    ub = (direction[0]*diff[1] - direction[1]*diff[0]) / d
    
    if ua >= 0 and 0 <= ub <= 1:
        intersection = origin + ua * direction
        dist = np.linalg.norm(intersection - origin)
        if dist > 1e-4:  # ← 여기서 너무 가까운 교차 무시
            return intersection
    return None

def compute_segment_normal(p1, p2, incident_dir):
    edge = p2 - p1
    rawN = np.array([-edge[1], edge[0]], dtype=float)
    if np.dot(rawN, incident_dir) > 0:
        rawN = -rawN
    return -rawN / np.linalg.norm(rawN)

# ────────────────────────────────────────────────────────────────────────────────
# 트렌치 노드 생성
nodes = generate_trench_nodes(points, num_nodes=500)
node_energy = np.zeros(len(nodes))  # 각 노드에 대한 누적 에너지 배열

# MDN 기반 궤적 시뮬레이션 (하나의 이온)
# pred=None 일 때도 traj를 리턴하도록 수정
def simulate_ion_trajectory_mdn(origin, direction, points, E_init, max_bounces=5, max_iter=50, n=-1):
    traj = [origin.copy()]
    collisions = []
    pos = origin.copy()
    dir_vec = direction.copy()
    E_in = E_init
    bounces = 0
    iter_count = 0
    
    while bounces < max_bounces and iter_count < max_iter:
        # 교차 검색
        nearest_inter, nearest_idx, min_dist = None, -1, np.inf
        
        iter_count += 1
        
        for i in range(len(points)-1):
            p1, p2 = points[i], points[i+1]
            inter = ray_segment_intersection(pos, dir_vec, p1, p2)
            if inter is not None:
                dist = np.linalg.norm(inter - pos)
                if 1e-8 < dist < min_dist:
                    min_dist, nearest_inter, nearest_idx = dist, inter, i
                    
# ──────────────────────────                    
        if nearest_inter is None:
            break
            
        if np.linalg.norm(pos - nearest_inter) < 1e-3:
            print(f"⚠️ Ion {n}: 제자리 충돌, 강제 종료")
            break  # 제자리에서 계속 튕기면 종료

        if iter_count >= max_iter:
            print(f"⚠️ Ion {n}: max_iter={max_iter} 도달, 루프 강제 종료")
            break
# ──────────────────────────            
        # 충돌점 기록
        pos = nearest_inter
        traj.append(pos.copy())
        collisions.append(pos.copy())

        # 법선 및 입사각
        p1, p2 = points[nearest_idx], points[nearest_idx+1]
        normal = compute_segment_normal(p1, p2, -dir_vec)
        cos_i = np.dot(-dir_vec, normal)
        theta_i = np.arccos(np.clip(cos_i, -1, 1))

# ──────────────────────────  
        # MDN 예측
        try:
            vx, vy, vz = convert_2d_to_3d(E_in, theta_i)
        except ValueError as e:
            print(f"⚠️ Ion {n}: {e} → 강제 종료")
            break  # 루프 종료
            
        pred = surrogate_predict(vx, vy, vz, E_in)
        if pred is None:
            # 주입된 지점은 현재 pos!
            return np.array(traj), np.array(collisions), pos.copy(), bounces, True
        
        vx_p, vy_p, vz_p, E_out = pred
        print(f"Converted 3D vector: vx={vx}, vy={vy}, vz={vz}")

        # 에너지 가드
        if np.isnan(E_out) or E_out <= 0:
            break

        proj = convert_3d_to_2d(vx_p, vy_p, vz_p)
        if proj is None or np.any(np.isnan(proj)) or np.any(np.isinf(proj)):
            print(f"⚠️ Ion {n}: convert_3d_to_2d 결과 이상 → 종료")
            break
            
        _, theta_o = proj
        if not np.isfinite(theta_o):
            print(f"⚠️ Ion {n}: theta_o가 유한하지 않음 (theta_o={theta_o}) → 종료")
            break
    
        if np.abs(theta_o) < 1e-3 or np.abs(np.pi - theta_o) < 1e-3:
            print(f"⚠️ Ion {n}: 반사각 극단값 (θ = {theta_o:.5f} rad) → 강제 종료")
            break
# ────────────────────────── 
        
        # 반사 벡터 계산
        tangent = (p2 - p1); tangent /= np.linalg.norm(tangent)
        y_axis = -normal
        ix = np.dot(dir_vec, tangent); iy = np.dot(dir_vec, y_axis)
        sign_ix = np.sign(ix) if ix != 0 else 1
        sign_iy = np.sign(iy) if iy != 0 else 1
        local_x = sign_ix * np.sin(theta_o)
        local_y = -sign_iy * np.cos(theta_o)
        refl_vec = local_x * tangent + local_y * y_axis
        refl_vec /= np.linalg.norm(refl_vec)

        # 상태 업데이트
        bounces += 1
        dir_vec = refl_vec
        E_in = E_out

    return np.array(traj), np.array(collisions), None, bounces, False

# ────────────────────────────────────────────────────────────────────────────────
# 여러 이온 루프 및 플로팅
num_ions = 1000
all_trajectories = []
injection_points = []

min_x, max_x = points[:,0].min(), points[:,0].max()
max_y = points[:,1].max()
E_init=50

ion_to_nodes = []  # 각 이온이 영향을 준 노드 목록 저장
ion_trajectories = []  # 이온 번호별 궤적 저장

for n in range(num_ions):
    origin = np.array([np.random.uniform(min_x+15, max_x-15), max_y+3])
    direction = np.random.uniform([-0.5, -5], [0.5, -0.5])
    direction /= np.linalg.norm(direction)

    traj, collisions, inj_pt, bounces, injected = simulate_ion_trajectory_mdn(
        origin, direction, points, E_init=E_init, n=n)
    
    ion_trajectories.append(traj)  # 항상 궤적 저장
    affected_nodes = set()
    
    if injected:
        print(f"Ion {n}: injected at {inj_pt}, bounces={bounces}")
        all_trajectories.append(traj)
        injection_points.append(inj_pt)

        # 주입된 경우, 충돌 지점(inj_pt)에 E_init 에너지 온전히 전달
        r = 3.0
        distances = np.linalg.norm(nodes - inj_pt, axis=1)
        nearby = np.where(distances < r)[0]

        # 충돌 지점 주변 노드에 에너지 기록
        for idx in nearby:
            weight = np.exp(-distances[idx] / r)
            node_energy[idx] += E_init * weight 
            affected_nodes.add(idx)

        if affected_nodes:
            ion_to_nodes.append(affected_nodes)

    elif collisions.size > 0:
        print(f"Ion {n}: {len(collisions)} collisions, bounces={bounces}")
        all_trajectories.append(traj)

        # 기존 충돌 로직 유지
        r = 3.0
        for pos in collisions:
            distances = np.linalg.norm(nodes - pos, axis=1)
            nearby = np.where(distances < r)[0]
            for idx in nearby:
                weight = np.exp(-distances[idx] / r)
                node_energy[idx] += E_init * weight
                affected_nodes.add(idx)

        ion_to_nodes.append(affected_nodes)

    else:
        print(f"Ion {n}: no collisions, skipping")

# ────────────────────────────────────────────────────────────────────────────────
# 이온 궤적 시각화

plt.figure(figsize=(8,8))

# 모든 궤적 그리기
for traj in all_trajectories:
    plt.plot(traj[:,0], traj[:,1], '-', linewidth=1, alpha=0.7)

# 트렌치 경계 한 번만 루프
for i in range(len(points)-1):
    label = 'Trench wall' if i == 0 else None
    plt.plot(points[i:i+2, 0], points[i:i+2, 1], 'k-', lw=2, label=label)

# 트렌치 노드 표시
plt.scatter(nodes[:,0], nodes[:,1], s=5, color='blue', label='Trench nodes')

# 주입 지점 표시
if injection_points:
    inj = np.vstack(injection_points)
    plt.scatter(inj[:,0], inj[:,1], marker='x', s=50, c='red', label='Injection Point')

# 축 및 범례
plt.xlabel('X'); plt.ylabel('Y')
plt.title(f'{len(all_trajectories)} Trajectories (red × = injected)')
plt.legend()
plt.axis('equal'); plt.grid(True)
plt.show(block=False)

# ────────────────────────────────────────────────────────────────────────────────
for idx, E in enumerate(node_energy):
    if E > 0:
        print(f"노드 {idx}: 누적 에너지 = {E:.2f}")

def find_ions_for_node(node_id):
    result = []
    for ion_idx, nodes_hit in enumerate(ion_to_nodes):
        if node_id in nodes_hit:
            result.append(ion_idx)
    return result

target_node = 240  # 분석하고 싶은 노드 번호
ion_list = find_ions_for_node(target_node)  # 이 노드에 영향을 준 이온들 찾기

plt.figure(figsize=(8, 8))
plt.title(f'the trajectory of ions that affected node {target_node}')
plt.xlabel('X')
plt.ylabel('Y')

# 트렌치 벽 그리기
for i in range(len(points) - 1):
    plt.plot(points[i:i+2, 0], points[i:i+2, 1], 'k-', lw=2)

# 선택한 노드 위치 표시 (빨간 X)
plt.scatter(nodes[target_node, 0], nodes[target_node, 1],
            s=80, color='red', marker='x', label='Target Node')

# 해당 노드에 영향을 준 이온들의 궤적만 그리기
for ion_id in ion_list:
    traj = ion_trajectories[ion_id]
    print(f"노드 {target_node}에 영향을 준 이온 {ion_id}, 궤적 길이: {len(traj)}")
    plt.plot(traj[:, 0], traj[:, 1], '-', linewidth=1.5, label=f'Ion {ion_id}')

plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show(block=False)

# ────────────────────────────────────────────────────────────────────────────────
# 누적 에너지 시각화 (컬러 스캐터 방식)

plt.figure(figsize=(8, 8))
plt.title("cumulative energy distribution")
plt.xlabel("X"); plt.ylabel("Y")

# 트렌치 경계
for i in range(len(points) - 1):
    plt.plot(points[i:i+2, 0], points[i:i+2, 1], 'k-', lw=2)

# 에너지 값으로 색상 표현
sc = plt.scatter(nodes[:, 0], nodes[:, 1],
                 c=node_energy, cmap='hot', s=15, marker='s')

plt.colorbar(sc, label='cumulative energy')
plt.axis('equal')
plt.grid(True)
plt.show(block=False)

input("Press Enter...")
