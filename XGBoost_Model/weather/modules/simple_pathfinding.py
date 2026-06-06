import numpy as np
from modules.hoist import haversine


def _terrain_cost(r, c, terrain):
    """경로 비용 함수: 경사도 중심, 능선 회피 보조"""
    slope = terrain["slope"][r, c]
    ridge = terrain["is_ridge"][r, c]
    # 경사도: 45도 기준 지수 증가 (완만한 구간은 낮은 비용)
    slope_cost = (slope / 45.0) ** 1.5
    ridge_cost  = 0.3 if ridge else 0.0
    return slope_cost + ridge_cost


def simple_path_with_obstacles(start, goal, terrain, dem_lats, dem_lons,
                                num_waypoints=10):
    """
    직선 경로 기반 경로 생성 — 비용 함수로 완만한 지형 우선 선택
    각 웨이포인트에서 ±4셀 탐색해 slope 비용이 낮은 지점으로 이동
    """
    r1, c1 = start
    r2, c2 = goal
    shape   = terrain["slope"].shape
    search  = 4   # 탐색 반경 (셀 수)

    rs = np.linspace(r1, r2, num_waypoints, dtype=int)
    cs = np.linspace(c1, c2, num_waypoints, dtype=int)

    refined_path = []
    for i, (r, c) in enumerate(zip(rs, cs)):
        # 격자 범위 체크
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            refined_path.append(refined_path[-1] if refined_path else (r, c))
            continue

        base_cost = _terrain_cost(r, c, terrain)

        # 경사도 임계치 이상이면 주변 탐색
        if terrain["slope"][r, c] > 30:
            best_r, best_c = r, c
            best_total = base_cost

            for dr in range(-search, search + 1):
                for dc in range(-search, search + 1):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < shape[0] and 0 <= nc < shape[1]):
                        continue
                    # 직선 경로에서 너무 벗어나지 않도록 이탈 패널티 부여
                    deviation = (abs(dr) + abs(dc)) * 1.5
                    total = _terrain_cost(nr, nc, terrain) + deviation
                    if total < best_total:
                        best_total = total
                        best_r, best_c = nr, nc

            refined_path.append((best_r, best_c))
        else:
            refined_path.append((r, c))

    return refined_path
