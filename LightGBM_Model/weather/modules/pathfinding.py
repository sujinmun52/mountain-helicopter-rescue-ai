import numpy as np
import heapq
from math import tan, radians, exp
from modules.hoist import haversine
from config import WIND_BLOCK, WIND_PENALTY_HIGH

def get_move_vector(current, neighbor):
    """이동 방향 단위벡터"""
    vec = np.array([neighbor[1] - current[1],
                    neighbor[0] - current[0]], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def compute_cost(current, neighbor, terrain, wind_field, dem_lats, dem_lons):
    """A* 격자 이동 비용 함수"""
    r1, c1 = current
    r2, c2 = neighbor

    # 기본 이동 거리
    dist = haversine(dem_lats[r1,c1], dem_lons[r1,c1],
                     dem_lats[r2,c2], dem_lons[r2,c2])

    # 경사도 페널티 (Tobler's hiking function 변형 — 계수 1.8로 상향해 급경사 회피 강화)
    slope_deg = terrain["slope"][r2, c2]
    slope_rad = radians(slope_deg)
    slope_cost = 1.8 / max(0.1, 0.6 * exp(-3.5 * abs(tan(slope_rad) + 0.05)))

    # 풍속 페널티 (계수 하향: 완만한 지형 우선 탐색 허용)
    ws = wind_field["ws"][r2, c2]
    if ws > WIND_BLOCK:
        return 9999  # 사실상 차단
    elif ws > WIND_PENALTY_HIGH:
        wind_cost = ws * 1.2
    else:
        wind_cost = ws * 0.5

    # 맞바람 페널티 (이동벡터 vs 풍향벡터 내적)
    move_vec = get_move_vector(current, neighbor)
    wind_vec = np.array([wind_field["u"][r2,c2],
                         wind_field["v"][r2,c2]])
    dot = np.dot(move_vec, wind_vec)
    headwind_cost = 1.0 + max(0, dot) * 1.5

    # 능선 페널티
    ridge_cost = 1.5 if terrain["is_ridge"][r2, c2] else 1.0

    return dist * slope_cost * (wind_cost / 10.0) * headwind_cost * ridge_cost

def heuristic(current, goal, dem_lats, dem_lons):
    """Haversine 직선거리 휴리스틱 (admissible 보장)"""
    r1, c1 = current
    r2, c2 = goal
    return haversine(dem_lats[r1,c1], dem_lons[r1,c1],
                     dem_lats[r2,c2], dem_lons[r2,c2])

def astar(start, goal, terrain, wind_field, dem_lats, dem_lons, max_iterations=50000):
    """
    A* 경로 탐색
    start, goal: (row, col) 튜플
    """
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, dem_lats, dem_lons)}

    rows, cols = dem_lats.shape
    # 8방향 이동
    directions = [(-1,-1),(-1,0),(-1,1),
                  (0,-1),        (0,1),
                  (1,-1), (1,0), (1,1)]

    iteration = 0
    while open_set and iteration < max_iterations:
        iteration += 1
        _, current = heapq.heappop(open_set)

        if current == goal:
            # 경로 역추적
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dr, dc in directions:
            nr, nc = current[0]+dr, current[1]+dc
            neighbor = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            cost = compute_cost(current, neighbor, terrain,
                                wind_field, dem_lats, dem_lons)
            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal, dem_lats, dem_lons)
                f_score[neighbor] = f
                heapq.heappush(open_set, (f, neighbor))

    if iteration >= max_iterations:
        print(f"경로 탐색 제한 도달 ({max_iterations}번 반복)")
    print("경로를 찾을 수 없습니다.")
    return []