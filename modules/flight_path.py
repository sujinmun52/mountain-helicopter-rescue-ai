"""
헬기 비행 경로 생성 모듈
- 119 소방서 → 착륙지점 구간은 헬기가 '날아가는' 경로
- 지상 A* 와 달리 산을 우회할 필요 없이 직선 + 안전고도(지형 최대고도 + 마진)
- 시각화 시 산을 뚫는 것처럼 보이지 않도록 z값을 함께 반환
"""
from __future__ import annotations
import numpy as np
from modules.hoist import haversine


DEFAULT_MARGIN_M = 150.0   # 지형 위 안전마진 (m)
DEFAULT_NUM_POINTS = 40    # 비행 경로 샘플링 점 수
DEFAULT_CORRIDOR_CELLS = 3 # 직선 좌우 ±N 셀의 지형 최대고도를 살피는 '회랑' 폭


def _bilinear_elev(dem_array, dem_lats, dem_lons, lat, lon):
    """가장 가까운 4셀 평균 고도 (간단 보간)."""
    dist = (dem_lats - lat) ** 2 + (dem_lons - lon) ** 2
    r, c = np.unravel_index(np.argmin(dist), dist.shape)
    rows, cols = dem_array.shape
    r0, r1 = max(0, r - 1), min(rows - 1, r + 1)
    c0, c1 = max(0, c - 1), min(cols - 1, c + 1)
    return float(np.nanmean(dem_array[r0:r1 + 1, c0:c1 + 1]))


def _corridor_max_elev(dem_array, dem_lats, dem_lons, lat, lon, half_width_cells):
    """직선 점 주변 (±half_width_cells) 회랑의 최대 고도 — 측면 봉우리 보호."""
    dist = (dem_lats - lat) ** 2 + (dem_lons - lon) ** 2
    r, c = np.unravel_index(np.argmin(dist), dist.shape)
    rows, cols = dem_array.shape
    r0 = max(0, r - half_width_cells)
    r1 = min(rows - 1, r + half_width_cells)
    c0 = max(0, c - half_width_cells)
    c1 = min(cols - 1, c + half_width_cells)
    patch = dem_array[r0:r1 + 1, c0:c1 + 1]
    return float(np.nanmax(patch))


def make_flight_path(
    start_latlon: tuple[float, float],
    end_latlon: tuple[float, float],
    dem_lats: np.ndarray,
    dem_lons: np.ndarray,
    dem_array: np.ndarray,
    margin_m: float = DEFAULT_MARGIN_M,
    num_points: int = DEFAULT_NUM_POINTS,
    corridor_cells: int = DEFAULT_CORRIDOR_CELLS,
) -> list[dict]:
    """
    119 → 착륙지점 비행 경로 생성.

    Args:
        start_latlon: (lat, lon) 출발지 (119 소방서)
        end_latlon:   (lat, lon) 도착지 (착륙지점)
        dem_lats, dem_lons, dem_array: 지형 격자
        margin_m: 지형 위 안전마진 (m)
        num_points: 경로 샘플링 점 수
        corridor_cells: 직선 양옆 ±N 셀의 최대 고도를 참고할 회랑 폭

    Returns:
        [{"lat", "lon", "alt_m", "terrain_m", "dist_m"}, ...]
        - alt_m   : 비행 고도 (지형 최대고도 + margin_m)
        - terrain_m: 해당 점 회랑의 지형 최대 고도
        - dist_m  : 출발지로부터 누적 거리
    """
    s_lat, s_lon = start_latlon
    e_lat, e_lon = end_latlon

    lats = np.linspace(s_lat, e_lat, num_points)
    lons = np.linspace(s_lon, e_lon, num_points)

    # 모든 점에서 회랑 내 최대 지형 고도 → 전 구간 최대값 + margin = 순항고도
    corridor_max = np.array([
        _corridor_max_elev(dem_array, dem_lats, dem_lons, la, lo, corridor_cells)
        for la, lo in zip(lats, lons)
    ])
    cruise_alt = float(np.nanmax(corridor_max)) + margin_m

    # 이륙·착륙 경사: 시작·끝 5개 점은 지형따라 점진 상승/하강
    n_ramp = max(2, num_points // 8)
    altitudes = np.full(num_points, cruise_alt, dtype=float)
    start_terrain = corridor_max[0]
    end_terrain = corridor_max[-1]
    for i in range(n_ramp):
        t = i / n_ramp
        altitudes[i] = start_terrain + margin_m * 0.3 + (cruise_alt - (start_terrain + margin_m * 0.3)) * t
        altitudes[-1 - i] = end_terrain + margin_m * 0.3 + (cruise_alt - (end_terrain + margin_m * 0.3)) * t

    # 누적 거리
    cum_dist = [0.0]
    for i in range(1, num_points):
        d = haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
        cum_dist.append(cum_dist[-1] + d)

    path = []
    for i in range(num_points):
        path.append({
            "lat": float(lats[i]),
            "lon": float(lons[i]),
            "alt_m": float(altitudes[i]),
            "terrain_m": float(corridor_max[i]),
            "dist_m": float(cum_dist[i]),
        })
    return path


def flight_path_from_grid(grid_path, dem_lats, dem_lons, dem_array,
                          margin_m: float = DEFAULT_MARGIN_M) -> list[dict]:
    """
    A* 비행 격자 경로 [(row, col), ...] → 비행 경로 dict 리스트.
    고도는 지형추종(셀 지형고도 + margin_m), 거리는 누적 haversine.

    Returns:
        [{"lat", "lon", "alt_m", "terrain_m", "dist_m"}, ...]
    """
    pts = []
    cum = 0.0
    prev = None
    for (r, c) in grid_path:
        lat = float(dem_lats[r, c])
        lon = float(dem_lons[r, c])
        terr = float(dem_array[r, c])
        if prev is not None:
            cum += haversine(prev[0], prev[1], lat, lon)
        pts.append({
            "lat": lat,
            "lon": lon,
            "alt_m": terr + margin_m,
            "terrain_m": terr,
            "dist_m": cum,
        })
        prev = (lat, lon)
    return pts
