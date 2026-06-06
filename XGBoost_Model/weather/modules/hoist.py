import numpy as np
from config import HOIST_MAX_WIND, HOIST_MAX_SLOPE, HOIST_MAX_CANOPY

def haversine(lat1, lon1, lat2, lon2):
    """두 위경도 간 거리 계산 (미터)"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def find_hoist_candidates(victim_gps, terrain, wind_field, dem_lats, dem_lons):
    """
    호이스트 가능 지점 필터링 후
    조난자와 가장 가까운 지점 반환
    """
    # 조건 필터
    cond_open = terrain["is_open"]
    cond_slope = terrain["slope"] < HOIST_MAX_SLOPE
    cond_wind = wind_field["ws"] < HOIST_MAX_WIND
    cond_canopy = terrain["canopy_height"] < HOIST_MAX_CANOPY

    candidate_mask = cond_open & cond_slope & cond_wind & cond_canopy
    candidate_indices = np.argwhere(candidate_mask)

    # 단계적 조건 완화 (조건을 하나씩 제거하며 재탐색)
    if len(candidate_indices) == 0:
        print("호이스트 1단계 완화: 풍속 조건 제거")
        candidate_indices = np.argwhere(cond_open & cond_slope & cond_canopy)

    if len(candidate_indices) == 0:
        print("호이스트 2단계 완화: 경사도 기준 +10도 확장")
        cond_slope_relax = terrain["slope"] < (HOIST_MAX_SLOPE + 10.0)
        candidate_indices = np.argwhere(cond_open & cond_slope_relax & cond_canopy)

    if len(candidate_indices) == 0:
        print("호이스트 3단계 완화: 수목 조건 제거")
        cond_slope_relax = terrain["slope"] < (HOIST_MAX_SLOPE + 10.0)
        candidate_indices = np.argwhere(cond_open & cond_slope_relax)

    if len(candidate_indices) == 0:
        print("호이스트 후보 없음: 개활지 조건만 적용")
        candidate_indices = np.argwhere(cond_open)

    if len(candidate_indices) == 0:
        print("호이스트 후보가 없습니다. 입력 데이터를 검토하세요.")
        return None

    # 조난자 GPS 기준 Haversine 거리 계산
    victim_lat, victim_lon = victim_gps["latitude"], victim_gps["longitude"]
    distances = []

    for idx in candidate_indices:
        r, c = idx
        clat = dem_lats[r, c]
        clon = dem_lons[r, c]
        dist = haversine(victim_lat, victim_lon, clat, clon)
        distances.append(dist)

    nearest_idx = candidate_indices[np.argmin(distances)]
    r, c = nearest_idx

    return {
        "row": r,
        "col": c,
        "latitude": dem_lats[r, c],
        "longitude": dem_lons[r, c],
        "distance_m": min(distances)
    }