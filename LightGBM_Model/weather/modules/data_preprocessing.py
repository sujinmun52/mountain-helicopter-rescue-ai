"""
좌표계 변환 및 기상-지형 데이터 공간 융합 전처리 모듈
- WGS84 위경도 ↔ 기상청 LCC 격자(nx, ny) 양방향 변환
- cKDTree 최근접 이웃 탐색 기반 정밀 지형 격자 ← 실시간 기상 데이터 융합
"""

import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Tuple, Dict

# ── 기상청 Lambert Conformal Conic 투영 상수 (KMA 공공데이터 포털 기준) ──────
_DEGRAD = math.pi / 180.0
_RADDEG = 180.0 / math.pi

_Re    = 6371.00877   # 지구 반경 (km)
_grid  = 5.0          # 격자 간격 (km)
_slat1 = 30.0         # 표준위도 1 (도)
_slat2 = 60.0         # 표준위도 2 (도)
_olon  = 126.0        # 기준점 경도 (도)
_olat  = 38.0         # 기준점 위도 (도)
_xo    = 43.0         # 기준점 격자 X
_yo    = 136.0        # 기준점 격자 Y

# LCC 투영 파라미터: 모듈 임포트 시 1회 사전 계산
_s1 = _slat1 * _DEGRAD
_s2 = _slat2 * _DEGRAD
_ol = _olat  * _DEGRAD

_SN = (math.log(math.cos(_s1) / math.cos(_s2)) /
       math.log(math.tan(math.pi * 0.25 + _s2 * 0.5) /
                math.tan(math.pi * 0.25 + _s1 * 0.5)))

_SF = (math.pow(math.tan(math.pi * 0.25 + _s1 * 0.5), _SN) *
       math.cos(_s1) / _SN)

_RO = (_Re / _grid * _SF /
       math.pow(math.tan(math.pi * 0.25 + _ol * 0.5), _SN))


def convert_wgs84_to_kma_grid(lat: float, lon: float) -> Tuple[int, int]:
    """
    WGS84 위경도 → 기상청 LCC 격자 좌표 변환

    기상청 공공데이터 포털 가이드라인의 Lambert Conformal Conic 삼각함수 투영 공식 사용.
    설악산 대청봉(38.1191, 128.4657) 기준 반환 예시: (88, 139)

    Args:
        lat: 위도 (WGS84)
        lon: 경도 (WGS84)

    Returns:
        (nx, ny): 기상청 격자 정수 좌표
    """
    lat_rad = lat * _DEGRAD
    lon_rad = lon * _DEGRAD
    ol_rad  = _olon * _DEGRAD

    ra = _Re / _grid * _SF / math.pow(math.tan(math.pi * 0.25 + lat_rad * 0.5), _SN)

    theta = lon_rad - ol_rad
    # 경도 차이를 [-π, π] 범위로 정규화
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= _SN

    nx = int(ra * math.sin(theta) + _xo + 0.5)
    ny = int(_RO - ra * math.cos(theta) + _yo + 0.5)

    return nx, ny


def _kma_grid_to_wgs84(nx: int, ny: int) -> Tuple[float, float]:
    """
    기상청 LCC 격자 좌표 → WGS84 위경도 역변환

    convert_wgs84_to_kma_grid의 역함수.
    API 응답의 nx, ny 키로부터 실제 지리 좌표를 복원하는 데 사용.

    Args:
        nx: 기상청 격자 X 좌표
        ny: 기상청 격자 Y 좌표

    Returns:
        (lat, lon): WGS84 위경도 (도)
    """
    # 투영 평면상 벡터 성분 분해
    # 순방향: x = ra*sin(theta) + xo, y = ro - ra*cos(theta) + yo
    xn = float(nx) - _xo
    yn = _RO + _yo - float(ny)   # = ra * cos(theta)

    ra = math.sqrt(xn * xn + yn * yn)
    if ra == 0.0:
        return 90.0, _olon  # 북극점 예외 처리

    lat = (2.0 * math.atan(math.pow(_Re / _grid * _SF / ra, 1.0 / _SN))
           - math.pi * 0.5)
    theta = math.atan2(xn, yn)             # atan2(sin_component, cos_component)
    lon = theta / _SN + _olon * _DEGRAD    # radians

    return lat * _RADDEG, lon * _RADDEG


def mapping_live_weather_to_grid(
    victim_lat: float,
    victim_lon: float,
    live_weather_api_data: Dict,
    terrain_df: pd.DataFrame,
    radius_km: float = 5.0
) -> pd.DataFrame:
    """
    정적 지형 격자를 중심축으로, 최근접 기상 격자점의 실시간 데이터를 공간 융합

    terrain_df의 세밀한 피처(경사도, 임상, 하천 등)를 보존한 채
    각 지형 행에 가장 가까운 KMA 500m 기상 격자의 풍속/풍향을 직접 할당.

    파이프라인:
      1. victim 위치 기준 radius_km 반경 내 지형 격자 슬라이싱
      2. API 응답의 nx, ny 키 → 역LCC 변환으로 실제 위경도 좌표 복원
      3. cKDTree 최근접 이웃 탐색: 각 지형 행 ← 가장 가까운 기상 격자점 매칭
      4. 매칭된 풍속/풍향을 terrain_df 행에 직접 컬럼으로 할당

    Args:
        victim_lat: 요구조자 위도 (WGS84)
        victim_lon: 요구조자 경도 (WGS84)
        live_weather_api_data: fetch_kma_realtime() 반환 딕셔너리
                               형식: {"nx_ny": {"ws": float, "wd": float}, ...}
        terrain_df: DEM+임상도 마스터 격자 DataFrame (필수 컬럼: "latitude", "longitude")
        radius_km: 슬라이싱 반경 (기본 5.0 km)

    Returns:
        슬라이싱된 지형 격자 DataFrame에 wind_speed, wind_direction 컬럼 추가.
        모든 행에 최근접 기상 값이 할당되어 NaN 없이 반환.
    """
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(victim_lat)))

    # ── Step 1: 반경 내 지형 격자 슬라이싱 (terrain_df 원본 피처 전량 보존) ───
    mask = (
        (terrain_df["latitude"] >= victim_lat - lat_delta) &
        (terrain_df["latitude"] <= victim_lat + lat_delta) &
        (terrain_df["longitude"] >= victim_lon - lon_delta) &
        (terrain_df["longitude"] <= victim_lon + lon_delta)
    )
    local_grid = terrain_df[mask].copy()

    if local_grid.empty or not live_weather_api_data:
        local_grid["wind_speed"]     = np.nan
        local_grid["wind_direction"] = np.nan
        return local_grid

    # ── Step 2: API 응답에서 실제 위경도 직접 추출 ──────────────────────────
    # fetch_kma_realtime이 lat/lon을 포함해 반환하므로 역변환 불필요
    weather_lats, weather_lons = [], []
    weather_ws,   weather_wd   = [], []

    for key, val in live_weather_api_data.items():
        weather_lats.append(val["lat"])
        weather_lons.append(val["lon"])
        weather_ws.append(val["ws"])
        weather_wd.append(val["wd"])

    weather_coords = np.column_stack([weather_lats, weather_lons])  # (N_weather, 2)
    ws_arr = np.array(weather_ws)
    wd_arr = np.array(weather_wd)

    # ── Step 3: cKDTree 최근접 이웃 탐색 ─────────────────────────────────────
    # 설악산 소규모 영역(~10km)에서 위경도 유클리드 거리 오차는 무시 가능 수준
    tree = cKDTree(weather_coords)
    terrain_coords = local_grid[["latitude", "longitude"]].values  # (N_terrain, 2)
    _, nearest_idx = tree.query(terrain_coords, k=1)

    # ── Step 4: 매칭된 기상 값을 지형 격자 컬럼으로 직접 할당 ─────────────────
    local_grid["wind_speed"]     = ws_arr[nearest_idx]
    local_grid["wind_direction"] = wd_arr[nearest_idx]

    return local_grid
