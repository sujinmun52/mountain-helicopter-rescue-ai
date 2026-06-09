"""
지형 인지형(Terrain-Aware) 기상 공간 보간 모듈
================================================

설악산처럼 표고차가 1,500 m 이상 벌어지는 복잡 산악에서는, 평면상의
2D IDW 나 Bilinear 보간만으로는 풍황을 재현할 수 없다. 산록의 AWS(예: 속초 18 m)
관측값을 대청봉(1,708 m) 격자에 그대로 가져다 쓰면 고지대 강풍을 심각하게 과소평가하기
때문이다.

본 모듈이 채택한 방법: **고도 보정 다변량 IDW (Elevation-aware / vertically-anisotropic
Multivariate IDW)** + **멱법칙 연직 프로파일 보정**.

  ┌─ 1단계: 다변량(3D) IDW ─────────────────────────────────────────────┐
  │  거리 = sqrt(dx² + dy² + (k·dz)²)  로 정의하여 "수평 거리"뿐 아니라     │
  │  "고도차(dz)"까지 가중치에 반영. k(IDW_VERTICAL_SCALE)가 클수록 같은     │
  │  고도대의 관측소가 더 큰 가중치를 가진다 → 대관령(772 m)이 고지대 격자를  │
  │  주도. 풍향은 0°/360° 불연속을 피하기 위해 반드시 u/v 벡터로 분해 후 보간. │
  └────────────────────────────────────────────────────────────────────┘
  ┌─ 2단계: 멱법칙 연직 보정 ────────────────────────────────────────────┐
  │  IDW 가 대표하는 "유효 관측 표고"에서 실제 목표 격자 표고까지 풍속을      │
  │  U(z) = U_ref·(z/z_ref)^α 프로파일로 보정. 상대 보정이라 폭주하지 않음.   │
  └────────────────────────────────────────────────────────────────────┘

왜 Kriging 이 아니라 다변량 IDW 인가?
  - 권역 AWS 가 5개소뿐이라 베리오그램(variogram) 추정이 통계적으로 불안정하다.
    (안정적 베리오그램 적합에는 통상 수십 개 이상의 표본점이 필요)
  - 표본이 늘어나면 'Kriging with External Drift(외부 추세=표고)' 또는
    'Regression Kriging' 으로 무손실 확장 가능하도록 거리/추세 분리 구조로 설계했다.

기존 코드(modules/weather.py: build_wind_field)와 동일한 반환 구조
{"ws", "u", "v", "wd"} 를 유지하여 호이스트 선정·A* 비용함수에 그대로 연동된다.
"""

import numpy as np
from config import (
    IDW_VERTICAL_SCALE, IDW_POWER, WIND_PROFILE_ALPHA, WIND_REF_HEIGHT,
)

_EARTH_R = 6_371_000.0  # 지구 반경(m)


def _stations_to_arrays(stations):
    """
    관측소 딕셔너리/리스트를 (lat, lon, elev, ws, u, v) ndarray로 변환.

    stations: fetch_kma_realtime() 반환 dict {"id": {...}} 또는 동일 형식 list.
    풍향(wd)은 기상 관례(북=0, 시계방향)로 u/v 분해:
        u = -sin(wd),  v = -cos(wd)   (바람이 '불어오는' 방향 기준 단위벡터)
    """
    if isinstance(stations, dict):
        stations = list(stations.values())

    lat  = np.array([s["lat"]  for s in stations], dtype=float)
    lon  = np.array([s["lon"]  for s in stations], dtype=float)
    elev = np.array([s.get("elev", 0.0) for s in stations], dtype=float)
    ws   = np.array([s["ws"]   for s in stations], dtype=float)
    wd   = np.radians(np.array([s["wd"] for s in stations], dtype=float))

    u = -np.sin(wd)
    v = -np.cos(wd)
    return lat, lon, elev, ws, u, v


def _latlon_to_meters(ref_lat, lat, lon, ref_lon):
    """
    소규모 권역(설악산 ~20 km)에 충분한 등거리 평면 근사로 위경도차를 미터로 환산.
    위도 1° ≈ 111,320 m, 경도는 cos(위도) 보정.
    """
    dlat_m = (lat - ref_lat) * (np.pi / 180.0) * _EARTH_R
    dlon_m = (lon - ref_lon) * (np.pi / 180.0) * _EARTH_R * np.cos(np.radians(ref_lat))
    return dlat_m, dlon_m


def interpolate_wind_terrain_aware(
    stations,
    target_lat,
    target_lon,
    target_elev,
    vertical_scale=IDW_VERTICAL_SCALE,
    power=IDW_POWER,
    profile_alpha=WIND_PROFILE_ALPHA,
    ref_height=WIND_REF_HEIGHT,
):
    """
    고도 보정 다변량 IDW 풍속/풍향 보간 (핵심 함수).

    Args:
        stations:      관측소 dict/list. 각 항목에 lat, lon, elev, ws, wd 필요.
        target_lat:    목표 위도 (스칼라 또는 임의 형상 ndarray)
        target_lon:    목표 경도 (target_lat 와 동일 형상)
        target_elev:   목표 표고 DEM (m, target_lat 와 동일 형상)
        vertical_scale: 수직 이방성 계수 k (수직 1 m ↔ 수평 k m)
        power:          IDW 거리 감쇠 지수
        profile_alpha:  멱법칙 연직 풍속 프로파일 지수
        ref_height:     관측 기준고도(지상 m)

    Returns:
        dict {"ws", "u", "v", "wd"} — 입력 target_* 와 동일 형상의 ndarray.
            ws : 풍속(m/s, 고도 보정 반영)
            u,v: 보간된 풍향 단위벡터 성분 (A* 맞바람 계산용)
            wd : 풍향(deg, 북=0 시계방향)
    """
    s_lat, s_lon, s_elev, s_ws, s_u, s_v = _stations_to_arrays(stations)

    if len(s_lat) == 0:
        raise ValueError("보간할 관측소 데이터가 없습니다.")

    shape = np.shape(target_lat)
    tlat = np.atleast_1d(np.asarray(target_lat, dtype=float)).ravel()
    tlon = np.atleast_1d(np.asarray(target_lon, dtype=float)).ravel()
    telev = np.atleast_1d(np.asarray(target_elev, dtype=float)).ravel()
    n_tgt = tlat.size

    # 출력 버퍼
    out_ws = np.empty(n_tgt)
    out_u  = np.empty(n_tgt)
    out_v  = np.empty(n_tgt)
    out_eff_elev = np.empty(n_tgt)  # IDW가 대표하는 '유효 관측 표고'

    ref_lat = float(np.mean(s_lat))
    ref_lon = float(np.mean(s_lon))
    s_dlat_m, s_dlon_m = _latlon_to_meters(ref_lat, s_lat, s_lon, ref_lon)

    for i in range(n_tgt):
        # ── 1단계: 다변량(3D) 거리 — 수평 + 수직 이방성 ──────────────────
        t_dlat_m, t_dlon_m = _latlon_to_meters(ref_lat, tlat[i], tlon[i], ref_lon)
        dx = s_dlon_m - t_dlon_m
        dy = s_dlat_m - t_dlat_m
        dz = (s_elev - telev[i]) * vertical_scale          # 고도차 → 수평 등가거리
        dist3d = np.sqrt(dx * dx + dy * dy + dz * dz)

        # 관측점과 목표점이 사실상 일치하면 해당 값 직접 사용 (0 division 방지)
        zero = dist3d < 1e-6
        if np.any(zero):
            w = zero.astype(float)
        else:
            w = 1.0 / np.power(dist3d, power)
        w_sum = w.sum()

        # ── 풍속·풍향벡터·유효표고 가중 평균 ─────────────────────────────
        out_ws[i]       = np.dot(w, s_ws)   / w_sum
        out_u[i]        = np.dot(w, s_u)    / w_sum
        out_v[i]        = np.dot(w, s_v)    / w_sum
        out_eff_elev[i] = np.dot(w, s_elev) / w_sum

    # ── 2단계: 멱법칙 연직 보정 (유효 관측표고 → 목표 표고) ──────────────────
    # U(z_target) = U_idw · ((z_target + ref_h) / (z_eff + ref_h))^α
    #   목표표고가 유효 관측표고보다 높으면 풍속 증폭, 낮으면 감쇠. 상대 보정이라 안정적.
    z_t = np.maximum(telev, 0.0) + ref_height
    z_e = np.maximum(out_eff_elev, 0.0) + ref_height
    out_ws = out_ws * np.power(z_t / z_e, profile_alpha)
    out_ws = np.clip(out_ws, 0.0, None)

    # ── 풍향 재합성 (u/v → deg) ──────────────────────────────────────────
    out_wd = (np.degrees(np.arctan2(-out_u, -out_v)) + 360.0) % 360.0

    return {
        "ws": out_ws.reshape(shape) if shape else float(out_ws[0]),
        "u":  out_u.reshape(shape)  if shape else float(out_u[0]),
        "v":  out_v.reshape(shape)  if shape else float(out_v[0]),
        "wd": out_wd.reshape(shape) if shape else float(out_wd[0]),
    }


def build_wind_field_terrain_aware(dem_lats, dem_lons, dem_array, stations):
    """
    기존 build_wind_field() 의 드롭인 대체 함수 (지형 인지형 버전).

    main.py 파이프라인에서 build_wind_field 대신 호출하면, 동일한 wind_field
    구조 {"ws","u","v"}(+ "wd")를 반환하므로 호이스트 선정·A*·평가 모듈을
    수정 없이 그대로 사용할 수 있다.

    Args:
        dem_lats, dem_lons: np.meshgrid 로 생성한 2D 위경도 격자
        dem_array:          동일 형상의 2D DEM 표고 배열(m)
        stations:           fetch_kma_realtime() 반환 dict (lat/lon/elev/ws/wd 포함)

    Returns:
        dict {"ws","u","v","wd"} — 모두 dem_lats 와 동일 형상의 2D 배열
    """
    return interpolate_wind_terrain_aware(
        stations, dem_lats, dem_lons, dem_array
    )
