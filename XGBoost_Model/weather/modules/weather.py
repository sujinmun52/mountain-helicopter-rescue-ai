import json
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from weather.config import KMA_API_KEY, SEORAK_LAT_MIN, SEORAK_LAT_MAX, SEORAK_LON_MIN, SEORAK_LON_MAX

def floor_to_5min(dt):
    """API 격자 생산 주기에 맞춰 5분 단위 내림"""
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

def get_base_time():
    """초단기실황 발표 기준시각 계산"""
    now = datetime.now()
    if now.minute < 40:
        base = now - timedelta(hours=1)
    else:
        base = now
    return base.strftime("%Y%m%d"), base.strftime("%H00")

# 설악산 권역 AWS 관측 지점 (지점번호 + 실제 위경도)
_SEORAK_STATIONS = {
    90:  {"name": "속초",   "lat": 38.2506, "lon": 128.5644},
    100: {"name": "대관령", "lat": 37.6764, "lon": 128.7183},
    105: {"name": "강릉",   "lat": 37.7514, "lon": 128.8908},
    211: {"name": "인제",   "lat": 38.0606, "lon": 128.1717},
    212: {"name": "홍천",   "lat": 37.6863, "lon": 127.8883},
}

def fetch_kma_realtime(target_time: str = None, nx_list=None, ny_list=None):
    """
    기상청 API허브 지상 AWS 관측 호출 (실시간 및 과거 시점 겸용)
    :param target_time: 'YYYYMMDDHHMM' 형식의 12자리 문자열 (None이면 현재 실시간)
    """
    if target_time is not None:
        try:
            base_dt = datetime.strptime(target_time, "%Y%m%d%H%M")
        except ValueError:
            raise ValueError("❌ target_time 형식은 반드시 'YYYYMMDDHHMM' 형태여야 합니다.")
        tm2 = base_dt.strftime("%Y%m%d%H%M")
        tm1 = (base_dt - timedelta(minutes=10)).strftime("%Y%m%d%H%M")
    else:
        now = datetime.now()
        tm2 = (now - timedelta(minutes=10)).strftime("%Y%m%d%H%M")
        tm1 = (now - timedelta(minutes=20)).strftime("%Y%m%d%H%M")

    base_url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
    wind_data = {}

    for stn_id, stn_info in _SEORAK_STATIONS.items():
        url = f"{base_url}?tm1={tm1}&tm2={tm2}&stn={stn_id}&disp=0&help=2&authKey={KMA_API_KEY}"
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
        except requests.exceptions.RequestException:
            continue

        try:
            raw_lines = [
                line.split() for line in res.text.splitlines()
                if line.strip() and not line.startswith("#")
            ]
            if not raw_lines:
                continue

            valid = None
            for row in reversed(raw_lines):
                try:
                    wd_candidate = float(row[2])
                    ws_candidate = float(row[3])
                except (IndexError, ValueError):
                    continue
                if ws_candidate < -50 or wd_candidate < -50:
                    continue
                if ws_candidate > 100 or wd_candidate > 360:
                    continue
                valid = (ws_candidate, wd_candidate)
                break

            if valid is None:
                continue

            ws, wd = valid
            wind_data[str(stn_id)] = {
                "ws": ws, "wd": wd, "lat": stn_info["lat"], "lon": stn_info["lon"]
            }
        except (IndexError, ValueError):
            continue

    return wind_data

def apply_elevation_wind_correction(grid_ws, dem, work_altitude=60.0):
    alpha = 0.27
    z_ref = 10.0
    effective_altitude = dem + work_altitude
    return grid_ws * (effective_altitude / z_ref) ** alpha

def multi_point_bias_correction(kma_points, kma_ws, obs_points):
    residuals = []
    residual_coords = []
    for obs in obs_points:
        obs_coord = np.array([obs["lon"], obs["lat"]])
        distances = np.linalg.norm(kma_points - obs_coord, axis=1)
        nearest_idx = np.argmin(distances)
        residual = obs["observed"] - kma_ws[nearest_idx]
        residuals.append(residual)
        residual_coords.append(obs_coord)
    residual_coords = np.array(residual_coords)
    residuals = np.array(residuals)
    bias_field = griddata(residual_coords, residuals, kma_points, method="linear", fill_value=0.0)
    return kma_ws + bias_field

def build_wind_field(dem_lats, dem_lons, kma_points, kma_ws, kma_wd, dem):
    wd_rad = np.radians(kma_wd)
    kma_u = -np.sin(wd_rad)
    kma_v = -np.cos(wd_rad)
    target = (dem_lons, dem_lats)

    grid_ws = griddata(kma_points, kma_ws, target, method="linear")
    nan_mask = np.isnan(grid_ws)
    if np.any(nan_mask):
        grid_ws[nan_mask] = griddata(kma_points, kma_ws, target, method="nearest")[nan_mask]

    grid_u = griddata(kma_points, kma_u, target, method="linear")
    grid_v = griddata(kma_points, kma_v, target, method="linear")
    nan_mask_u = np.isnan(grid_u)
    if np.any(nan_mask_u):
        grid_u[nan_mask_u] = griddata(kma_points, kma_u, target, method="nearest")[nan_mask_u]
        grid_v[nan_mask_u] = griddata(kma_points, kma_v, target, method="nearest")[nan_mask_u]

    grid_ws_corrected = apply_elevation_wind_correction(grid_ws, dem)
    return {"ws": grid_ws_corrected, "u": grid_u, "v": grid_v}