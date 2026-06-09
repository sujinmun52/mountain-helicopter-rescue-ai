import json
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from config import KMA_API_KEY, SEORAK_LAT_MIN, SEORAK_LAT_MAX, SEORAK_LON_MIN, SEORAK_LON_MAX

def floor_to_5min(dt):
    """API 격자 생산 주기에 맞춰 5분 단위 내림"""
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)

def get_base_time():
    """초단기실황 발표 기준시각 계산 (매시 40분 이후 → 현재 정시)"""
    now = datetime.now()
    if now.minute < 40:
        base = now - timedelta(hours=1)
    else:
        base = now
    return base.strftime("%Y%m%d"), base.strftime("%H00")

# 설악산 권역 AWS 관측 지점 (기상청 API허브 지점번호 + 실제 위경도 + 관측소 표고)
# elev(관측소 해발고도, m)는 지형 인지형 보간의 수직 이방성 계산에 필수.
# 특히 대관령(772m)은 고지대 풍황을 대표하는 핵심 앵커 관측소.
_SEORAK_STATIONS = {
    90:  {"name": "속초",   "lat": 38.2506, "lon": 128.5644, "elev": 18.1},
    100: {"name": "대관령", "lat": 37.6764, "lon": 128.7183, "elev": 772.4},
    105: {"name": "강릉",   "lat": 37.7514, "lon": 128.8908, "elev": 26.0},
    211: {"name": "인제",   "lat": 38.0606, "lon": 128.1717, "elev": 200.2},
    212: {"name": "홍천",   "lat": 37.6863, "lon": 127.8883, "elev": 140.9},
}


def fetch_kma_realtime(nx_list=None, ny_list=None):
    """
    기상청 API허브 지상 AWS 실시간 관측 호출
    반환: {"지점번호": {"ws": float, "wd": float, "lat": float, "lon": float}}
    """
    now = datetime.now()
    # 기상청 데이터 동기화 딜레이(~15분)를 고려해 현재 기준 20분~10분 전 구간 요청
    tm2 = (now - timedelta(minutes=10)).strftime("%Y%m%d%H%M")
    tm1 = (now - timedelta(minutes=20)).strftime("%Y%m%d%H%M")

    base_url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
    wind_data = {}

    for stn_id, stn_info in _SEORAK_STATIONS.items():
        url = (f"{base_url}?tm1={tm1}&tm2={tm2}"
               f"&stn={stn_id}&disp=0&help=2&authKey={KMA_API_KEY}")

        masked_url = url.replace(KMA_API_KEY, "***") if KMA_API_KEY else url
        print(f"  [요청 URL] {masked_url}")

        # ── 네트워크 요청 ────────────────────────────────────────────────────
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code
            msg = f"  [HTTP {code}] {stn_info['name']}({stn_id}): {e.response.reason}"
            if code == 403:
                msg += " → apihub.kma.go.kr에서 해당 API 활용신청 필요"
            print(msg)
            continue
        except requests.exceptions.RequestException as e:
            print(f"  [네트워크 오류] {stn_info['name']}({stn_id}): {type(e).__name__}")
            continue

        # ── 텍스트 응답 파싱 ─────────────────────────────────────────────────
        # 응답 형식(help=2, disp=0): 헤더 없는 순수 데이터행
        # 컬럼 순서: datetime STN WD1 WS1 WDS WSS ...
        try:
            raw_lines = [
                line.split() for line in res.text.splitlines()
                if line.strip() and not line.startswith("#")
            ]
            if not raw_lines:
                print(f"  [데이터 없음] {stn_info['name']}({stn_id})")
                continue

            # 결측치(-50 이하) 제거 후 유효 행만 추출 — 시간 역순으로 최신값 우선
            valid = None
            for row in reversed(raw_lines):
                try:
                    wd_candidate = float(row[2])
                    ws_candidate = float(row[3])
                except (IndexError, ValueError):
                    continue
                # KMA 결측 마커: -50 이하(예: -99.9) 또는 물리적으로 불가능한 값 제외
                if ws_candidate < -50 or wd_candidate < -50:
                    continue
                if ws_candidate > 100 or wd_candidate > 360:
                    continue
                valid = (ws_candidate, wd_candidate)
                break

            if valid is None:
                print(f"  [결측값] {stn_info['name']}({stn_id}): "
                      f"유효 관측행 없음 (전체 {len(raw_lines)}행 모두 결측)")
                continue

            ws, wd = valid
            wind_data[str(stn_id)] = {
                "ws":   ws,
                "wd":   wd,
                "lat":  stn_info["lat"],
                "lon":  stn_info["lon"],
                "elev": stn_info["elev"],   # 지형 인지형 보간용 관측소 표고
            }
            print(f"  [수신] {stn_info['name']}({stn_id}): "
                  f"풍속={ws:.1f}m/s 풍향={wd:.0f}°")

        except (IndexError, ValueError) as e:
            print(f"  [파싱 오류] {stn_info['name']}({stn_id}): {e}")

    return wind_data

def apply_elevation_wind_correction(grid_ws, dem, work_altitude=60.0):
    """
    Power Law로 고도별 풍속 보정
    U(z) = U_ref * (z / z_ref) ^ alpha
    산악 지형 alpha = 0.27
    """
    alpha = 0.27
    z_ref = 10.0
    effective_altitude = dem + work_altitude
    return grid_ws * (effective_altitude / z_ref) ** alpha

def multi_point_bias_correction(kma_points, kma_ws, obs_points):
    """
    다중 관측지점 기반 IDW 잔차 보정
    obs_points: [{"lon": ..., "lat": ..., "observed": ...}, ...]
    """
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

    # IDW로 전체 격자에 잔차 분배
    bias_field = griddata(residual_coords, residuals, kma_points, method="linear", fill_value=0.0)
    return kma_ws + bias_field

def build_wind_field(dem_lats, dem_lons, kma_points, kma_ws, kma_wd, dem):
    """
    u/v 분해 후 15m 격자로 보간
    풍향 보간 순서: u/v 먼저 분해 → 보간 (각도 직접 보간 금지)
    """
    # u/v 분해 (기상 관례: 북=0, 시계방향)
    wd_rad = np.radians(kma_wd)
    kma_u = -np.sin(wd_rad)
    kma_v = -np.cos(wd_rad)

    target = (dem_lons, dem_lats)

    # 풍속 보간
    grid_ws = griddata(kma_points, kma_ws, target, method="linear")
    nan_mask = np.isnan(grid_ws)
    if np.any(nan_mask):
        grid_ws[nan_mask] = griddata(kma_points, kma_ws, target, method="nearest")[nan_mask]

    # u/v 보간
    grid_u = griddata(kma_points, kma_u, target, method="linear")
    grid_v = griddata(kma_points, kma_v, target, method="linear")
    nan_mask_u = np.isnan(grid_u)
    if np.any(nan_mask_u):
        grid_u[nan_mask_u] = griddata(kma_points, kma_u, target, method="nearest")[nan_mask_u]
        grid_v[nan_mask_u] = griddata(kma_points, kma_v, target, method="nearest")[nan_mask_u]

    # 고도 보정 적용
    grid_ws_corrected = apply_elevation_wind_correction(grid_ws, dem)

    return {
        "ws": grid_ws_corrected,
        "u": grid_u,
        "v": grid_v
    }