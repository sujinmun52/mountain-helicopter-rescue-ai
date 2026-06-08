import os
import sys
import requests
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import joblib
from dotenv import load_dotenv

# 1. 인프라 및 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, '.env'), override=True)
project_root = os.path.dirname(current_dir)

try:
    from weather.config import KMA_API_KEY
except ImportError:
    KMA_API_KEY = os.getenv("KMA_API_KEY", "YOUR_API_KEY_HERE")

# ==============================================================================
# [ENGINE] 실시간 AWS 공간 선형 보간 및 대기 경계층 멱법칙 보정 모듈
# ==============================================================================
_SEORAK_STATIONS = {
    90:  {"name": "속초",   "lat": 38.2506, "lon": 128.5644},
    100: {"name": "대관령", "lat": 37.6764, "lon": 128.7183},
    105: {"name": "강릉",   "lat": 37.7514, "lon": 128.8908},
    211: {"name": "인제",   "lat": 38.0606, "lon": 128.1717},
    212: {"name": "홍천",   "lat": 37.6863, "lon": 127.8883},
}

def fetch_kma_realtime():
    now = pd.Timestamp.now()
    tm2 = (now - pd.Timedelta(minutes=10)).strftime("%Y%m%d%H%M")
    tm1 = (now - pd.Timedelta(minutes=20)).strftime("%Y%m%d%H%M")
    base_url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
    wind_data = {}

    for stn_id, stn_info in _SEORAK_STATIONS.items():
        url = f"{base_url}?tm1={tm1}&tm2={tm2}&stn={stn_id}&disp=0&help=2&authKey={KMA_API_KEY}"
        try:
            res = requests.get(url, timeout=5)
            raw_lines = [l.split() for l in res.text.splitlines() if l.strip() and not l.startswith("#")]
            if not raw_lines: continue
            valid = None
            for row in reversed(raw_lines):
                try: wd, ws = float(row[2]), float(row[3])
                except: continue
                if -50 < ws < 100 and -50 < wd <= 360:
                    valid = (ws, wd)
                    break
            if valid:
                wind_data[str(stn_id)] = {"ws": valid[0], "wd": valid[1], "lat": stn_info["lat"], "lon": stn_info["lon"]}
        except: continue
    return wind_data

def apply_elevation_wind_correction(grid_ws, dem, work_altitude=60.0):
    return grid_ws * ((dem + work_altitude) / 10.0) ** 0.27

# ── [STEP 1] 작전 기체 투입 인터페이스 ─────────────────────────────────────
print("\n" + "="*70)
print("=== [산악 구조 관제 시스템 - Random Forest] 출격 기체 제원 설정 ===")
print("="*70)
heli_input = input("현재 작전에 투입할 헬기 기종의 번호를 입력하세요 (1 또는 2): ").strip()
deployed_heli = "small" if heli_input == "1" else "large"
heli_kor_name = "소형 구급 헬기 (Small)" if heli_input == "1" else "대형 수송 헬기 (Large)"

# 공용 지형 데이터셋 연동
XGB_DATA_DIR = os.path.join(project_root, 'XGBoost_Model', 'dataset')
df_master = pd.read_parquet(os.path.join(XGB_DATA_DIR, 'terrain_base.parquet'))

models = {
    "landing": joblib.load(os.path.join(current_dir, 'models', f'rf_{deployed_heli}_landing_model.joblib')),
    "hoist":   joblib.load(os.path.join(current_dir, 'models', f'rf_{deployed_heli}_hoist_model.joblib'))
}
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50}
feature_columns = ['elevation', 'slope_deg', 'tree_density', 'tree_height', 'wind_speed', 'wind_dir_sin', 'wind_dir_cos', 'land_0', 'land_1', 'land_2']

# ── [STEP 2] 전술 연산 코어 커널 ──────────────────────────────────────────
def find_best_rescue_tactics(rescue_lat, rescue_lon, heli_type, search_radius_meters=500):
    print(f"\n[작전 개시] 구조 요청 지점 (위도: {rescue_lat:.5f}, 경도: {rescue_lon:.5f}) 실황 탐색...")
    candidates = df_master.copy()
    candidates['dist_to_rescue'] = np.sqrt((candidates['latitude'] - rescue_lat) ** 2 + (candidates['longitude'] - rescue_lon) ** 2)
    candidates['dist_from_base_km'] = np.sqrt((candidates['latitude'] - FIRE_STATION["latitude"]) ** 2 + (candidates['longitude'] - FIRE_STATION["longitude"]) ** 2) * 110.0

    candidates = candidates[candidates['dist_to_rescue'] * 110000 <= search_radius_meters].copy()

    # [스트레스 테스트 처리 제어 레이어]
    if candidates.empty:
        print(f" ⚠️  [스트레스 테스트 가동] 가상의 최악 지형 노드를 빌드 주입합니다.")
        synthetic_data = {
            'latitude': [rescue_lat, rescue_lat + 0.0002, rescue_lat - 0.0002], 'longitude': [rescue_lon, rescue_lon - 0.0002, rescue_lon + 0.0002],
            'elevation': [1620.0, 1685.0, 1708.0], 'slope_deg': [2, 2, 2], 'tree_density': [3, 3, 3], 'tree_height': [2, 2, 2],
            'land_0': [1, 1, 1], 'land_1': [0, 0, 0], 'land_2': [0, 0, 0]
        }
        candidates = pd.DataFrame(synthetic_data)
        candidates['dist_to_rescue'] = np.sqrt((candidates['latitude'] - rescue_lat) ** 2 + (candidates['longitude'] - rescue_lon) ** 2)
        candidates['dist_from_base_km'] = np.sqrt((candidates['latitude'] - FIRE_STATION["latitude"]) ** 2 + (candidates['longitude'] - FIRE_STATION["longitude"]) ** 2) * 110.0
    else:
        candidates = candidates[candidates['land_2'] != 1].copy()

    # 실시간 기상 다각 보간 융합
    try:
        live_weather = fetch_kma_realtime()
        stn_points = np.array([[v["lon"], v["lat"]] for v in live_weather.values()])
        stn_ws, stn_wd = np.array([v["ws"] for v in live_weather.values()]), np.array([v["wd"] for v in live_weather.values()])
        stn_u, stn_v = -np.sin(np.radians(stn_wd)), -np.cos(np.radians(stn_wd))
        target_coords = (candidates['longitude'].values, candidates['latitude'].values)
        
        grid_ws = griddata(stn_points, stn_ws, target_coords, method="linear")
        grid_u  = griddata(stn_points, stn_u, target_coords, method="linear")
        grid_v  = griddata(stn_points, stn_v, target_coords, method="linear")
        
        nan_mask = np.isnan(grid_ws)
        if np.any(nan_mask):
            grid_ws[nan_mask] = griddata(stn_points, stn_ws, target_coords, method="nearest")[nan_mask]
            grid_u[nan_mask]  = griddata(stn_points, stn_u, target_coords, method="nearest")[nan_mask]
            grid_v[nan_mask]  = griddata(stn_points, stn_v, target_coords, method="nearest")[nan_mask]
            
        candidates['wind_direction'] = np.degrees(np.arctan2(-grid_u, -grid_v)) % 360.0
        candidates['wind_speed'] = apply_elevation_wind_correction(grid_ws, candidates['elevation'].values)
    except:
        candidates['wind_speed'], candidates['wind_direction'] = 8.5, 225.0

    candidates['wind_dir_sin'] = np.sin(np.radians(candidates['wind_direction'])).astype(np.float32)
    candidates['wind_dir_cos'] = np.cos(np.radians(candidates['wind_direction'])).astype(np.float32)

    # 동적 평점 연산
    flight_heading = np.degrees(np.arctan2(candidates['longitude'] - FIRE_STATION["longitude"], candidates['latitude'] - FIRE_STATION["latitude"])) % 360
    angle_diff     = np.abs(flight_heading - candidates['wind_direction']) % 360
    candidates['wind_dir_score'] = np.select([(angle_diff < 45) | (angle_diff >= 315), (angle_diff >= 135) & (angle_diff < 225)], [1.0, 0.0], default=0.5)

    ELEVATION_MAP = {0: 1.0, 1: 0.70, 2: 0.35}
    elevation_grade = np.select([candidates['elevation'] < 500, (candidates['elevation'] >= 500) & (candidates['elevation'] < 1200)], [0, 1], default=2)
    candidates['elevation_score'] = np.vectorize(ELEVATION_MAP.get)(elevation_grade).astype('float32')

    candidates['wind_score'] = np.select([candidates['wind_speed'] < 5.0, (candidates['wind_speed'] >= 5.0) & (candidates['wind_speed'] < 10.0), (candidates['wind_speed'] >= 10.0) & (candidates['wind_speed'] < 15.0)], [1.0, 0.6, 0.2], default=0.0)
    slope_map, density_map, height_map = {0: 1.0, 1: 0.7, 2: 0.35}, {0: 1.0, 1: 0.85, 2: 0.21, 3: 0.0}, {0: 1.0, 1: 0.61, 2: 0.31}
    candidates['slope_score']        = candidates['slope_deg'].map(slope_map)
    candidates['tree_density_score'] = candidates['tree_density'].map(density_map)
    candidates['tree_height_score']  = candidates['tree_height'].map(height_map)

    if heli_type == "small":
        candidates['score_landing'] = (candidates['slope_score'] * 0.42 + candidates['tree_density_score'] * 0.10 + candidates['tree_height_score'] * 0.08 + candidates['wind_score'] * 0.20 + candidates['wind_dir_score'] * 0.12 + candidates['elevation_score'] * 0.08)
        candidates['score_hoist']   = (candidates['slope_score'] * 0.12 + candidates['tree_density_score'] * 0.15 + candidates['tree_height_score'] * 0.20 + candidates['wind_score'] * 0.30 + candidates['wind_dir_score'] * 0.15 + candidates['elevation_score'] * 0.08)
    else:
        candidates['score_landing'] = (candidates['slope_score'] * 0.46 + candidates['tree_density_score'] * 0.14 + candidates['tree_height_score'] * 0.08 + candidates['wind_score'] * 0.12 + candidates['wind_dir_score'] * 0.08 + candidates['elevation_score'] * 0.12)
        candidates['score_hoist']   = (candidates['slope_score'] * 0.10 + candidates['tree_density_score'] * 0.24 + candidates['tree_height_score'] * 0.18 + candidates['wind_score'] * 0.18 + candidates['wind_dir_score'] * 0.12 + candidates['elevation_score'] * 0.18)

    # 🎯 Random Forest 모델 추론 가동 (RF는 원래 1D Class Label을 반환하므로 argmax 디코딩 레이어가 필요 없음)
    X_candidates = candidates[feature_columns].astype(np.float32)
    candidates['pred_landing'] = models["landing"].predict(X_candidates).astype(np.int32)
    candidates['pred_hoist']   = models["hoist"].predict(X_candidates).astype(np.int32)

    def filter_spatial_diversity(sorted_df):
        selected = []
        for _, row in sorted_df.iterrows():
            if not selected: selected.append(row)
            else:
                if all(np.sqrt((row['latitude'] - s['latitude'])**2 + (row['longitude'] - s['longitude'])**2) >= 0.000455 for s in selected): selected.append(row)
            if len(selected) == 3: break
        return pd.DataFrame(selected) if selected else pd.DataFrame()

    best_landing = filter_spatial_diversity(candidates.sort_values(by=['pred_landing', 'score_landing', 'dist_to_rescue'], ascending=[True, False, True]))
    best_hoist   = filter_spatial_diversity(candidates.sort_values(by=['pred_hoist', 'score_hoist', 'dist_to_rescue'], ascending=[True, False, True]))

    target_labels = {0: "안전 - 작전 원활", 1: "주의 - 조건부 작전", 2: "위험 - 작전 불가"}
    w_map         = {1.0: '정풍', 0.0: '역풍', 0.5: '측풍'}

    print("\n" + "="*115); print(f"[산악 구조 헬기 전술 통제 시스템] 실시간 관제 리포트 - 기종: {heli_kor_name}"); print("="*115)
    print(f"\n[A안: 기체 직접 안착(Landing) 추천 좌표 목록 (최대 3순위)]\n" + "-" * 115)
    if not best_landing.empty:
        for rank, (_, r) in enumerate(best_landing.iterrows(), 1):
            w_rel = w_map.get(r['wind_dir_score'], '측풍')
            print(f" {rank}순위 추천지 -> 좌표: ({r['latitude']:.5f}, {r['longitude']:.5f})\n    [전술 거리] 조난자까지: 약 {r['dist_to_rescue'] * 110000:.1f}m | 소방기지로부터: 약 {r['dist_from_base_km']:.2f}km\n    [AI 안전성] 등급: {target_labels[int(r['pred_landing'])]} | 전술 적합도 점수: {r['score_landing']:.4f}점\n    [현장 실황] 풍속: {r['wind_speed']:.1f}m/s ({w_rel}) | 계측고도: {r['elevation']:.1f}m\n" + "-" * 80)

    print(f"\n[B안: 제자리 비행 호이스트(Hoist) 강하 추천 좌표 목록 (최대 3순위)]\n" + "-" * 115)
    if not best_hoist.empty:
        for rank, (_, r) in enumerate(best_hoist.iterrows(), 1):
            w_rel = w_map.get(r['wind_dir_score'], '측풍')
            orig_land = 0 if r.get('land_0', 0) == 1 else (1 if r.get('land_1', 0) == 1 else 2)
            print(f" {rank}순위 구조지 -> 좌표: ({r['latitude']:.5f}, {r['longitude']:.5f})\n    [전술 거리] 조난자까지: 약 {r['dist_to_rescue'] * 110000:.1f}m | 소방기지로부터: 약 {r['dist_from_base_km']:.2f}km\n    [AI 안전성] 등급: {target_labels[int(r['pred_hoist'])]} | 전술 적합도 점수: {r['score_hoist']:.4f}점\n    [현장 실황] 풍속: {r['wind_speed']:.1f}m/s ({w_rel}) | 수목 높이등급: {int(r['tree_height'])} | 수목밀도: {int(r['tree_density'])} | 지형형태: {orig_land} | 계측고도: {r['elevation']:.1f}m\n" + "-" * 80)
            
    return best_landing, best_hoist

example_lat, example_lon = 38.1270, 128.4662
find_best_rescue_tactics(example_lat, example_lon, deployed_heli, search_radius_meters=500)