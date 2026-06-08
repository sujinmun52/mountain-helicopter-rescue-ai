"""
LightGBM 듀얼 추론 엔진 기반 실시간 산악 구조 관제 및 전술 추천 스크립트 (실시간 AWS 기상 보간 통합판)
"""
import os
import sys
import json
import shutil
import requests
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from dotenv import load_dotenv

# ── 1. 인프라 및 경로 설정 ──────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, '.env'), override=True)
project_root = os.path.dirname(current_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

try:
    from weather.config import KMA_API_KEY
except ImportError:
    KMA_API_KEY = os.getenv("KMA_API_KEY", "YOUR_API_KEY_HERE")

# ==============================================================================
# [ENGINE] 상단 기상청 AWS 실시간 정보 수집 및 수리 공간 보간 모듈
# ==============================================================================
_SEORAK_STATIONS = {
    90:  {"name": "속초",   "lat": 38.2506, "lon": 128.5644},
    100: {"name": "대관령", "lat": 37.6764, "lon": 128.7183},
    105: {"name": "강릉",   "lat": 37.7514, "lon": 128.8908},
    211: {"name": "인제",   "lat": 38.0606, "lon": 128.1717},
    212: {"name": "홍천",   "lat": 37.6863, "lon": 127.8883},
}

def fetch_kma_realtime(target_time: str = None):
    """기상청 API허브 지상 AWS 관측 데이터 실시간 호출"""
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
            res = requests.get(url, timeout=5)
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
    """지상 10m 기준 오차 풍속을 대기 경계층 멱법칙(Power Law) 기반으로 헬기 운항 고도로 보정"""
    alpha = 0.27
    z_ref = 10.0
    effective_altitude = dem + work_altitude
    return grid_ws * (effective_altitude / z_ref) ** alpha


# ==============================================================================
# [STEP 1] 출격 헬기 기종 선택 (소형 / 대형 고정 관제 UI)
# ==============================================================================
print("\n" + "="*70)
print("=== [산악 구조 관제 시스템 - LightGBM] 출격 기체 제원 설정 ===")
print("="*70)
print(" 1. 소형 구급 헬기 (Small Helicopter) - 돌풍 리스크 민감형")
print(" 2. 대형 수송 헬기 (Large Helicopter) - 지형 공간/하강풍 리스크 민감형")
print("="*70)

heli_input = input("현재 작전에 투입할 헬기 기종의 번호를 입력하세요 (1 또는 2): ").strip()

if heli_input == "1":
    deployed_heli = "small"
    heli_kor_name = "소형 구급 헬기 (Small)"
elif heli_input == "2":
    deployed_heli = "large"
    heli_kor_name = "대형 수송 헬기 (Large)"
else:
    sys.exit("에러: 1번(소형) 또는 2번(대형) 중 하나만 선택해야 합니다. 프로그램을 종료합니다.")

print(f"\n[기종 고정 완료] 작전 기체: 【 {heli_kor_name} 】")
print("해당 기체 전용 신규 가중치 기반 LightGBM 듀얼 AI 추론 엔진을 로드합니다.\n")


# ==============================================================================
# [STEP 2] 마스터 데이터 캐시 및 LightGBM 모델 인프라 로드
# ==============================================================================
print("데이터 매트릭스 및 LightGBM 모델 인프라 로드 중...")

df_master = pd.read_parquet(os.path.join(current_dir, 'dataset', 'terrain_base.parquet'))

def load_lgbm_model(model_key: str, tactic: str) -> lgb.Booster:
    """LightGBM 네이티브 포맷(.txt) 모델을 Windows 한글 경로 버그를 우회하여 로드합니다."""
    path = os.path.join(current_dir, 'models', f'lgbm_{model_key}_{tactic}_model.txt')
    if not os.path.exists(path):
        sys.exit(f"에러: 모델 파일을 찾을 수 없습니다. 경로를 확인하세요: {path}")
    
    temp_file = f"temp_load_lgbm_{model_key}_{tactic}.txt"
    shutil.copyfile(path, temp_file)
    
    try:
        booster = lgb.Booster(model_file=temp_file)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    return booster

models = {
    "landing": load_lgbm_model(deployed_heli, "landing"),
    "hoist":   load_lgbm_model(deployed_heli, "hoist"),
}

FIRE_STATION = {"latitude": 38.25, "longitude": 128.50}

# 10대 고정 추론 입력 피처 리스트
feature_columns = [
    'elevation', 'slope_deg', 'tree_density', 'tree_height',
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos',
    'land_0', 'land_1', 'land_2'
]


# ==============================================================================
# [STEP 3] 실시간 착륙/호이스트 신규 가중치 반영 전술 연산 커널
# ==============================================================================
def find_best_rescue_tactics(rescue_lat: float, rescue_lon: float,
                             heli_type: str, search_radius_meters: int = 500):
    print(f"\n[작전 개시] 구조 요청 지점 (위도: {rescue_lat:.5f}, 경도: {rescue_lon:.5f}) 실황 탐색...")

    candidates = df_master.copy()
    candidates['dist_to_rescue']    = np.sqrt((candidates['latitude'] - rescue_lat) ** 2 + (candidates['longitude'] - rescue_lon) ** 2)
    candidates['dist_from_base_km'] = np.sqrt((candidates['latitude'] - FIRE_STATION["latitude"]) ** 2 + (candidates['longitude'] - FIRE_STATION["longitude"]) ** 2) * 110.0

    # 🔥 [공간 가속 연산 패치]: 3000만 행 griddata 연산 오버헤드를 막기 위해 거리 반경 컷 선행 차단
    candidates = candidates[candidates['dist_to_rescue'] * 110000 <= search_radius_meters].copy()

    # 🎯 [스트레스 테스트 예외 처리 패치]
    # 사각지대 좌표 대입으로 데이터프레임이 빌 경우 최악 지형 조건 가상 격자를 주입하여 파이프라인 무결성을 보장합니다.
    if candidates.empty:
        print(f" ⚠️  안내: 입력하신 테스트 좌표 주변 반경 {search_radius_meters}m 이내에 실존 지형 격자가 없습니다.")
        print(f" ➔  [스트레스 테스트 모드 가동] 시스템 방어선 검증을 위해 가상의 최악 지형 노드를 생성해 파이프라인에 주입합니다.")
        
        synthetic_data = {
            'latitude': [rescue_lat, rescue_lat + 0.0002, rescue_lat - 0.0002],
            'longitude': [rescue_lon, rescue_lon - 0.0002, rescue_lon + 0.0002],
            'elevation': [1620.0, 1685.0, 1708.0],     # 아고산대 최고 위험 고도 강제 스케일링
            'slope_deg': [2, 2, 2],                     # 최악의 사면 등급
            'tree_density': [3, 3, 3],                  # 밀집 초과 수목 등급
            'tree_height': [2, 2, 2],                   # 최고 수목 높이
            'land_0': [1, 1, 1],
            'land_1': [0, 0, 0],
            'land_2': [0, 0, 0]
        }
        candidates = pd.DataFrame(synthetic_data)
        candidates['dist_to_rescue'] = np.sqrt((candidates['latitude'] - rescue_lat) ** 2 + (candidates['longitude'] - rescue_lon) ** 2)
        candidates['dist_from_base_km'] = np.sqrt((candidates['latitude'] - FIRE_STATION["latitude"]) ** 2 + (candidates['longitude'] - FIRE_STATION["longitude"]) ** 2) * 110.0
    else:
        # 하천 구역 배제 예외 처리 규칙 안전 통제 레이어
        candidates = candidates[candidates['land_2'] != 1].copy()

    if candidates.empty:
        print(f"조난 지점 반경 {search_radius_meters}m 이내에 유효 격자 데이터가 존재하지 않습니다.")
        return None

    # 🎯 [실시간 AWS 기상 인프라 다각 공간 보간 엔진 결합]
    try:
        live_weather = fetch_kma_realtime()
        if not live_weather or len(live_weather) < 2:
            raise ValueError("실시간 연동 가능한 외부 관측소 스트림 개수 부족")
            
        stn_points = np.array([[v["lon"], v["lat"]] for v in live_weather.values()])
        stn_ws     = np.array([v["ws"] for v in live_weather.values()])
        stn_wd     = np.array([v["wd"] for v in live_weather.values()])
        
        # 풍향 데이터 각도 단절 우회를 위한 삼각함수 공간 U, V 성분 벡터 찢기
        stn_wd_rad = np.radians(stn_wd)
        stn_u      = -np.sin(stn_wd_rad)
        stn_v      = -np.cos(stn_wd_rad)
        
        target_coords = (candidates['longitude'].values, candidates['latitude'].values)
        
        # 선형(Linear) 기하 격자 보간 연산 수행
        grid_ws = griddata(stn_points, stn_ws, target_coords, method="linear")
        grid_u  = griddata(stn_points, stn_u, target_coords, method="linear")
        grid_v  = griddata(stn_points, stn_v, target_coords, method="linear")
        
        # Convex Hull 외곽 경계 지역 결손(NaN) 발생 시 Nearest 보완 백업
        nan_mask = np.isnan(grid_ws)
        if np.any(nan_mask):
            grid_ws[nan_mask] = griddata(stn_points, stn_ws, target_coords, method="nearest")[nan_mask]
            grid_u[nan_mask]  = griddata(stn_points, stn_u, target_coords, method="nearest")[nan_mask]
            grid_v[nan_mask]  = griddata(stn_points, stn_v, target_coords, method="nearest")[nan_mask]
            
        # 벡터 공간에서 다시 0~360도 실치수 기상 각도로 아크탄젠트 복원
        candidates['wind_direction'] = np.degrees(np.arctan2(-grid_u, -grid_v)) % 360.0
        
        # 대기 경계층 보정 수식 결합
        candidates['wind_speed'] = apply_elevation_wind_correction(grid_ws, candidates['elevation'].values)
        print(" -> [AWS 기상 융합 완료] 실시간 관측 인프라 다각 선형 보간 및 대기 경계층 멱법칙 보정 완료.")
        
    except Exception as e:
        print(f"실시간 날씨 연산 에러({e}). 안전 우회 모드로 캐시 풍장을 가상 할당합니다.")
        candidates['wind_speed'] = 7.5
        candidates['wind_direction'] = 240.0

    # 풍향 공간 인코딩 및 실시간 진입 정풍 평점, 밀도고도 확장 연산
    candidates['wind_dir_rad'] = np.radians(candidates['wind_direction'])
    candidates['wind_dir_sin'] = np.sin(candidates['wind_dir_rad']).astype(np.float32)
    candidates['wind_dir_cos'] = np.cos(candidates['wind_dir_rad']).astype(np.float32)

    # 실시간 정풍 평점 동적 계산
    flight_heading = np.degrees(np.arctan2(candidates['longitude'] - FIRE_STATION["longitude"], candidates['latitude'] - FIRE_STATION["latitude"])) % 360
    angle_diff     = np.abs(flight_heading - candidates['wind_direction']) % 360
    candidates['wind_dir_score'] = np.select(
        [(angle_diff < 45) | (angle_diff >= 315), (angle_diff >= 135) & (angle_diff < 225)],
        [1.0, 0.0], default=0.5
    )

    # 실시간 밀도고도 평점 계산 (설악산 최고봉 1708m 기준 변곡 스케일링)
    candidates['altitude_score'] = 1.0 - (candidates['elevation'] / 1708.0) * 0.4

    # 기본 리스크 맵 인프라 연산
    candidates['wind_score'] = np.select(
        [candidates['wind_speed'] < 5.0, (candidates['wind_speed'] >= 5.0) & (candidates['wind_speed'] < 10.0), (candidates['wind_speed'] >= 10.0) & (candidates['wind_speed'] < 15.0)],
        [1.0, 0.6, 0.2], default=0.0
    )
    slope_map, density_map, height_map = {0: 1.0, 1: 0.7, 2: 0.35}, {0: 1.0, 1: 0.85, 2: 0.21, 3: 0.0}, {0: 1.0, 1: 0.61, 2: 0.31}
    candidates['slope_score']        = candidates['slope_deg'].map(slope_map)
    candidates['tree_density_score'] = candidates['tree_density'].map(density_map)
    candidates['tree_height_score']  = candidates['tree_height'].map(height_map)

    # 지형/기상 가중치 테이블 세부 연산
    if heli_type == "small":
        candidates['score_landing'] = (
            candidates['slope_score']        * 0.42 + candidates['tree_density_score'] * 0.10 +
            candidates['tree_height_score']  * 0.08 + candidates['wind_score']         * 0.20 +
            candidates['wind_dir_score']     * 0.12 + candidates['altitude_score']     * 0.08
        )
        candidates['score_hoist'] = (
            candidates['slope_score']        * 0.12 + candidates['tree_density_score'] * 0.15 +
            candidates['tree_height_score']  * 0.20 + candidates['wind_score']         * 0.30 +
            candidates['wind_dir_score']     * 0.15 + candidates['altitude_score']     * 0.08
        )
    else:  # large
        candidates['score_landing'] = (
            candidates['slope_score']        * 0.46 + candidates['tree_density_score'] * 0.14 +
            candidates['tree_height_score']  * 0.08 + candidates['wind_score']         * 0.12 +
            candidates['wind_dir_score']     * 0.08 + candidates['altitude_score']     * 0.12
        )
        candidates['score_hoist'] = (
            candidates['slope_score']        * 0.10 + candidates['tree_density_score'] * 0.24 +
            candidates['tree_height_score']  * 0.18 + candidates['wind_score']         * 0.18 +
            candidates['wind_dir_score']     * 0.12 + candidates['altitude_score']     * 0.18
        )

    # ── LightGBM Core 추론 연산 가동 ──────────────────────────────────────────
    X_candidates = candidates[feature_columns].astype(np.float32).values

    # 클래스별 확률 행렬(N, 3) 스캔 수집
    prob_landing = models["landing"].predict(X_candidates)
    prob_hoist   = models["hoist"].predict(X_candidates)

    # 확률 매트릭스 argmax 차원 디코딩 결착 (0, 1, 2)
    candidates['pred_landing'] = np.argmax(prob_landing, axis=1).astype(np.int32)
    candidates['pred_hoist']   = np.argmax(prob_hoist, axis=1).astype(np.int32)

    # 반경 50m 이격 안전 보장 공간 다양성 필터 정의
    def filter_spatial_diversity(sorted_df: pd.DataFrame, min_sep_deg: float = 0.000455) -> pd.DataFrame:
        selected = []
        for _, row in sorted_df.iterrows():
            if not selected:
                selected.append(row)
            else:
                if all(np.sqrt((row['latitude'] - s['latitude'])**2 + (row['longitude'] - s['longitude'])**2) >= min_sep_deg for s in selected):
                    selected.append(row)
            if len(selected) == 3:  
                break
        return pd.DataFrame(selected) if selected else pd.DataFrame()

    best_landing = filter_spatial_diversity(candidates.sort_values(by=['pred_landing', 'score_landing', 'dist_to_rescue'], ascending=[True, False, True]))
    best_hoist   = filter_spatial_diversity(candidates.sort_values(by=['pred_hoist', 'score_hoist', 'dist_to_rescue'], ascending=[True, False, True]))

    target_labels = {0: "안전 - 작전 원활", 1: "주의 - 조건부 작전", 2: "위험 - 작전 불가"}
    w_map         = {1.0: '정풍', 0.0: '역풍', 0.5: '측풍'}

    print("\n" + "="*115)
    print(f"[산악 구조 헬기 전술 통제 시스템 - LightGBM] 실시간 관제 리포트 - 기종: {heli_kor_name}")
    print("="*115)

    # A안 출력
    print(f"\n[A안: 기체 직접 안착(Landing) 추천 좌표 목록 (최대 3순위)]")
    print("-" * 115)
    if not best_landing.empty:
        for rank, (_, r) in enumerate(best_landing.iterrows(), 1):
            w_rel = w_map.get(r['wind_dir_score'], '측풍')
            print(f" {rank}순위 추천지 -> 좌표: ({r['latitude']:.5f}, {r['longitude']:.5f})")
            print(f"    [전술 거리] 조난자까지: 약 {r['dist_to_rescue'] * 110000:.1f}m | 소방기지로부터: 약 {r['dist_from_base_km']:.2f}km")
            print(f"    [AI 안전성] 등급: {target_labels[int(r['pred_landing'])]} | 전술 적합도 점수: {r['score_landing']:.4f}점")
            print(f"    [현장 실황] 풍속: {r['wind_speed']:.1f}m/s ({w_rel}) | 실효고도: {r['elevation']:.1f}m (밀도리스크점수: {r['altitude_score']:.2f}점)")
            print("-" * 80)
    else:
        print(" 현재 조건 및 지형 제약 하에 안전한 기체 안착 격자가 없습니다.")

    # B안 출력
    print(f"\n[B안: 제자리 비행 호이스트(Hoist) 강하 추천 좌표 목록 (최대 3순위)]")
    print("-" * 115)
    if not best_hoist.empty:
        for rank, (_, r) in enumerate(best_hoist.iterrows(), 1):
            w_rel = w_map.get(r['wind_dir_score'], '측풍')
            orig_land = 0 if r.get('land_0', 0) == 1 else (1 if r.get('land_1', 0) == 1 else 2)
            print(f" {rank}순위 구조지 -> 좌표: ({r['latitude']:.5f}, {r['longitude']:.5f})")
            print(f"    [전술 거리] 조난자까지: 약 {r['dist_to_rescue'] * 110000:.1f}m | 소방기지로부터: 약 {r['dist_from_base_km']:.2f}km")
            print(f"    [AI 안전성] 등급: {target_labels[int(r['pred_hoist'])]} | 전술 적합도 점수: {r['score_hoist']:.4f}점")
            print(f"    [현장 실황] 풍속: {r['wind_speed']:.1f}m/s ({w_rel}) | 수목 높이등급: {int(r['tree_height'])} | 수목밀도: {int(r['tree_density'])} | 지형형태: {orig_land} | 실효고도: {r['elevation']:.1f}m")
            print("-" * 80)
    else:
        print(" 현재 조건 및 지형 제약 하에 안전한 호이스트 작전 공간이 없습니다.")

    print("\n" + "="*115)
    return best_landing, best_hoist


# ==============================================================================
# 고정된 최악 지점 가상 좌표 대입 스트레스 테스트 가동
# ==============================================================================
sample_rescue_point = df_master.sample(1)
example_lat = sample_rescue_point['latitude'].values[0]
example_lon = sample_rescue_point['longitude'].values[0]

print(f"[검증 가동] 지정된 극한의 테스트 구조 요청 지점(Stress Test):")
print(f"위도: {example_lat:.5f}, 경도: {example_lon:.5f}")

l_spots, h_spots = find_best_rescue_tactics(
    rescue_lat=example_lat,
    rescue_lon=example_lon,
    heli_type=deployed_heli,
    search_radius_meters=500
)