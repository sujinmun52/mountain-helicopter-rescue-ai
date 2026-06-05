import os
import sys
from dotenv import load_dotenv

# 1. 인프라 및 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, '.env'), override=True)
project_root = os.path.dirname(current_dir)
config_candidates = [os.path.join(current_dir, 'weather'), os.path.join(project_root, 'weather'), current_dir, project_root]

for folder in config_candidates:
    if os.path.exists(os.path.join(folder, 'config.py')):
        sys.path.insert(0, folder)
        break
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

import pandas as pd
import numpy as np
import xgboost as xgb
from weather.modules.weather import fetch_kma_realtime
from weather.modules.data_preprocessing import mapping_live_weather_to_grid

# ==============================================================================
# [STEP 1] 출격 헬기 기종 선택 (소형 / 대형 고정 관제)
# ==============================================================================
print("\n" + "="*70)
print("=== [산악 구조 관제 시스템 - XGBoost GPU] 출격 기체 제원 설정 ===")
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
    print("\n에러: 1번(소형) 또는 2번(대형) 중 하나만 선택해야 합니다. 프로그램을 종료합니다.")
    sys.exit()

print(f"\n[기종 고정 완료] 작전 기체: 【 {heli_kor_name} 】")
print("해당 기체 전용 신규 가중치 기반 XGBoost 듀얼 AI 추론 엔진을 로드합니다.\n")


# ==============================================================================
# 📌 [STEP 2] 마스터 데이터 및 신규 피처 대응 구조 모델 로드
# ==============================================================================
print("데이터 매트릭스 및 XGBoost 모델 인프라 로드 중")
df_master = pd.read_csv(r'RF_Model\dataset\processed_seoraksan_master.csv')

def load_xgb_model(model_key: str, tactic: str) -> xgb.Booster:
    path = fr'RF_Model\xgb_{model_key}_{tactic}_model.ubj'
    booster = xgb.Booster()
    booster.load_model(path)
    booster.set_param({"device": "cuda"})  # GPU 실시간 추론 바인딩
    return booster

models = {
    "landing": load_xgb_model(deployed_heli, "landing"),
    "hoist":   load_xgb_model(deployed_heli, "hoist"),
}

FIRE_STATION = {"latitude": 38.25, "longitude": 128.50}

# 새 항공 역학 레이어 가중치 학습에 쓰인 12대 추론 변수 리스트
feature_columns = [
    'elevation', 'slope_deg', 'tree_density', 'tree_height',
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos', 'wind_dir_score', 'altitude_score',
    'land_0', 'land_1', 'land_2'
]


# ==============================================================================
# 🚀 [STEP 3] 실시간 착륙/호이스트 신규 가중치 반영 전술 연산 커널
# ==============================================================================
def find_best_rescue_tactics(rescue_lat: float, rescue_lon: float,
                             heli_type: str, search_radius_meters: int = 500):
    print(f"\n[작전 개시] 구조 요청 지점 (위도: {rescue_lat:.5f}, 경도: {rescue_lon:.5f}) 실황 탐색...")

    try:
        live_weather = fetch_kma_realtime()
        candidates = mapping_live_weather_to_grid(rescue_lat, rescue_lon, live_weather, df_master, radius_km=5.0)
    except Exception as e:
        print(f"실시간 날씨 연산 에러({e}). 마스터 캐시 데이터로 우회 연산합니다.")
        search_radius_deg = (search_radius_meters / 110.0) * 0.001
        candidates = df_master[
            (df_master['latitude']  >= rescue_lat  - search_radius_deg) &
            (df_master['latitude']  <= rescue_lat  + search_radius_deg) &
            (df_master['longitude'] >= rescue_lon  - search_radius_deg) &
            (df_master['longitude'] <= rescue_lon  + search_radius_deg)
        ].copy()

    if candidates.empty or ("wind_speed" not in candidates.columns):
        print("반경 내 유효 지형 데이터가 없거나 기상 피처가 유실되었습니다.")
        return None

    # 풍향 공간 인코딩 및 실시간 진입 정풍 평점, 밀도고도 확장 연산
    candidates['wind_direction']  = candidates['wind_direction'].fillna(0.0)
    candidates['wind_dir_rad']    = np.radians(candidates['wind_direction'])
    candidates['wind_dir_sin']    = np.sin(candidates['wind_dir_rad'])
    candidates['wind_dir_cos']    = np.cos(candidates['wind_dir_rad'])

    candidates['dist_to_rescue']    = np.sqrt((candidates['latitude'] - rescue_lat) ** 2 + (candidates['longitude'] - rescue_lon) ** 2)
    candidates['dist_from_base_km'] = np.sqrt((candidates['latitude'] - FIRE_STATION["latitude"]) ** 2 + (candidates['longitude'] - FIRE_STATION["longitude"]) ** 2) * 110.0

    # 실시간 정풍 평점 동적 계산
    delta_lat      = candidates['latitude']  - FIRE_STATION["latitude"]
    delta_lon      = candidates['longitude'] - FIRE_STATION["longitude"]
    flight_heading = np.degrees(np.arctan2(delta_lon, delta_lat)) % 360
    angle_diff     = np.abs(flight_heading - candidates['wind_direction']) % 360
    candidates['wind_dir_score'] = np.select(
        [(angle_diff < 45) | (angle_diff >= 315), (angle_diff >= 135) & (angle_diff < 225)],
        [1.0, 0.0], default=0.5
    )

    # 실시간 밀도고도 평점 계산 (설악산 최고봉 1708m 기준)
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

    # XGBoost GPU 매트릭스 추론 가동 (12대 입력 변수)
    X_candidates = candidates[feature_columns].astype(np.float32)
    d_candidates = xgb.DMatrix(X_candidates)

    candidates['pred_landing'] = models["landing"].predict(d_candidates).astype(np.int32)
    candidates['pred_hoist']   = models["hoist"].predict(d_candidates).astype(np.int32)

    # 🎯 반경 50m 이격 안전 보장 필터 (물리적 거리 기준 정밀 차단)
    # 50m를 위경도 평면상의 유클리드 거리 임계치로 변환: 50 / 110,000 m = 약 0.0004545도
    def filter_spatial_diversity(sorted_df: pd.DataFrame, min_sep_deg: float = 0.000455) -> pd.DataFrame:
        selected = []
        for _, row in sorted_df.iterrows():
            if not selected:
                selected.append(row)
            else:
                # 하위 순위 지점이 기존에 확정된 상위 지점들의 반경 50m 이내에 걸리는지 동적 검사
                if all(np.sqrt((row['latitude'] - s['latitude'])**2 + (row['longitude'] - s['longitude'])**2) >= min_sep_deg for s in selected):
                    selected.append(row)
            if len(selected) == 3:  # 최대 3순위까지만 수집 후 조기 중단
                break
        return pd.DataFrame(selected) if selected else pd.DataFrame()

    best_landing = filter_spatial_diversity(candidates.sort_values(by=['pred_landing', 'score_landing', 'dist_to_rescue'], ascending=[True, False, True]))
    best_hoist   = filter_spatial_diversity(candidates.sort_values(by=['pred_hoist', 'score_hoist', 'dist_to_rescue'], ascending=[True, False, True]))

    target_labels = {0: "안전 - 작전 원활", 1: "주의 - 조건부 작전", 2: "위험 - 작전 불가"}
    w_map         = {1.0: '정풍', 0.0: '역풍', 0.5: '측풍'}

    print("\n" + "="*115)
    print(f"[산악 구조 헬기 전술 통제 시스템] 실시간 관제 리포트 - 기종: {heli_kor_name}")
    print("="*115)

    # 🎯 A안 출력: 3순위 안착 착륙 후보지 리포트 통합 루프
    print(f"\n[A안: 기체 직접 안착(Landing) 추천 좌표 목록 (최대 3순위)]")
    print("-" * 115)
    if not best_landing.empty:
        for rank, (_, r) in enumerate(best_landing.iterrows(), 1):
            w_rel = w_map.get(r['wind_relation_score'], '측풍')
            print(f" {rank}순위 추천지 -> 좌표: ({r['latitude']:.5f}, {r['longitude']:.5f})")
            print(f"    [전술 거리] 조난자까지: 약 {r['dist_to_rescue'] * 110000:.1f}m | 소방기지로부터: 약 {r['dist_from_base_km']:.2f}km")
            print(f"    [AI 안전성] 등급: {target_labels[int(r['pred_landing'])]} | 전술 적합도 점수: {r['score_landing']:.4f}점")
            print(f"    [현장 실황] 풍속: {r['wind_speed']:.1f}m/s ({w_rel}) | 실효고도: {r['elevation']:.1f}m (밀도리스크점수: {r['altitude_score']:.2f}점)")
            print("-" * 80)
    else:
        print(" 현재 조건 하에 안전한 기체 안착 격자가 없습니다.")

    # 🎯 B안 출력: 3순위 호이스트 강하 후보지 리포트 통합 루프
    print(f"\n[B안: 제자리 비행 호이스트(Hoist) 강하 추천 좌표 목록 (최대 3순위)]")
    print("-" * 115)
    if not best_hoist.empty:
        for rank, (_, r) in enumerate(best_hoist.iterrows(), 1):
            w_rel = w_map.get(r['wind_relation_score'], '측풍')
            orig_land = 0 if r.get('land_0', 0) == 1 else (1 if r.get('land_1', 0) == 1 else 2)
            print(f" {rank}순위 구조지 -> 좌표: ({r['latitude']:.5f}, {r['longitude']:.5f})")
            print(f"    [전술 거리] 조난자까지: 약 {r['dist_to_rescue'] * 110000:.1f}m | 소방기지로부터: 약 {r['dist_from_base_km']:.2f}km")
            print(f"    [AI 안전성] 등급: {target_labels[int(r['pred_hoist'])]} | 전술 적합도 점수: {r['score_hoist']:.4f}점")
            print(f"    [현장 실황] 풍속: {r['wind_speed']:.1f}m/s | 수목 높이등급: {int(r['tree_height'])} (수직클리어런스 확보) | 수목밀도: {int(r['tree_density'])} | 지형형태: {orig_land}")
            print("-" * 80)
    else:
        print(" 현재 조건 하에 안전한 호이스트 작전 공간이 없습니다.")

    print("\n" + "="*115)
    return best_landing, best_hoist


# ==============================================================================
# 무작위 조난 지점 샘플링 후 작전 가동
# ==============================================================================
sample_rescue_point = df_master.sample(1)
example_lat = sample_rescue_point['latitude'].values[0]
example_lon = sample_rescue_point['longitude'].values[0]

print(f"[무작위 매칭] 마스터 맵에서 추출된 실제 테스트 구조 지점:")
print(f"위도: {example_lat:.5f}, 경도: {example_lon:.5f}")

l_spots, h_spots = find_best_rescue_tactics(
    rescue_lat=example_lat,
    rescue_lon=example_lon,
    heli_type=deployed_heli,
    search_radius_meters=500
)