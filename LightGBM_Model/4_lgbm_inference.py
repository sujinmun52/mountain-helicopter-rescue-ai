"""
LightGBM 듀얼 추론 엔진 기반 실시간 산악 구조 관제 및 전술 추천 스크립트
"""
import os
import sys
import shutil
from dotenv import load_dotenv

# ── 1. 인프라 및 경로 설정 ──────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, '.env'), override=True)
_PROJECT_ROOT = os.path.dirname(_DIR)

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _DIR)

import numpy as np
import pandas as pd
import lightgbm as lgb
from weather.modules.weather import fetch_kma_realtime
from weather.modules.data_preprocessing import mapping_live_weather_to_grid

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

# 기존 대용량 CSV 대신 전처리된 경량화 Parquet 레이어 파일 바인딩
df_master = pd.read_parquet(os.path.join(_DIR, 'dataset', 'terrain_base.parquet'))

def load_lgbm_model(model_key: str, tactic: str) -> lgb.Booster:
    """LightGBM 네이티브 포맷(.txt) 모델을 Windows 한글 경로 버그를 우회하여 로드합니다."""
    path = os.path.join(_DIR, 'models', f'lgbm_{model_key}_{tactic}_model.txt')
    if not os.path.exists(path):
        sys.exit(f"에러: 모델 파일을 찾을 수 없습니다. 경로를 확인하세요: {path}")
    
    # [핵심 패치] 한글 경로 문자열 파괴 방지를 위한 파일 시스템 레이어 우회
    # 파이썬 커널을 이용해 한글이 없는 안전한 파일명으로 복사본을 생성합니다.
    temp_file = f"temp_load_lgbm_{model_key}_{tactic}.txt"
    shutil.copyfile(path, temp_file)
    
    try:
        # C++ 코어 엔진에는 영문 임시 파일 경로만 넘겨주어 입출력 오류를 원천 차단합니다.
        booster = lgb.Booster(model_file=temp_file)
    finally:
        # 인메모리에 모델 구조가 탑재된 직후, 생성했던 임시 복사 파일은 즉시 삭제합니다.
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    return booster

models = {
    "landing": load_lgbm_model(deployed_heli, "landing"),
    "hoist":   load_lgbm_model(deployed_heli, "hoist"),
}

FIRE_STATION = {"latitude": 38.25, "longitude": 128.50}

# 12대 고정 추론 입력 피처 리스트
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

    try:
        live_weather = fetch_kma_realtime()
        candidates = mapping_live_weather_to_grid(rescue_lat, rescue_lon, live_weather, df_master, radius_km=5.0)
    except Exception as e:
        print(f"실시간 날씨 연산 에러({e}). 마스터 캐시 데이터로 우회 연산합니다.")
        candidates = df_master.copy()

    if candidates.empty or ("wind_speed" not in candidates.columns):
        print("반경 내 유효 지형 데이터가 없거나 기상 피처가 유실되었습니다.")
        return None

    # 풍향 공간 인코딩 및 실시간 진입 정풍 평점, 밀도고도 확장 연산
    candidates['wind_direction'] = candidates['wind_direction'].fillna(0.0)
    candidates['wind_dir_rad']   = np.radians(candidates['wind_direction'])
    candidates['wind_dir_sin']   = np.sin(candidates['wind_dir_rad'])
    candidates['wind_dir_cos']   = np.cos(candidates['wind_dir_rad'])

    candidates['dist_to_rescue']    = np.sqrt((candidates['latitude'] - rescue_lat) ** 2 + (candidates['longitude'] - rescue_lon) ** 2)
    candidates['dist_from_base_km'] = np.sqrt((candidates['latitude'] - FIRE_STATION["latitude"]) ** 2 + (candidates['longitude'] - FIRE_STATION["longitude"]) ** 2) * 110.0

    # 1차 반경 필터링 (설정한 미터 범위 내 격자만 슬라이싱)
    candidates = candidates[candidates['dist_to_rescue'] * 110000 <= search_radius_meters].copy()

    # [하천 구역 안전 차단 패치] land_type이 2(land_2 == 1)인 모든 격자를 후보군에서 원천 배제
    candidates = candidates[candidates['land_2'] != 1].copy()

    if candidates.empty:
        print(f"조난 지점 반경 {search_radius_meters}m 이내에 하천 구역을 제외한 유효 격자 데이터가 존재하지 않습니다.")
        return None

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
            candidates['tree_height_score']  * 0.18 + candidates['wind_speed']         * 0.18 +
            candidates['wind_dir_score']     * 0.12 + candidates['altitude_score']     * 0.18
        )

    # ── LightGBM Core 추론 연산 가동 ──────────────────────────────────────────
    X_candidates = candidates[feature_columns].astype(np.float32).values

    # LightGBM Booster.predict() 결과는 클래스별 확률 행렬(N, 3)을 반환함
    prob_landing = models["landing"].predict(X_candidates)
    prob_hoist   = models["hoist"].predict(X_candidates)

    # 확률 행렬에서 가장 값이 큰 인덱스를 클래스 레이블(0, 1, 2)로 최종 확정 (argmax 필수)
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
            if len(selected) == 3:  # 최대 3순위까지만 수집 후 조기 중단
                break
        return pd.DataFrame(selected) if selected else pd.DataFrame()

    best_landing = filter_spatial_diversity(candidates.sort_values(by=['pred_landing', 'score_landing', 'dist_to_rescue'], ascending=[True, False, True]))
    best_hoist   = filter_spatial_diversity(candidates.sort_values(by=['pred_hoist', 'score_hoist', 'dist_to_rescue'], ascending=[True, False, True]))

    target_labels = {0: "안전 - 작전 원활", 1: "주의 - 조건부 작전", 2: "위험 - 작전 불가"}
    w_map         = {1.0: '정풍', 0.0: '역풍', 0.5: '측풍'}

    print("\n" + "="*115)
    print(f"[산악 구조 헬기 전술 통제 시스템 - LightGBM] 실시간 관제 리포트 - 기종: {heli_kor_name}")
    print("="*115)

    # A안 출력: 3순위 안착 착륙 후보지 리포트 통합 루프
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

    # B안 출력: 3순위 호이스트 강하 후보지 리포트 통합 루프
    print(f"\n[B안: 제자리 비행 호이스트(Hoist) 강하 추천 좌표 목록 (최대 3순위)]")
    print("-" * 115)
    if not best_hoist.empty:
        for rank, (_, r) in enumerate(best_hoist.iterrows(), 1):
            w_rel = w_map.get(r['wind_dir_score'], '측풍')
            orig_land = 0 if r.get('land_0', 0) == 1 else (1 if r.get('land_1', 0) == 1 else 2)
            print(f" {rank}순위 구조지 -> 좌표: ({r['latitude']:.5f}, {r['longitude']:.5f})")
            print(f"    [전술 거리] 조난자까지: 약 {r['dist_to_rescue'] * 110000:.1f}m | 소방기지로부터: 약 {r['dist_from_base_km']:.2f}km")
            print(f"    [AI 안전성] 등급: {target_labels[int(r['pred_hoist'])]} | 전술 적합도 점수: {r['score_hoist']:.4f}점")
            print(f"    [현장 실황] 풍속: {r['wind_speed']:.1f}m/s | 수목 높이등급: {int(r['tree_height'])} | 수목밀도: {int(r['tree_density'])} | 지형형태: {orig_land}")
            print("-" * 80)
    else:
        print(" 현재 조건 및 지형 제약 하에 안전한 호이스트 작전 공간이 없습니다.")

    print("\n" + "="*115)
    return best_landing, best_hoist


# ==============================================================================
# 무작위 조난 지점 샘플링 후 시뮬레이션 기동
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