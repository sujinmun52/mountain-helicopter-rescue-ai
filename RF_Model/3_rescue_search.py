import os
import sys
from dotenv import load_dotenv

# 1. 인프라 및 경로 설정 (자동 탐색 및 절대 경로 바인딩)
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
import joblib
from weather.modules.weather import fetch_kma_realtime
from weather.modules.data_preprocessing import mapping_live_weather_to_grid

# ==============================================================================
# [STEP 1] 작전 출격 헬기 기종 선택 인터페이스 (소형 / 대형 고정)
# ==============================================================================
print("\n" + "="*70)
print("[산악 구조 관제 시스템 - Random Forest] 출격 기체 제원 설정")
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

print(f"\n[기종 고정 완료] 작전 기체 명세: 【 {heli_kor_name} 】")
print("이에 따라 해당 기체 전용 듀얼 AI 전술 추론 엔진을 로드합니다.\n")


# ==============================================================================
# [STEP 2] 인프라 및 선택 기종 전용 AI 모델 파일 탑재
# ==============================================================================
print("인프라 설정에 맞춰 데이터 및 제원별 AI 모델 로드 중...")

# 🎯 [경로 최적화]: 마스터 데이터 로드 절대 경로 매핑
master_dataset_path = os.path.join(current_dir, 'dataset', 'processed_seoraksan_master_rf.csv')
df_master = pd.read_csv(master_dataset_path)

# 🎯 [경로 최적화]: 모델 직렬화 파일 로드 절대 경로 매핑
landing_model_path = os.path.join(current_dir, f'rf_{deployed_heli}_landing_model.joblib')
hoist_model_path = os.path.join(current_dir, f'rf_{deployed_heli}_hoist_model.joblib')

models = {
    "landing": joblib.load(landing_model_path),
    "hoist":   joblib.load(hoist_model_path)
}

# 119 소방구급센터 고정 좌표 성분
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50}

# 🎯 [피처 동기화]: 학습 단계와 일치하도록 수목 데이터 2종을 포함한 완전한 10대 독립 변수 리스트 선언
feature_columns = [
    'elevation', 'slope_deg', 'tree_density', 'tree_height',
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos',
    'land_0', 'land_1', 'land_2'
]


# ==============================================================================
# [STEP 3] 선택 기종 기반 착륙/호이스트 듀얼 전술 연산 커널
# ==============================================================================
def find_best_rescue_tactics(rescue_lat, rescue_lon, heli_type, search_radius_meters=500):
    print(f"\n[작전 개시] 구조 요청 지점 (위도: {rescue_lat:.5f}, 경도: {rescue_lon:.5f}) 주변 실황 탐색 시작...")
    
    try:
        live_weather = fetch_kma_realtime()
        candidates = mapping_live_weather_to_grid(rescue_lat, rescue_lon, live_weather, df_master, radius_km=5.0)
    except Exception as e:
        print(f"실시간 날씨 연산 에러({e}). 마스터본에 캐싱된 데이터 기반으로 우회 연산합니다.")
        candidates = df_master.copy()

    if candidates.empty or ("wind_speed" not in candidates.columns):
        print("반경 내 유효 지형 데이터가 없거나 기상 피처가 유실되었습니다.")
        return None

    # 풍향 데이터 삼각함수 공간 주기성 인코딩
    candidates['wind_direction'] = candidates['wind_direction'].fillna(0.0)
    candidates['wind_dir_rad'] = np.radians(candidates['wind_direction'])
    candidates['wind_dir_sin'] = np.sin(candidates['wind_dir_rad'])
    candidates['wind_dir_cos'] = np.cos(candidates['wind_dir_rad'])
    
    # 조난자 및 베이스 기지 거리 연산
    candidates['dist_to_rescue'] = np.sqrt((candidates['latitude'] - rescue_lat)**2 + (candidates['longitude'] - rescue_lon)**2)
    candidates['dist_from_base_km'] = np.sqrt((candidates['latitude'] - FIRE_STATION["latitude"])**2 + (candidates['longitude'] - FIRE_STATION["longitude"])**2) * 110.0
    
    # 작전 영역 제한 (반경 필터링)
    candidates = candidates[candidates['dist_to_rescue'] * 110000 <= search_radius_meters].copy()

    # 하천 구역 배제 예외 규칙 적용
    candidates = candidates[candidates['land_2'] != 1].copy()

    if candidates.empty:
        print(f"조난 지점 반경 {search_radius_meters}m 이내에 하천 구역을 제외한 유효 격자 데이터가 존재하지 않습니다.")
        return None

    # 🎯 [알고리즘 동기화]: 실시간 정풍/역풍 평점 연산 (`wind_dir_score`)
    delta_lat = candidates['latitude'] - FIRE_STATION["latitude"]
    delta_lon = candidates['longitude'] - FIRE_STATION["longitude"]
    flight_heading = np.degrees(np.arctan2(delta_lon, delta_lat)) % 360
    angle_diff = np.abs(flight_heading - candidates['wind_direction']) % 360
    candidates['wind_dir_score'] = np.select(
        [(angle_diff < 45) | (angle_diff >= 315), (angle_diff >= 135) & (angle_diff < 225)],
        [1.0, 0.0], default=0.5
    ).astype('float32')

    # 🎯 [알고리즘 동기화]: 실시간 풍속 평점 연산 (`wind_score`)
    candidates['wind_score'] = np.select(
        [candidates['wind_speed'] < 5.0,
         (candidates['wind_speed'] >= 5.0) & (candidates['wind_speed'] < 10.0),
         (candidates['wind_speed'] >= 10.0) & (candidates['wind_speed'] < 15.0)],
        [1.0, 0.6, 0.2], default=0.0
    ).astype('float32')

    # 🎯 [알고리즘 동기화]: 도메인 학술적 3단계 고도 구간 라벨링 및 스코어 매핑 수식 주입
    ELEVATION_MAP = {0: 1.0, 1: 0.70, 2: 0.35}
    elevation_grade = np.select(
        [candidates['elevation'] < 500,
         (candidates['elevation'] >= 500) & (candidates['elevation'] < 1200)],
        [0, 1], default=2
    )
    candidates['elevation_score'] = np.vectorize(ELEVATION_MAP.get)(elevation_grade).astype('float32')
    
    # 지형인프라 점수 매핑 구조 복원
    slope_map   = {0: 1.0, 1: 0.70, 2: 0.35}
    density_map = {0: 1.0, 1: 0.85, 2: 0.21, 3: 0.00}
    height_map  = {0: 1.0, 1: 0.61, 2: 0.31}
    candidates['slope_score']        = candidates['slope_deg'].map(slope_map).astype('float32')
    candidates['tree_density_score'] = candidates['tree_density'].map(density_map).astype('float32')
    candidates['tree_height_score']  = candidates['tree_height'].map(height_map).astype('float32')
    
    # 🎯 [알고리즘 동기화]: 개편된 전문가 의사결정 수식 테이블 독립 반영 (Static/Dynamic 완전 결합)
    if heli_type == "small":
        candidates['score_landing'] = (
            candidates['slope_score']        * 0.42 + candidates['tree_density_score'] * 0.10 +
            candidates['tree_height_score']  * 0.08 + candidates['wind_score']         * 0.20 +
            candidates['wind_dir_score']     * 0.12 + candidates['elevation_score']    * 0.08
        )
        candidates['score_hoist'] = (
            candidates['slope_score']        * 0.12 + candidates['tree_density_score'] * 0.15 +
            candidates['tree_height_score']  * 0.20 + candidates['wind_score']         * 0.30 +
            candidates['wind_dir_score']     * 0.15 + candidates['elevation_score']    * 0.08
        )
    else:  # large
        candidates['score_landing'] = (
            candidates['slope_score']        * 0.46 + candidates['tree_density_score'] * 0.14 +
            candidates['tree_height_score']  * 0.08 + candidates['wind_score']         * 0.12 +
            candidates['wind_dir_score']     * 0.08 + candidates['elevation_score']    * 0.12
        )
        candidates['score_hoist'] = (
            candidates['slope_score']        * 0.10 + candidates['tree_density_score'] * 0.24 +
            candidates['tree_height_score']  * 0.18 + candidates['wind_score']         * 0.18 +
            candidates['wind_dir_score']     * 0.12 + candidates['elevation_score']    * 0.18
        )

    # 4. 실시간 듀얼 모델 병렬 추론 (10대 피처 정합성 확보 완료)
    X_candidates = candidates[feature_columns].astype(np.float32)
    candidates['pred_landing'] = models["landing"].predict(X_candidates)
    candidates['pred_hoist'] = models["hoist"].predict(X_candidates)
    
    # 55m 공간 이격 안전 보장 알고리즘
    def filter_spatial_diversity(sorted_df, min_sep_deg=0.0005):
        selected_spots = []
        for _, row in sorted_df.iterrows():
            if len(selected_spots) == 0:
                selected_spots.append(row)
            else:
                is_diverse = True
                for selected in selected_spots:
                    dist = np.sqrt((row['latitude'] - selected['latitude'])**2 + (row['longitude'] - selected['longitude'])**2)
                    if dist < min_sep_deg:
                        is_diverse = False
                        break
                if is_diverse:
                    selected_spots.append(row)
            if len(selected_spots) == 3:
                break
        return pd.DataFrame(selected_spots) if selected_spots else pd.DataFrame()

    best_landing = filter_spatial_diversity(candidates.sort_values(by=['pred_landing', 'score_landing', 'dist_to_rescue'], ascending=[True, False, True]))
    best_hoist   = filter_spatial_diversity(candidates.sort_values(by=['pred_hoist', 'score_hoist', 'dist_to_rescue'], ascending=[True, False, True]))

    # 5. 선택 기종 집중 관제 리포트 종합 출력
    target_labels = {0: "안전 - 작전 원활", 1: "주의 - 조건부 작전", 2: "위험 - 작전 불가"}
    w_map = {1.0: '정풍', 0.0: '역풍', 0.5: '측풍'}
    
    print("\n" + "="*115)
    print(f"[산악 구조 헬기 전술 통제 시스템] 실시간 관제 리포트 - 기종: {heli_kor_name}")
    print("="*115)
    
    # A안 출력
    print(f"\n[A안: 기체 직접 안착(Landing) 추천 좌표 목록 (최대 3순위)]")
    print("-" * 115)
    if not best_landing.empty:
        for rank, (_, r) in enumerate(best_landing.iterrows(), 1):
            w_rel = w_map.get(r['wind_dir_score'], '측풍')
            print(f" {rank}순위 추천지 -> 좌표: ({r['latitude']:.5f}, {r['longitude']:.5f})")
            print(f"    [전술 거리] 조난자까지: 약 {r['dist_to_rescue'] * 110000:.1f}m | 소방기지로부터: 약 {r['dist_from_base_km']:.2f}km")
            print(f"    [AI 안전성] 등급: {target_labels[int(r['pred_landing'])]} | 통합 전술 적합도 점수: {r['score_landing']:.4f}점")
            # 🎯 [UI/UX 정돈]: 지형고도점수를 걷어내고 직관적인 계측고도(m)만 출력
            print(f"    [현장 기상] 풍속: {r['wind_speed']:.1f}m/s ({w_rel}) | 고도: {r['elevation']:.1f}m | 경사등급: {int(r['slope_deg'])}")
            print("-" * 80)
    else:
        print(" 현재 현장 기상 및 경사도 조건 하에 안전한 기체 안착 격자가 존재하지 않습니다.")

    # B안 출력
    print(f"\n[B안: 제자리 비행 호이스트(Hoist) 강하 추천 좌표 목록 (최대 3순위)]")
    print("-" * 115)
    if not best_hoist.empty:
        for rank, (_, r) in enumerate(best_hoist.iterrows(), 1):
            w_rel = w_map.get(r['wind_dir_score'], '측풍')
            print(f" {rank}순위 구조지 -> 좌표: ({r['latitude']:.5f}, {r['longitude']:.5f})")
            print(f"    [전술 거리] 조난자까지: 약 {r['dist_to_rescue'] * 110000:.1f}m | 소방기지로부터: 약 {r['dist_from_base_km']:.2f}km")
            print(f"    [AI 안전성] 등급: {target_labels[int(r['pred_hoist'])]} | 통합 전술 적합도 점수: {r['score_hoist']:.4f}점")
            orig_land = 0 if r.get('land_0', 0) == 1 else (1 if r.get('land_1', 0) == 1 else 2)
            # 🎯 [UI/UX 정돈]: 후미부의 군더더기를 제거하고 깔끔하게 계측고도(m)로 가독성 고도화
            print(f"    [현장 기상] 풍속: {r['wind_speed']:.1f}m/s ({w_rel}) | 고도: {r['elevation']:.1f}m | 수목밀도등급: {int(r['tree_density'])} | 지형형태: {orig_land}")
            print("-" * 80)
    else:
        print(" 현재 산악 돌풍 및 수목 패널티로 인해 안전한 호이스트 작전 공간 확보 불가.")
        
    print("\n" + "="*115)
    return best_landing, best_hoist

# ==============================================================================
# 데이터셋 내부에서 무작위로 조난 지점 샘플 1개 추출 및 기종 맞춤 작전 가동
# ==============================================================================
sample_rescue_point = df_master.sample(1)
example_lat = sample_rescue_point['latitude'].values[0]
example_lon = sample_rescue_point['longitude'].values[0]

print(f"[무작위 매칭] 마스터 맵에서 추출된 실제 테스트 구조 지점:")
print(f" -> 위도: {example_lat:.5f}, 경도: {example_lon:.5f}")

l_spots, h_spots = find_best_rescue_tactics(rescue_lat=example_lat, rescue_lon=example_lon, heli_type=deployed_heli, search_radius_meters=500)