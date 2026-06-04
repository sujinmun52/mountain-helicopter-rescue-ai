import os
import sys
from dotenv import load_dotenv

# 1. 현재 폴더(RF_Model) 바로 옆에 있는 .env에서 기상청 인증키 즉시 메모리에 주입
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, '.env'), override=True)

# 2. config.py가 숨어있을 만한 모든 후보 폴더 경로 리스트업
project_root = os.path.dirname(current_dir)
config_candidates = [
    os.path.join(current_dir, 'weather'),   # 후보 A: RF_Model\weather
    os.path.join(project_root, 'weather'),  # 후보 B: mountain-helicopter-rescue-ai\weather
    current_dir,                            # 후보 C: RF_Model\
    project_root                            # 후보 D: mountain-helicopter-rescue-ai\
]

# 3. 루프를 돌며 config.py를 실제로 가지고 있는 폴더를 찾아내어 0순위 경로로 바인딩
config_mounted = False
print("\n🔍 === [config.py 위치 지능형 자동 탐색 시스템 가동] ===")
for folder in config_candidates:
    potential_config = os.path.join(folder, 'config.py')
    if os.path.exists(potential_config):
        sys.path.insert(0, folder)
        print(f"✅ [엔진 매핑 성공] config.py 발견 및 등록 완료!")
        print(f"📍 진짜 살아있던 경로: {folder}\n")
        config_mounted = True
        break
    else:
        print(f"❌ 여기에는 config.py가 없음: {folder}")

if not config_mounted:
    print("\n⚠️ [경고] 모든 후보지를 뒤졌으나 config.py를 찾지 못했습니다.")
    print("💡 파일명이 혹시 대문자(Config.py)이거나 철자가 틀리지 않았는지 꼭 확인하세요!\n")

# 기본 프로젝트 루트 및 현재 디렉토리도 탐색 보조 경로로 추가
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

import pandas as pd
import numpy as np
import joblib
from weather.modules.weather import fetch_kma_realtime
from weather.modules.data_preprocessing import mapping_live_weather_to_grid

print("📌 데이터 매트릭스 및 듀얼 전술 AI 인프라 로드 중...")
df_master = pd.read_csv(r'RF_Model\dataset\processed_seoraksan_master.csv')
rf_landing = joblib.load(r'RF_Model\rf_landing_model.joblib')
rf_hoist = joblib.load(r'RF_Model\rf_hoist_model.joblib')

# 119 소방구급센터 고정 좌표 성분 (진입각 및 기지 거리 산출용)
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50}

feature_columns = [
    'elevation', 'slope_deg', 'land_0', 'land_1', 'land_2', 
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos'
]

def find_best_rescue_tactics(rescue_lat, rescue_lon, search_radius_meters=500):
    print(f"\n[작전 개시] 구조 요청 지점 (위도: {rescue_lat:.5f}, 경도: {rescue_lon:.5f}) 주변 실황 탐색 시작...")
    
    # 1. 기상청 API 호출을 통한 실시간 지형-기상 격자 공간 매핑
    try:
        live_weather = fetch_kma_realtime()
        candidates = mapping_live_weather_to_grid(rescue_lat, rescue_lon, live_weather, df_master, radius_km=5.0)
    except Exception as e:
        print(f"⚠️ 실시간 날씨 연산 에러({e}). 마스터본에 캐싱된 날씨 데이터 기반으로 우회 연산합니다.")
        search_radius_deg = (search_radius_meters / 110.0) * 0.001
        candidates = df_master[
            (df_master['latitude'] >= rescue_lat - search_radius_deg) & (df_master['latitude'] <= rescue_lat + search_radius_deg) &
            (df_master['longitude'] >= rescue_lon - search_radius_deg) & (df_master['longitude'] <= rescue_lon + search_radius_deg)
        ].copy()

    if candidates.empty or ("wind_speed" not in candidates.columns):
        print("❌ 반경 내 유효 지형 데이터가 없거나 기상 피처가 유실되었습니다.")
        return None

    # 2. 실시간 풍향 데이터 주기성 인코딩 및 요구조자 물리적 거리 연산
    candidates['wind_direction'] = candidates['wind_direction'].fillna(0.0)
    candidates['wind_dir_rad'] = np.radians(candidates['wind_direction'])
    candidates['wind_dir_sin'] = np.sin(candidates['wind_dir_rad'])
    candidates['wind_dir_cos'] = np.cos(candidates['wind_dir_rad'])
    
    # [거리 연산 A] 추천 후보 격자부터 조난자 실제 위치까지의 거리 (m 단위)
    candidates['dist_to_rescue'] = np.sqrt((candidates['latitude'] - rescue_lat)**2 + (candidates['longitude'] - rescue_lon)**2)
    
    # 🎯 [거리 연산 B] 119 소방 기지(출발지)로부터 추천 후보 격자까지의 실제 비행 거리 (원거리이므로 km 단위 처리)
    candidates['dist_from_base_km'] = np.sqrt((candidates['latitude'] - FIRE_STATION["latitude"])**2 + (candidates['longitude'] - FIRE_STATION["longitude"])**2) * 110.0
    
    # [3.5번] 기지 진입 방위각(Approach Vector) 연산을 통한 정풍/역풍 관계 점수화
    delta_lat = candidates['latitude'] - FIRE_STATION["latitude"]
    delta_lon = candidates['longitude'] - FIRE_STATION["longitude"]
    flight_heading = np.degrees(np.arctan2(delta_lon, delta_lat)) % 360
    angle_diff = np.abs(flight_heading - candidates['wind_direction']) % 360
    
    candidates['wind_relation_score'] = np.select(
        [(angle_diff < 45) | (angle_diff >= 315), (angle_diff >= 135) & (angle_diff < 225)],
        [1.0, 0.0], default=0.5  # 정풍: 1.0점, 역풍: 0.0점, 측풍: 0.5점
    )

    # 4. 실시간 풍속 데이터 4단계 리스크 점수화 및 가중치 수식 업데이트
    candidates['wind_score'] = np.select(
        [candidates['wind_speed'] < 5.0, (candidates['wind_speed'] >= 5.0) & (candidates['wind_speed'] < 10.0), (candidates['wind_speed'] >= 10.0) & (candidates['wind_speed'] < 15.0)],
        [1.0, 0.6, 0.2], default=0.0
    )
    # 풍속 리스크(70%) + 기체 방위 관계(30%)를 결합한 통합 기상 점수 도출
    candidates['final_weather_score'] = candidates['wind_score'] * 0.70 + candidates['wind_relation_score'] * 0.30
    
    # [사용자 전술 배분율 대입]
    candidates['total_landing_score'] = (
        candidates['slope_score'] * 0.55 + candidates['tree_density_score'] * 0.25 + 
        candidates['final_weather_score'] * 0.13 + candidates['tree_height_score'] * 0.07
    )
    candidates['total_hoist_score'] = (
        candidates['final_weather_score'] * 0.45 + candidates['slope_score'] * 0.30 + 
        candidates['tree_density_score'] * 0.18 + candidates['tree_height_score'] * 0.07
    )

    # 5. 듀얼 모델 독립 병렬 추론
    X_candidates = candidates[feature_columns].astype(np.float32)
    candidates['pred_target_landing'] = rf_landing.predict(X_candidates)
    candidates['pred_target_hoist'] = rf_hoist.predict(X_candidates)
    
    # 공간적 격리 안전장치 헬퍼 함수 (최소 55m 거리두기 완벽 차단)
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

    # 🛫 [A안] 직접 착륙 후보지 도출 정렬
    best_landing = filter_spatial_diversity(candidates.sort_values(by=['pred_target_landing', 'total_landing_score', 'dist_to_rescue'], ascending=[True, False, True]))

    # 🏗️ [B안] 호이스트 후보지 도출 정렬
    best_hoist = filter_spatial_diversity(candidates.sort_values(by=['pred_target_hoist', 'total_hoist_score', 'dist_to_rescue'], ascending=[True, False, True]))

    # 🖨️ 6. 실시간 통제 관제 리포트 종합 출력
    target_labels = {0: "🟢 작전 원활 (안전)", 1: "🟡 조건부 작전 (주의)", 2: "🔴 작전 불가 (위험)"}
    print("\n" + "="*90)
    print("🚁 [산악 구조 헬기 복합 전술 통제 시스템] 실시간 기상 대응 브리핑")
    print("="*90)
    
    print("\n🛫 [A안: 기체 직접 안착(Landing) 추천 좌표]")
    print("-" * 90)
    if not best_landing.empty:
        for i, (_, row) in enumerate(best_landing.iterrows(), 1):
            w_rel = '🟢정풍' if row['wind_relation_score']==1.0 else ('🔴역풍' if row['wind_relation_score']==0.0 else '🟡측풍')
            print(f" 📌 {i}순위 착륙지 -> 좌표: ({row['latitude']:.5f}, {row['longitude']:.5f})")
            print(f"    [전술 거리] 📍조난자까지: 약 {row['dist_to_rescue'] * 110000:.1f}m | 🏥기지(출발지)로부터: 약 {row['dist_from_base_km']:.2f}km")
            print(f"    [AI 안전성] 등급: {target_labels[int(row['pred_target_landing'])]} | 통합 적합도 점수: {row['total_landing_score']:.4f}점")
            print(f"    [실황 기상/지형] 현장풍속: {row['wind_speed']:.1f}m/s ({w_rel}) | 고도: {row['elevation']:.1f}m | 경사등급: {int(row['slope_deg'])}")
    else:
        print(" ❌ 현재 기상 및 경사도 조건 하에 안전한 기체 안착 격자가 존재하지 않습니다.")

    print("\n🏗️ [B안: 제자리 비행 호이스트(Hoist) 강하 추천 좌표]")
    print("-" * 90)
    if not best_hoist.empty:
        for i, (_, row) in enumerate(best_hoist.iterrows(), 1):
            w_rel = '🟢정풍' if row['wind_relation_score']==1.0 else ('🔴역풍' if row['wind_relation_score']==0.0 else '🟡측풍')
            print(f" 📌 {i}순위 구조지 -> 좌표: ({row['latitude']:.5f}, {row['longitude']:.5f})")
            print(f"    [전술 거리] 📍조난자까지: 약 {row['dist_to_rescue'] * 110000:.1f}m | 🏥기지(출발지)로부터: 약 {row['dist_from_base_km']:.2f}km")
            print(f"    [AI 안전성] 등급: {target_labels[int(row['pred_target_hoist'])]} | 통합 적합도 점수: {row['total_hoist_score']:.4f}점")
            orig_land = 0 if row.get('land_0', 0) == 1 else (1 if row.get('land_1', 0) == 1 else 2)
            print(f"    [실황 기상/지형] 현장풍속: {row['wind_speed']:.1f}m/s ({w_rel}) | 고도: {row['elevation']:.1f}m | 수목밀도등급: {int(row['tree_density'])} | 지형형태: {orig_land}")
    else:
        print(" ❌ 현재 산악 돌풍 및 수목 수관고 패널티로 인해 안전한 호이스트 공간 확보 불가.")
        
    print("\n" + "="*90)
    return best_landing, best_hoist

# ==============================================================================
# 데이터셋 내부에서 무작위로 진짜 조난 지점 샘플 1개 추출 및 가동
# ==============================================================================
sample_rescue_point = df_master.sample(1)
example_lat = sample_rescue_point['latitude'].values[0]
example_lon = sample_rescue_point['longitude'].values[0]

print(f"🎲 [무작위 매칭] 마스터 맵에서 추출된 실제 테스트 구조 지점:")
print(f" ➡️ 위도: {example_lat:.5f}, 경도: {example_lon:.5f}")

l_spots, h_spots = find_best_rescue_tactics(rescue_lat=example_lat, rescue_lon=example_lon, search_radius_meters=500)