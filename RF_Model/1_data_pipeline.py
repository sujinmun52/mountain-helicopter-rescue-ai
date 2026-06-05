import os
import sys
from dotenv import load_dotenv

# 1. 인프라 설정 (경로 탐색 및 .env 로드)
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
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from weather.modules.weather import fetch_kma_realtime
from weather.modules.data_preprocessing import mapping_live_weather_to_grid

DATA_MODE = 'EXTERNAL'
print(f"🎬 [파이프라인 가동] 현재 설정된 모드: {DATA_MODE}")

# --- [STEP 1] 모드별 지형 데이터 원본 로드 ---
if DATA_MODE == 'INTERNAL':
    print("\n--- 1. 내부 데이터셋 로드 및 병합 ---")
    df1 = pd.read_csv(r'RF_Model\dataset\Final_seoraksan_part1_v2_sujin_260602.csv')
    df2 = pd.read_csv(r'RF_Model\dataset\Final_seoraksan_part2_v2_sujin_260602.csv')
    df3 = pd.read_csv(r'RF_Model\dataset\Final_seoraksan_part3_v2_sujin_260602.csv')
    df = pd.concat([df1, df2, df3], ignore_index=True)
    
    coords = df[['longitude', 'latitude']]
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['zone'] = kmeans.fit_predict(coords)
    train_mask = (df['zone'] != 0)
    test_mask = (df['zone'] == 0)

elif DATA_MODE == 'EXTERNAL':
    print("\n--- 1. 외부 데이터셋 로드 및 병합 ---")
    train1 = pd.read_csv(r'RF_Model\dataset\train_excel_01.csv')
    train2 = pd.read_csv(r'RF_Model\dataset\train_excel_02.csv')
    train_df = pd.concat([train1, train2], ignore_index=True)
    train_df['is_test_flag'] = False
    
    test_df = pd.read_csv(r'RF_Model\dataset\test_set.csv')
    test_df['is_test_flag'] = True
    
    df = pd.concat([train_df, test_df], ignore_index=True)
    df['zone'] = -1

# --- [STEP 1.5] 과거 특정 '기간' 기상청 API 시계열 분할 매핑 (데이터 증강) ---
print("\n--- 2.5. 기상청 API 특정 기간(Time-Range) 기상장 멀티 매핑 파이프라인 가동 ---")
START_PERIOD = "2025-10-01 00:00:00"
END_PERIOD   = "2025-10-06 23:00:00"
SAMPLING_FREQ = "6h"  # 6시간 간격으로 데이터 압축 및 시뮬레이션 효율 극대화

time_slots = pd.date_range(start=START_PERIOD, end=END_PERIOD, freq=SAMPLING_FREQ)
print(f"📅 총 {len(time_slots)}개의 기상 타임스탬프 슬롯 생성 완료. (주기: {SAMPLING_FREQ})")

base_df = df.copy()
compiled_period_dfs = []
center_lat, center_lon = base_df['latitude'].mean(), base_df['longitude'].mean()

for i, dt in enumerate(time_slots, 1):
    target_string = dt.strftime("%Y%m%d%H%M")
    print(f" 🔄 [{i}/{len(time_slots)}] 타임스탬프 처리 중 ➡️ {dt.strftime('%Y-%m-%d %H:%M')}")
    try:
        snapshot_weather = fetch_kma_realtime(target_time=target_string)
        if not snapshot_weather:
            continue
        sub_df = mapping_live_weather_to_grid(center_lat, center_lon, snapshot_weather, base_df.copy(), radius_km=50.0)
        sub_df['snapshot_timestamp'] = target_string
        compiled_period_dfs.append(sub_df)
    except Exception as e:
        print(f"  ❌ {target_string} 시점 공간 결합 에러: {e}")

if compiled_period_dfs:
    df = pd.concat(compiled_period_dfs, ignore_index=True)
    print(f"\n📊 [시공간 통합 완료] 연속 기간 데이터 병합 성공!")
    print(f" ➡️ 최종 파이프라인 매트릭스 크기: {df.shape[0]}행 × {df.shape[1]}열")
else:
    print("❌ [치명적 오류] 기간 내에 수집된 기상 데이터가 전혀 없습니다.")
    sys.exit()

# ==============================================================================
# 🎯 [핵심 패치] 4,080만 행으로 완전히 확장된 df 크기에 맞춰 마스크 최신화!
# ==============================================================================
if DATA_MODE == 'INTERNAL':
    train_mask = (df['zone'] != 0)
    test_mask = (df['zone'] == 0)
elif DATA_MODE == 'EXTERNAL':
    train_mask = (df['is_test_flag'] == False)
    test_mask = (df['is_test_flag'] == True)
# ==============================================================================

df['wind_direction'] = df['wind_direction'].fillna(0.0)
df['wind_dir_rad'] = np.radians(df['wind_direction'])
df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])

# --- [STEP 2] 헬기 제원별 2x2 항공 교범 가중치 대입 및 4대 독립 라벨링 ---
print("\n--- 3. 헬기 제원별 4대 전술 정답 라벨링 및 적합도 점수 산정 ---")
df['wind_score'] = np.select(
    [df['wind_speed'] < 5.0, (df['wind_speed'] >= 5.0) & (df['wind_speed'] < 10.0), (df['wind_speed'] >= 10.0) & (df['wind_speed'] < 15.0)],
    [1.0, 0.6, 0.2], default=0.0
)

slope_map, density_map, height_map = {0: 1.0, 1: 0.7, 2: 0.35}, {0: 1.0, 1: 0.85, 2: 0.21, 3: 0.0}, {0: 1.0, 1: 0.61, 2: 0.31}
df['slope_score'] = df['slope_deg'].map(slope_map)
df['tree_density_score'] = df['tree_density'].map(density_map)
df['tree_height_score'] = df['tree_height'].map(height_map)

# 🛸 소형 헬기 전술 점수 및 라벨 (바람 패널티 강화)
df['score_small_landing'] = df['slope_score'] * 0.45 + df['wind_score'] * 0.25 + df['tree_density_score'] * 0.20 + df['tree_height_score'] * 0.10
df['target_small_landing'] = np.select([(df['score_small_landing'] >= 0.80), (df['score_small_landing'] >= 0.55) & (df['score_small_landing'] < 0.80)], [0, 1], default=2)

df['score_small_hoist'] = df['wind_score'] * 0.55 + df['slope_score'] * 0.20 + df['tree_density_score'] * 0.15 + df['tree_height_score'] * 0.10
df['target_small_hoist'] = np.select([(df['score_small_hoist'] >= 0.80), (df['score_small_hoist'] >= 0.55) & (df['score_small_hoist'] < 0.80)], [0, 1], default=2)

# 🛸 대형 헬기 전술 점수 및 라벨 (지형/공간 및 하강풍 패널티 강화)
df['score_large_landing'] = df['slope_score'] * 0.60 + df['tree_density_score'] * 0.22 + df['tree_height_score'] * 0.10 + df['wind_score'] * 0.08
df['target_large_landing'] = np.select([(df['score_large_landing'] >= 0.80), (df['score_large_landing'] >= 0.55) & (df['score_large_landing'] < 0.80)], [0, 1], default=2)

df['score_large_hoist'] = df['wind_score'] * 0.40 + df['tree_density_score'] * 0.25 + df['slope_score'] * 0.25 + df['tree_height_score'] * 0.10
df['target_large_hoist'] = np.select([(df['score_large_hoist'] >= 0.80), (df['score_large_hoist'] >= 0.55) & (df['score_large_hoist'] < 0.80)], [0, 1], default=2)

# --- [STEP 3] 공간적 버퍼 존 및 마스킹 ---
print("\n--- 4. 공간 누수 차단 버퍼 필터링 ---")
if DATA_MODE == 'INTERNAL':
    test_coords = KDTree(df[test_mask][['longitude', 'latitude']].values)
    train_coords = df[train_mask][['longitude', 'latitude']].values
    distances, _ = test_coords.query(train_coords, k=1, workers=-1)
    safe_train_indices = df[train_mask].index[distances > 0.001]
elif DATA_MODE == 'EXTERNAL':
    safe_train_indices = df[train_mask].index

df['is_train_final'] = False
df.loc[safe_train_indices, 'is_train_final'] = True
df['is_test'] = test_mask

df = pd.get_dummies(df, columns=['land_type'], prefix='land', dtype=int)

output_path = r'RF_Model\dataset\processed_seoraksan_master.csv'
df.to_csv(output_path, index=False)
print(f"\n🎉 마스터 파일 내보내기 성공 ➡️ {output_path}")