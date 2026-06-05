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
import gc
from weather.modules.weather import fetch_kma_realtime
from weather.modules.data_preprocessing import mapping_live_weather_to_grid

print("[파이프라인 가동] 고정 모드: EXTERNAL (Random Forest 베이스라인 전처리 가속 버전)")

# --- [STEP 1] 외부 데이터셋 로드 및 기본 뼈대 통합 ---
print("\n--- 1. 외부 데이터셋 로드 및 병합 ---")
train1 = pd.read_csv(r'RF_Model\dataset\train_excel_01.csv')
train2 = pd.read_csv(r'RF_Model\dataset\train_excel_02.csv')
train_df = pd.concat([train1, train2], ignore_index=True)
train_df['is_test_flag'] = False

test_df = pd.read_csv(r'RF_Model\dataset\test_set.csv')
test_df['is_test_flag'] = True

df = pd.concat([train_df, test_df], ignore_index=True)
df['zone'] = -1

# 메모리 확보를 위한 초기 파편 제거
del train1, train2, train_df, test_df
gc.collect()

# 속도 최적화: 시계열 확장 전 원-핫 인코딩 선배치 및 8비트 정수 변환
df = pd.get_dummies(df, columns=['land_type'], prefix='land', dtype=np.int8)

# 메모리 다이어트: 수치 데이터 다운캐스팅
float_cols = ['latitude', 'longitude', 'elevation']
int8_cols = ['slope_deg', 'tree_density', 'tree_height', 'zone']

for col in float_cols:
    if col in df.columns:
        df[col] = df[col].astype(np.float32)
for col in int8_cols:
    if col in df.columns:
        df[col] = df[col].astype(np.int8)


# --- [STEP 1.5] 과거 특정 '기간' 기상청 API 시계열 분할 매핑 (데이터 증강) ---
print("\n--- 2. 기상청 API 특정 기간 기상장 멀티 매핑 파이프라인 가동 ---")
START_PERIOD = "2025-10-01 00:00:00"
END_PERIOD   = "2025-10-03 23:00:00"
SAMPLING_FREQ = "3h"

time_slots = pd.date_range(start=START_PERIOD, end=END_PERIOD, freq=SAMPLING_FREQ)
print(f"총 {len(time_slots)}개의 기상 타임스탬프 슬롯 생성 완료. (주기: {SAMPLING_FREQ})")

base_df = df.copy()
compiled_period_dfs = []
center_lat, center_lon = base_df['latitude'].mean(), base_df['longitude'].mean()

for i, dt in enumerate(time_slots, 1):
    target_string = dt.strftime("%Y%m%d%H%M")
    print(f"[{i}/{len(time_slots)}] 타임스탬프 처리 중 {dt.strftime('%Y-%m-%d %H:%M')}")
    try:
        snapshot_weather = fetch_kma_realtime(target_time=target_string)
        if not snapshot_weather:
            continue
        sub_df = mapping_live_weather_to_grid(center_lat, center_lon, snapshot_weather, base_df.copy(), radius_km=50.0)
        sub_df['snapshot_timestamp'] = target_string
        
        # 루프 내부 결합본 다운캐스팅으로 피크 메모리 억제
        for col in sub_df.columns:
            if sub_df[col].dtype == np.float64:
                sub_df[col] = sub_df[col].astype(np.float32)
            elif sub_df[col].dtype == np.int64:
                sub_df[col] = sub_df[col].astype(np.int32)
                
        compiled_period_dfs.append(sub_df)
    except Exception as e:
        print(f"{target_string} 시점 공간 결합 에러: {e}")

del base_df
gc.collect()

if compiled_period_dfs:
    df = pd.concat(compiled_period_dfs, ignore_index=True)
    print(f"\n[시공간 통합 완료] 연속 기간 데이터 병합 성공!")
    print(f"최종 파이프라인 매트릭스 크기: {df.shape[0]}행 × {df.shape[1]}열")
    del compiled_period_dfs
    gc.collect()
else:
    print("기간 내에 수집된 기상 데이터가 전혀 없습니다. 파이프라인을 종료합니다.")
    sys.exit()


# ==============================================================================
# [마스크 최신화 및 기상 피처 공간 매핑]
# ==============================================================================
df['wind_direction'] = df['wind_direction'].fillna(0.0).astype(np.float32)
df['wind_dir_rad'] = np.radians(df['wind_direction']).astype(np.float32)
df['wind_dir_sin'] = np.sin(df['wind_dir_rad']).astype(np.float32)
df['wind_dir_cos'] = np.cos(df['wind_dir_rad']).astype(np.float32)


# --- [STEP 2] 헬기 제원별 과거 항공 가중치 기준 라벨링 ---
print("\n--- 3. 헬기 제원별 4대 전술 정답 라벨링 및 적합도 점수 산정 ---")
df['wind_score'] = np.select(
    [df['wind_speed'] < 5.0, (df['wind_speed'] >= 5.0) & (df['wind_speed'] < 10.0), (df['wind_speed'] >= 10.0) & (df['wind_speed'] < 15.0)],
    [1.0, 0.6, 0.2], default=0.0
).astype(np.float32)

slope_map, density_map, height_map = {0: 1.0, 1: 0.7, 2: 0.35}, {0: 1.0, 1: 0.85, 2: 0.21, 3: 0.0}, {0: 1.0, 1: 0.61, 2: 0.31}
df['slope_score'] = df['slope_deg'].map(slope_map).astype(np.float32)
df['tree_density_score'] = df['tree_density'].map(density_map).astype(np.float32)
df['tree_height_score'] = df['tree_height'].map(height_map).astype(np.float32)

# 소형 헬기 과거 전술 공식 복원
df['score_small_landing'] = (df['slope_score'] * 0.45 + df['wind_score'] * 0.25 + df['tree_density_score'] * 0.20 + df['tree_height_score'] * 0.10).astype(np.float32)
df['target_small_landing'] = np.select([(df['score_small_landing'] >= 0.80), (df['score_small_landing'] >= 0.55) & (df['score_small_landing'] < 0.80)], [0, 1], default=2).astype(np.int32)

df['score_small_hoist'] = (df['wind_score'] * 0.55 + df['slope_score'] * 0.20 + df['tree_density_score'] * 0.15 + df['tree_height_score'] * 0.10).astype(np.float32)
df['target_small_hoist'] = np.select([(df['score_small_hoist'] >= 0.80), (df['score_small_hoist'] >= 0.55) & (df['score_small_hoist'] < 0.80)], [0, 1], default=2).astype(np.int32)

# 대형 헬기 과거 전술 공식 복원
df['score_large_landing'] = (df['slope_score'] * 0.60 + df['tree_density_score'] * 0.22 + df['tree_height_score'] * 0.10 + df['wind_score'] * 0.08).astype(np.float32)
df['target_large_landing'] = np.select([(df['score_large_landing'] >= 0.80), (df['score_large_landing'] >= 0.55) & (df['score_large_landing'] < 0.80)], [0, 1], default=2).astype(np.int32)

df['score_large_hoist'] = (df['wind_score'] * 0.40 + df['tree_density_score'] * 0.25 + df['slope_score'] * 0.25 + df['tree_height_score'] * 0.10).astype(np.float32)
df['target_large_hoist'] = np.select([(df['score_large_hoist'] >= 0.80), (df['score_large_hoist'] >= 0.55) & (df['score_large_hoist'] < 0.80)], [0, 1], default=2).astype(np.int32)


# --- [STEP 3] 외부 데이터 전용 고속 마스킹 변환 ---
print("\n--- 4. 외부 데이터셋 전용 최종 인덱스 분할 마스킹 ---")
# .loc 레이블 루프 검색을 완전 삭제하고 넘파이 논리 조건 대입으로 1초 내에 연산 완료
df['is_train_final'] = (df['is_test_flag'] == False)
df['is_test']        = (df['is_test_flag'] == True)

if 'is_test_flag' in df.columns:
    df = df.drop(columns=['is_test_flag'])
gc.collect()

# XGBoost 마스터 파일과 충돌 및 오염을 방지하기 위해 파일명 분리 지정
output_path = r'RF_Model\dataset\processed_seoraksan_master_rf.csv'
df.to_csv(output_path, index=False)
print(f"\n마스터 파일 내보내기 성공: {output_path}")