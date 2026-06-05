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
from weather.modules.weather import fetch_kma_realtime
from weather.modules.data_preprocessing import mapping_live_weather_to_grid

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


# --- [STEP 1.5] 과거 특정 '기간' 기상청 API 시계열 분할 매핑 (데이터 증강) ---
print("\n--- 2. 기상청 API 특정 기간 데이터 맵핑 파이프라인 시작 ---")
START_PERIOD   = "2025-10-01 00:00:00"
END_PERIOD     = "2025-10-06 23:00:00"
SAMPLING_FREQ  = "6h"

time_slots = pd.date_range(start=START_PERIOD, end=END_PERIOD, freq=SAMPLING_FREQ)
print(f"총 {len(time_slots)}개의 기상 타임스탬프 슬롯 생성 완료. (주기: {SAMPLING_FREQ})")

base_df = df.copy()
compiled_period_dfs = []
center_lat = base_df['latitude'].mean()
center_lon = base_df['longitude'].mean()

for i, dt in enumerate(time_slots, 1):
    target_string = dt.strftime("%Y%m%d%H%M")
    print(f"[{i}/{len(time_slots)}] 타임스탬프 처리 중 {dt.strftime('%Y-%m-%d %H:%M')}")
    try:
        snapshot_weather = fetch_kma_realtime(target_time=target_string)
        if not snapshot_weather:
            continue
        sub_df = mapping_live_weather_to_grid(center_lat, center_lon, snapshot_weather, base_df.copy(), radius_km=50.0)
        sub_df['snapshot_timestamp'] = target_string
        compiled_period_dfs.append(sub_df)
    except Exception as e:
        print(f"{target_string} 시점 공간 결합 에러: {e}")

if compiled_period_dfs:
    df = pd.concat(compiled_period_dfs, ignore_index=True)
    print(f"\n[시공간 통합 완료] 연속 기간 데이터 병합 성공!")
    print(f"최종 파이프라인 매트릭스 크기: {df.shape[0]}행 × {df.shape[1]}열")
else:
    print("기간 내에 수집된 기상 데이터가 전혀 없습니다. 파이프라인을 종료합니다.")
    sys.exit()


# ==============================================================================
# 🎯 [마스크 최신화 및 기상 피처 공간 매핑]
# ==============================================================================
train_mask = (df['is_test_flag'] == False)
test_mask  = (df['is_test_flag'] == True)

df['wind_direction']  = df['wind_direction'].fillna(0.0)
df['wind_dir_rad']    = np.radians(df['wind_direction'])
df['wind_dir_sin']    = np.sin(df['wind_dir_rad'])
df['wind_dir_cos']    = np.cos(df['wind_dir_rad'])


# ==============================================================================
# 🛸 [항공 전술 피처] 신규 변수 산출 (wind_dir_score & altitude_score)
# ==============================================================================
# 1. 119 소방구급센터 고정 좌표 성분을 이용한 정풍 진입각(wind_dir_score) 계산
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50}
delta_lat = df['latitude'] - FIRE_STATION["latitude"]
delta_lon = df['longitude'] - FIRE_STATION["longitude"]
flight_heading = np.degrees(np.arctan2(delta_lon, delta_lat)) % 360
angle_diff = np.abs(flight_heading - df['wind_direction']) % 360

df['wind_dir_score'] = np.select(
    [(angle_diff < 45) | (angle_diff >= 315), (angle_diff >= 135) & (angle_diff < 225)],
    [1.0, 0.0], default=0.5  # 정풍 진입: 1.0점, 역풍 진입: 0.0점, 측풍 진입: 0.5점
)

# 2. 밀도고도 변수(altitude_score) 추가: 설악산 최고 고도(1708m) 기준 고도 리스크 계산
df['altitude_score'] = 1.0 - (df['elevation'] / 1708.0) * 0.4


# --- [STEP 2] 헬기 제원별 4대 전술 라벨링 ---
print("\n--- 3. 헬기 제원별 4가지 방법 정답 라벨링 및 적합도 점수 산정 ---")
df['wind_score'] = np.select(
    [df['wind_speed'] < 5.0,
     (df['wind_speed'] >= 5.0)  & (df['wind_speed'] < 10.0),
     (df['wind_speed'] >= 10.0) & (df['wind_speed'] < 15.0)],
    [1.0, 0.6, 0.2], default=0.0
)

slope_map   = {0: 1.0, 1: 0.7,  2: 0.35}
density_map = {0: 1.0, 1: 0.85, 2: 0.21, 3: 0.0}
height_map  = {0: 1.0, 1: 0.61, 2: 0.31}

df['slope_score']        = df['slope_deg'].map(slope_map)
df['tree_density_score'] = df['tree_density'].map(density_map)
df['tree_height_score']  = df['tree_height'].map(height_map)


# ==============================================================================
# 📊 [새 가중치 행렬 공식 완벽 대입]
# ==============================================================================
# 🛸 [1] 소형 안착 착륙 모델 (L_Light)
df['score_small_landing'] = (
    df['slope_score']        * 0.42 +
    df['tree_density_score'] * 0.10 +
    df['tree_height_score']  * 0.08 +
    df['wind_score']         * 0.20 +
    df['wind_dir_score']     * 0.12 +
    df['altitude_score']     * 0.08
)
df['target_small_landing'] = np.select(
    [(df['score_small_landing'] >= 0.80),
     (df['score_small_landing'] >= 0.55) & (df['score_small_landing'] < 0.80)],
    [0, 1], default=2
)

# 🛸 [2] 대형 안착 착륙 모델 (L_Heavy)
df['score_large_landing'] = (
    df['slope_score']        * 0.46 +
    df['tree_density_score'] * 0.14 +
    df['tree_height_score']  * 0.08 +
    df['wind_score']         * 0.12 +
    df['wind_dir_score']     * 0.08 +
    df['altitude_score']     * 0.12
)
df['target_large_landing'] = np.select(
    [(df['score_large_landing'] >= 0.80),
     (df['score_large_landing'] >= 0.55) & (df['score_large_landing'] < 0.80)],
    [0, 1], default=2
)

# 🛸 [3] 소형 강하 호이스트 모델 (H_Light)
df['score_small_hoist'] = (
    df['slope_score']        * 0.12 +
    df['tree_density_score'] * 0.15 +
    df['tree_height_score']  * 0.20 +
    df['wind_score']         * 0.30 +
    df['wind_dir_score']     * 0.15 +
    df['altitude_score']     * 0.08
)
df['target_small_hoist'] = np.select(
    [(df['score_small_hoist'] >= 0.80),
     (df['score_small_hoist'] >= 0.55) & (df['score_small_hoist'] < 0.80)],
    [0, 1], default=2
)

# 🛸 [4] 대형 강하 호이스트 모델 (H_Heavy)
df['score_large_hoist'] = (
    df['slope_score']        * 0.10 +
    df['tree_density_score'] * 0.24 +
    df['tree_height_score']  * 0.18 +
    df['wind_score']         * 0.18 +
    df['wind_dir_score']     * 0.12 +
    df['altitude_score']     * 0.18
)
df['target_large_hoist'] = np.select(
    [(df['score_large_hoist'] >= 0.80),
     (df['score_large_hoist'] >= 0.55) & (df['score_large_hoist'] < 0.80)],
    [0, 1], default=2
)


# --- [STEP 3] 외부 데이터 전용 데이터 마스크 주입 ---
print("\n--- 4. 외부 데이터셋 전용 최종 인덱스 분할 마스킹 ---")
safe_train_indices = df[train_mask].index

df['is_train_final'] = False
df.loc[safe_train_indices, 'is_train_final'] = True
df['is_test'] = test_mask

df = pd.get_dummies(df, columns=['land_type'], prefix='land', dtype=int)


# ==============================================================================
# ✅ [XGBoost 데이터 타입 매칭] 정수형 라벨 명시적 캐스팅 및 임시 플래그 청소
# ==============================================================================
label_cols = ['target_small_landing', 'target_small_hoist',
              'target_large_landing', 'target_large_hoist']
for col in label_cols:
    if col in df.columns:
        df[col] = df[col].astype(np.int32)

if 'is_test_flag' in df.columns:
    df = df.drop(columns=['is_test_flag'])

output_path = r'RF_Model\dataset\processed_seoraksan_master.csv'
df.to_csv(output_path, index=False)
print(f"\n파일 출력 성공: {output_path}")