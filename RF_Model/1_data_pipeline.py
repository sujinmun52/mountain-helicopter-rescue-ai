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

# ==============================================================================
# [동기화 셋업] 전문가 채점 시스템 상수 및 가중치 매트릭스 정의
# ==============================================================================
SLOPE_MAP    = {0: 1.0, 1: 0.70, 2: 0.35}
DENSITY_MAP  = {0: 1.0, 1: 0.85, 2: 0.21, 3: 0.00}
HEIGHT_MAP   = {0: 1.0, 1: 0.61, 2: 0.31}
ELEVATION_MAP = {0: 1.0, 1: 0.70, 2: 0.35}
MAX_ELEV     = 1708.0
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50}

TACTIC_WEIGHTS = {
    "small_landing": {
        "static":  {"slope": 0.42, "density": 0.10, "height": 0.08, "elevation": 0.08},
        "dynamic": {"wind_score": 0.20, "wind_dir_score": 0.12},
    },
    "small_hoist": {
        "static":  {"slope": 0.12, "density": 0.15, "height": 0.20, "elevation": 0.08},
        "dynamic": {"wind_score": 0.30, "wind_dir_score": 0.15},
    },
    "large_landing": {
        "static":  {"slope": 0.46, "density": 0.14, "height": 0.08, "elevation": 0.12},
        "dynamic": {"wind_score": 0.12, "wind_dir_score": 0.08},
    },
    "large_hoist": {
        "static":  {"slope": 0.10, "density": 0.24, "height": 0.18, "elevation": 0.18},
        "dynamic": {"wind_score": 0.18, "wind_dir_score": 0.12},
    },
}

# ==============================================================================
# [엔진 이식] 전문가 채점 모듈 연산 커널 함수화
# ==============================================================================
def compute_static_terrain_scores(df):
    """기상과 무관한 정적 지형 점수를 연산."""
    elevation_grade = np.select(
        [df['elevation'] < 500, (df['elevation'] >= 500) & (df['elevation'] < 1200)],
        [0, 1], default=2
    ).astype('int8')

    df['slope_score']        = df['slope_deg'].map(SLOPE_MAP).astype('float32')
    df['tree_density_score'] = df['tree_density'].map(DENSITY_MAP).astype('float32')
    df['tree_height_score']  = df['tree_height'].map(HEIGHT_MAP).astype('float32')
    df['elevation_score']    = np.vectorize(ELEVATION_MAP.get)(elevation_grade).astype('float32')

    for key, w in TACTIC_WEIGHTS.items():
        s = w["static"]
        df[f'static_score_{key}'] = (
            df['slope_score']        * s["slope"]   +
            df['tree_density_score'] * s["density"] +
            df['tree_height_score']  * s["height"]  +
            df['elevation_score']    * s["elevation"]
        ).astype('float32')
    return df

def compute_wind_score(wind_speed_arr: np.ndarray) -> np.ndarray:
    return np.select(
        [wind_speed_arr < 5.0, (wind_speed_arr >= 5.0) & (wind_speed_arr < 10.0), (wind_speed_arr >= 10.0) & (wind_speed_arr < 15.0)],
        [1.0, 0.6, 0.2], default=0.0
    ).astype('float32')

def compute_wind_dir_score(lat: np.ndarray, lon: np.ndarray, wind_dir: np.ndarray) -> np.ndarray:
    heading    = np.degrees(np.arctan2(lon - FIRE_STATION["longitude"], lat - FIRE_STATION["latitude"])) % 360
    angle_diff = np.abs(heading - wind_dir) % 360
    return np.select(
        [(angle_diff < 45) | (angle_diff >= 315), (angle_diff >= 135) & (angle_diff < 225)],
        [1.0, 0.0], default=0.5
    ).astype('float32')

def compute_targets(df):
    """정적 성분 + 동적 성분 결합 -> 최종 점수 및 동기화된 3클래스 라벨 부여."""
    for key, w in TACTIC_WEIGHTS.items():
        d = w["dynamic"]
        score = (
            df[f'static_score_{key}']
            + df['wind_score']      * d["wind_score"]
            + df['wind_dir_score']  * d["wind_dir_score"]
        ).astype('float32')
        df[f'score_{key}']  = score
        df[f'target_{key}'] = np.select(
            [score >= 0.80, (score >= 0.55) & (score < 0.80)],
            [0, 1], default=2
        ).astype('int8')
    return df


# --- [STEP 1] 외부 데이터셋 로드 및 기본 뼈대 통합 ──────────────────────────────
print("\n--- 1. 외부 데이터셋 로드 및 병합 ---")
# 🎯 [버그 해결]: 하드코딩 수식을 current_dir 지향형 절대 경로로 격상하여 경로 에러 차단
train1_path = os.path.join(current_dir, 'dataset', 'train_excel_01.csv')
train2_path = os.path.join(current_dir, 'dataset', 'train_excel_02.csv')
test_path   = os.path.join(current_dir, 'dataset', 'test_set.csv')

train1 = pd.read_csv(train1_path)
train2 = pd.read_csv(train2_path)
train_df = pd.concat([train1, train2], ignore_index=True)
train_df['is_test_flag'] = False

test_df = pd.read_csv(test_path)
test_df['is_test_flag'] = True

df = pd.concat([train_df, test_df], ignore_index=True)
df['zone'] = -1

del train1, train2, train_df, test_df
gc.collect()

df = pd.get_dummies(df, columns=['land_type'], prefix='land', dtype=np.int8)

float_cols = ['latitude', 'longitude', 'elevation']
int8_cols = ['slope_deg', 'tree_density', 'tree_height', 'zone']

for col in float_cols:
    if col in df.columns:
        df[col] = df[col].astype(np.float32)
for col in int8_cols:
    if col in df.columns:
        df[col] = df[col].astype(np.int8)


# --- [STEP 2] 과거 특정 '기간' 기상청 API 시계열 분할 매핑 (데이터 증강) ───────────
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


# --- [STEP 3] 🎯 동기화된 전문가용 가중치 기반 정답 라벨링 가동 ──────────────────────
print("\n--- 3. 헬기 제원별 4대 전술 동기화 정답 라벨링 및 적합도 점수 산정 ---")

# 1. 기상 가속 벡터 연산 주입
df['wind_score'] = compute_wind_score(df['wind_speed'].values)
df['wind_dir_score'] = compute_wind_dir_score(df['latitude'].values, df['longitude'].values, df['wind_direction'].values)

# 2. 정적 지형 통합 점수 빌드
df = compute_static_terrain_scores(df)

# 3. 최종 스코어 및 동일 타겟(0, 1, 2) 추출
df = compute_targets(df)


# --- [STEP 4] 외부 데이터 전용 고속 마스킹 변환 및 저장 ──────────────────────────
print("\n--- 4. 외부 데이터셋 전용 최종 인덱스 분할 마스킹 ---")
df['is_train_final'] = (df['is_test_flag'] == False)
df['is_test']        = (df['is_test_flag'] == True)

if 'is_test_flag' in df.columns:
    df = df.drop(columns=['is_test_flag'])
gc.collect()

# 🎯 [버그 해결]: 내보낼 마스터 파일의 저장 위치도 패키지 내부 절대 경로로 일치화
output_path = os.path.join(current_dir, 'dataset', 'processed_seoraksan_master_rf.csv')
df.to_csv(output_path, index=False)
print(f"\n마스터 파일 내보내기 성공: {output_path}")