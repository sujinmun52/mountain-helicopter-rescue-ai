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
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from weather.modules.weather import fetch_kma_realtime
from weather.modules.data_preprocessing import mapping_live_weather_to_grid

DATA_MODE = 'EXTERNAL'  # 👈 내부 정렬 테스트 시 'INTERNAL'로 변경하세요.

print(f"🎬 [파이프라인 가동] 현재 설정된 모드: {DATA_MODE}")

# --- [STEP 1] 모드별 데이터 로드 및 마스크(Train/Test) 설정 ---
if DATA_MODE == 'INTERNAL':
    print("\n--- 1. 내부 데이터셋 로드 및 병합 ---")
    df1 = pd.read_csv(r'RF_Model\dataset\Final_seoraksan_part1_v2_sujin_260602.csv')
    df2 = pd.read_csv(r'RF_Model\dataset\Final_seoraksan_part2_v2_sujin_260602.csv')
    df3 = pd.read_csv(r'RF_Model\dataset\Final_seoraksan_part3_v2_sujin_260602.csv')
    df = pd.concat([df1, df2, df3], ignore_index=True)
    
    print("\n--- 2. KMeans 기반 공간적 블록 분리 ---")
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
    
    train_mask = (df['is_test_flag'] == False)
    test_mask = (df['is_test_flag'] == True)
    df = df.drop(columns=['is_test_flag'])

else:
    raise ValueError("❌ DATA_MODE는 반드시 'INTERNAL' 또는 'EXTERNAL' 중 하나여야 합니다.")


# --- [STEP 1.5] 실시간 기상청 API 데이터 동적 공간 융합 및 수치형 풍향 인코딩 ---
print("\n--- 2.5. 기상청 API 실시간 바람장 매핑 및 삼각함수 벡터 인코딩 ---")
try:
    live_weather_api_data = fetch_kma_realtime()
    center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
    df = mapping_live_weather_to_grid(center_lat, center_lon, live_weather_api_data, df, radius_km=50.0)
except Exception as e:
    print(f"⚠️ 기상 API 호출 실패 혹은 모듈 에러({e}). 기본값(2.0m/s, 북풍)으로 강제 폴백합니다.")
    df['wind_speed'] = 2.0
    df['wind_direction'] = 0.0

# 💡 [상황 A 대응] 주기성 해결을 위한 라디안 변환 후 성분 분해 (Sine/Cosine 변환)
df['wind_direction'] = df['wind_direction'].fillna(0.0)
df['wind_dir_rad'] = np.radians(df['wind_direction'])
df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])


# --- [STEP 2] 4단계 풍속 분기점 적용 및 사용자 지정 가중치 라벨링 ---
print("\n--- 3. 듀얼 정답 라벨링 및 작전별 적합도 점수 산정 (사용자 가중치 반영) ---")

# 💨 [항공 교범 기준] 풍속 4단계 리스크 점수 매핑 (안전 마진 조율)
df['wind_score'] = np.select(
    [
        df['wind_speed'] < 5.0,                                      # 🟢 안전 (1.0점)
        (df['wind_speed'] >= 5.0) & (df['wind_speed'] < 10.0),       # 🟡 주의 (0.6점)
        (df['wind_speed'] >= 10.0) & (df['wind_speed'] < 15.0)       # 🟠 위험 (0.2점)
    ],
    [1.0, 0.6, 0.2], 
    default=0.0                                                      # 🔴 제한 (15m/s 이상 작전취소)
)

# 지형/식생 점수 매핑 (정형화)
slope_map = {0: 1.0, 1: 0.7, 2: 0.35}
density_map = {0: 1.0, 1: 0.85, 2: 0.21, 3: 0.0}
height_map = {0: 1.0, 1: 0.61, 2: 0.31}

df['slope_score'] = df['slope_deg'].map(slope_map)
df['tree_density_score'] = df['tree_density'].map(density_map)
df['tree_height_score'] = df['tree_height'].map(height_map)

# 🛫 [가닥 A] 기체 직접 착륙(Landing) -> 경사도(0.55) + 밀도(0.25) + 풍속(0.13) + 높이(0.07)
df['total_landing_score'] = (
    df['slope_score'] * 0.55 + 
    df['tree_density_score'] * 0.25 + 
    df['wind_score'] * 0.13 + 
    df['tree_height_score'] * 0.07
)
cond_landing = [
    (df['total_landing_score'] >= 0.80),
    (df['total_landing_score'] >= 0.55) & (df['total_landing_score'] < 0.80),
    (df['total_landing_score'] < 0.55)
]
df['target_landing'] = np.select(cond_landing, [0, 1, 2], default=2)

# 🏗️ [가닥 B] 호이스트(Hoist) 강하 -> 풍속(0.45) + 경사도(0.30) + 밀도(0.18) + 높이(0.07)
df['total_hoist_score'] = (
    df['wind_score'] * 0.45 + 
    df['slope_score'] * 0.30 + 
    df['tree_density_score'] * 0.18 + 
    df['tree_height_score'] * 0.07
)
cond_hoist = [
    (df['total_hoist_score'] >= 0.80),
    (df['total_hoist_score'] >= 0.55) & (df['total_hoist_score'] < 0.80),
    (df['total_hoist_score'] < 0.55)
]
df['target_hoist'] = np.select(cond_hoist, [0, 1, 2], default=2)


# --- [STEP 3] 하이브리드 버퍼 존 필터링 (모드별 차등 적용) ---
print("\n--- 4. 하이브리드 버퍼 존 필터링 (공간 누수 차단) ---")
if DATA_MODE == 'INTERNAL':
    test_coords = KDTree(df[test_mask][['longitude', 'latitude']].values)
    train_coords = df[train_mask][['longitude', 'latitude']].values
    distances, _ = test_coords.query(train_coords, k=1, workers=-1)

    buffer_distance = 0.001
    safe_train_indices = df[train_mask].index[distances > buffer_distance]
    print(f"경계선 인접 구역에서 삭제된 데이터 (누수 차단): {len(df[train_mask]) - len(safe_train_indices)}개")
elif DATA_MODE == 'EXTERNAL':
    print("💡 외부 데이터셋은 이미 자체 누수 처리가 완료되어 있어 버퍼 필터링을 생략합니다.")
    safe_train_indices = df[train_mask].index

df['is_train_final'] = False
df.loc[safe_train_indices, 'is_train_final'] = True
df['is_test'] = test_mask
print(f"최종 학습 데이터 수: {len(df[df['is_train_final'] == True])}개")


# --- [STEP 3.5] land_type 명목형 범주 원-핫 인코딩 추가 ---
df = pd.get_dummies(df, columns=['land_type'], prefix='land', dtype=int)


# --- [STEP 4] 최종 파일 내보내기 ---
output_path = r'RF_Model\dataset\processed_seoraksan_master.csv'
df.to_csv(output_path, index=False)
print(f"\n🎉 [{DATA_MODE} 모드 완료] 기상 통합 마스터 파일 출력 성공: {output_path}")