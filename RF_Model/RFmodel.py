import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from scipy.spatial import KDTree

# 1. 파일 읽어오기
df1 = pd.read_csv(r'EDA\최종\Final_seoraksan(wind_speedX)_part1_v2_dawon_260602.csv')
df2 = pd.read_csv(r'EDA\최종\Final_seoraksan(wind_speedX)_part2_v2_dawon_260602.csv')
df3 = pd.read_csv(r'EDA\최종\Final_seoraksan(wind_speedX)_part3_v2_dawon_260602.csv')
df = pd.concat([df1, df2, df3], ignore_index=True)

print("원본 데이터 컬럼 확인:")
print(df.columns.tolist())

print("\n--- 정답 라벨링 및 적합도 점수 산정 ---")

# 변경된 세부 점수 매핑 매트릭스 적용 (정형화)
slope_map = {0: 1.0, 1: 0.7, 2: 0.35}
density_map = {0: 1.0, 1: 0.85, 2: 0.21, 3: 0.0}  # 0~3 단계 유지
height_map = {0: 1.0, 1: 0.61, 2: 0.31}

df['slope_score'] = df['slope_deg'].map(slope_map)
df['tree_density_score'] = df['tree_density'].map(density_map)
df['tree_height_score'] = df['tree_height'].map(height_map)

# 새로운 가중치 반영 (합이 1.0인 착륙 적합도 점수 생성)
df['total_landing_score'] = (
    df['slope_score'] * 0.55 + 
    df['tree_density_score'] * 0.33 + 
    df['tree_height_score'] * 0.12
)

# 착륙 적합도 기반 최종 3단계 타겟(target) 분류 규칙
score_conditions = [
    (df['total_landing_score'] >= 0.85),                                      # 0: 착륙 가능
    (df['total_landing_score'] >= 0.65) & (df['total_landing_score'] < 0.85), # 1: 조건부 착륙
    (df['total_landing_score'] < 0.65)                                        # 2: 착륙 불가
]
score_choices = [0, 1, 2]
df['target'] = np.select(score_conditions, score_choices, default=2)


print("\n공간적 블록 분리 (KMeans)")

# 위경도 좌표 기준 클러스터링 컬럼 생성
coords = df[['longitude', 'latitude']]
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['zone'] = kmeans.fit_predict(coords)

# 0번 구역을 Test, 나머지 구역을 Train으로 분리
train_df = df[df['zone'] != 0]
test_df = df[df['zone'] == 0]


print("\n하이브리드 버퍼 존 경계면 데이터 필터링")

test_coords = KDTree(test_df[['longitude', 'latitude']].values)
train_coords = train_df[['longitude', 'latitude']].values

distances, _ = test_coords.query(train_coords, k=1, workers=-1)

# 0.001도 = 약 110m 완충지대 격리
buffer_distance = 0.001
train_df_final = train_df[distances > buffer_distance]

dropped_count = len(train_df) - len(train_df_final)
print(f"최초 후보 학습 데이터: {len(train_df)}개")
print(f"경계선 인접 구역에서 삭제된 데이터 (누수 차단): {dropped_count}개")
print(f"버퍼 존 적용 후 최종 학습 데이터: {len(train_df_final)}개")


print("\n피처(X)와 타겟(y) 분리")

# 중간 점수 연산 컬럼 및 원본 훼손 방지용 드롭 리스트 구성
exclude_columns = [
    'target', 'zone', 'latitude', 'longitude', 
    'total_landing_score', 'slope_score', 'tree_density_score', 'tree_height_score', 
    'tree_density', 'tree_height'  # slope_deg는 원본 연속형 수치 학습을 위해 유지
]

X_train = train_df_final.drop(columns=exclude_columns).astype(np.float32)
y_train = train_df_final['target']

X_test = test_df.drop(columns=exclude_columns).astype(np.float32)
y_test = test_df['target']

print("최종 입력 피처 목록:", X_train.columns.tolist())
print(f"학습 데이터 수: {len(X_train)} | 검증 데이터 수: {len(X_test)}")

# RF 모델 설계
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=4,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

# 모델 학습 시작
print("\n학습 시작")
rf_model.fit(X_train, y_train)
print("학습 완료")

# 성능 검증 평가
y_pred = rf_model.predict(X_test)
print("\n[검증 데이터셋 평가 결과]")
print(classification_report(y_test, y_pred))

def find_best_landing_spots(rescue_lat, rescue_lon, search_radius_meters=500):
    print(f"\n[작전 개시] 구조 요청 지점 (위도: {rescue_lat}, 경도: {rescue_lon}) 주변 탐색 시작...")
    
    # 1. 위경도 '도' 단위 변환 (약 110m = 0.001도 공식 활용)
    search_radius_deg = (search_radius_meters / 110.0) * 0.001
    
    # 2. 전체 데이터(df)에서 구조 지점 반경 내에 있는 격자들만 1차 필터링
    # 하버사인 공식 대신 대용량 고속 연산을 위해 유클리드 거리 사각형 마스킹 사용
    lat_min, lat_max = rescue_lat - search_radius_deg, rescue_lat + search_radius_deg
    lon_min, lon_max = rescue_lon - search_radius_deg, rescue_lon + search_radius_deg
    
    candidates = df[
        (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
        (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)
    ].copy()
    
    if len(candidates) == 0:
        print("❌ 해당 반경 내에 존재하는 지형 격자 데이터가 없습니다. 탐색 반경을 넓혀주세요.")
        return None

    # 3. 구조 지점으로부터 각 후보지까지의 실제 직선 거리 계산 (정렬용)
    candidates['dist_to_rescue'] = np.sqrt(
        (candidates['latitude'] - rescue_lat)**2 + 
        (candidates['longitude'] - rescue_lon)**2
    )
    
    # 4. ❌ 기존 코드: X_candidates = candidates.drop(columns=exclude_columns, errors='ignore').astype(np.float32)
    #    ⭕ 변경 코드: 학습 데이터(X_train)의 컬럼 목록과 순서를 그대로 강제 매핑합니다.
    X_candidates = candidates[X_train.columns].astype(np.float32)

    # 5. 우리 AI 모델을 동원해 주변 격자들의 위험도(target) 예측하기
    candidates['pred_target'] = rf_model.predict(X_candidates)
    
    # [수정] 우선순위 조건에 맞춰 전체 후보를 먼저 정렬합니다.
    sorted_candidates = candidates.sort_values(
        by=['pred_target', 'total_landing_score', 'dist_to_rescue'],
        ascending=[True, False, True]
    )
    
    # 6. ⭐ [공간적 다양성 필터링] 바로 옆 격자가 중복 추천되는 것을 방지
    best_spots_list = []
    
    # 💡 최소 거리 두기 설정 (위경도 '도' 단위)
    # 0.0005도 = 약 55m | 0.0009도 = 약 100m
    # 조종사가 전혀 다른 대안지로 인식할 수 있도록 최소 55m 이상 떨어지도록 설정합니다.
    min_separation_deg = 0.0005 
    
    for idx, row in sorted_candidates.iterrows():
        if len(best_spots_list) == 0:
            # 1등 자리는 무조건 채택
            best_spots_list.append(row)
        else:
            # 이미 채택된 대안들과 거리가 충분히 떨어져 있는지 검사
            is_diverse_enough = True
            for selected in best_spots_list:
                distance = np.sqrt(
                    (row['latitude'] - selected['latitude'])**2 + 
                    (row['longitude'] - selected['longitude'])**2
                )
                if distance < min_separation_deg:
                    is_diverse_enough = False
                    break # 너무 가까우면 이 격자는 탈락(Pass)
            
            if is_diverse_enough:
                best_spots_list.append(row)
                
        # 최적의 대안지 3곳을 모두 찾으면 루프 종료
        if len(best_spots_list) == 3:
            break
            
    # 리스트를 데이터프레임으로 복원
    best_spots = pd.DataFrame(best_spots_list)
    
    # 7. 결과 출력 매핑용 딕셔너리
    target_labels = {0: "🟢 착륙 가능 (최적)", 1: "🟡 조건부 착륙 가능", 2: "🔴 착륙 불가 (위험)"}
    
    print(f"\n✨ 구조 지점 반경 {search_radius_meters}m 내 최선의 독립적 착륙 후보지 TOP 3 결과:")
    print("=" * 85)
    for i, (_, row) in enumerate(best_spots.iterrows(), 1):
        print(f"📌 [후보 {i}순위]")
        print(f" - 좌표: (위도: {row['latitude']:.5f}, 경도: {row['longitude']:.5f})")
        print(f" - AI 예측 등급: {target_labels[int(row['pred_target'])]}")
        print(f" - 구조 지점과의 거리: 약 {row['dist_to_rescue'] * 110000:.1f} 미터")
        print(f" - 상세 지형 정보: 고도 {row['elevation']:.1f}m | 경사등급 {int(row['slope_deg'])} | land_type {int(row['land_type'])}")
        print("-" * 85)
        
    return best_spots

# 1. 데이터셋 내부에서 무작위로 진짜 조난 지점 샘플 1개 추출
sample_rescue_point = df.sample(1, random_state=100)

# [⚠️ 핵심 수정] 하드코딩으로 덮어씌우는 대신, 샘플링된 데이터의 실제 위경도 값을 추출합니다.
example_lat = sample_rescue_point['latitude'].values[0]
example_lon = sample_rescue_point['longitude'].values[0]

print(f"💡 데이터셋에서 매칭된 실제 구조 지점 -> 위도: {example_lat:.5f}, 경도: {example_lon:.5f}")

# 2. 함수 가동 (구조 지점 주변 반경 500m 이내 최적 착륙지 탐색)
top_3_spots = find_best_landing_spots(rescue_lat=example_lat, rescue_lon=example_lon, search_radius_meters=500)