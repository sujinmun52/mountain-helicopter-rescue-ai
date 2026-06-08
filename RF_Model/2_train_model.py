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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ==============================================================================
# [STEP 1] 대형 데이터 로드 전, 안전을 위한 인터랙티브 모델 선택 메뉴
# ==============================================================================
print("\n" + "="*60)
print("[산악 구조 AI] 독립 기체 제원별 전술 모델 선택 훈련 가동")
print("="*60)
print(" 1. 소형 안착 착륙 모델 (Small Helicopter Landing)")
print(" 2. 소형 강하 호이스트 모델 (Small Helicopter Hoist)")
print(" 3. 대형 안착 착륙 모델 (Large Helicopter Landing)")
print(" 4. 대형 강하 호이스트 모델 (Large Helicopter Hoist)")
print("="*60)

user_input = input("학습을 진행할 모델의 번호를 입력하세요 (1 ~ 4): ").strip()

# 입력값 검증 및 매핑
tactics_map = {
    "1": ("small_landing", "target_small_landing", "소형 착륙 모델"),
    "2": ("small_hoist",   "target_small_hoist",   "소형 호이스트 모델"),
    "3": ("large_landing", "target_large_landing", "대형 착륙 모델"),
    "4": ("large_hoist",   "target_large_hoist",   "대형 호이스트 모델")
}

if user_input not in tactics_map:
    print("\n에러: 1번부터 4번 사이의 올바른 숫자만 입력해야 합니다. 프로그램을 종료합니다.")
    sys.exit()

# 선택된 제원 정보 추출
model_key, target_column, model_kor_name = tactics_map[user_input]

print(f"\n[선택 확정] 이번 회차에는 【 {model_kor_name} 】 딱 하나만 독립 집중 학습합니다.")
print("만약 중간에 문제가 발생하더라도 이 모델 외의 다른 가중치 파일은 안전하게 보존됩니다.\n")


# ==============================================================================
# [STEP 2] 선택 완료 후 비로소 무거운 마스터 데이터셋 로드 시작
# ==============================================================================
print("--- 1. 대용량 복합 시공간 마스터 데이터셋 로드 (PyArrow 병렬 가속 연산) ---")

# 🎯 [피처 복원]: 누락되었던 수목 밀도('tree_density')와 높이('tree_height')를 추가하여 완전한 10대 독립 변수 구축
feature_columns = [
    'elevation', 'slope_deg', 'tree_density', 'tree_height',
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos',
    'land_0', 'land_1', 'land_2'
]

# 이번 학습에 필요한 열만 저격 지정하여 메모리 누수 원천 차단
load_columns = feature_columns + [target_column, 'is_train_final', 'is_test']

# 🎯 [경로 최적화]: 하드코딩 수식을 제거하고 인프라 절대 경로 매핑으로 안전성 보장
csv_path = os.path.join(current_dir, 'dataset', 'processed_seoraksan_master_rf.csv')

df = pd.read_csv(
    csv_path,
    engine='pyarrow',
    usecols=load_columns
)

train_df = df[df['is_train_final'] == True]
test_df = df[df['is_test'] == True]

X_train = train_df[feature_columns].astype(np.float32)
X_test = test_df[feature_columns].astype(np.float32)

print(f"훈련셋 데이터: {len(X_train):,}개 | 검증셋 데이터: {len(X_test):,}개")


# ==============================================================================
# [STEP 3] 선택된 타겟 단 하나만 브루트 포스 방어선 구축 후 훈련
# ==============================================================================
print(f"\n[{model_key.upper()} 단독 가동] 랜덤 포레스트 학습 시작...")

# 하이브리드 아키텍처 최적화 패치
rf_model = RandomForestClassifier(
    n_estimators=400, 
    max_depth=10, 
    min_samples_leaf=4,
    class_weight='balanced', 
    n_jobs=6,  # 고성능 P코어 개수 체계 바인딩
    random_state=42,
    max_samples=4000000  # 메모리 부족 차단용 하위 샘플링 버퍼
)

# 선택된 타겟 컬럼 하나만 타겟팅하여 피팅(Fit)
rf_model.fit(X_train, train_df[target_column])

# 평가지표 출력
print(f"\n[검증 결과 분석 - {model_key.upper()}]")
print(classification_report(test_df[target_column], rf_model.predict(X_test), target_names=["안전(0)", "주의(1)", "위험(2)"]))

# 🎯 [경로 최적화]: 모델 직렬화 내보내기도 패키지 내부 절대 경로로 통일
output_model_path = os.path.join(current_dir, f'rf_{model_key}_model.joblib')
joblib.dump(rf_model, output_model_path)

print(f"\n[완료] 【 {model_kor_name} 】 단독 학습 및 직렬화 완료 -> {output_model_path}")
print("==============================================================================")