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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

print("--- 1. 기상 통합 마스터 데이터 로드 ---")
df = pd.read_csv(r'RF_Model\dataset\processed_seoraksan_master.csv')

train_df = df[df['is_train_final'] == True]
test_df = df[df['is_test'] == True]

# 실시간 변동 기상 피처 3종을 정식 피처 구조에 매핑
feature_columns = [
    'elevation', 'slope_deg', 'land_0', 'land_1', 'land_2', 
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos'
]

X_train = train_df[feature_columns].astype(np.float32)
X_test = test_df[feature_columns].astype(np.float32)

print(f"학습 데이터 수: {len(X_train)}개 | 검증 데이터 수: {len(X_test)}개\n")


# 🛫 [MODEL 1] 직접 착륙(Landing) 위험도 예측 모델 학습
print("--- 2-1. [착륙 모델] 랜덤 포레스트 학습 시작 ---")
rf_landing = RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_leaf=4,
    class_weight='balanced', n_jobs=-1, random_state=42
)
rf_landing.fit(X_train, train_df['target_landing'])

print("\n📊 [검증 데이터셋 평가 결과 - 착륙 모델]")
print(classification_report(test_df['target_landing'], rf_landing.predict(X_test)))
joblib.dump(rf_landing, r'RF_Model\rf_landing_model.joblib')
print("🎉 착륙 모델 직렬화 성공!\n")


# 🏗️ [MODEL 2] 제자리 비행 호이스트(Hoist) 위험도 예측 모델 학습
print("--- 2-2. [호이스트 모델] 랜덤 포레스트 학습 시작 ---")
rf_hoist = RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_leaf=4,
    class_weight='balanced', n_jobs=-1, random_state=42
)
rf_hoist.fit(X_train, train_df['target_hoist'])

print("\n📊 [검증 데이터셋 평가 결과 - 호이스트 모델]")
print(classification_report(test_df['target_hoist'], rf_hoist.predict(X_test)))
joblib.dump(rf_hoist, r'RF_Model\rf_hoist_model.joblib')
print("🎉 호이스트 모델 직렬화 성공!")