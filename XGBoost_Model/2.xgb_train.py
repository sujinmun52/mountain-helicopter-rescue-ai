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
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import joblib

# ==============================================================================
# [STEP 1] 인터랙티브 모델 선택 메뉴
# ==============================================================================
print("\n" + "="*60)
print("헬기 제원별 4가지 방법 모델 선택 학습")
print("="*60)
print(" 1. 소형 안착 착륙 모델 (Small Helicopter Landing - L_Light)")
print(" 2. 소형 강하 호이스트 모델 (Small Helicopter Hoist - H_Light)")
print(" 3. 대형 안착 착륙 모델 (Large Helicopter Landing - L_Heavy)")
print(" 4. 대형 강하 호이스트 모델 (Large Helicopter Hoist - H_Heavy)")
print("="*60)

user_input = input("학습을 진행할 모델의 번호를 입력하세요 (1 ~ 4): ").strip()

tactics_map = {
    "1": ("small_landing", "target_small_landing", "소형 착륙 모델(L_Light)"),
    "2": ("small_hoist",   "target_small_hoist",   "소형 호이스트 모델(H_Light)"),
    "3": ("large_landing", "target_large_landing", "대형 착륙 모델(L_Heavy)"),
    "4": ("large_hoist",   "target_large_hoist",   "대형 호이스트 모델(H_Heavy)"),
}

if user_input not in tactics_map:
    print("\n에러: 1번부터 4번 사이의 올바른 숫자만 입력해야 합니다. 프로그램을 종료합니다.")
    sys.exit()

model_key, target_column, model_kor_name = tactics_map[user_input]


# ==============================================================================
# [STEP 1.5] 연산 장치(GPU / CPU) 선택 메뉴
# ==============================================================================
print("\n" + "="*60)
print("연산 장치 선택")
print("="*60)
print(" 1. GPU 가속 모드 (NVIDIA CUDA)")
print(" 2. CPU 모드")
print("="*60)

proc_input = input("학습에 사용할 장치의 번호를 입력하세요 (1 또는 2): ").strip()

if proc_input == "1":
    device_param = "cuda"
    proc_name = "GPU 가속 (cuda)"
elif proc_input == "2":
    device_param = "cpu"
    proc_name = "CPU"
else:
    print("\n에러: 1번 또는 2번 중 하나만 선택해야 합니다. 프로그램을 종료합니다.")
    sys.exit()

print(f"\n이번 회차: 【 {model_kor_name} 】 XGBoost {proc_name} 집중 학습\n")


# ==============================================================================
# [STEP 2] 마스터 데이터셋 로드
# ==============================================================================
print("--- 1. 대용량 복합 시공간 마스터 데이터셋 로드 (잠시 기다려주세요) ---")
df = pd.read_csv(r'RF_Model\dataset\processed_seoraksan_master.csv')

train_df = df[df['is_train_final'] == True].copy()
test_df  = df[df['is_test']       == True].copy()

# 항공 교범 가중치 매트릭스 연산에 개입하는 12대 독립 변수 컬럼 정의
feature_columns = [
    'elevation', 'slope_deg', 'tree_density', 'tree_height',
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos', 'wind_dir_score', 'altitude_score',
    'land_0', 'land_1', 'land_2'
]

X_train = train_df[feature_columns].astype(np.float32)
y_train = train_df[target_column].astype(np.int32)

X_test  = test_df[feature_columns].astype(np.float32)
y_test  = test_df[target_column].astype(np.int32)

print(f"훈련셋: {X_train.shape[0]:,}개 | 검증셋: {X_test.shape[0]:,}개")

# 클래스 불균형 보정 (balanced)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)


# ==============================================================================
# [STEP 3] XGBoost DMatrix 변환 후 선택된 장치로 학습
# ==============================================================================
print(f"\n[{model_key.upper()} XGBoost 가동] 학습 시작...")

dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    "device":            device_param,    # 선택된 연산 장치 동적 주입
    "tree_method":       "hist",          # 히스토그램 알고리즘 (CPU/GPU 공용 효율적 타겟)
    "objective":         "multi:softmax", # 3클래스 전술 리스크 분류
    "num_class":         3, 
    "eval_metric":       "mlogloss",
    "max_depth":         6, 
    "learning_rate":     0.1, 
    "subsample":         0.8, 
    "colsample_bytree":  0.8, 
    "min_child_weight":  4, 
    "gamma":             0.1, 
    "lambda":            1.0, 
    "seed":              42,
    "nthread":           -1
}

callbacks = [
    xgb.callback.EarlyStopping(
        rounds=30,
        metric_name="mlogloss",
        data_name="eval",
        save_best=True,
    )
]

evals = [(dtrain, "train"), (dtest, "eval")]

xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=400,
    evals=evals,
    callbacks=callbacks,
    verbose_eval=10,
)

# ==============================================================================
# [STEP 4] 검증 리포트 출력 및 저장
# ==============================================================================
print(f"\n[검증 결과 분석 - {model_key.upper()} XGBoost]")
y_pred = xgb_model.predict(dtest).astype(np.int32)
print(classification_report(y_test, y_pred, target_names=["안전(0)", "주의(1)", "위험(2)"]))

# 저장 경로 정의
output_model_path_ubj    = fr'RF_Model\xgb_{model_key}_model.ubj'
output_model_path_joblib = fr'RF_Model\xgb_{model_key}_model.joblib'

xgb_model.save_model(output_model_path_ubj)
joblib.dump(xgb_model, output_model_path_joblib)

print(f"\n【 {model_kor_name} 】 XGBoost 학습 및 파일 저장 완료!")
print(f"네이티브 보안 포맷 보존: {output_model_path_ubj}")
print("="*60)