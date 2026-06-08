"""
Polars 지연 스캔 및 분산 Parquet 배치를 활용한 LightGBM 가속 학습 스크립트 (GPU/CPU 대응)
"""
import os
import sys
import shutil
from dotenv import load_dotenv

# -- 1. 인프라 및 경로 설정 ------------------------------------------------──
_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, '.env'), override=True)
_PROJECT_ROOT = os.path.dirname(_DIR)

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _DIR)

import numpy as np
import polars as pl
import lightgbm as lgb
import joblib
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

from preprocess.score_utils import FEATURE_COLUMNS

BATCH_DIR = os.path.join(_DIR, 'dataset', 'batches')
MODEL_DIR = os.path.join(_DIR, 'models')


# -- [STEP 1] 인터랙티브 모델 및 장치(GPU/CPU) 선택 ---------------------------
print("\n" + "="*60)
print(" 헬기 제원별 4가지 전술 모델 선택 학습 (LightGBM)")
print("="*60)
TACTICS = {
    "1": ("small_landing", "소형 착륙 모델(L_Light)"),
    "2": ("small_hoist",   "소형 호이스트 모델(H_Light)"),
    "3": ("large_landing", "대형 착륙 모델(L_Heavy)"),
    "4": ("large_hoist",   "대형 호이스트 모델(H_Heavy)"),
}
for k, (_, name) in TACTICS.items():
    print(f" {k}. {name}")

sel = input("학습할 모델 번호 입력 (1~4): ").strip()
if sel not in TACTICS:
    sys.exit("에러: 1번부터 4번 사이의 올바른 숫자만 입력해야 합니다.")
model_key, model_name = TACTICS[sel]
target_col = f'target_{model_key}'

dev = input("연산 장치 선택 (1=GPU / 2=CPU): ").strip()
device = {"1": "gpu", "2": "cpu"}.get(dev)
if not device:
    sys.exit("에러: 1(GPU) 또는 2(CPU)를 입력하세요.")

print(f"\n[선택 확정] {model_name} LightGBM {device.upper()} 모드 가동 및 학습 시작\n")


# -- [STEP 2] Polars LazyFrame 기반 고속 분산 스캔 및 로드 ----------------──
print("--- 1. Polars 지연 스캔 엔진 가동 (*.parquet 배칭 구조) ---")
load_cols = FEATURE_COLUMNS + [target_col, 'is_train_final', 'is_test']

# 물리적으로 쪼개진 모든 배치 파일을 가상으로 연결하여 스캔 메타데이터 생성
lazy_all = (
    pl.scan_parquet(os.path.join(BATCH_DIR, "*.parquet"))
      .select(load_cols)
)

lazy_train = lazy_all.filter(pl.col('is_train_final')).drop(['is_train_final', 'is_test'])
lazy_test  = lazy_all.filter(pl.col('is_test')).drop(['is_train_final', 'is_test'])

# Polars Streaming 규격 매핑(engine="streaming")을 이용해 Pandas 데이터프레임으로 수집
print("-> 훈련 데이터셋 로드 중...")
train_pd = lazy_train.collect(engine="streaming").to_pandas()
X_train  = train_pd[FEATURE_COLUMNS].values.astype('float32')
y_train  = train_pd[target_col].values.astype('int8')

print("-> 검증 데이터셋 로드 중...")
test_pd  = lazy_test.collect(engine="streaming").to_pandas()
X_test   = test_pd[FEATURE_COLUMNS].values.astype('float32')
y_test   = test_pd[target_col].values.astype('int8')

print(f"   - 훈련셋 크기: {X_train.shape[0]:,}행 | 검증셋 크기: {X_test.shape[0]:,}행")

# 불균형 데이터셋 방어를 위한 클래스 가중치 계산
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 변환이 끝난 원본 Pandas 데이터프레임은 즉시 메모리에서 해제
del train_pd, test_pd
import gc; gc.collect()


# -- [STEP 3] LightGBM 전용 Dataset 변환 및 하이퍼파라미터 세팅 --------------
dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights, free_raw_data=False)
dtest  = lgb.Dataset(X_test,  label=y_test,  reference=dtrain,      free_raw_data=False)

# 하이브리드 코어 속도 최적화를 위한 스레드 제어 분기 연산
# CPU 모드일 때는 정예 P코어 개수인 6개로 스레드를 할당하여 E코어 병목 현상을 방지합니다.
computed_threads = 6 if device == "cpu" else -1

# 하이퍼파라미터 구조화 (선택된 연산 장치 동적 결합)
params = {
    "objective":         "multiclass",
    "num_class":         3,
    "metric":            "multi_logloss",
    
    # -- [핵심 1] 입력받은 장치(GPU / CPU) 및 스레드 튜닝 파라미터 동적 바인딩 ----
    "device":            device,
    "num_threads":       computed_threads,

    # -- 트리 구조 및 규제 제약 설정 -----------------------------------------
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 20,          
    "min_child_weight":  1e-3,         
    "min_gain_to_split": 0.0,         
    "max_bin":           255,         

    # -- 서브샘플링 및 속도 제어 ---------------------------------------------
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "learning_rate":     0.1,
    "seed":              42,
    "verbose":           -1,          # 불필요한 로그 생략으로 속도 추가 가속
}

callbacks = [
    lgb.early_stopping(stopping_rounds=30, verbose=True),
    lgb.log_evaluation(period=10),
]


# -- [STEP 4] LightGBM 훈련 가동 ---------------------------------------------
print(f"--- 2. LightGBM 코어 엔진 훈련 시작 ---")
lgbm_model = lgb.train(
    params,
    dtrain,
    num_boost_round=200,
    valid_sets=[dtrain, dtest],
    valid_names=["train", "eval"],
    callbacks=callbacks,
)


# -- [STEP 5] 검증 리포트 분석 및 멀티 포맷 모델 저장 ------------------------
print(f"\n[검증 결과 분석 - {model_key.upper()} LightGBM]")

# LightGBM의 multiclass predict 결과는 확률 행렬이므로 argmax 추출 처리
y_prob = lgbm_model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1).astype(np.int32)

print(classification_report(y_test, y_pred, target_names=["안전(0)", "주의(1)", "위험(2)"]))
print(f"최적의 얼리 스토핑 부스팅 라운드 수: {lgbm_model.best_iteration}회")

# 모델 저장 폴더 인프라 유효성 보장
os.makedirs(MODEL_DIR, exist_ok=True)
output_txt    = os.path.join(MODEL_DIR, f'lgbm_{model_key}_model.txt')
output_joblib = os.path.join(MODEL_DIR, f'lgbm_{model_key}_model.joblib')

# -- [핵심 2] Windows 한글 절대 경로 인코딩 파괴 버그 우회 프로세스 -----------
# 영문 임시 파일로 C++ 저장 단계를 무결하게 통과시킨 후 파이어폴 파일 시스템을 통해 이동시킵니다.
temp_file = f"temp_lgbm_{model_key}_model.txt"
lgbm_model.save_model(temp_file)

if os.path.exists(temp_file):
    shutil.move(temp_file, output_txt)
    print(f"네이티브 텍스트 포맷 보존 성공: {output_txt}")
else:
    print("경고: C++ 내부 문제로 임시 파일 전송에 실패했습니다. 기본 저장 방식으로 우회합니다.")
    lgbm_model.save_model(output_txt)

# 파이썬 고유 라이브러리 인터페이스인 joblib은 자체 인코딩 처리가 되므로 정상 저장 처리
joblib.dump(lgbm_model, output_joblib)

print(f"\n[완료] {model_name} LightGBM 학습 및 파일 내보내기 완료")
print(f"  -> 직렬화 바이너리 포맷: {output_joblib}")
print("="*60)