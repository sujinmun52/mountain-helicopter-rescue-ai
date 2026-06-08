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
import matplotlib.pyplot as plt
import platform
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from preprocess.score_utils import FEATURE_COLUMNS

# 🎯 [인프라 셋업] 맷플롯립 한글 깨짐 방지 및 마이너스 기호 깨짐 해결 글로벌 패치
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우 (맑은 고딕)
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'     # 맥 (애플 고딕)
else:
    plt.rcParams['font.family'] = 'NanumBarunGothic' # 리눅스/우분투 (나눔바른고딕)
plt.rcParams['axes.unicode_minus'] = False

BATCH_DIR = os.path.join(_DIR, 'dataset', 'batches')
MODEL_DIR = os.path.join(_DIR, 'models')


# ==============================================================================
# [이식 모듈 1] 원천 행렬 비교 연산 기반 커스텀 평가지표 리포트 함수
# ==============================================================================
def calculate_manual_metrics(y_true, y_pred, num_classes=3):
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    
    class_labels = {0: "안전(0)", 1: "주의(1)", 2: "위험(2)"}
    evaluation_report = {}
    
    print("\n" + "="*60)
    print("원천 행렬 비교 연산 기반 커스텀 평가지표 리포트 (LightGBM)")
    print("="*60)
    
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        tn = np.sum((y_true != c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        evaluation_report[c] = {
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Precision": precision, "Recall": recall, "F1-Score": f1_score
        }
        
        print(f"클래스 명세: {class_labels[c]}")
        print(f"  - True Positive (정확한 분류) : {tp:,}개")
        print(f"  - True Negative (정확한 기각) : {tn:,}개")
        print(f"  - False Positive (위험 오탐지) : {fp:,}개")
        print(f"  - False Negative (안전 오탐지) : {fn:,}개")
        print(f"  -> 자체 정밀도: {precision:.4f} | 재현율: {recall:.4f} | F1: {f1_score:.4f}")
        print("-" * 60)
        
    return evaluation_report


# ==============================================================================
# [이식 모듈 2] 다중 클래스 OvR 방식의 ROC 및 PR 곡선 대시보드 플로팅 함수
# ==============================================================================
def plot_evaluation_curves(y_true, y_prob, model_name="LightGBM"):
    n_classes = 3
    y_true_binarized = label_binarize(y_true, classes=[0, 1, 2])
    
    class_labels = {0: "Class 0: Safe (안전)", 1: "Class 1: Caution (주의)", 2: "Class 2: Danger (위험)"}
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # 1. Left Panel: ROC Curve
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i], lw=2.5, label=f"{class_labels[i]} (AUC = {roc_auc:.4f})")
        
    ax1.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.5, label='Random Guess (AUC = 0.50)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, labelpad=6)
    ax1.set_ylabel('True Positive Rate (TPR / Sensitivity)', fontsize=11, labelpad=6)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # 2. Right Panel: Precision-Recall Curve
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_prob[:, i])
        ap_score = average_precision_score(y_true_binarized[:, i], y_prob[:, i])
        ax2.plot(recall, precision, color=colors[i], lw=2.5, label=f"{class_labels[i]} (AP = {ap_score:.4f})")
        
        # 불균형 클래스 분포 지분율 기반의 개별 베이스라인 플로팅
        baseline = np.sum(y_true_binarized[:, i]) / len(y_true)
        ax2.axhline(y=baseline, color=colors[i], linestyle='--', alpha=0.35)
        
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (재현율)', fontsize=11, labelpad=6)
    ax2.set_ylabel('Precision (정밀도)', fontsize=11, labelpad=6)
    ax2.set_title('Precision-Recall (PR) Curve', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.suptitle(f'[{model_name.upper()}] Performance Evaluation Dashboard', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(MODEL_DIR, f"curves_lgbm_{model_name.lower()}.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[시각화 내보내기 성공] LightGBM ROC/PR 통합 대시보드가 저장되었습니다 -> {output_path}")
    plt.show()


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

lazy_all = (
    pl.scan_parquet(os.path.join(BATCH_DIR, "*.parquet"))
      .select(load_cols)
)

lazy_train = lazy_all.filter(pl.col('is_train_final')).drop(['is_train_final', 'is_test'])
lazy_test  = lazy_all.filter(pl.col('is_test')).drop(['is_train_final', 'is_test'])

print("-> 훈련 데이터셋 로드 중...")
train_pd = lazy_train.collect(engine="streaming").to_pandas()
X_train  = train_pd[FEATURE_COLUMNS].values.astype('float32')
y_train  = train_pd[target_col].values.astype('int8')

print("-> 검증 데이터셋 로드 중...")
test_pd  = lazy_test.collect(engine="streaming").to_pandas()
X_test   = test_pd[FEATURE_COLUMNS].values.astype('float32')
y_test   = test_pd[target_col].values.astype('int8')

print(f"   - 훈련셋 크기: {X_train.shape[0]:,}행 | 검증셋 크기: {X_test.shape[0]:,}행")

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

del train_pd, test_pd
import gc; gc.collect()


# -- [STEP 3] LightGBM 전용 Dataset 변환 및 하이퍼파라미터 세팅 --------------
dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights, free_raw_data=False)
dtest  = lgb.Dataset(X_test,  label=y_test,  reference=dtrain,       free_raw_data=False)

computed_threads = 6 if device == "cpu" else -1

params = {
    "objective":         "multiclass",
    "num_class":         3,
    "metric":            "multi_logloss",
    "device":            device,
    "num_threads":       computed_threads,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 20,          
    "min_child_weight":  1e-3,         
    "min_gain_to_split": 0.0,         
    "max_bin":           255,         
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "learning_rate":     0.1,
    "seed":              42,
    "verbose":           -1,          
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

# 1. 확률 행렬 수집 후 최종 하드 라벨 디코딩
y_prob = lgbm_model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1).astype(np.int32)

# 2. 사이킷런 표준 classification_report 출력
print(classification_report(y_test, y_pred, target_names=["안전(0)", "주의(1)", "위험(2)"]))
print(f"최적의 얼리 스토핑 부스팅 라운드 수: {lgbm_model.best_iteration}회")

# 3. 🎯 [추가]: 원천 비교 연산 기반 수제 평가지표 리포트(TP, TN, FP, FN) 엔진 가동
manual_metrics = calculate_manual_metrics(y_test, y_pred, num_classes=3)

# 4. 🎯 [추가]: softprob 결과 매트릭스를 기반으로 한 다중 클래스 ROC/PR 대시보드 저장 가동
plot_evaluation_curves(y_test, y_prob, model_name=model_key)

# 모델 저장 폴더 인프라 유효성 보장
os.makedirs(MODEL_DIR, exist_ok=True)
output_txt    = os.path.join(MODEL_DIR, f'lgbm_{model_key}_model.txt')
output_joblib = os.path.join(MODEL_DIR, f'lgbm_{model_key}_model.joblib')

# Windows 한글 경로 인코딩 버그 우회 프로세스
temp_file = f"temp_lgbm_{model_key}_model.txt"
lgbm_model.save_model(temp_file)

if os.path.exists(temp_file):
    shutil.move(temp_file, output_txt)
    print(f"네이티브 텍스트 포맷 보존 성공: {output_txt}")
else:
    print("경고: 임시 파일 생성 실패로 우회 저장합니다.")
    lgbm_model.save_model(output_txt)

joblib.dump(lgbm_model, output_joblib)

print(f"\n[완료] {model_name} LightGBM 학습 및 파일 내보내기 완료")
print(f"  -> 직렬화 바이너리 포맷: {output_joblib}")
print("="*60)