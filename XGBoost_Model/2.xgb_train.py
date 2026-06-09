"""
[소거 실험 대조군] 오직 도메인 파생 피처 엔지니어링(Feature 추가)만 반영한 XGBoost 학습 스크립트
"""
import os
import sys
import gc  # 메모리 OOM 방어를 위한 가비지 컬렉션 모듈 선언 완비
from dotenv import load_dotenv

# -- 1. 인프라 및 경로 설정 ------------------------------------------------──
_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, '.env'), override=True)
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import platform
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from preprocess.score_utils import FEATURE_COLUMNS, TARGET_COLUMNS

# 맷플롯립 한글 깨짐 방지 글로벌 패치
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

BATCH_DIR   = os.path.join(_DIR, 'dataset', 'batches')
MODEL_DIR   = os.path.join(_DIR, 'models')
CHUNK_ROWS  = 500_000


# ==============================================================================
# [유지 항목] 순수 날것의 독립 변수 기반 비선형 파생 피처 엔지니어링 (치팅 전면 배제)
# ==============================================================================
def apply_advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """인간의 규칙 점수를 전면 배제하고, 순수 지형/기상 물리 변수들의 상호작용 피처만 생성합니다."""
    # 1. 수목 복합 위험도 (tree_risk): 밀도와 높이의 비선형 결합
    df['tree_risk'] = (df['tree_density'] * df['tree_height']).astype('float32')
    
    # 2. 대기 기류 리스크 (aero_risk): 고고도 강풍구역 산악파 발생 맥락 매핑
    df['aero_risk'] = (df['elevation'] * df['wind_speed']).astype('float32')
    
    # 3. 경사 사면-풍속 복합 리스크 (slope_wind_risk): 사면 리스크와 돌풍 리스크의 시너지 매핑
    df['slope_wind_risk'] = (df['slope_deg'] * df['wind_speed']).astype('float32')
    
    return df


def calculate_manual_metrics(y_true, y_pred, num_classes=3):
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    class_labels = {0: "안전(0)", 1: "주의(1)", 2: "위험(2)"}
    evaluation_report = {}
    
    print("\n" + "="*60)
    print("원천 행렬 비교 연산 기반 커스텀 평가지표 리포트")
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


def plot_evaluation_curves(y_true, y_prob, model_key):
    n_classes = 3
    y_true_binarized = label_binarize(y_true, classes=[0, 1, 2])
    class_labels = {0: "Class 0: Safe (안전)", 1: "Class 1: Caution (주의)", 2: "Class 2: Danger (위험)"}
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
        ax1.plot(fpr, tpr, color=colors[i], lw=2.5, label=f"{class_labels[i]} (AUC = {auc(fpr, tpr):.4f})")
    ax1.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.5, label='Random Guess (AUC = 0.50)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (TPR / Sensitivity)', fontsize=11)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_prob[:, i])
        ax2.plot(recall, precision, color=colors[i], lw=2.5, label=f"{class_labels[i]} (AP = {average_precision_score(y_true_binarized[:, i], y_prob[:, i]):.4f})")
        baseline = np.sum(y_true_binarized[:, i]) / len(y_true)
        ax2.axhline(y=baseline, color=colors[i], linestyle='--', alpha=0.35)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (재현율)', fontsize=11)
    ax2.set_ylabel('Precision (정밀도)', fontsize=11)
    ax2.set_title('Precision-Recall (PR) Curve', fontsize=12, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.suptitle(f'[XGB_{model_key.upper()}_Feature] Performance Dashboard', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"curves_xgb_{model_key.lower()}_feature.png"), dpi=300)
    plt.close()


# ── [STEP 1] 모델 선택 ────────────────────────────────────────────────────
print("\n" + "="*60)
TACTICS = {
    "1": ("small_landing", "소형 착륙 모델(L_Light)"),
    "2": ("small_hoist",   "소형 호이스트 모델(H_Light)"),
    "3": ("large_landing", "대형 착륙 모델(L_Heavy)"),
    "4": ("large_hoist",   "대형 호이스트 모델(H_Heavy)"),
}
for k, (_, name) in TACTICS.items():
    print(f" {k}. {name}")

sel = input("번호 입력 (1~4): ").strip()
if sel not in TACTICS: sys.exit("올바른 번호를 입력하세요.")
model_key, model_name = TACTICS[sel]
target_col = f'target_{model_key}'

dev = input("장치 (1=GPU / 2=CPU): ").strip()
device = "cuda" if dev == "1" else "cpu"

print(f"\n[실험군 학습] {model_name} XGBoost {device.upper()} 피처 확장 전용 파이프라인 가동\n")

# ── [STEP 2] Polars 지연 스캔 및 확장 13대 피처 추출 ──────────────────────
load_cols = FEATURE_COLUMNS + [target_col, 'is_train_final', 'is_test']

lazy_all = pl.scan_parquet(os.path.join(BATCH_DIR, "*.parquet")).select(load_cols)
lazy_train = lazy_all.filter(pl.col('is_train_final')).drop(['is_train_final', 'is_test'])
lazy_test  = lazy_all.filter(pl.col('is_test')).drop(['is_train_final', 'is_test'])

print("검증셋 수집 및 파생 피처 엔지니어링 처리 중...")
test_pd = lazy_test.collect(engine="streaming").to_pandas()
test_pd = apply_advanced_feature_engineering(test_pd)

# 🎯 확장된 13대 피처 매트릭스 리스트 동기화 활용
EXTENDED_FEATURES = FEATURE_COLUMNS + ['tree_risk', 'aero_risk', 'slope_wind_risk']

X_test  = test_pd[EXTENDED_FEATURES].values.astype('float32')
y_test  = test_pd[target_col].values.astype('int32')
dtest   = xgb.DMatrix(X_test, label=y_test)
print(f"   검증셋 매트릭스 차원: {X_test.shape[0]:,}행 × {X_test.shape[1]}열")
del test_pd

print("훈련셋 수집 및 다운캐스팅 가동...")
train_pd = lazy_train.collect(engine="streaming").to_pandas()
train_pd = apply_advanced_feature_engineering(train_pd)
X_train = train_pd[EXTENDED_FEATURES].values.astype('float32')
y_train = train_pd[target_col].values.astype('int32')
print(f"   훈련셋 매트릭스 차원: {X_train.shape[0]:,}행 × {X_train.shape[1]}열")

del train_pd
gc.collect() 


# ── [STEP 3] DataIter: 순정 상태(가중치 배제) 청크 반복자 가동 ───────────────
class BaselineParquetChunkIter(xgb.DataIter):
    def __init__(self, X, y, chunk_size):
        self._X = X
        self._y = y
        self._step = chunk_size
        self._n = len(X)
        self._cur = 0
        super().__init__()

    def next(self, input_data):
        if self._cur >= self._n:
            return 0
        end = min(self._cur + self._step, self._n)
        
        # 🎯 [소거 조치 2]: 비용 가중치(weight) 레이어를 완전히 걷어내고 순수 데이터셋만 전송
        input_data(
            data=self._X[self._cur:end],
            label=self._y[self._cur:end]
        )
        self._cur = end
        return 1

    def reset(self):
        self._cur = 0

print("QuantileDMatrix 인프라 구조체 생성 중 (피처 확장 스트리밍 주입)...")
it = BaselineParquetChunkIter(X_train, y_train, CHUNK_ROWS)
dtrain = xgb.QuantileDMatrix(it)


# ==============================================================================
# [소거 조치 3] Optuna 베이지안 최적화 하이퍼파라미터 오토 튜닝 엔진 전면 삭제
# ==============================================================================


# ── [STEP 4] 베이스라인 고정 하이퍼파라미터 기반 피팅 가동 ────────────────────
print(f"\n--- 베이스라인 기본 하이퍼파라미터 고정 기준 훈련 개시 ---")
baseline_params = {
    "device": device, 
    "tree_method": "hist",
    "objective": "multi:softprob", 
    "num_class": 3, 
    "eval_metric": "mlogloss",
    "max_depth": 6, 
    "learning_rate": 0.1,
    "subsample": 0.8, 
    "colsample_bytree": 0.8,
    "min_child_weight": 4, 
    "gamma": 0.1, 
    "lambda": 1.0,
    "seed": 42, 
    "nthread": 6,
}

model = xgb.train(
    baseline_params, dtrain,
    num_boost_round=200,
    evals=[(dtrain, "train"), (dtest, "eval")],
    callbacks=[xgb.callback.EarlyStopping(30, metric_name="mlogloss", data_name="eval", save_best=True)],
    verbose_eval=20,
)


# ── [STEP 5] 다차원 검증 및 성능 곡선 플로팅 ────────────────────────────────
y_prob = model.predict(dtest)
y_pred = np.argmax(y_prob, axis=1).astype('int32')

print(f"\n[소거실험 검증 분석 - XGB_{model_key.upper()}_Feature_Only]")
print(classification_report(y_test, y_pred, target_names=["안전(0)", "주의(1)", "위험(2)"]))

# 1. 바닥부터 계산하는 수제 혼동행렬 지표 리포팅
manual_metrics = calculate_manual_metrics(y_test, y_pred, num_classes=3)

# 2. 고해상도 대시보드 플롯 파일 생성 자동 가동
plot_evaluation_curves(y_test, y_prob, model_key)

# 실험군 전용 고유 파일명 명세 저장결착
model.save_model(os.path.join(MODEL_DIR, f'xgb_{model_key}_feature_model.ubj'))
joblib.dump(model, os.path.join(MODEL_DIR, f'xgb_{model_key}_feature_model.joblib'))
print(f"\n[대조군 훈련 완료] 피처 확장 단독 처리 모델 및 대시보드 저장 성공.")
print("="*60)