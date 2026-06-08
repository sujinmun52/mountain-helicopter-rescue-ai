import os
import sys
from dotenv import load_dotenv

_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, '.env'), override=True)
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import numpy as np
import polars as pl
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import platform


from preprocess.score_utils import FEATURE_COLUMNS, TARGET_COLUMNS

BATCH_DIR   = os.path.join(_DIR, 'dataset', 'batches')
MODEL_DIR   = os.path.join(_DIR, 'models')
CHUNK_ROWS  = 500_000


def calculate_manual_metrics(y_true, y_pred, num_classes=3):
    """
    내장 함수 없이 정답 배열과 예측 배열을 비교하여 
    각 클래스별 TP, TN, FP, FN을 직접 계산하는 함수
    """
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


def plot_evaluation_curves(y_true, y_prob, model_name="XGBoost"):
    """
    [추가 모듈] 다중 클래스 OvR 방식의 ROC 및 PR 곡선을 생성하여 시각화 파일로 내보내는 함수
    """
    n_classes = 3
    # OvR 곡선 묘사를 위한 타겟 이진화 원-핫 레이아웃 인코딩
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
    
    output_path = os.path.join(MODEL_DIR, f"curves_{model_name.lower()}.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[시각화 내보내기 성공] ROC/PR 통합 대시보드가 저장되었습니다 -> {output_path}")
    plt.show()


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
if sel not in TACTICS:
    sys.exit("올바른 번호를 입력하세요.")
model_key, model_name = TACTICS[sel]
target_col = f'target_{model_key}'

dev = input("장치 (1=GPU / 2=CPU): ").strip()
device = {"1": "cuda", "2": "cpu"}.get(dev)
if not device:
    sys.exit("1 또는 2를 입력하세요.")

print(f"\n{model_name} {device.upper()} 학습 시작\n")

# ── [STEP 2] Polars LazyFrame 지연 스캔 ──────────────────────────────────
load_cols = FEATURE_COLUMNS + [target_col, 'is_train_final', 'is_test']

lazy_all = (
    pl.scan_parquet(os.path.join(BATCH_DIR, "*.parquet"))
      .select(load_cols)
)
lazy_train = lazy_all.filter(pl.col('is_train_final')).drop(['is_train_final', 'is_test'])
lazy_test  = lazy_all.filter(pl.col('is_test')).drop(['is_train_final', 'is_test'])

print("검증셋 수집 중 (Polars streaming)...")
test_pd = lazy_test.collect(engine="streaming").to_pandas()
X_test  = test_pd[FEATURE_COLUMNS].values.astype('float32')
y_test  = test_pd[target_col].values.astype('int32')
dtest   = xgb.DMatrix(X_test, label=y_test)
print(f"   검증셋: {len(test_pd):,}행")
del test_pd

# ── [STEP 3] DataIter: 훈련셋 청크 스트리밍 주입 ─────────────────────────
class ParquetChunkIter(xgb.DataIter):
    def __init__(self, lazy: pl.LazyFrame, feat_cols, tgt_col, chunk):
        self._df   = lazy.collect(engine="streaming").to_pandas()
        self._feat = feat_cols
        self._tgt  = tgt_col
        self._step = chunk
        self._n    = len(self._df)
        self._cur  = 0
        super().__init__()

    def next(self, input_data):
        if self._cur >= self._n:
            return 0
        end   = min(self._cur + self._step, self._n)
        chunk = self._df.iloc[self._cur:end]
        input_data(
            data  = chunk[self._feat].values.astype('float32'),
            label = chunk[self._tgt].values.astype('int32'),
        )
        self._cur = end
        return 1

    def reset(self):
        self._cur = 0

print("DataIter 초기화 중...")
it     = ParquetChunkIter(lazy_train, FEATURE_COLUMNS, target_col, CHUNK_ROWS)
dtrain = xgb.QuantileDMatrix(it)

# ── [STEP 4] 학습 ─────────────────────────────────────────────────────────
params = {
    "device": device, "tree_method": "hist",
    "objective": "multi:softprob",  # 🎯 [수정]: ROC/PR 연산을 위해 softmax를 softprob로 교체
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 6, "learning_rate": 0.1,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 4, "gamma": 0.1, "lambda": 1.0,
    "seed": 42, "nthread": 6,
}
model = xgb.train(
    params, dtrain,
    num_boost_round=200,
    evals=[(dtrain, "train"), (dtest, "eval")],
    callbacks=[xgb.callback.EarlyStopping(30, metric_name="mlogloss",
                                           data_name="eval", save_best=True)],
    verbose_eval=10,
)

# ── [STEP 5] 검증 및 저장 ────────────────────────────────────────────────
# 🎯 [수정]: 확률 행렬(N, 3)을 먼저 수집한 뒤 argmax로 하드 레이블 복원하도록 파이프라인 전면 개편
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우 (맑은 고딕)
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'     # 맥 (애플 고딕)
else:
    plt.rcParams['font.family'] = 'NanumBarunGothic' # 리눅스/우분투 (나눔바른고딕)

# 그래프 내부 마이너스(-) 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

y_prob = model.predict(dtest)
y_pred = np.argmax(y_prob, axis=1).astype('int32')

# 1. 사이킷런 표준 지표 출력
print(classification_report(y_test, y_pred, target_names=["안전(0)", "주의(1)", "위험(2)"]))

# 2. 바닥부터 직접 정산하는 수제 행렬 검증 연산 가동
manual_metrics = calculate_manual_metrics(y_test, y_pred, num_classes=3)

# 3. 🎯 [추가 주입]: 확률 매트릭스를 기반으로 다중 클래스 ROC 및 PR 곡선 그래프 플로팅 가동
plot_evaluation_curves(y_test, y_prob, model_name=model_key)

os.makedirs(MODEL_DIR, exist_ok=True)
model.save_model(os.path.join(MODEL_DIR, f'xgb_{model_key}_model.ubj'))
joblib.dump(model, os.path.join(MODEL_DIR, f'xgb_{model_key}_model.joblib'))
print(f"\n{model_name} 저장 완료 → XGBoost_Model/models/")
