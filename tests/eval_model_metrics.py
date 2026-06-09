"""[검증] XGBoost 4모드 모델의 ML 완성도 평가 (test셋).
PDF 기준: 분류 → Accuracy / F1-Score / Confusion Matrix.
"""
import sys, os, glob
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from modules.rescue_zone_inference import _load_models, FEATURE_COLUMNS

BATCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "XGBoost_Model", "dataset", "batches")
CLASSES = ["안전(0)", "주의(1)", "위험(2)"]

# test셋만 로드 (is_test=True)
files = sorted(glob.glob(os.path.join(BATCH_DIR, "*.parquet")))
target_cols = [f"target_{s}_{t}" for s in ("small", "large") for t in ("landing", "hoist")]
lazy = pl.scan_parquet(files).filter(pl.col("is_test"))
df = lazy.select(FEATURE_COLUMNS + target_cols).collect(engine="streaming").to_pandas()
print(f"test셋: {len(df):,}행\n")

X = df[FEATURE_COLUMNS].astype("float32")
dmat = xgb.DMatrix(X)

print("=" * 70)
print(f"{'모드':<16}{'Accuracy':>10}{'F1(macro)':>12}{'F1(위험)':>12}")
print("=" * 70)
results = {}
for size in ("small", "large"):
    models = _load_models(size)
    for tactic in ("landing", "hoist"):
        key = f"{size}_{tactic}"
        y_true = df[f"target_{key}"].values.astype(int)
        pred = models[tactic].predict(dmat)
        y_pred = pred.argmax(axis=1) if pred.ndim == 2 else pred.astype(int)

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_danger = f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
        results[key] = (acc, f1m, y_true, y_pred)
        print(f"{key:<16}{acc:>10.3f}{f1m:>12.3f}{f1_danger:>12.3f}")

print("=" * 70)

# 대표 모드 1개 상세 (large_landing)
key = "large_landing"
acc, f1m, y_true, y_pred = results[key]
print(f"\n=== [{key}] Confusion Matrix (행=실제, 열=예측) ===")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print(f"{'':>10}{'예측안전':>10}{'예측주의':>10}{'예측위험':>10}")
for i, row in enumerate(cm):
    print(f"{CLASSES[i]:>10}" + "".join(f"{v:>10,}" for v in row))
print(f"\n[{key}] 클래스별 상세:")
print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0, digits=3))

# 라벨 분포 (불균형 확인)
print("=== test셋 라벨 분포 (large_landing) ===")
vals, cnts = np.unique(y_true, return_counts=True)
for v, c in zip(vals, cnts):
    print(f"  {CLASSES[v]}: {c:,} ({100*c/len(y_true):.1f}%)")
