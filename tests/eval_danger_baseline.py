"""[변별력 검정] 위험지역 6곳의 위험점수가 설악산 '무작위 지점'보다 유의하게 높은가?
표본 6개 한계를 보완 — 위험지역이 baseline 대비 변별되는지 통계적으로 확인.
"""
import sys, os, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from modules.rescue_zone_inference import _load_master, _load_models, FEATURE_COLUMNS

HELI = "large"
master = _load_master()

# softprob 기대위험도(연속): P(주의)*0.5 + P(위험)*1.0  (0~1, 높을수록 위험)
def expected_risk(cells):
    cells = cells.copy()
    cells["wind_speed"] = 6.5
    cells["wind_dir_sin"] = np.sin(np.radians(270.0)).astype("float32")
    cells["wind_dir_cos"] = np.cos(np.radians(270.0)).astype("float32")
    dmat = xgb.DMatrix(cells[FEATURE_COLUMNS].astype("float32"))
    models = _load_models(HELI)
    er = np.zeros(len(cells))
    for tactic in ("landing", "hoist"):
        p = models[tactic].predict(dmat)
        er = np.maximum(er, p[:, 1]*0.5 + p[:, 2]*1.0)  # 보수적(더 위험한 전술)
    return er

# 1) 위험지역 6곳 (영역 내)
dz = pd.read_csv("data/설악산_위험지역.csv", encoding="utf-8-sig")
dz = dz[dz["영역내"] == True].reset_index(drop=True)
tree = cKDTree(master[["latitude", "longitude"]].values)
_, idx = tree.query(dz[["lat", "lon"]].values)
risk_dz = expected_risk(master.iloc[idx])

# 2) baseline: 설악산 무작위 지점 N개 (하천 land_2 제외 = 착륙 후보지만)
pool = master[master["land_2"] != 1] if "land_2" in master.columns else master
base = pool.sample(n=5000, random_state=42)
risk_base = expected_risk(base)

# 3) 비교
print("=== 위험점수(기대위험도 0~1) 비교 ===")
print(f"  위험지역 6곳:   평균 {risk_dz.mean():.3f}  (개별: {', '.join(f'{x:.2f}' for x in risk_dz)})")
print(f"  무작위 5000곳:  평균 {risk_base.mean():.3f}  (중앙값 {np.median(risk_base):.3f})")

# 위험지역 각 점이 baseline 분포에서 상위 몇 %인지
pct = [100*(risk_base < r).mean() for r in risk_dz]
print(f"\n  위험지역이 baseline 분포의 상위 백분위: {[f'{p:.0f}%' for p in pct]}")
print(f"  평균 백분위: {np.mean(pct):.0f}% (50%=평균수준, 높을수록 변별력↑)")

# Mann-Whitney U (분포 차이 유의성)
u, p = mannwhitneyu(risk_dz, risk_base, alternative="greater")
print(f"\n  Mann-Whitney U 검정 (위험지역 > 무작위): p = {p:.4f}")
print(f"  → {'유의함 (변별력 있음)' if p < 0.05 else '유의하지 않음 (변별력 약함)'}")

# 효과크기 (위험지역 평균이 baseline 분포에서 어디)
print(f"\n  baseline 위험비율(>0.5): {100*(risk_base>0.5).mean():.0f}%  "
      f"vs 위험지역(>0.5): {100*(risk_dz>0.5).mean():.0f}%")
