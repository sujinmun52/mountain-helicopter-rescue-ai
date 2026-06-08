"""[진단] Stage 2 XGBoost 추론이 '제대로' 들어갔는지 점검.
 - parquet 컬럼/피처 존재 확인
 - 실시간 풍속 민감도 (모델이 바람에 반응하는가)
 - 후보 risk_class 분포 (trivial 여부)
 - risk_class(모델) vs risk_score(계산) 일관성
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import modules.rescue_zone_inference as rzi
from modules.rescue_zone_inference import (_load_master, _load_models,
                                           FEATURE_COLUMNS, TACTIC_WEIGHTS,
                                           compute_wind_score, compute_wind_dir_score)

HELI = "large"
GPS = {"latitude": 38.1192, "longitude": 128.4652}
R = 500.0

master = _load_master()
print("=== [1] parquet 컬럼 점검 ===")
print("총 행:", len(master), "/ 컬럼 수:", len(master.columns))
miss = [c for c in FEATURE_COLUMNS if c not in master.columns]
print("FEATURE_COLUMNS 누락:", miss if miss else "없음 ✅")
for key in (f"{HELI}_landing", f"{HELI}_hoist"):
    col = f"static_score_{key}"
    print(f"  static_score 컬럼 [{col}]:", "있음 ✅" if col in master.columns else "❌ 없음")

# 반경 후보 추출 (추론 코드와 동일 로직 일부 재현)
v_lat, v_lon = GPS["latitude"], GPS["longitude"]
deg = R / 111_000.0
box = master[(master["latitude"].between(v_lat-deg, v_lat+deg)) &
             (master["longitude"].between(v_lon-deg, v_lon+deg))].copy()
dlat = (box["latitude"]-v_lat)*111_000.0
dlon = (box["longitude"]-v_lon)*111_000.0*np.cos(np.radians(v_lat))
box["d"] = np.sqrt(dlat**2+dlon**2)
cand = box[box["d"] <= R].copy()
if "land_2" in cand.columns:
    cand = cand[cand["land_2"] != 1].copy()
print("\n반경 내 후보 셀:", len(cand))
print("land 원핫 합계(0/1/2):",
      [int(cand[c].sum()) for c in ("land_0","land_1","land_2") if c in cand])

models = _load_models(HELI)

def predict_at_wind(ws, wd=270.0):
    c = cand.copy()
    c["wind_speed"] = float(ws)
    c["wind_direction"] = float(wd)
    wr = np.radians(c["wind_direction"].values)
    c["wind_dir_sin"] = np.sin(wr).astype("float32")
    c["wind_dir_cos"] = np.cos(wr).astype("float32")
    X = c[FEATURE_COLUMNS].astype("float32")
    dmat = xgb.DMatrix(X)
    out = {}
    for tactic in ("landing","hoist"):
        pred = models[tactic].predict(dmat)
        cls = pred.argmax(axis=1)
        out[tactic] = cls
    return out, c

print("\n=== [2] 실시간 풍속 민감도 (모델 예측 클래스 분포) ===")
print("  풍속  | landing(안전0/주의1/위험2) | hoist(0/1/2)")
for ws in (0.0, 5.0, 10.0, 15.0, 25.0):
    out, _ = predict_at_wind(ws)
    lb = np.bincount(out["landing"], minlength=3)
    hb = np.bincount(out["hoist"], minlength=3)
    print(f"  {ws:5.1f} |  {lb[0]:4d}/{lb[1]:4d}/{lb[2]:4d}        |  {hb[0]:4d}/{hb[1]:4d}/{hb[2]:4d}")

print("\n=== [3] risk_class(모델) vs risk_score(계산) 일관성 (풍속 7m/s) ===")
out, c = predict_at_wind(7.0)
c["wind_score"] = compute_wind_score(c["wind_speed"].values)
c["wind_dir_score"] = compute_wind_dir_score(c["latitude"].values, c["longitude"].values,
                                             c["wind_direction"].values)
for tactic in ("landing","hoist"):
    key = f"{HELI}_{tactic}"
    d = TACTIC_WEIGHTS[key]["dynamic"]
    rs = (c[f"static_score_{key}"].values
          + c["wind_score"].values*d["wind_score"]
          + c["wind_dir_score"].values*d["wind_dir_score"])
    cls = out[tactic]
    # 라벨 규약: 0=안전(>=0.80),1=주의(>=0.55),2=위험(<0.55)
    score_label = np.where(rs>=0.80, 0, np.where(rs>=0.55, 1, 2))
    agree = np.mean(score_label == cls)*100
    print(f"  [{key}] risk_score 범위 {rs.min():.2f}~{rs.max():.2f} | "
          f"모델클래스 분포 {np.bincount(cls,minlength=3)} | "
          f"score라벨과 일치율 {agree:.0f}%")
