"""[실용 예측력 검증] 국립공원 '위험지역'(실제 정답) vs 모델 위험 예측 대조.
설악산 위험지역 전체를 추출·저장하고, 모델이 '위험/주의'로 보는지 검증.
"""
import sys, os, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree
from modules.rescue_zone_inference import _load_master, _load_models, FEATURE_COLUMNS

SRC = "data/국립공원공단_국립공원 위험지역 공간데이터_20230212.csv"
OUT = "data/설악산_위험지역.csv"
HELI = "large"
CLASSES = ["안전(0)", "주의(1)", "위험(2)"]

# 1) 설악산 위험지역 추출 + 좌표 파싱
dz = pd.read_csv(SRC, encoding="cp949")
dz = dz[dz["국립공원명"].str.contains("설악", na=False)].copy()
def parse_pt(s):
    m = re.search(r"POINT\(([\d.]+)\s+([\d.]+)\)", str(s))
    return (float(m.group(2)), float(m.group(1))) if m else (None, None)
dz[["lat", "lon"]] = dz["GIS위치"].apply(lambda s: pd.Series(parse_pt(s)))
dz = dz.dropna(subset=["lat", "lon"]).reset_index(drop=True)
print(f"설악산 위험지역 전체: {len(dz)}개")

# 데이터 영역 내외 플래그
master = _load_master()
LAT0, LAT1 = master["latitude"].min(), master["latitude"].max()
LON0, LON1 = master["longitude"].min(), master["longitude"].max()
dz["영역내"] = (dz.lat.between(LAT0, LAT1)) & (dz.lon.between(LON0, LON1))
print(f"  데이터 영역 내: {int(dz['영역내'].sum())}개 / 밖(경계외삽): {int((~dz['영역내']).sum())}개")

# 2) 깔끔한 CSV로 저장
dz_save = dz[["고유번호", "명칭", "위험유형", "lat", "lon", "영역내"]]
dz_save.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"  저장 → {OUT}\n")

# 3) 각 지점 → 가장 가까운 parquet 셀 피처 (7곳 전부)
tree = cKDTree(master[["latitude", "longitude"]].values)
_, idx = tree.query(dz[["lat", "lon"]].values)
cells = master.iloc[idx].copy().reset_index(drop=True)
cells["wind_speed"] = 6.5
wd = 270.0
cells["wind_dir_sin"] = np.sin(np.radians(wd)).astype("float32")
cells["wind_dir_cos"] = np.cos(np.radians(wd)).astype("float32")

# 4) 모델 예측 (landing/hoist 중 더 위험한 등급 = 보수적)
models = _load_models(HELI)
dmat = xgb.DMatrix(cells[FEATURE_COLUMNS].astype("float32"))
p_l = models["landing"].predict(dmat); rc_l = p_l.argmax(axis=1) if p_l.ndim == 2 else p_l.astype(int)
p_h = models["hoist"].predict(dmat);   rc_h = p_h.argmax(axis=1) if p_h.ndim == 2 else p_h.astype(int)
risk_class = np.maximum(rc_l, rc_h)

# 5) 집계
print("=== 위험지역 7곳에 대한 모델 예측 ===")
for c in range(3):
    n = int(np.sum(risk_class == c))
    print(f"  {CLASSES[c]}: {n}개 ({100*n/len(risk_class):.0f}%)")
print(f"\n[실용 예측력] '주의+위험' 예측: {np.mean(risk_class>=1)*100:.0f}% | '위험' 예측: {np.mean(risk_class>=2)*100:.0f}%")

print("\n=== 상세 (실제 위험지역 → 모델 예측) ===")
for i in range(len(dz)):
    flag = "" if dz.iloc[i]["영역내"] else " [경계외삽]"
    print(f"  {CLASSES[risk_class[i]]:<8} {dz.iloc[i]['명칭']}{flag}")
