import os
import sys
from dotenv import load_dotenv

_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, '.env'), override=True)
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

import pandas as pd
from preprocess.terrain_base import build as build_terrain
from preprocess.batch_runner  import run  as run_batches

TERRAIN_PATH = os.path.join(_DIR, 'dataset', 'terrain_base.parquet')

# ── Step 1: 지형 정적 기반 (최초 1회만 빌드, 이후 재사용) ────────────────
if os.path.exists(TERRAIN_PATH):
    print(f"[Step 1] terrain_base 재사용: {TERRAIN_PATH}")
    terrain_df = pd.read_parquet(TERRAIN_PATH)
else:
    print("[Step 1] terrain_base 최초 빌드...")
    terrain_df = build_terrain()

print(f"  지형 격자: {len(terrain_df):,}개")

# ── Step 2: 타임스탬프별 배치 Parquet 분산 저장 ───────────────────────────
run_batches(
    terrain_df,
    start = "2025-10-01 00:00:00",
    end   = "2025-10-03 23:00:00",
    freq  = "3h",
)

print("\n[완료] XGBoost_Model/dataset/batches/ 확인 후 2_xgb_train.py 실행")