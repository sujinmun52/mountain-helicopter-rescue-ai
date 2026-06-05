"""
원본 CSV(RF_Model/dataset/)를 로드해 정적 지형 피처를 연산하고
XGBoost_Model/dataset/terrain_base.parquet 으로 저장.
기상 데이터와 완전히 분리되므로 이후 배치 루프에서 재연산 불필요.
"""
import os
import sys
import numpy as np
import pandas as pd

# XGBoost_Model 루트를 sys.path 에 등록
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_XGBOOST_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_DIR = os.path.dirname(_XGBOOST_DIR)
sys.path.insert(0, _XGBOOST_DIR)

from preprocess.score_utils import compute_static_terrain_scores

# 경로 상수
CSV_TRAIN1 = os.path.join(_PROJECT_DIR, 'RF_Model', 'dataset', 'train_excel_01.csv')
CSV_TRAIN2 = os.path.join(_PROJECT_DIR, 'RF_Model', 'dataset', 'train_excel_02.csv')
CSV_TEST   = os.path.join(_PROJECT_DIR, 'RF_Model', 'dataset', 'test_set.csv')
OUTPUT     = os.path.join(_XGBOOST_DIR, 'dataset', 'terrain_base.parquet')


def build():
    print("[terrain_base] CSV 로드 중...")
    t1   = pd.read_csv(CSV_TRAIN1)
    t2   = pd.read_csv(CSV_TRAIN2)
    test = pd.read_csv(CSV_TEST)

    t1['is_test_flag'] = False
    t2['is_test_flag'] = False
    test['is_test_flag'] = True

    df = pd.concat([t1, t2, test], ignore_index=True)
    df['zone'] = np.int8(-1)
    del t1, t2, test

    df = pd.get_dummies(df, columns=['land_type'], prefix='land', dtype=np.int8)

    for col in ['latitude', 'longitude', 'elevation']:
        df[col] = df[col].astype(np.float32)
    for col in ['slope_deg', 'tree_density', 'tree_height', 'zone']:
        if col in df.columns:
            df[col] = df[col].astype(np.int8)

    df = compute_static_terrain_scores(df)

    df['is_train_final'] = ~df['is_test_flag']
    df['is_test']        =  df['is_test_flag']
    df = df.drop(columns=['is_test_flag'])

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_parquet(OUTPUT, index=False, compression='snappy')
    print(f"[terrain_base] 저장 완료 → {OUTPUT}")
    print(f"  격자 수: {len(df):,}개 | 컬럼 수: {df.shape[1]}")
    return df


if __name__ == '__main__':
    build()