"""
LightGBM용 정적 지형 데이터 빌더
"""
import os
import sys
import numpy as np
import pandas as pd

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_LGBM_DIR    = os.path.dirname(_THIS_DIR)
_PROJECT_DIR = os.path.dirname(_LGBM_DIR)
sys.path.insert(0, _LGBM_DIR)

from preprocess.score_utils import compute_static_terrain_scores

# 경로 정의 (RF_Model 폴더 안의 공용 원본 데이터셋 참조)
CSV_TRAIN1 = os.path.join(_PROJECT_DIR, 'RF_Model', 'dataset', 'train_excel_01.csv')
CSV_TRAIN2 = os.path.join(_PROJECT_DIR, 'RF_Model', 'dataset', 'train_excel_02.csv')
CSV_TEST   = os.path.join(_PROJECT_DIR, 'RF_Model', 'dataset', 'test_set.csv')
OUTPUT     = os.path.join(_LGBM_DIR, 'dataset', 'terrain_base.parquet')

def build():
    print("[LightGBM terrain_base] 외부 데이터셋 원본 CSV 로드 중...")
    t1   = pd.read_csv(CSV_TRAIN1)
    t2   = pd.read_csv(CSV_TRAIN2)
    test = pd.read_csv(CSV_TEST)

    t1['is_test_flag'] = False
    t2['is_test_flag'] = False
    test['is_test_flag'] = True

    df = pd.concat([t1, t2, test], ignore_index=True)
    df['zone'] = np.int8(-1)
    del t1, t2, test

    # 토양유형 원-핫 인코딩 경량화
    df = pd.get_dummies(df, columns=['land_type'], prefix='land', dtype=np.int8)

    # 데이터 타입 다운캐스팅 메모리 최적화
    for col in ['latitude', 'longitude', 'elevation']:
        df[col] = df[col].astype(np.float32)
    for col in ['slope_deg', 'tree_density', 'tree_height', 'zone']:
        if col in df.columns:
            df[col] = df[col].astype(np.int8)

    # 정적 지형 스코어 함수 호출
    df = compute_static_terrain_scores(df)

    # 최종 데이터셋 마스킹 설정
    df['is_train_final'] = ~df['is_test_flag']
    df['is_test']        =  df['is_test_flag']
    df = df.drop(columns=['is_test_flag'])

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_parquet(OUTPUT, index=False, compression='snappy')
    print(f"[LightGBM terrain_base] 빌드 및 저장 완료 → {OUTPUT}")
    print(f"   격자 수: {len(df):,}개 | 최종 피처 컬럼 수: {df.shape[1]}")
    return df

if __name__ == '__main__':
    build()