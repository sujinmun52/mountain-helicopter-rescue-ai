"""
수정된 batch_runner.py: 시계열 API 호출 제거 및 정적 풍속 증강(Augmentation) 파이프라인.
OOM(메모리 초과) 방지를 위해 위험 지형은 다운샘플링하고, 가상의 풍속 시나리오를 Cross Join하여 생성.
"""
import os
import sys
import gc
import numpy as np
import pandas as pd

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_XGBOOST_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_DIR = os.path.dirname(_XGBOOST_DIR)
sys.path.insert(0, _XGBOOST_DIR)
sys.path.insert(0, _PROJECT_DIR)

from preprocess.score_utils import (
    compute_wind_score, compute_wind_dir_score,
    compute_targets, FEATURE_COLUMNS, TARGET_COLUMNS,
)

BATCH_DIR = os.path.join(_XGBOOST_DIR, 'dataset', 'batches')

def run(terrain_df: pd.DataFrame) -> list[str]:
    """
    단일 지형 데이터를 받아 증강된 학습용 Parquet 파일을 생성하는 메인 함수.
    """
    os.makedirs(BATCH_DIR, exist_ok=True)
    out_path = os.path.join(BATCH_DIR, f"augmented_train_data.parquet")
    saved = []

    print(f"\n[batch_runner] 데이터 증강 및 전처리 시작 (원본: {len(terrain_df):,}행)")

    # ── [STEP 1] 지형 기반 계층화 다운샘플링 (Stratified Sampling) ──
    is_hard_block = (terrain_df['slope_deg'] >= 45.0) | (terrain_df['tree_density'] >= 2.0)
    
    df_danger = terrain_df[is_hard_block].copy()
    df_safe_caution = terrain_df[~is_hard_block].copy()

    # 위험군은 10%만 남기고 버림 (명백한 위험지의 과적합 및 데이터 폭발 방지)
    df_danger_sampled = df_danger.sample(frac=0.1, random_state=42)
    
    print(f"  • 안전/주의 후보군 보존: {len(df_safe_caution):,}행 (100%)")
    print(f"  • 명백한 위험군 샘플링: {len(df_danger):,}행 -> {len(df_danger_sampled):,}행 (10%)")

    # ── [STEP 2] 그룹별 풍속 증강 (Cross Join) ──
    wind_scenarios_full = pd.DataFrame({'wind_speed': [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]})
    wind_scenarios_basic = pd.DataFrame({'wind_speed': [0.0, 15.0]})

    print("  • 가상 풍속 시나리오 Cross Join 진행 중...")
    df_safe_aug = df_safe_caution.merge(wind_scenarios_full, how='cross')
    df_danger_aug = df_danger_sampled.merge(wind_scenarios_basic, how='cross')

    sdf = pd.concat([df_safe_aug, df_danger_aug], ignore_index=True)
    
    # ── [STEP 3] 풍향(Vector) 등 동적 피처 생성 ──
    np.random.seed(42)
    sdf['wind_direction'] = np.random.randint(0, 360, size=len(sdf)).astype('float32')
    sdf['wind_dir_sin'] = np.sin(np.radians(sdf['wind_direction'])).astype('float32')
    sdf['wind_dir_cos'] = np.cos(np.radians(sdf['wind_direction'])).astype('float32')
    
    sdf['flight_heading'] = np.random.randint(0, 360, size=len(sdf))
    angle_diff = np.abs(sdf['flight_heading'] - sdf['wind_direction']) % 360
    sdf['wind_dir_score'] = np.select(
        [(angle_diff < 45) | (angle_diff >= 315), (angle_diff >= 135) & (angle_diff < 225)],
        [1.0, 0.0], default=0.5
    )

    print(f"  • 데이터 증강 완료 (최종 병합: {len(sdf):,}행)")

    # ── [STEP 4] 타겟 라벨링 및 저장 ──
    try:
        # 🎯 [버그 해결]: Cross Join으로 베이킹된 wind_speed 배열을 전문가 스코어로 인코딩 가동!
        sdf['wind_score'] = compute_wind_score(sdf['wind_speed'].values)

        # 이제 결손 없는 완벽한 상태에서 타겟 공식 연산 커널 가동
        sdf = compute_targets(sdf)

        # Train / Test (8:2) 무작위 분할
        sdf['is_test'] = np.random.rand(len(sdf)) < 0.2
        sdf['is_train_final'] = ~sdf['is_test']

        # 필요 컬럼만 선별
        keep = [c for c in FEATURE_COLUMNS + TARGET_COLUMNS + ['is_train_final', 'is_test'] if c in sdf.columns]
        
        sdf[keep].to_parquet(out_path, index=False, compression='snappy')
        print(f"\n[batch_runner] 완료: 증강된 데이터가 성공적으로 저장되었습니다. -> {os.path.basename(out_path)}")
        saved.append(out_path)

    except Exception as e:
        print(f"\n[에러 발생] 타겟 연산 또는 저장 중 문제 발생: {e}")
    finally:
        del sdf, df_safe_aug, df_danger_aug
        gc.collect()

    return saved