"""
타임스탬프 루프: 기상 API 호출 → 격자 매핑 → 슬롯별 Parquet 즉시 저장.
compiled_period_dfs 누적 없음 → OOM 원천 차단.
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
from XGBoost_Model.weather.modules.weather import fetch_kma_realtime
from XGBoost_Model.weather.modules.data_preprocessing import mapping_live_weather_to_grid

BATCH_DIR = os.path.join(_XGBOOST_DIR, 'dataset', 'batches')


def run(terrain_df: pd.DataFrame,
        start: str = "2025-10-01 00:00:00",
        end:   str = "2025-10-03 23:00:00",
        freq:  str = "3h") -> list[str]:

    os.makedirs(BATCH_DIR, exist_ok=True)

    slots      = pd.date_range(start=start, end=end, freq=freq)
    center_lat = terrain_df['latitude'].mean()
    center_lon = terrain_df['longitude'].mean()
    saved      = []

    print(f"[batch_runner] {len(slots)}개 슬롯 처리 시작 (주기: {freq})")

    for i, dt in enumerate(slots, 1):
        ts       = dt.strftime("%Y%m%d%H%M")
        out_path = os.path.join(BATCH_DIR, f"ts_{ts}.parquet")

        # 재실행 시 이미 처리된 슬롯 스킵 (멱등성)
        if os.path.exists(out_path):
            print(f"  [{i}/{len(slots)}] SKIP: {dt.strftime('%Y-%m-%d %H:%M')}")
            saved.append(out_path)
            continue

        print(f"  [{i}/{len(slots)}] {dt.strftime('%Y-%m-%d %H:%M')} 처리 중...", end=' ')
        try:
            weather = fetch_kma_realtime(target_time=ts)
            if not weather:
                print("기상 없음, 스킵")
                continue

            sdf = mapping_live_weather_to_grid(
                center_lat, center_lon, weather, terrain_df, radius_km=50.0
            )
            sdf['snapshot_timestamp'] = ts

            # 동적 기상 피처 연산 (슬롯 단위 경량 처리)
            wd = sdf['wind_direction'].fillna(0.0).values.astype('float32')
            sdf['wind_dir_sin']   = np.sin(np.radians(wd)).astype('float32')
            sdf['wind_dir_cos']   = np.cos(np.radians(wd)).astype('float32')
            sdf['wind_score']     = compute_wind_score(sdf['wind_speed'].values)
            sdf['wind_dir_score'] = compute_wind_dir_score(
                sdf['latitude'].values, sdf['longitude'].values, wd
            )

            # 정적 성분 + 동적 성분 브로드캐스팅 → 타깃 라벨
            sdf = compute_targets(sdf)

            # 필요 컬럼만 선별 저장
            keep = [c for c in
                    FEATURE_COLUMNS + TARGET_COLUMNS +
                    ['snapshot_timestamp', 'is_train_final', 'is_test']
                    if c in sdf.columns]
            sdf[keep].to_parquet(out_path, index=False, compression='snappy')

            print(f"{sdf.shape[0]:,}행 → {os.path.basename(out_path)}")
            saved.append(out_path)

        except Exception as e:
            print(f"에러: {e}")
        finally:
            try:
                del sdf
            except NameError:
                pass
            gc.collect()

    print(f"\n[batch_runner] 완료: {len(saved)}/{len(slots)} 슬롯 저장")
    return saved