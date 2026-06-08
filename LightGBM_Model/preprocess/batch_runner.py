"""
LightGBM용 시계열 기상 매핑 및 분산 배치 러너 Engine
"""
import os
import sys
import gc
import numpy as np
import pandas as pd

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_LGBM_DIR    = os.path.dirname(_THIS_DIR)
_PROJECT_DIR = os.path.dirname(_LGBM_DIR)
sys.path.insert(0, _LGBM_DIR)
sys.path.insert(0, _PROJECT_DIR)

from preprocess.score_utils import (
    compute_wind_score, compute_wind_dir_score,
    compute_targets, FEATURE_COLUMNS, TARGET_COLUMNS,
)
# LightGBM 폴더 하위의 기상 모듈 호출 경로 명시
from LightGBM_Model.weather.modules.weather import fetch_kma_realtime
from LightGBM_Model.weather.modules.data_preprocessing import mapping_live_weather_to_grid

BATCH_DIR = os.path.join(_LGBM_DIR, 'dataset', 'batches')

def run(terrain_df: pd.DataFrame,
        start: str = "2025-10-01 00:00:00",
        end:   str = "2025-10-03 23:00:00",
        freq:  str = "3h") -> list[str]:

    os.makedirs(BATCH_DIR, exist_ok=True)

    slots      = pd.date_range(start=start, end=end, freq=freq)
    center_lat = terrain_df['latitude'].mean()
    center_lon = terrain_df['longitude'].mean()
    saved      = []

    print(f"[LightGBM batch_runner] {len(slots)}개 기상 타임스탬프 슬롯 처리 가동 (주기: {freq})")

    for i, dt in enumerate(slots, 1):
        ts       = dt.strftime("%Y%m%d%H%M")
        out_path = os.path.join(BATCH_DIR, f"ts_{ts}.parquet")

        # 멱등성 보장 검사 (이미 로컬에 연산된 파일은 건너뛰기)
        if os.path.exists(out_path):
            print(f"   [{i}/{len(slots)}] SKIP: {dt.strftime('%Y-%m-%d %H:%M')}")
            saved.append(out_path)
            continue

        print(f"   [{i}/{len(slots)}] {dt.strftime('%Y-%m-%d %H:%M')} 공간 결합 중...", end=' ')
        try:
            # 1. API 데이터 요청
            weather = fetch_kma_realtime(target_time=ts)
            if not weather:
                print("기상 수집 데이터 없음, 다음 슬롯으로 스킵")
                continue

            # 2. 실시간 공간 그리드 크로스 매핑
            sdf = mapping_live_weather_to_grid(
                center_lat, center_lon, weather, terrain_df, radius_km=50.0
            )
            sdf['snapshot_timestamp'] = ts

            # 3. 동적 기상 변수 인코딩 및 삼각함수 가속 벡터 연산

            # [동적 기상 피처 및 삼각함수 가속 벡터 연산]
            wd = sdf['wind_direction'].fillna(0.0).values.astype('float32')
            sdf['wind_dir_sin']   = np.sin(np.radians(wd)).astype('float32')
            sdf['wind_dir_cos']   = np.cos(np.radians(wd)).astype('float32')
            sdf['wind_score']     = compute_wind_score(sdf['wind_speed'].values)
            sdf['wind_dir_score'] = compute_wind_dir_score(
                sdf['latitude'].values, sdf['longitude'].values, wd
            )

            # [정적 static_score_* 선형 결합 성분에 동적 벡터 결합 및 자동 라벨링]
            sdf = compute_targets(sdf)

            # [필수 인코딩 피처 선별 추출 및 Parquet 저장]
            dynamic_land_cols = [c for c in sdf.columns if c.startswith('land_')]
            keep = [c for c in
                    FEATURE_COLUMNS + dynamic_land_cols + TARGET_COLUMNS +
                    ['snapshot_timestamp', 'is_train_final', 'is_test']
                    if c in sdf.columns]
            
            # 중복 컬럼 유일화 처리 후 스내피 압축 저장
            keep = list(set(keep))
            sdf[keep].to_parquet(out_path, index=False, compression='snappy')

            print(f"{sdf.shape[0]:,}행 매트릭스 완료 → {os.path.basename(out_path)}")
            saved.append(out_path)

        except Exception as e:
            print(f"에러 발생 시점 Skip ({ts}): {e}")
        finally:
            try:
                del sdf
            except NameError:
                pass
            gc.collect()

    print(f"\n[LightGBM batch_runner] 전체 작업 완료: {len(saved)}/{len(slots)} 슬롯 빌드 성공")
    return saved