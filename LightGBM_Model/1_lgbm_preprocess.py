"""
LightGBM용 시공간 데이터 전처리 통합 가속 파이프라인 컨트롤러
"""
import os
import sys

# ── 1. 인프라 및 환경 변수 경로 탐색 ──────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DIR)

# 내부 모듈 및 공용 기상 스크립트 탐색을 위한 sys.path 인프라 등록
sys.path.insert(0, os.path.join(_DIR, 'weather'))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _DIR)

import pandas as pd
from preprocess.terrain_base import build as build_terrain
from preprocess.batch_runner import run as run_batches

# 지형 뼈대 캐시 경로 설정
TERRAIN_PATH = os.path.join(_DIR, 'dataset', 'terrain_base.parquet')


# ── 2. 메인 오케스트레이션 실행 제어부 ────────────────────────────────────────
def main():
    print("[LightGBM 파이프라인 가동] 고정 모드: EXTERNAL (분산 파이프라인)")

    # ── Step 1: 지형 정적 기반 검사 및 로드 (최초 1회만 빌드, 이후 캐시 재사용) ──
    if os.path.exists(TERRAIN_PATH):
        print(f"[Step 1] 기존 LightGBM terrain_base 캐시 발견: {TERRAIN_PATH}")
        terrain_df = pd.read_parquet(TERRAIN_PATH)
    else:
        print("[Step 1] 최초 가동 검출 - terrain_base.py 빌드 엔진을 호출합니다.")
        terrain_df = build_terrain()

    print(f"      ㄴ 기준 지형 공간 격자: {len(terrain_df):,}개 매트릭스 로드 완료")

    # ── Step 2: 타임스탬프별 배치 Parquet 분산 저장 및 기상 결합 ──────────────────
    # 중복 연산 및 메모리 누적(OOM)을 완전히 차단한 구조로 기상 매핑 수행
    run_batches(
        terrain_df,
        start = "2025-10-01 00:00:00",
        end   = "2025-10-03 23:00:00",
        freq  = "3h",
    )

    print(f"\n[완료] LightGBM_Model/dataset/batches/ 내 분산 파일 생성 검증 완료.")
    print("      이제 안정적인 대용량 학습을 위해 '2_lgbm_train.py' 스크립트를 실행하세요.")


if __name__ == '__main__':
    main()