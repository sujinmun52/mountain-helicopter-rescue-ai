import os
import sys
import pandas as pd
from dotenv import load_dotenv

# 1. 인프라 및 패키지 탐색 경로 최적화
_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, '.env'), override=True)
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

# 개편된 batch_runner 모듈에서 단일 인자용 run 함수 안전 바인딩
try:
    from preprocess.batch_runner import run as run_batches
except ImportError:
    try:
        sys.path.insert(0, os.path.join(_DIR, 'preprocess'))
        from preprocess.batch_runner import run as run_batches
    except ImportError as e:
        sys.exit(f"❌ 에러: batch_runner.py 모듈을 임포트할 수 없습니다: {e}")

print("=" * 60)
print("[XGBoost 전처리 파이프라인] 정적 시나리오 증강 엔진 가동 준비")
print("=" * 60)

# 2. 원천 지형 베이스 캐시 데이터(terrain_base.parquet) 절대 경로 탐색
# 환경에 따른 경로 유연성을 확보하기 위해 멀티 레이어 후보군 검사
possible_terrain_paths = [
    os.path.join(_DIR, 'dataset', 'terrain_base.parquet'),
    os.path.join(os.path.dirname(_DIR), 'XGBoost_Model', 'dataset', 'terrain_base.parquet'),
    os.path.join(_DIR, 'preprocess', 'dataset', 'terrain_base.parquet')
]

terrain_df = None
for path in possible_terrain_paths:
    if os.path.exists(path):
        print(f" ➔ 원천 지형 파크웨이 파일 탐색 성공: {path}")
        terrain_df = pd.read_parquet(path)
        break

if terrain_df is None:
    print("\n❌ [치명적 에러] 원천 지형 데이터베이스(terrain_base.parquet)를 찾을 수 없습니다.")
    print("의도된 데이터 폴더 구조에 파일이 존재하는지 확인해 주세요. 탐색 실패 경로 리스트:")
    for p in possible_terrain_paths:
        print(f"   - {p}")
    sys.exit()

# 3. 개편된 1대1 매핑 구조 커널 점화 (치팅 스코어 배제 및 Cross Join 증강 가동)
print(f" ➔ {len(terrain_df):,}행 규모의 지형 데이터프레임을 batch_runner 구조체로 주입합니다.")
print(" ➔ 가상 풍속 시나리오 Cross Join 및 계층화 다운샘플링 프로세스 가동...\n")

try:
    saved_files = run_batches(terrain_df)
    
    # 4. 최종 파일 생성 유효성 더블 체크 보장 레이어
    if saved_files and os.path.exists(saved_files[0]):
        print("\n" + "=" * 60)
        print("🎉 [전처리 파이프라인 완주 성공]")
        print(f" ➔ 마스터 배치 안착 완료: {saved_files[0]}")
        print(" ➔ 이제 2.xgb_train.py를 실행하여 오토 튜닝 본 학습을 터뜨리시면 됩니다.")
        print("=" * 60)
    else:
        print("\n❌ [경고] batch_runner 커널은 가동되었으나, 최종 Parquet 물리 파일 생성에 실패했습니다.")
        print("상단 콘솔 로그에 '[에러 발생] 타겟 연산 또는 저장 중 문제 발생' 텍스트가 찍혔는지 확인하세요.")
        print("score_utils.py의 compute_targets 내부 로직과 컬럼 정합성이 깨졌을 수 있습니다.")

except Exception as e:
    print(f"\n❌ [상위 스크립트 예외 발생] 전처리 파이프라인 구동 중 전역 크래시: {e}")