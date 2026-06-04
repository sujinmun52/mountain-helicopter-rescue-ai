from dotenv import load_dotenv
import os

load_dotenv()

KMA_API_KEY = os.getenv("KMA_API_KEY")
FOREST_API_KEY = os.getenv("KMA_API_KEY")

# 설악산 영역 바운딩 박스
SEORAK_LAT_MIN = 38.05
SEORAK_LAT_MAX = 38.20
SEORAK_LON_MIN = 128.35
SEORAK_LON_MAX = 128.55

# DEM 해상도
DEM_RESOLUTION = 15  # 미터

# 호이스트 작업 조건 임계치
HOIST_MAX_WIND = 18.0    # m/s  (15→18: 산악 강풍 구간에서 후보지 확보)
HOIST_MAX_SLOPE = 40.0   # degrees  (35→40: 험준지 호이스트 가능 영역 확대)
HOIST_ALTITUDE = 60.0    # 작업고도 m
HOIST_MAX_CANOPY = 12.0  # 수고 m   (8→12: 일반 수림까지 허용)

# A* 풍속 임계치
WIND_BLOCK = 15.0
WIND_PENALTY_HIGH = 10.0

# ──────────────────────────────────────────────────────────────────────────
# 지형 인지형 기상 보간(Terrain-Aware Interpolation) 파라미터
# ──────────────────────────────────────────────────────────────────────────
# 수직 이방성 계수: "수직 1 m 는 수평 IDW_VERTICAL_SCALE m 와 동등한 거리"로 환산.
#   설악산처럼 표고차가 큰 지형에서, 같은 고도대의 관측소에 더 큰 가중치를 부여하기 위함.
#   값이 클수록 고도차가 보간 가중치에 더 크게 반영됨(고도 동질성 강조).
IDW_VERTICAL_SCALE = 180.0
IDW_POWER          = 2.0    # IDW 거리 감쇠 지수 (2.0 = 표준 역거리 제곱)
WIND_PROFILE_ALPHA = 0.27   # 멱법칙(Power Law) 연직 풍속 프로파일 지수 (복잡 산악 ≈ 0.25~0.30)
WIND_REF_HEIGHT    = 10.0   # AWS 관측 기준고도(지상 10 m)

# ──────────────────────────────────────────────────────────────────────────
# Risk Scoring 정규화 기준값 (각 피처를 물리적 한계로 [0,1] 위험도로 환산)
# ──────────────────────────────────────────────────────────────────────────
SLOPE_RISK_REF    = 30.0    # 경사도 위험 포화 기준(도). 착륙 한계각 부근.
WS_RISK_REF       = 18.0    # 풍속 위험 포화 기준(m/s). 산악 호이스트 상한.
DENSITY_RISK_REF  = 100.0   # 임목 밀도(%) 또는 라벨 최대값. 데이터 스케일에 맞게 조정.
CANOPY_RISK_REF   = 20.0    # 수고 위험 포화 기준(m).
ELEV_BASE         = 200.0   # 밀도고도 위험 산정 기준 하단(m) — 산록부.
ELEV_MAX          = 1708.0  # 설악산 대청봉(m) — 위험 산정 상단.