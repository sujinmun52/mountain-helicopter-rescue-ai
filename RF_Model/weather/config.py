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