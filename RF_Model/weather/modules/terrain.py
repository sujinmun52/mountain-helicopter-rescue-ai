import numpy as np

def calculate_slope(dem, resolution=15):
    """경사도 계산 (degrees)"""
    dy, dx = np.gradient(dem, resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)

def calculate_aspect(dem):
    """사면 방향 계산 (degrees, 북=0 시계방향)"""
    dy, dx = np.gradient(dem)
    aspect = np.degrees(np.arctan2(-dy, dx))
    aspect = (aspect + 360) % 360
    return aspect

def calculate_curvature(dem, resolution=15):
    """지형 곡률 계산 (능선/계곡 판별용)"""
    dy, dx = np.gradient(dem, resolution)
    ddy, _ = np.gradient(dy, resolution)
    _, ddx = np.gradient(dx, resolution)
    return ddy + ddx

def build_terrain_layer(dem_array, forest_map, resolution=15):
    """
    DEM + 임상도 기반 지형 레이어 구성
    forest_map: 전처리된 임상도 (개활지 코드 = 0)
    """
    slope = calculate_slope(dem_array, resolution)
    aspect = calculate_aspect(dem_array)
    curvature = calculate_curvature(dem_array, resolution)

    threshold = 0.02
    is_ridge = curvature < -threshold   # 능선: 풍속 증폭
    is_valley = curvature > threshold   # 계곡: 협곡 효과

    # 임상도 기반 개활지 판별 (코드 0 = 무임목지/개활지)
    is_open = forest_map == 0

    # 수고 추정 (임상도 밀도 코드 기반, 없으면 0으로 처리)
    canopy_height = np.where(is_open, 0.0, 10.0)

    return {
        "slope": slope,
        "aspect": aspect,
        "is_ridge": is_ridge,
        "is_valley": is_valley,
        "is_open": is_open,
        "canopy_height": canopy_height,
        "dem": dem_array
    }