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

def build_terrain_layer(dem_array, forest_map, resolution=15, tree_density=None):
    """
    DEM + 임상도 기반 지형 레이어 구성
    forest_map: 전처리된 임상도 (개활지 코드 = 0)
    tree_density: (선택) 실측 임관피도 격자(0~1). 주어지면 그대로 저장,
                  없으면 forest_map 으로 근사(개활지 0.0 / 임목지 0.6).
                  → 보행 A* 의 PDF 밀도 risk(graded)에 사용.
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

    # 임관피도(0~1): 실측 보간값 우선, 없으면 임상도 근사
    if tree_density is not None:
        density = np.clip(np.nan_to_num(np.asarray(tree_density, dtype=float), nan=0.0),
                          0.0, 1.0)
    else:
        density = np.where(is_open, 0.0, 0.6)

    return {
        "slope": slope,
        "aspect": aspect,
        "is_ridge": is_ridge,
        "is_valley": is_valley,
        "is_open": is_open,
        "canopy_height": canopy_height,
        "tree_density": density,
        "dem": dem_array
    }