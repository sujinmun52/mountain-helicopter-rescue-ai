import numpy as np

def calculate_slope(dem, resolution=15, cell_m=None):
    """경사도 계산 (degrees)

    cell_m : None | float | (dy_m, dx_m)
        격자의 '실제' 셀 간격(m). 경사 = arctan(|∇고도|)는 셀 간격에 민감하므로,
        보간 격자처럼 셀이 비등방(행≠열)일 때는 축별 실제 간격을 넘겨야 정확하다.
        - None        → 기존 호환: resolution(=15m)을 양축에 사용
        - float       → 양축 동일 간격
        - (dy_m, dx_m)→ 행(axis0)·열(axis1) 간격 분리 지정
        근거: 격자 셀 간격은 (영역폭/격자수)로 결정되며, 본 파이프라인의
        조난자 ±0.15° / 200격자 구성에서는 실측 약 81m(행)·140m(열)로
        하드코딩 15m와 불일치 → 미보정 시 경사가 수 배 과대 계산됨.
    """
    if cell_m is None:
        dy_m = dx_m = resolution
    elif np.isscalar(cell_m):
        dy_m = dx_m = float(cell_m)
    else:
        dy_m, dx_m = float(cell_m[0]), float(cell_m[1])
    dy, dx = np.gradient(dem, dy_m, dx_m)
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

def build_terrain_layer(dem_array, forest_map, resolution=15, tree_density=None,
                        cell_m=None):
    """
    DEM + 임상도 기반 지형 레이어 구성
    forest_map: 전처리된 임상도 (개활지 코드 = 0)
    tree_density: (선택) 실측 임관피도 격자(0~1). 주어지면 그대로 저장,
                  없으면 forest_map 으로 근사(개활지 0.0 / 임목지 0.6).
                  → 보행 A* 의 PDF 밀도 risk(graded)에 사용.
    cell_m: (선택) 격자 실제 셀 간격(m) — float 또는 (dy_m, dx_m).
            주어지면 '경사' 계산에 반영(정확도↑). 곡률/능선(is_ridge)은
            튜닝된 임계치(0.02)가 resolution=15 기준이라 기존 거동을 보존하기
            위해 의도적으로 resolution 으로 계산한다(능선 판별 일관성 유지).
    """
    slope = calculate_slope(dem_array, resolution, cell_m=cell_m)
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