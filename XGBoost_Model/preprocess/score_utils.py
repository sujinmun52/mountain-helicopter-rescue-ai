import numpy as np

SLOPE_MAP   = {0: 1.0, 1: 0.70, 2: 0.35}
DENSITY_MAP = {0: 1.0, 1: 0.85, 2: 0.21, 3: 0.00}
HEIGHT_MAP  = {0: 1.0, 1: 0.61, 2: 0.31}
ELEVATION_MAP = {0: 1.0, 1: 0.70, 2: 0.35}
MAX_ELEV    = 1708.0
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50}

# 전술별 (정적 가중치, 동적 가중치) 분리 정의
TACTIC_WEIGHTS = {
    "small_landing": {
        "static":  {"slope": 0.42, "density": 0.10, "height": 0.08, "elevation": 0.08},
        "dynamic": {"wind_score": 0.20, "wind_dir_score": 0.12},
    },
    "small_hoist": {
        "static":  {"slope": 0.12, "density": 0.15, "height": 0.20, "elevation": 0.08},
        "dynamic": {"wind_score": 0.30, "wind_dir_score": 0.15},
    },
    "large_landing": {
        "static":  {"slope": 0.46, "density": 0.14, "height": 0.08, "elevation": 0.12},
        "dynamic": {"wind_score": 0.12, "wind_dir_score": 0.08},
    },
    "large_hoist": {
        "static":  {"slope": 0.10, "density": 0.24, "height": 0.18, "elevation": 0.18},
        "dynamic": {"wind_score": 0.18, "wind_dir_score": 0.12},
    },
}

FEATURE_COLUMNS = [
    'elevation', 'slope_deg', 'tree_density', 'tree_height',
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos',
    'land_0', 'land_1', 'land_2',
]

TARGET_COLUMNS = [
    'target_small_landing', 'target_small_hoist',
    'target_large_landing', 'target_large_hoist',
]


def compute_static_terrain_scores(df):
    """기상과 무관한 정적 지형 점수를 terrain_base 빌드 시 1회만 연산."""

    elevation_grade = np.select(
        [df['elevation'] < 500,
         (df['elevation'] >= 500) & (df['elevation'] < 1200)],
        [0, 1], default=2
    ).astype('int8')

    df['slope_score']        = df['slope_deg'].map(SLOPE_MAP).astype('float32')
    df['tree_density_score'] = df['tree_density'].map(DENSITY_MAP).astype('float32')
    df['tree_height_score']  = df['tree_height'].map(HEIGHT_MAP).astype('float32')
    df['elevation_score'] = np.vectorize(ELEVATION_MAP.get)(elevation_grade).astype('float32')

    for key, w in TACTIC_WEIGHTS.items():
        s = w["static"]
        df[f'static_score_{key}'] = (
            df['slope_score']        * s["slope"]   +
            df['tree_density_score'] * s["density"] +
            df['tree_height_score']  * s["height"]  +
            df['elevation_score']    * s["elevation"]
        ).astype('float32')
    return df


def compute_wind_score(wind_speed_arr: np.ndarray) -> np.ndarray:
    return np.select(
        [wind_speed_arr < 5.0,
         (wind_speed_arr >= 5.0)  & (wind_speed_arr < 10.0),
         (wind_speed_arr >= 10.0) & (wind_speed_arr < 15.0)],
        [1.0, 0.6, 0.2], default=0.0
    ).astype('float32')


def compute_wind_dir_score(lat: np.ndarray, lon: np.ndarray,
                           wind_dir: np.ndarray) -> np.ndarray:
    heading    = np.degrees(np.arctan2(lon - FIRE_STATION["longitude"],
                                       lat - FIRE_STATION["latitude"])) % 360
    angle_diff = np.abs(heading - wind_dir) % 360
    return np.select(
        [(angle_diff < 45) | (angle_diff >= 315),
         (angle_diff >= 135) & (angle_diff < 225)],
        [1.0, 0.0], default=0.5
    ).astype('float32')


def compute_targets(df):
    """정적 성분 + 동적 성분 브로드캐스팅 → 최종 점수 및 3클래스 라벨."""
    for key, w in TACTIC_WEIGHTS.items():
        d = w["dynamic"]
        score = (
            df[f'static_score_{key}']
            + df['wind_score']      * d["wind_score"]
            + df['wind_dir_score']  * d["wind_dir_score"]
        ).astype('float32')
        df[f'score_{key}']  = score
        df[f'target_{key}'] = np.select(
            [score >= 0.80, (score >= 0.55) & (score < 0.80)],
            [0, 1], default=2
        ).astype('int8')
    return df