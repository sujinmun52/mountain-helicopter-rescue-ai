import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# 1. 사용자 입력 신고 지점
# ============================================================

INCIDENT_LAT = 38.1195
INCIDENT_LON = 128.4652
SEARCH_RADIUS_M = 500


# ============================================================
# 2. 팀원이 분리한 Train/Test 파일
# ============================================================

TRAIN_FILES = [
    "train_excel_01.csv",
    "train_excel_02.csv"
]

TEST_FILES = [
    "test_set.csv"
]


# ============================================================
# 3. 컬럼명 설정
# ============================================================

LAND_COL = "land_type"
SLOPE_COL = "slope_deg"
DENSITY_COL = "tree_density"
HEIGHT_COL = "tree_height"


# ============================================================
# 4. 파일 로드 함수
# ============================================================

def load_csv_files(file_list, split_name):
    df_list = []

    for file_path in file_list:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"{split_name} 파일을 찾을 수 없습니다: {file_path}"
            )

        temp_df = pd.read_csv(file_path)
        temp_df["source_file"] = file_path
        temp_df["split"] = split_name
        df_list.append(temp_df)

    combined_df = pd.concat(df_list, ignore_index=True)

    print(f"[{split_name}] 로드 완료")
    print(f"파일 개수: {len(file_list)}")
    print(f"총 행 개수: {len(combined_df):,}")

    return combined_df


train_df = load_csv_files(TRAIN_FILES, "train")
test_df = load_csv_files(TEST_FILES, "test")


print("\n============================================================")
print("1. Train/Test 데이터 로드 결과")
print("============================================================")
print(f"Train 행 개수: {len(train_df):,}")
print(f"Test 행 개수 : {len(test_df):,}")
print("Train 컬럼:", train_df.columns.tolist())
print("Test 컬럼 :", test_df.columns.tolist())


# ============================================================
# 5. 위도/경도 컬럼 자동 탐지
# ============================================================

def detect_coordinate_cols(df):
    longitude_candidates = [
        "longitude", "lon", "lng", "x", "경도", "Longitude", "LONGITUDE"
    ]

    latitude_candidates = [
        "latitude", "lat", "y", "위도", "Latitude", "LATITUDE"
    ]

    lon_col = None
    lat_col = None

    for col in longitude_candidates:
        if col in df.columns:
            lon_col = col
            break

    for col in latitude_candidates:
        if col in df.columns:
            lat_col = col
            break

    if lon_col is None or lat_col is None:
        raise ValueError(
            "위도/경도 컬럼을 찾지 못했습니다.\n"
            f"현재 컬럼 목록: {df.columns.tolist()}"
        )

    return lon_col, lat_col


LON_COL, LAT_COL = detect_coordinate_cols(train_df)

print("\n============================================================")
print("2. 좌표 컬럼 확인")
print("============================================================")
print(f"경도 컬럼: {LON_COL}")
print(f"위도 컬럼: {LAT_COL}")


# ============================================================
# 6. 전처리 함수
# ============================================================

def preprocess_rescue_df(df, dataset_name):
    df = df.copy()

    required_cols = [
        LAND_COL,
        SLOPE_COL,
        DENSITY_COL,
        HEIGHT_COL,
        LON_COL,
        LAT_COL
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{dataset_name} 데이터에 필수 컬럼이 없습니다: {missing_cols}"
        )

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_ranges = {
        LAND_COL: [0, 1, 2],
        SLOPE_COL: [0, 1, 2, 3, 4],
        DENSITY_COL: [0, 1, 2, 3],
        HEIGHT_COL: [0, 1, 2, 3]
    }

    for col, valid_values in valid_ranges.items():
        unique_values = sorted(df[col].dropna().unique())
        invalid_values = sorted(set(unique_values) - set(valid_values))

        if invalid_values:
            raise ValueError(
                f"{dataset_name} 데이터의 {col} 컬럼에 비정상 값이 있습니다.\n"
                f"허용값: {valid_values}\n"
                f"비정상 값 예시: {invalid_values[:20]}\n"
                "CSV 컬럼이 밀려서 읽혔거나 잘못 저장됐는지 확인하세요."
            )

    land_score_map = {
        0: 1.00,
        1: 0.55,
        2: 0.20
    }

    slope_score_map = {
        0: 1.00,
        1: 0.70,
        2: 0.35
    }

    density_score_map = {
        0: 1.00,
        1: 0.85,
        2: 0.21
    }

    height_score_map = {
        0: 1.00,
        1: 0.61,
        2: 0.31,
        3: 0.10
    }

    df["land_score"] = df[LAND_COL].map(land_score_map)
    df["slope_score"] = df[SLOPE_COL].map(slope_score_map)
    df["density_score"] = df[DENSITY_COL].map(density_score_map)
    df["height_score"] = df[HEIGHT_COL].map(height_score_map)

    before_count = len(df)

    df_valid = df[
        df[LON_COL].notna()
        & df[LAT_COL].notna()
        & df[LAND_COL].notna()
        & df[SLOPE_COL].notna()
        & df[DENSITY_COL].notna()
        & df[HEIGHT_COL].notna()
        & df["land_score"].notna()
        & df["slope_score"].notna()
        & df["density_score"].notna()
        & df["height_score"].notna()
    ].copy()

    df_valid["rescue_score"] = (
        df_valid["slope_score"] * 0.55
        + df_valid["density_score"] * 0.33
        + df_valid["height_score"] * 0.12
    )

    # 구조적으로 구조가 어려운 후보 제거
    # slope_deg 3, 4 제거
    # tree_density 3 제거
    before_filter_count = len(df_valid)

    df_valid = df_valid[
        ~df_valid[SLOPE_COL].isin([3, 4])
        & ~df_valid[DENSITY_COL].isin([3])
    ].copy()

    after_filter_count = len(df_valid)

    print(f"\n[{dataset_name}] 전처리 완료")
    print(f"원본 행 개수: {before_count:,}")
    print(f"결측/비정상 제거 후 행 개수: {before_filter_count:,}")
    print(f"구조 불가 조건 제거 후 행 개수: {after_filter_count:,}")
    print(f"최종 제외 행 개수: {before_count - after_filter_count:,}")

    return df_valid


train_valid = preprocess_rescue_df(train_df, "train")
test_valid = preprocess_rescue_df(test_df, "test")


# ============================================================
# 7. XGBoost 학습
# ============================================================
# 중요:
# 위도/경도는 feature에 넣지 않음
# train_test_split 사용하지 않음
# ============================================================

feature_cols = [
    LAND_COL,
    SLOPE_COL,
    DENSITY_COL,
    HEIGHT_COL
]

X_train = train_valid[feature_cols].copy()
y_train = train_valid["rescue_score"].copy()

X_test = test_valid[feature_cols].copy()
y_test = test_valid["rescue_score"].copy()


print("\n============================================================")
print("3. 학습/테스트 데이터 구성")
print("============================================================")
print("모델 입력 feature:", feature_cols)
print("제외한 정보: 위도/경도")
print("이유: 좌표를 넣으면 지형 특성이 아니라 특정 위치를 외울 수 있음")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")


model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)


# ============================================================
# 8. Test 데이터 성능 평가
# ============================================================

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\n============================================================")
print("4. Test 데이터 기준 모델 성능")
print("============================================================")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"R2  : {r2:.6f}")


# ============================================================
# 9. 추천 후보 데이터 구성
# ============================================================
# 실사용 추천 목적:
# train + test 전체 후보지에서 최적 위치를 찾음
# ============================================================

candidate_df = pd.concat(
    [train_valid, test_valid],
    ignore_index=True
)

X_candidate = candidate_df[feature_cols].copy()

candidate_df["xgb_pred_score"] = model.predict(X_candidate)
candidate_df["xgb_pred_score"] = candidate_df["xgb_pred_score"].clip(0, 1)


# ============================================================
# 10. Haversine 거리 계산 함수
# ============================================================

def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# ============================================================
# 11. 신고 지점 기준 거리 계산
# ============================================================

candidate_df["distance_m"] = haversine_distance_m(
    INCIDENT_LAT,
    INCIDENT_LON,
    candidate_df[LAT_COL],
    candidate_df[LON_COL]
)

nearby = candidate_df[
    candidate_df["distance_m"] <= SEARCH_RADIUS_M
].copy()


print("\n============================================================")
print("5. 신고 지점 기준 반경 내 후보")
print("============================================================")
print(f"신고 지점 위도: {INCIDENT_LAT}")
print(f"신고 지점 경도: {INCIDENT_LON}")
print(f"검색 반경: {SEARCH_RADIUS_M} m")
print(f"반경 내 후보 수: {len(nearby):,}")


# ============================================================
# 12. 반경 내 Top 3 선정
# ============================================================

if len(nearby) == 0:
    print("\n반경 내 후보가 없습니다.")
    print("대신 전체 후보 중 신고 지점과 가장 가까운 후보 5개를 출력합니다.")

    nearest = candidate_df.sort_values(
        by="distance_m",
        ascending=True
    ).head(5).copy()

    nearest["rank"] = range(1, len(nearest) + 1)

    nearest_output = nearest.rename(
        columns={
            LAT_COL: "latitude",
            LON_COL: "longitude"
        }
    )

    output_cols = [
        "rank",
        "latitude",
        "longitude",
        "distance_m",
        "xgb_pred_score"
    ]

    print("\n============================================================")
    print("6. 가장 가까운 후보 5개")
    print("============================================================")
    print(nearest_output[output_cols].to_string(index=False))

else:
    top3 = nearby.sort_values(
        by=[
            "xgb_pred_score",
            "rescue_score",
            "distance_m"
        ],
        ascending=[
            False,
            False,
            True
        ]
    ).head(3).copy()

    top3["rank"] = range(1, len(top3) + 1)

    top3_output = top3.rename(
        columns={
            LAT_COL: "latitude",
            LON_COL: "longitude"
        }
    )

    output_cols = [
        "rank",
        "latitude",
        "longitude",
        "elevation",
        "distance_m",
        "xgb_pred_score"
    ]

    print("\n============================================================")
    print("6. 신고 지점 기준 반경 내 최적 위치 Top 3")
    print("============================================================")
    print(top3_output[output_cols].to_string(index=False))
