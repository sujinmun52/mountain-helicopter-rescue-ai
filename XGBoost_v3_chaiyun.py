import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from realtime_weather_scoring import apply_realtime_weather_scoring


# ============================================================
# 1. 사용자 입력 신고 지점
# ============================================================

INCIDENT_LAT = 38.2499
INCIDENT_LON = 128.40678
SEARCH_RADIUS_M = 500


# ============================================================
# 2. 팀원이 분리한 Train/Test 파일
# ============================================================
# 주의:
# 팀원이 미리 train/test를 나누어 준 상황이므로
# train_test_split은 사용하지 않는다.
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


# train 기준으로 좌표 컬럼을 탐지한다.
# 단, 이후 전처리 단계에서 test에도 같은 컬럼이 있는지 검사한다.
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
            f"{dataset_name} 데이터에 필수 컬럼이 없습니다: {missing_cols}\n"
            f"현재 컬럼 목록: {df.columns.tolist()}"
        )

    # 숫자형 변환
    # errors='coerce'로 변환 실패 값은 NaN 처리
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # elevation 컬럼이 있으면 숫자형으로 변환
    # 없으면 realtime_weather_scoring 쪽에서 고도 보정을 자동으로 건너뛸 수 있게 둔다.
    if "elevation" in df.columns:
        df["elevation"] = pd.to_numeric(df["elevation"], errors="coerce")

    # 원본 행 수
    before_count = len(df)

    # 좌표 및 기본 필수 컬럼 결측 제거
    df_valid = df[
        df[LON_COL].notna()
        & df[LAT_COL].notna()
        & df[LAND_COL].notna()
        & df[SLOPE_COL].notna()
        & df[DENSITY_COL].notna()
        & df[HEIGHT_COL].notna()
    ].copy()

    after_na_count = len(df_valid)

    # ========================================================
    # 값 범위 검사
    # ========================================================
    # slope_deg 3, 4 및 tree_density 3은 데이터 값으로는 존재 가능하지만,
    # 본 모델에서는 구조적으로 구조가 어려운 후보로 보고 이후 제거한다.
    # 따라서 여기서는 "비정상 값"이 아니라 "제거 대상 정상 범주"로 취급한다.
    # ========================================================

    valid_ranges = {
        LAND_COL: [0, 1, 2],
        SLOPE_COL: [0, 1, 2, 3, 4],
        DENSITY_COL: [0, 1, 2, 3],
        HEIGHT_COL: [0, 1, 2, 3]
    }

    for col, valid_values in valid_ranges.items():
        unique_values = sorted(df_valid[col].dropna().unique())
        invalid_values = sorted(set(unique_values) - set(valid_values))

        if invalid_values:
            raise ValueError(
                f"{dataset_name} 데이터의 {col} 컬럼에 비정상 값이 있습니다.\n"
                f"허용값: {valid_values}\n"
                f"비정상 값 예시: {invalid_values[:20]}\n"
                "CSV 컬럼이 밀려서 읽혔거나 잘못 저장됐는지 확인하세요."
            )

    # ========================================================
    # 구조적으로 구조가 어려운 후보 제거
    # ========================================================
    # slope_deg 3, 4 제거
    # tree_density 3 제거
    # ========================================================

    before_filter_count = len(df_valid)

    df_valid = df_valid[
        ~df_valid[SLOPE_COL].isin([3, 4])
        & ~df_valid[DENSITY_COL].isin([3])
    ].copy()

    after_filter_count = len(df_valid)

    # ========================================================
    # 점수 매핑
    # ========================================================

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

    df_valid["land_score"] = df_valid[LAND_COL].map(land_score_map)
    df_valid["slope_score"] = df_valid[SLOPE_COL].map(slope_score_map)
    df_valid["density_score"] = df_valid[DENSITY_COL].map(density_score_map)
    df_valid["height_score"] = df_valid[HEIGHT_COL].map(height_score_map)

    before_score_na_count = len(df_valid)

    df_valid = df_valid[
        df_valid["land_score"].notna()
        & df_valid["slope_score"].notna()
        & df_valid["density_score"].notna()
        & df_valid["height_score"].notna()
    ].copy()

    after_score_na_count = len(df_valid)

    # ========================================================
    # 구조 적합성 점수 산정
    # ========================================================
    # 주의:
    # 이 rescue_score는 실제 구조 성공 여부가 아니라,
    # 지형 조건을 기반으로 사람이 정의한 구조 적합성 점수이다.
    #
    # 따라서 XGBoost 성능은 "현실 구조 성공 예측 성능"이라기보다
    # "정의된 점수 체계를 얼마나 잘 근사하는지"로 해석해야 한다.
    # ========================================================

    df_valid["rescue_score"] = (
        df_valid["land_score"] * 0.20
        + df_valid["slope_score"] * 0.45
        + df_valid["density_score"] * 0.25
        + df_valid["height_score"] * 0.10
    )

    print(f"\n[{dataset_name}] 전처리 완료")
    print(f"원본 행 개수: {before_count:,}")
    print(f"좌표/필수값 결측 제거 후 행 개수: {after_na_count:,}")
    print(f"구조 불가 조건 제거 전 행 개수: {before_filter_count:,}")
    print(f"구조 불가 조건 제거 후 행 개수: {after_filter_count:,}")
    print(f"점수 결측 제거 전 행 개수: {before_score_na_count:,}")
    print(f"점수 결측 제거 후 행 개수: {after_score_na_count:,}")
    print(f"최종 제외 행 개수: {before_count - after_score_na_count:,}")

    if len(df_valid) == 0:
        raise ValueError(
            f"{dataset_name} 전처리 결과 유효한 행이 없습니다. "
            "컬럼값 범위, 결측치, 구조 불가 조건을 확인하세요."
        )

    return df_valid


train_valid = preprocess_rescue_df(train_df, "train")
test_valid = preprocess_rescue_df(test_df, "test")


# ============================================================
# 7. XGBoost 학습
# ============================================================
# 중요:
# 위도/경도는 feature에 넣지 않는다.
# train_test_split은 사용하지 않는다.
#
# 좌표를 feature로 넣으면 지형 특성 자체가 아니라
# 특정 위치를 외우는 모델이 될 수 있다.
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

model.fit(X_train, y_train)


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

print("\n[해석 주의]")
print("현재 rescue_score는 실제 구조 성공 관측값이 아니라")
print("land/slope/density/height 점수와 가중치로 정의한 지형 적합성 점수입니다.")
print("따라서 위 성능은 실제 구조 성공 예측 성능이 아니라,")
print("정의된 점수 체계를 모델이 얼마나 잘 근사했는지를 의미합니다.")


# ============================================================
# 9. 추천 후보 데이터 구성
# ============================================================
# 주의:
# 현재 추천 후보는 test_set 내 후보만 대상으로 한다.
# 전체 공간 후보지를 대상으로 추천하려면 별도의 전체 후보지 파일을 사용해야 한다.
# ============================================================

candidate_df = test_valid.copy()

X_candidate = candidate_df[feature_cols].copy()

candidate_df["xgb_pred_score"] = model.predict(X_candidate)
candidate_df["xgb_pred_score"] = candidate_df["xgb_pred_score"].clip(0, 1)


# ============================================================
# 10. 실시간 기상청 AWS 풍향/풍속 기반 최종 점수 보정
# ============================================================
# 중요:
# 실시간 기상은 XGBoost feature로 넣지 않는다.
# historical training data에 동일 시점의 기상 관측값이 없으므로,
# 학습 feature가 아니라 사후 안전 보정 계수로 사용한다.
#
# final_score = xgb_pred_score * weather_factor
# ============================================================

candidate_df = apply_realtime_weather_scoring(
    candidate_df=candidate_df,
    lat_col=LAT_COL,
    lon_col=LON_COL,
    elevation_col="elevation",
    terrain_score_col="xgb_pred_score",
    final_score_col="final_score",
    fallback_ws=2.0,
    fallback_wd=0.0,
    use_elevation_correction=True,
    work_altitude=60.0,
    alpha=0.27,
    z_ref=10.0,
    max_wind_speed=20.0
)

print("\n============================================================")
print("5. 실시간 기상 보정 완료")
print("============================================================")

if "final_score" in candidate_df.columns:
    print("final_score 생성 완료")

if "weather_factor" in candidate_df.columns:
    print(
        "weather_factor 범위:",
        f"{candidate_df['weather_factor'].min():.3f}",
        "~",
        f"{candidate_df['weather_factor'].max():.3f}"
    )

if "corrected_ws" in candidate_df.columns:
    print(
        "고도 보정 풍속 corrected_ws 범위:",
        f"{candidate_df['corrected_ws'].min():.3f}",
        "~",
        f"{candidate_df['corrected_ws'].max():.3f}",
        "m/s"
    )


# ============================================================
# 11. Haversine 거리 계산 함수
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
# 12. 신고 지점 기준 거리 계산
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
print("6. 신고 지점 기준 반경 내 후보")
print("============================================================")
print(f"신고 지점 위도: {INCIDENT_LAT}")
print(f"신고 지점 경도: {INCIDENT_LON}")
print(f"검색 반경: {SEARCH_RADIUS_M} m")
print(f"반경 내 후보 수: {len(nearby):,}")


# ============================================================
# 13. 출력 컬럼 구성 함수
# ============================================================

def get_output_cols(output_df):
    base_cols = [
        "rank",
        "latitude",
        "longitude"
    ]

    optional_cols = [
        "elevation",
        "distance_m",
        "xgb_pred_score",
        "weather_factor",
        "final_score",
        "rescue_score",
        "nearest_station_id",
        "nearest_station_name",
        "station_ws",
        "station_wd",
        "corrected_ws",
        "corrected_wd",
        "weather_station_distance_deg"
    ]

    output_cols = base_cols.copy()

    for col in optional_cols:
        if col in output_df.columns:
            output_cols.append(col)

    return output_cols


# ============================================================
# 14. 반경 내 Top 3 선정
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

    output_cols = get_output_cols(nearest_output)

    print("\n============================================================")
    print("7. 가장 가까운 후보 5개")
    print("============================================================")
    print(nearest_output[output_cols].to_string(index=False))

else:
    top3 = nearby.sort_values(
        by=[
            "final_score",
            "xgb_pred_score",
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

    output_cols = get_output_cols(top3_output)

    print("\n============================================================")
    print("7. 신고 지점 기준 반경 내 최적 위치 Top 3")
    print("============================================================")
    print(top3_output[output_cols].to_string(index=False))


# ============================================================
# 15. Feature Importance 출력
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values(
    by="importance",
    ascending=False
)

print("\n============================================================")
print("8. XGBoost Feature Importance")
print("============================================================")
print(importance_df.to_string(index=False))


