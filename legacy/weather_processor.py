def convert_wgs84_to_kma_grid(lat: float, lon: float) -> tuple[int, int]:
    """
    WGS84 위경도 → 기상청 LCC 격자 좌표 변환

    기상청 공공데이터 포털 가이드라인의 Lambert Conformal Conic 삼각함수 투영 공식 사용.
    설악산 대청봉(38.1191, 128.4657) 기준 반환 예시: (88, 139)

    Args:
        lat: 위도 (WGS84)
        lon: 경도 (WGS84)

    Returns:
        (nx, ny): 기상청 격자 정수 좌표
    """
    lat_rad = lat * _DEGRAD
    lon_rad = lon * _DEGRAD
    ol_rad  = _olon * _DEGRAD

    ra = _Re / _grid * _SF / math.pow(math.tan(math.pi * 0.25 + lat_rad * 0.5), _SN)

    theta = lon_rad - ol_rad
    # 경도 차이를 [-π, π] 범위로 정규화
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= _SN

    nx = int(ra * math.sin(theta) + _xo + 0.5)
    ny = int(_RO - ra * math.cos(theta) + _yo + 0.5)

    return nx, ny
def mapping_live_weather_to_grid(
    victim_lat: float,
    victim_lon: float,
    live_weather_api_data: Dict,
    terrain_df: pd.DataFrame,
    radius_km: float = 5.0
) -> pd.DataFrame:
    """
    요구조자 GPS 기준 반경 내 지형 격자에 실시간 기상 데이터 매핑

    파이프라인:
      1. victim 위치 기준 radius_km 반경 내 지형 격자 슬라이싱
      2. 각 격자 위경도 → 기상청 (nx, ny) 실시간 변환 (임시 컬럼 추가)
      3. live_weather_api_data와 Left Join → wind_speed, wind_direction 컬럼 매핑

    Args:
        victim_lat: 요구조자 위도 (WGS84)
        victim_lon: 요구조자 경도 (WGS84)
        live_weather_api_data: fetch_kma_realtime() 반환 딕셔너리
                               형식: {"nx_ny": {"ws": float, "wd": float}, ...}
        terrain_df: DEM+임상도 마스터 격자 DataFrame (필수 컬럼: "lat", "lon")
        radius_km: 슬라이싱 반경 (기본 5.0 km)

    Returns:
        슬라이싱된 지형 격자 DataFrame에 nx, ny, wind_speed, wind_direction 컬럼 추가.
        대응 기상 데이터가 없는 격자는 wind_speed/wind_direction = NaN.
    """
    # 위경도 1도 ≈ 111 km 기준으로 bounding box 델타 계산
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(victim_lat)))

    # ── Step 1: 반경 내 지형 격자 슬라이싱 ───────────────────────────────────
    mask = (
        (terrain_df["lat"] >= victim_lat - lat_delta) &
        (terrain_df["lat"] <= victim_lat + lat_delta) &
        (terrain_df["lon"] >= victim_lon - lon_delta) &
        (terrain_df["lon"] <= victim_lon + lon_delta)
    )
    local_grid = terrain_df[mask].copy()

    if local_grid.empty:
        return local_grid

    # ── Step 2: 각 격자 위경도 → nx, ny 변환 (임시 컬럼) ────────────────────
    grid_coords = local_grid.apply(
        lambda row: convert_wgs84_to_kma_grid(row["lat"], row["lon"]), axis=1
    )
    local_grid["nx"] = grid_coords.apply(lambda t: t[0])
    local_grid["ny"] = grid_coords.apply(lambda t: t[1])
    local_grid["_kma_key"] = (
        local_grid["nx"].astype(str) + "_" + local_grid["ny"].astype(str)
    )

    # ── Step 3: 기상 API 데이터 → DataFrame 변환 후 Left Join ────────────────
    if not live_weather_api_data:
        local_grid["wind_speed"]     = float("nan")
        local_grid["wind_direction"] = float("nan")
        return local_grid.drop(columns=["_kma_key"])

    weather_df = pd.DataFrame([
        {
            "_kma_key":       key,
            "wind_speed":     val["ws"],
            "wind_direction": val["wd"],
        }
        for key, val in live_weather_api_data.items()
    ]).drop_duplicates(subset=["_kma_key"])

    result = local_grid.merge(weather_df, on="_kma_key", how="left")
    result = result.drop(columns=["_kma_key"])

    return result
