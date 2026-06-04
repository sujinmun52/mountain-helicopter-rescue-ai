import numpy as np
import pandas as pd
from modules.terrain import build_terrain_layer
from modules.hoist import find_hoist_candidates
from modules.pathfinding import astar
from modules.simple_pathfinding import simple_path_with_obstacles
from modules.evaluation.metrics import evaluate_path
from modules.vworld_3d import create_vworld_3d_mission_map
from modules.helicopter_mission_map import create_helicopter_mission_folium_map
from modules.weather import fetch_kma_realtime, build_wind_field
from modules.data_preprocessing import convert_wgs84_to_kma_grid, mapping_live_weather_to_grid

# 119 소방구급센터 좌표 (설악산 북쪽 인제 소방서)
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50, "name": "인제 소방서"}

def latlon_to_grid(lat, lon, dem_lats, dem_lons):
    """위경도를 가장 가까운 격자 (row, col)로 변환"""
    dist = (dem_lats - lat)**2 + (dem_lons - lon)**2
    r, c = np.unravel_index(np.argmin(dist), dist.shape)
    return int(r), int(c)

def load_seorak_data():
    """설악산 CSV 데이터 통합 로드"""
    dfs = []
    for part in [1, 2, 3]:
        file = f"data/Final_seoraksan_part{part}_v2_sujin_260602.csv"
        df = pd.read_csv(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # -9999는 결측값이므로 NaN으로 처리 (선택적 컬럼만)
    for col in ['elevation', 'slope_deg', 'tree_height']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(-9999, np.nan)
    
    # tree_density 처리 (문자열 컬럼)
    df['tree_density'] = pd.to_numeric(df['tree_density'], errors='coerce')
    df['tree_density'] = df['tree_density'].replace(-9999, np.nan)
    
    # 필수 컬럼 중 하나라도 NaN이면 제거
    df = df.dropna(subset=['longitude', 'latitude', 'elevation'])
    return df


def create_3d_visualization(dem_lats, dem_lons, dem_array, terrain, full_path, 
                           path_penalties, fire_station, landing_point, victim_gps):
    """3D plotly 시각화"""
    import plotly.graph_objects as go
    
    # 경로상 penalty 값으로 색상 정의
    path_coords = np.array([[dem_lats[r, c], dem_lons[r, c], dem_array[r, c]] 
                            for r, c in full_path])
    path_penalties_vals = np.array([p["penalty"] for p in path_penalties])
    
    fig = go.Figure()
    
    # DEM 표면
    fig.add_trace(go.Surface(
        z=dem_array,
        x=dem_lons[0],
        y=dem_lats[:, 0],
        colorscale='Viridis',
        name='지형 고도',
        opacity=0.7
    ))
    
    # 경로 (색상: penalty에 따라)
    fig.add_trace(go.Scatter3d(
        x=path_coords[:, 1],
        y=path_coords[:, 0],
        z=path_coords[:, 2],
        mode='lines+markers',
        line=dict(
            color=path_penalties_vals,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="패널티<br>(높을수록<br>위험)")
        ),
        marker=dict(size=3),
        name='구조 경로',
        hovertemplate='<b>위치</b><br>위도: %{y:.4f}<br>경도: %{x:.4f}<br>고도: %{z:.0f}m<extra></extra>'
    ))
    
    # 119 센터
    fire_row, fire_col = latlon_to_grid(fire_station["latitude"], fire_station["longitude"],
                                         dem_lats, dem_lons)
    fire_z = dem_array[fire_row, fire_col]
    fig.add_trace(go.Scatter3d(
        x=[fire_station["longitude"]],
        y=[fire_station["latitude"]],
        z=[fire_z],
        mode='markers',
        marker=dict(size=15, color='green', symbol='diamond'),
        name='119 센터',
        hovertext=f"{fire_station['name']}"
    ))
    
    # 착륙지점
    fig.add_trace(go.Scatter3d(
        x=[landing_point["longitude"]],
        y=[landing_point["latitude"]],
        z=[dem_array[int(landing_point["row"]), int(landing_point["col"])]],
        mode='markers',
        marker=dict(size=15, color='blue', symbol='square'),
        name='호이스트 착륙지점'
    ))
    
    # 조난자 위치
    victim_row, victim_col = latlon_to_grid(victim_gps["latitude"], victim_gps["longitude"],
                                             dem_lats, dem_lons)
    victim_z = dem_array[victim_row, victim_col]
    fig.add_trace(go.Scatter3d(
        x=[victim_gps["longitude"]],
        y=[victim_gps["latitude"]],
        z=[victim_z],
        mode='markers',
        marker=dict(size=15, color='red', symbol='cross'),
        name='조난자'
    ))
    
    fig.update_layout(
        title='3D 산악 구조 경로 시각화<br><sub>색상: 경사도/풍속/숲 패널티의 합산</sub>',
        scene=dict(
            xaxis_title='경도 (°E)',
            yaxis_title='위도 (°N)',
            zaxis_title='고도 (m)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1200,
        height=800,
        showlegend=True
    )
    
    fig.write_html("output_map_3d.html")
    print("3D 지도 저장: output_map_3d.html")

def create_2d_visualization(path, victim_gps, landing_point, wind_field, dem_lats, dem_lons, 
                           path_penalties):
    """2D folium 시각화 (패널티 표시)"""
    import folium
    from folium import plugins
    
    center_lat = victim_gps["latitude"]
    center_lon = victim_gps["longitude"]
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # 경로상 패널티에 따른 색상 그라디언트
    path_coords = [[dem_lats[r, c], dem_lons[r, c]] for r, c in path]
    penalties = [p["penalty"] for p in path_penalties]
    max_penalty = max(penalties) if penalties else 1
    
    for i, ((r, c), penalty) in enumerate(zip(path, penalties)):
        # 패널티에 따른 색상 (0:초록색, 1:빨간색)
        penalty_ratio = penalty / max_penalty
        if penalty_ratio < 0.5:
            color = 'green'
        elif penalty_ratio < 0.7:
            color = 'orange'
        else:
            color = 'red'
        
        # 마커 추가
        folium.CircleMarker(
            location=[dem_lats[r, c], dem_lons[r, c]],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"지점 {i}<br>경사도: {path_penalties[i]['slope']:.1f}°<br>" +
                  f"풍속: {path_penalties[i]['wind_speed']:.1f}m/s<br>" +
                  f"패널티: {penalty:.2f}"
        ).add_to(m)
    
    # 경로 폴리라인
    folium.PolyLine(
        locations=path_coords,
        color='blue',
        weight=3,
        opacity=0.8,
        tooltip='구조 경로'
    ).add_to(m)
    
    # 조난자 마커
    folium.Marker(
        location=[victim_gps["latitude"], victim_gps["longitude"]],
        popup="조난자 위치",
        icon=folium.Icon(color="red", icon="exclamation-sign")
    ).add_to(m)
    
    # 착륙지점 마커
    folium.Marker(
        location=[landing_point["latitude"], landing_point["longitude"]],
        popup=f"호이스트 착륙지점\n거리: {landing_point['distance_m']:.0f}m",
        icon=folium.Icon(color="blue", icon="helicopter")
    ).add_to(m)
    
    m.save("output_map_2d_penalty.html")
    print("2D 패널티 지도 저장: output_map_2d_penalty.html")


def main(victim_csv_path: str):
    # 1. 조난자 GPS 로드
    df_victim = pd.read_csv(victim_csv_path)
    victim = df_victim.iloc[0]
    victim_gps = {"latitude": victim["latitude"], "longitude": victim["longitude"]}
    print(f"조난자 위치: {victim_gps}")

    # 2. 설악산 데이터 로드
    df_terrain = load_seorak_data()
    print(f"전체 지형 데이터: {len(df_terrain)} 행")

    # 조난자 근처 지역만 필터링 (±0.15도 범위로 확대 - 119센터 포함)
    lat_range = 0.15
    lon_range = 0.15
    df_terrain = df_terrain[
        (df_terrain['latitude'] >= victim_gps['latitude'] - lat_range) &
        (df_terrain['latitude'] <= victim_gps['latitude'] + lat_range) &
        (df_terrain['longitude'] >= victim_gps['longitude'] - lon_range) &
        (df_terrain['longitude'] <= victim_gps['longitude'] + lon_range)
    ]
    print(f"필터링된 지형 데이터: {len(df_terrain)} 행")
    
    if len(df_terrain) < 10:
        print("유효한 데이터가 부족합니다.")
        return
    
    # 보간 성능 개선을 위해 샘플링 (최대 3만 포인트)
    if len(df_terrain) > 30000:
        df_terrain = df_terrain.sample(n=30000, random_state=42)
        print(f"샘플링된 지형 데이터: {len(df_terrain)} 행")

    # 3. DEM과 기본 격자 구성
    lon_min, lon_max = df_terrain['longitude'].min(), df_terrain['longitude'].max()
    lat_min, lat_max = df_terrain['latitude'].min(), df_terrain['latitude'].max()
    print(f"위경도 범위: lat[{lat_min:.4f}, {lat_max:.4f}], lon[{lon_min:.4f}, {lon_max:.4f}]")

    # 300x300 격자로 보간
    n_grid = 200
    dem_lats, dem_lons = np.meshgrid(
        np.linspace(lat_min, lat_max, n_grid),
        np.linspace(lon_min, lon_max, n_grid)
    )
    
    # DEM 배열 구성 (elevation으로부터 보간)
    from scipy.interpolate import griddata
    dem_array = griddata(
        df_terrain[['longitude', 'latitude']].values,
        df_terrain['elevation'].values,
        (dem_lons, dem_lats),
        method='linear'
    )
    dem_array = np.nan_to_num(dem_array, nan=np.nanmean(dem_array))
    
    # 임상도 구성 (tree_density와 tree_height로부터)
    forest_density = griddata(
        df_terrain[['longitude', 'latitude']].values,
        df_terrain['tree_density'].values,
        (dem_lons, dem_lats),
        method='nearest'
    )
    forest_height = griddata(
        df_terrain[['longitude', 'latitude']].values,
        df_terrain['tree_height'].values,
        (dem_lons, dem_lats),
        method='nearest'
    )
    forest_map = np.where(forest_density > 0.5, 1, 0)  # 임목 밀도 기준

    # 4. 지형 레이어 구성
    terrain = build_terrain_layer(dem_array, forest_map)

    # 5. 기상청 API허브 실시간 바람장 생성
    from config import KMA_API_KEY
    if KMA_API_KEY:
        print(f"[API 키 확인] KMA_API_KEY 정상 로드 (끝 4자리: ...{KMA_API_KEY[-4:]})")
    else:
        print("[API 키 확인] ❌ KMA_API_KEY 미설정 — .env 파일을 확인하세요. 폴백 모드로 진행합니다.")

    print("기상청 API허브 AWS 관측 호출 중...")
    live_weather_api_data = fetch_kma_realtime()
    print(f"  → 수신된 지점 수: {len(live_weather_api_data)}개")

    # cKDTree 최근접 이웃 기반 공간 융합: terrain_df 피처 보존 + 실시간 기상 컬럼 추가
    processed_df = mapping_live_weather_to_grid(
        victim_gps["latitude"], victim_gps["longitude"],
        live_weather_api_data, df_terrain
    )

    # 공간 융합 결과에서 numpy 배열 추출 (nearest-neighbor 보장으로 NaN 없음)
    if processed_df.empty or processed_df["wind_speed"].isna().all():
        # API 전체 실패 시 5×5=25포인트 기본값 격자 생성 (griddata 삼각분할 최소 요건 충족)
        print("[경고] 기상청 API 응답 없음. 5×5 기본값 격자(2.0 m/s, 북풍)로 폴백합니다.")
        lat_pts = np.linspace(victim_gps["latitude"]  - 0.1, victim_gps["latitude"]  + 0.1, 5)
        lon_pts = np.linspace(victim_gps["longitude"] - 0.1, victim_gps["longitude"] + 0.1, 5)
        lon_grid, lat_grid = np.meshgrid(lon_pts, lat_pts)
        kma_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        kma_ws     = np.full(25, 2.0)   # 8.0→2.0: 고도보정 후에도 임계치 이하 유지
        kma_wd     = np.full(25, 0.0)
    else:
        kma_points = processed_df[["longitude", "latitude"]].values
        kma_ws     = processed_df["wind_speed"].values
        kma_wd     = processed_df["wind_direction"].values

    # u/v 분해·고도보정·DEM 격자 보간 → 기존 파이프라인과 동일한 wind_field 구조 반환
    wind_field = build_wind_field(dem_lats, dem_lons, kma_points, kma_ws, kma_wd, dem_array)

    # 6. 호이스트 착륙지점 선정
    landing_point = find_hoist_candidates(victim_gps, terrain, wind_field,
                                           dem_lats, dem_lons)
    if landing_point is None:
        print("호이스트 착륙지점을 찾을 수 없습니다.")
        return

    print(f"착륙지점: {landing_point['latitude']:.4f}, {landing_point['longitude']:.4f}")
    print(f"조난자까지 거리: {landing_point['distance_m']:.0f}m")

    # 7. 119 센터에서 착륙지점으로의 경로 탐색 (빠른 계산)
    fire_station_grid = latlon_to_grid(FIRE_STATION["latitude"], FIRE_STATION["longitude"],
                                        dem_lats, dem_lons)
    landing_grid = (landing_point["row"], landing_point["col"])
    
    print(f"\n=== 경로 분석 ===")
    print(f"119 센터: {FIRE_STATION['latitude']:.4f}, {FIRE_STATION['longitude']:.4f}")
    print(f"119 → 착륙지점 경로 생성 중...")
    path_to_landing = simple_path_with_obstacles(fire_station_grid, landing_grid, 
                                                  terrain, dem_lats, dem_lons, num_waypoints=30)
    
    # 8. 착륙지점에서 조난자까지의 경로
    victim_grid = latlon_to_grid(victim_gps["latitude"], victim_gps["longitude"],
                                  dem_lats, dem_lons)
    print(f"착륙지점 → 조난자 경로 생성 중...")
    path_to_victim = simple_path_with_obstacles(landing_grid, victim_grid, 
                                                 terrain, dem_lats, dem_lons, num_waypoints=20)
    
    # 전체 경로 통합
    full_path = path_to_landing + path_to_victim[1:]  # 중복 제거
    
    # 9. 경로상 패널티 분석
    path_penalties = []
    for r, c in full_path:
        slope = terrain["slope"][r, c]
        wind_speed = wind_field["ws"][r, c]
        is_forest = terrain["is_open"][r, c] == 0
        is_ridge = terrain["is_ridge"][r, c]
        
        # 종합 페널티 계산 (경사도 중심, 풍속 캡 처리)
        penalty  = (slope / 45.0) ** 1.5 * 0.6          # 경사도: 45도 기준 지수 증가
        penalty += min(wind_speed, 15.0) / 15.0 * 0.3   # 풍속: 15m/s 캡, 비중 축소
        penalty += (0.5 if is_forest else 0)             # 숲: 2.0→0.5
        penalty += (0.3 if is_ridge else 0)              # 능선: 1.5→0.3
        
        path_penalties.append({
            "row": r, "col": c,
            "slope": slope,
            "wind_speed": wind_speed,
            "is_forest": is_forest,
            "is_ridge": is_ridge,
            "penalty": penalty
        })
    
    # 10. V-World 기반 3D 미션 맵 생성
    print("V-World 3D 미션 맵 생성 중...")
    create_vworld_3d_mission_map(full_path, dem_lats, dem_lons, dem_array, path_penalties,
                                   FIRE_STATION, landing_point, victim_gps, terrain)
    
    # 11. Folium 기반 헬기 미션 플래닝 지도 (실무용)
    print("Folium 헬기 미션 지도 생성 중...")
    create_helicopter_mission_folium_map(full_path, dem_lats, dem_lons, dem_array, path_penalties,
                                         FIRE_STATION, landing_point, victim_gps, terrain)
    
    # 12. 기존 Plotly 3D 시각화 (참고용)
    print("Plotly 3D 지도 생성 중...")
    create_3d_visualization(dem_lats, dem_lons, dem_array, terrain, 
                           full_path, path_penalties, 
                           FIRE_STATION, landing_point, victim_gps)
    
    # 13. 2D 패널티 맵
    create_2d_visualization(full_path, victim_gps, landing_point, wind_field, 
                           dem_lats, dem_lons, path_penalties)
    
    # 14. 경로 평가
    if full_path:
        print("\n=== 경로 평가 ===")
        evaluate_path(full_path, terrain, wind_field, dem_lats, dem_lons)

if __name__ == "__main__":
    main("data/victim_gps.csv")
