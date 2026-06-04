import sys
# Windows 콘솔(cp949)에서 유니코드(•, ✅, ° 등) 출력 시 인코딩 오류 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd
from modules.terrain import build_terrain_layer
from modules.pathfinding import astar, estimate_path_time
from modules.simple_pathfinding import simple_path_with_obstacles
from modules.evaluation.metrics import evaluate_path
from modules.vworld_3d import create_vworld_3d_mission_map
from modules.helicopter_mission_map import create_helicopter_mission_folium_map
from modules.weather import fetch_kma_realtime
from modules.weather_interpolation import build_wind_field_terrain_aware
from modules.rescue_zone import select_rescue_zone           # [Stage 2] ML 셸(MOCK)
from modules.hoist import haversine

# 구조대 베이스(출발지) — 119 소방구급센터 (설악산 북쪽 인제 소방서)
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50, "name": "인제 소방서"}

# Stage 2 구조구역 추출 반경
RESCUE_RADIUS_M = 500.0

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
    
    import os
    os.makedirs("outputs", exist_ok=True)
    fig.write_html("outputs/output_map_3d.html")
    print("3D 지도 저장: outputs/output_map_3d.html")

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
    
    import os
    os.makedirs("outputs", exist_ok=True)
    m.save("outputs/output_map_2d_penalty.html")
    print("2D 패널티 지도 저장: outputs/output_map_2d_penalty.html")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    데이터 준비 (Stage 2 입력 생성)                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def prepare_terrain_grid(victim_gps):
    """
    설악산 CSV → DEM/임상 격자 + 지형 레이어 구성.
    Stage 2(구조구역 선정)와 Stage 3(경로)가 공통으로 쓰는 지형 격자 묶음을 만든다.

    Returns:
        dict {dem_lats, dem_lons, dem_array, terrain, df_terrain} 또는 None(데이터 부족)
    """
    from scipy.interpolate import griddata

    df_terrain = load_seorak_data()
    print(f"전체 지형 데이터: {len(df_terrain)} 행")

    # 조난자 근처 지역만 필터링 (±0.15도 — 구조대 베이스 포함)
    lat_range = lon_range = 0.15
    df_terrain = df_terrain[
        (df_terrain['latitude']  >= victim_gps['latitude']  - lat_range) &
        (df_terrain['latitude']  <= victim_gps['latitude']  + lat_range) &
        (df_terrain['longitude'] >= victim_gps['longitude'] - lon_range) &
        (df_terrain['longitude'] <= victim_gps['longitude'] + lon_range)
    ]
    print(f"필터링된 지형 데이터: {len(df_terrain)} 행")
    if len(df_terrain) < 10:
        print("유효한 데이터가 부족합니다.")
        return None

    if len(df_terrain) > 30000:
        df_terrain = df_terrain.sample(n=30000, random_state=42)
        print(f"샘플링된 지형 데이터: {len(df_terrain)} 행")

    lon_min, lon_max = df_terrain['longitude'].min(), df_terrain['longitude'].max()
    lat_min, lat_max = df_terrain['latitude'].min(), df_terrain['latitude'].max()

    n_grid = 200
    dem_lats, dem_lons = np.meshgrid(
        np.linspace(lat_min, lat_max, n_grid),
        np.linspace(lon_min, lon_max, n_grid)
    )
    dem_array = griddata(
        df_terrain[['longitude', 'latitude']].values, df_terrain['elevation'].values,
        (dem_lons, dem_lats), method='linear'
    )
    dem_array = np.nan_to_num(dem_array, nan=np.nanmean(dem_array))

    forest_density = griddata(
        df_terrain[['longitude', 'latitude']].values, df_terrain['tree_density'].values,
        (dem_lons, dem_lats), method='nearest'
    )
    forest_map = np.where(forest_density > 0.5, 1, 0)
    terrain = build_terrain_layer(dem_array, forest_map)

    return {
        "dem_lats": dem_lats, "dem_lons": dem_lons, "dem_array": dem_array,
        "terrain": terrain, "df_terrain": df_terrain,
    }


def _radius_mask(victim_gps, dem_lats, dem_lons, radius_m):
    """조난자 GPS 기준 radius_m 반경 내 격자 셀 마스크(2D bool)."""
    dist = haversine(victim_gps["latitude"], victim_gps["longitude"], dem_lats, dem_lons)
    return dist <= radius_m


def _route_with_tobler(start, goal, terrain, wind_field, dem_lats, dem_lons):
    """
    Tobler's Hiking Function + 풍속 페널티 A*(modules.pathfinding.astar) 경로 탐색.
    A* 가 경로를 못 찾으면 웨이포인트 기반 단순 경로로 폴백(파이프라인 보호).
    """
    path = astar(start, goal, terrain, wind_field, dem_lats, dem_lons)
    if not path:
        print("  [폴백] A* 실패 → 웨이포인트 단순 경로 사용")
        path = simple_path_with_obstacles(start, goal, terrain, dem_lats, dem_lons,
                                          num_waypoints=30)
    return path


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  [Stage 1] 환자 신고 & GPS 수신                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
def stage1_patient_intake(victim_csv_path):
    """조난자 실시간 GPS(위도/경도) 수신."""
    print("\n[Stage 1] 환자 신고 & GPS 수신")
    df_victim = pd.read_csv(victim_csv_path)
    victim = df_victim.iloc[0]
    victim_gps = {"latitude": float(victim["latitude"]), "longitude": float(victim["longitude"])}
    print(f"  • 조난자 GPS: {victim_gps}")
    return victim_gps


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  [Stage 2] AI 구조구역 선정 (MOCK) — modules/rescue_zone.py 셸 호출      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def stage2_select_rescue_zone(victim_gps, grid_bundle, radius_m=RESCUE_RADIUS_M):
    """
    500 m 반경 격자를 추출해 Stage 2 셸(select_rescue_zone)에 전달하고,
    목적지 노드(착륙/호이스트 지점)를 돌려받는다.
    ⚠ 실제 ML 통합 시에도 이 함수는 변경 불필요 — 셸 내부만 교체.
    """
    print("\n[Stage 2] AI 구조구역 선정 (현재: MOCK)")
    mask = _radius_mask(victim_gps, grid_bundle["dem_lats"], grid_bundle["dem_lons"], radius_m)
    grid_data = {
        "radius_m": radius_m,
        "radius_mask": mask,
        "dem_lats": grid_bundle["dem_lats"],
        "dem_lons": grid_bundle["dem_lons"],
        "dem_array": grid_bundle["dem_array"],
        "terrain": grid_bundle["terrain"],
    }
    return select_rescue_zone(victim_gps, grid_data)   # ← ML 교체 지점


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  [Stage 3] 동적 경로 모델링 (기상 융합 + Tobler A*)                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def stage3_path_modeling(victim_gps, destination, grid_bundle):
    """
    KMA 실시간 풍황을 지형 인지형 보간으로 융합하고, 구조대 베이스 → 목적지 노드까지
    Tobler 보행함수 + 풍속 페널티 A* 로 최적 우회 경로와 예상 소요시간(ETA)을 산출.

    Returns:
        dict {full_path, path_penalties, wind_field, eta_min}
    """
    print("\n[Stage 3] 동적 경로 모델링 (기상 융합 + Tobler A*)")
    dem_lats, dem_lons = grid_bundle["dem_lats"], grid_bundle["dem_lons"]
    dem_array, terrain = grid_bundle["dem_array"], grid_bundle["terrain"]

    # ── 3-1. KMA 실시간 풍황 → 지형 인지형(고도 보정 다변량 IDW) 바람장 ──────
    from config import KMA_API_KEY
    print(f"  • KMA API 키: {'정상' if KMA_API_KEY else '미설정(폴백 진행)'}")
    live = fetch_kma_realtime()
    print(f"  • 수신 관측소: {len(live)}개")
    if live:
        stations = live
    else:
        print("  • [경고] KMA 응답 없음 → 표고대별 합성 관측소로 폴백")
        stations = {
            "low":  {"lat": victim_gps["latitude"], "lon": victim_gps["longitude"],
                     "elev": 200.0,  "ws": 2.0, "wd": 0.0},
            "high": {"lat": victim_gps["latitude"] + 0.05, "lon": victim_gps["longitude"],
                     "elev": 1400.0, "ws": 6.0, "wd": 0.0},
        }
    wind_field = build_wind_field_terrain_aware(dem_lats, dem_lons, dem_array, stations)

    # ── 3-2. 베이스 → 목적지 노드 경로 (Tobler A*) ──────────────────────────
    base_grid = latlon_to_grid(FIRE_STATION["latitude"], FIRE_STATION["longitude"],
                               dem_lats, dem_lons)
    dest_grid = (destination["row"], destination["col"])
    print(f"  • 베이스={FIRE_STATION['name']} → 목적지 노드 경로 탐색")
    path_to_dest = _route_with_tobler(base_grid, dest_grid, terrain, wind_field,
                                      dem_lats, dem_lons)

    # ── 3-3. 목적지 노드 → 조난자 지상 접근 경로 (연속성 확보) ───────────────
    victim_grid = latlon_to_grid(victim_gps["latitude"], victim_gps["longitude"],
                                 dem_lats, dem_lons)
    path_to_victim = _route_with_tobler(dest_grid, victim_grid, terrain, wind_field,
                                       dem_lats, dem_lons)
    full_path = path_to_dest + path_to_victim[1:]

    # ── 3-4. 경로 패널티 + Tobler ETA ───────────────────────────────────────
    path_penalties = []
    for r, c in full_path:
        slope = terrain["slope"][r, c]
        ws = wind_field["ws"][r, c]
        is_forest = terrain["is_open"][r, c] == 0
        is_ridge = terrain["is_ridge"][r, c]
        penalty  = (slope / 45.0) ** 1.5 * 0.6
        penalty += min(ws, 15.0) / 15.0 * 0.3
        penalty += (0.5 if is_forest else 0)
        penalty += (0.3 if is_ridge else 0)
        path_penalties.append({"row": r, "col": c, "slope": slope, "wind_speed": ws,
                               "is_forest": is_forest, "is_ridge": is_ridge,
                               "penalty": penalty})

    eta_min = estimate_path_time(full_path, terrain, dem_lats, dem_lons)
    print(f"  • 경로 노드 수: {len(full_path)}, 예상 소요시간(ETA): {eta_min:.1f}분")

    return {"full_path": full_path, "path_penalties": path_penalties,
            "wind_field": wind_field, "eta_min": eta_min}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  [Stage 4] 3D 관제 시각화 (VWorld WebGL)                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
def stage4_visualization(victim_gps, destination, route, grid_bundle):
    """생성된 3D 경로/마커를 VWorld 3D 지형 뷰어(+보조 지도)로 스트리밍 시각화."""
    print("\n[Stage 4] 3D 관제 시각화 (VWorld WebGL)")
    dem_lats, dem_lons = grid_bundle["dem_lats"], grid_bundle["dem_lons"]
    dem_array, terrain = grid_bundle["dem_array"], grid_bundle["terrain"]
    full_path, path_penalties = route["full_path"], route["path_penalties"]

    print("  • VWorld 3D 미션 맵 생성 중...")
    create_vworld_3d_mission_map(full_path, dem_lats, dem_lons, dem_array, path_penalties,
                                 FIRE_STATION, destination, victim_gps, terrain)
    print("  • Folium 헬기 미션 지도 생성 중...")
    create_helicopter_mission_folium_map(full_path, dem_lats, dem_lons, dem_array, path_penalties,
                                         FIRE_STATION, destination, victim_gps, terrain)
    print("  • Plotly 3D / 2D 보조 지도 생성 중...")
    create_3d_visualization(dem_lats, dem_lons, dem_array, terrain, full_path, path_penalties,
                            FIRE_STATION, destination, victim_gps)
    create_2d_visualization(full_path, victim_gps, destination, route["wind_field"],
                            dem_lats, dem_lons, path_penalties)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     4단계 파이프라인 오케스트레이터                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def run_pipeline(victim_csv_path: str):
    """Stage 1 → 2(MOCK) → 3 → 4 데이터 흐름 실행."""
    # Stage 1
    victim_gps = stage1_patient_intake(victim_csv_path)

    # 공통 지형 격자 준비 (Stage 2 입력)
    grid_bundle = prepare_terrain_grid(victim_gps)
    if grid_bundle is None:
        return

    # Stage 2 (MOCK) → 목적지 노드
    destination = stage2_select_rescue_zone(victim_gps, grid_bundle)

    # Stage 3 → 경로 + ETA
    route = stage3_path_modeling(victim_gps, destination, grid_bundle)
    if not route["full_path"]:
        print("경로를 생성할 수 없습니다.")
        return

    # Stage 4 → 시각화
    stage4_visualization(victim_gps, destination, route, grid_bundle)

    # 경로 평가(참고)
    print("\n=== 경로 평가 ===")
    evaluate_path(route["full_path"], grid_bundle["terrain"], route["wind_field"],
                  grid_bundle["dem_lats"], grid_bundle["dem_lons"])


# 하위호환 별칭 (기존 호출부 보호)
main = run_pipeline

if __name__ == "__main__":
    run_pipeline("data/victim_gps.csv")