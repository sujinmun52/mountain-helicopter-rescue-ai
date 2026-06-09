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
from modules.pathfinding import A_star_helicopter, A_star_rescuer, estimate_path_time
from modules.simple_pathfinding import simple_path_with_obstacles
from modules.evaluation.metrics import evaluate_path
from modules.vworld_3d import create_vworld_3d_mission_map
from modules.helicopter_mission_map import create_helicopter_mission_folium_map
from modules.weather import fetch_kma_realtime
from modules.weather_interpolation import build_wind_field_terrain_aware
from modules.rescue_zone import select_rescue_zone           # [Stage 2] ML 셸(MOCK)
from modules.hoist import haversine
from modules.flight_path import make_flight_path, flight_path_from_grid, estimate_flight_time  # 119→착륙 비행경로

# 구조대 베이스(출발지) — 119 소방구급센터 (설악산 북쪽 인제 소방서)
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50, "name": "인제 소방서"}

# Stage 2 구조구역 추출 반경
RESCUE_RADIUS_M = 500.0

# 미션 기본 파라미터 (PDF 도메인 점수표 4모드)
MISSION_MODE = "hoist"   # "landing" | "hoist"
HELI_SIZE    = "heavy"   # "light"   | "heavy"
FLIGHT_MARGIN_M = 150.0  # 비행 안전마진 (지형 최대고도 + margin)

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
                           path_penalties, fire_station, landing_point, victim_gps,
                           flight_path=None, alt_points=None):
    """3D plotly 시각화.

    alt_points: 2·3순위 대체 구조 후보 [{latitude, longitude, row, col, rank, mode, ...}].
                범례에서 토글(기본 숨김 'legendonly')하는 별도 트레이스로 표시.
    """
    import plotly.graph_objects as go

    # 경로 좌표 (위도, 경도, 고도)
    path_coords = np.array([[dem_lats[r, c], dem_lons[r, c], dem_array[r, c]]
                            for r, c in full_path])

    fig = go.Figure()

    # 119 → 착륙지점 비행 경로 (주황 점선, 안전고도)
    if flight_path:
        def _fl_reason(p):
            rs = []
            ws = p.get("wind_speed")
            if ws is not None and ws > 8:    rs.append(f"강풍 {ws:.0f}m/s")
            if p.get("is_ridge"):            rs.append("능선 난류")
            return " · ".join(rs) if rs else "양호(순항)"
        fl_cd = [[p.get("wind_speed", 0), p.get("wind_dir", 0), p["alt_m"], _fl_reason(p)]
                 for p in flight_path]
        fig.add_trace(go.Scatter3d(
            x=[p["lon"] for p in flight_path],
            y=[p["lat"] for p in flight_path],
            z=[p["alt_m"] for p in flight_path],
            mode='lines+markers',
            line=dict(color='#ff6b35', width=8), marker=dict(size=3, color='#ff6b35'),
            name='비행 경로 (119→착륙)',
            customdata=fl_cd,
            hovertemplate=('<b>비행 · %{customdata[3]}</b><br>고도 %{customdata[2]:.0f}m · '
                           '풍속 %{customdata[0]:.1f}m/s · 풍향 %{customdata[1]:.0f}°<extra></extra>')
        ))
    
    # DEM 표면 — x/y는 2D 격자 전체(경도·위도)를 넘겨야 표면이 제대로 렌더됨.
    #   (이 프로젝트 meshgrid 구조상 dem_lons[0]·dem_lats[:,0]은 상수 배열이라 표면이 뭉침)
    fig.add_trace(go.Surface(
        z=dem_array,
        x=dem_lons,
        y=dem_lats,
        colorscale='Earth',
        name='지형 고도',
        opacity=0.78,
        showscale=False
    ))
    
    # 도보 경로 — 선(회색 베이스) + penalty 구간별 3색 마커 + hover(우회 이유)
    def _risk_reason(p):
        parts = []
        if p["slope"] > 30:      parts.append(f"급경사 {p['slope']:.0f}°")
        if p["wind_speed"] > 8:  parts.append(f"강풍 {p['wind_speed']:.0f}m/s")
        if p.get("is_forest"):   parts.append("밀림")
        if p.get("is_ridge"):    parts.append("능선 난류")
        return " · ".join(parts) if parts else "양호(저위험)"

    fig.add_trace(go.Scatter3d(
        x=path_coords[:, 1], y=path_coords[:, 0], z=path_coords[:, 2],
        mode='lines', line=dict(color='#555555', width=4),
        name='도보 경로 (착륙→조난자)', hoverinfo='skip'))

    # penalty 구간: 안전<0.5 / 주의 0.5~0.8 / 위험>0.8 (도보 penalty 실제 분포 반영)
    bands = [("안전 (패널티<0.5)", "#22c55e", lambda v: v < 0.5),
             ("주의 (0.5~0.8)",   "#eab308", lambda v: 0.5 <= v < 0.8),
             ("위험 (>0.8)",       "#ef4444", lambda v: v >= 0.8)]
    for label, color, cond in bands:
        idx = [i for i, p in enumerate(path_penalties) if cond(p["penalty"])]
        if not idx:
            continue
        cdata = [[path_penalties[i]["penalty"], path_penalties[i]["slope"],
                  path_penalties[i]["wind_speed"], path_penalties[i].get("wind_dir", 0),
                  _risk_reason(path_penalties[i])] for i in idx]
        fig.add_trace(go.Scatter3d(
            x=path_coords[idx, 1], y=path_coords[idx, 0], z=path_coords[idx, 2],
            mode='markers', marker=dict(size=6, color=color),
            name=label, customdata=cdata,
            hovertemplate=('<b>%{customdata[4]}</b><br>패널티 %{customdata[0]:.2f}<br>'
                           '경사 %{customdata[1]:.0f}° · 풍속 %{customdata[2]:.1f}m/s · '
                           '풍향 %{customdata[3]:.0f}°<extra></extra>')))
    
    # 119 센터
    fire_row, fire_col = latlon_to_grid(fire_station["latitude"], fire_station["longitude"],
                                         dem_lats, dem_lons)
    fire_z = dem_array[fire_row, fire_col]
    # 이륙선: 119 지면 → 비행 시작 고도 (마커와 비행경로가 끊겨 보이지 않도록 연결)
    if flight_path:
        fig.add_trace(go.Scatter3d(
            x=[fire_station["longitude"], flight_path[0]["lon"]],
            y=[fire_station["latitude"], flight_path[0]["lat"]],
            z=[float(fire_z), flight_path[0]["alt_m"]],
            mode='lines', line=dict(color='#ff6b35', width=5, dash='dot'),
            name='이륙', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(
        x=[fire_station["longitude"]],
        y=[fire_station["latitude"]],
        z=[fire_z],
        mode='markers',
        marker=dict(size=15, color='green', symbol='diamond'),
        name='119 센터',
        hovertext=f"{fire_station['name']}"
    ))
    
    # 구조 지점 — 전술(mode)에 따라 라벨/심볼 구분 (landing=착륙, hoist=호이스트)
    _mode = str(landing_point.get("mode", ""))
    if "hoist" in _mode:
        spot_name, spot_symbol = "호이스트 지점", "circle"
    else:
        spot_name, spot_symbol = "착륙 지점", "square"
    fig.add_trace(go.Scatter3d(
        x=[landing_point["longitude"]],
        y=[landing_point["latitude"]],
        z=[dem_array[int(landing_point["row"]), int(landing_point["col"])]],
        mode='markers',
        marker=dict(size=15, color='blue', symbol=spot_symbol),
        name=f"{spot_name} (1순위 · {landing_point.get('distance_m', 0):.0f}m)",
        hovertext=f"1순위 · 조난자까지 {landing_point.get('distance_m', 0):.0f}m",
        hoverinfo='text'
    ))

    # 2·3순위 대체 후보 — 외부 버튼으로 토글(기본 '표시'). 범례에는 넣지 않아
    #   클릭 가능 여부가 헷갈리지 않게 한다. _alt_idx로 버튼 restyle 타겟을 지정.
    _alt_idx = None
    if alt_points:
        _alt_label = "호이스트" if "hoist" in _mode else "착륙"
        _alt_idx = len(fig.data)
        fig.add_trace(go.Scatter3d(
            x=[a["longitude"] for a in alt_points],
            y=[a["latitude"] for a in alt_points],
            z=[dem_array[int(a["row"]), int(a["col"])] for a in alt_points],
            mode='markers',
            marker=dict(size=11, color='#7c3aed', symbol=spot_symbol,
                        line=dict(color='white', width=1)),
            customdata=[[a.get("rank", 0), a.get("score", 0),
                         a.get("distance_m", 0)] for a in alt_points],
            name=f"{_alt_label} 2·3순위 (대체)",
            visible=True, showlegend=False,     # 기본 표시 · 범례 제외(버튼으로만 제어)
            hovertemplate=('<b>%{customdata[0]}순위 대체 후보</b><br>'
                           '적합도 %{customdata[1]:.3f} · 조난자까지 %{customdata[2]:.0f}m'
                           '<extra></extra>')
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
    
    # z축 상한 = 지형 최대고도와 비행 순항고도 중 큰 값(+여유) → 비행경로 클리핑 방지
    _z_top = float(np.nanmax(dem_array))
    if flight_path:
        _z_top = max(_z_top, max(p["alt_m"] for p in flight_path))
    _z_top *= 1.05

    fig.update_layout(
        title=dict(
            text='3D 산악 구조 경로 시각화<br><sub>설악산 지형 · 비행경로(119→착륙) · 도보경로(착륙→조난자)</sub>',
            x=0.5, xanchor='center', font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='경도 (°E)',
            yaxis_title='위도 (°N)',
            # z축 범위를 비행 순항고도까지 확장 → 비행경로가 잘리지 않고 전부 보임
            zaxis=dict(title='고도 (m)', range=[0, _z_top]),
            aspectratio=dict(x=1, y=1, z=0.5),   # 지형 입체감(수직 강조 과대 방지)
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.0)),
            dragmode='turntable'                 # 기본 드래그=회전(버튼으로 pan 전환)
        ),
        # 범례를 좌상단 안쪽으로 배치 + 반투명 배경 → 마커/컬러바와 겹침 해소
        legend=dict(
            x=0.01, y=0.98, xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.7)', bordercolor='#cccccc', borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        width=1200, height=800, showlegend=True
    )

    import os
    os.makedirs("outputs", exist_ok=True)
    fig.write_html("outputs/output_map_3d.html")
    print("3D 지도 저장: outputs/output_map_3d.html")

    # 발표용 정적 3D 이미지(PNG) — VWorld 대체. kaleido 미설치 시 graceful skip.
    try:
        fig.write_image("outputs/output_map_3d.png", width=1600, height=900, scale=2)
        print("3D 이미지 저장: outputs/output_map_3d.png (발표용)")
    except Exception as e:
        print(f"  [건너뜀] 3D PNG — {e} (pip install kaleido 시 생성)")

    return fig   # 인터랙티브(회전/줌) 표시용 — Gradio gr.Plot 등에서 사용


def create_2d_visualization(path, victim_gps, landing_point, wind_field, dem_lats, dem_lons,
                           path_penalties, flight_path=None):
    """2D folium 시각화 (패널티 표시)"""
    import folium
    from folium import plugins

    center_lat = victim_gps["latitude"]
    center_lon = victim_gps["longitude"]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # 119 → 착륙지점 비행 경로 (주황 점선)
    if flight_path:
        folium.PolyLine(
            locations=[[p["lat"], p["lon"]] for p in flight_path],
            color='#ff6b35', weight=3, opacity=0.8, dash_array='8, 6',
            tooltip='비행 경로 (119→착륙)'
        ).add_to(m)
    
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
    
    # 구조 지점 마커 — 전술(mode)에 따라 라벨 구분
    _spot2d = "호이스트 지점" if "hoist" in str(landing_point.get("mode", "")) else "착륙 지점"
    folium.Marker(
        location=[landing_point["latitude"], landing_point["longitude"]],
        popup=f"{_spot2d}\n거리: {landing_point['distance_m']:.0f}m",
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

    # 격자 '실제' 셀 간격(m) 측정 → 경사 계산 정확도 확보.
    # meshgrid(linspace, linspace) 구성상 셀이 비등방(행≠열)이므로 축별로 측정한다.
    # (조난자 ±0.15°/200격자 → 실측 약 81m(행)·140m(열)로, 하드코딩 15m와 불일치)
    r0 = dem_lats.shape[0] // 2
    c0 = dem_lats.shape[1] // 2
    dy_m = haversine(dem_lats[r0, c0], dem_lons[r0, c0],
                     dem_lats[r0 + 1, c0], dem_lons[r0 + 1, c0])   # 행(axis0) 간격
    dx_m = haversine(dem_lats[r0, c0], dem_lons[r0, c0],
                     dem_lats[r0, c0 + 1], dem_lons[r0, c0 + 1])   # 열(axis1) 간격
    print(f"  • 격자 실제 셀 간격: 행≈{dy_m:.0f}m, 열≈{dx_m:.0f}m (경사 계산에 반영)")

    # 실측 임관피도(0~1) 격자를 함께 전달 → 보행 A* 의 graded 밀도 risk 복원
    terrain = build_terrain_layer(dem_array, forest_map, tree_density=forest_density,
                                  cell_m=(dy_m, dx_m))

    return {
        "dem_lats": dem_lats, "dem_lons": dem_lons, "dem_array": dem_array,
        "terrain": terrain, "df_terrain": df_terrain,
    }


def _radius_mask(victim_gps, dem_lats, dem_lons, radius_m):
    """조난자 GPS 기준 radius_m 반경 내 격자 셀 마스크(2D bool)."""
    dist = haversine(victim_gps["latitude"], victim_gps["longitude"], dem_lats, dem_lons)
    return dist <= radius_m


def _route_walk(start, goal, terrain, dem_lats, dem_lons, size=HELI_SIZE):
    """
    착륙지점 → 조난자 지상 보행 경로 (Tobler 시간 + 임상 PDF risk A*).
    A* 가 경로를 못 찾으면 웨이포인트 단순 경로로 폴백(파이프라인 보호).
    """
    path = A_star_rescuer(start, goal, terrain, size=size)
    if not path:
        print("  [폴백] 보행 A* 실패 → 웨이포인트 단순 경로 사용")
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
# ║  [Stage 2] AI 구조구역 선정 (MOCK) — modules/rescue_zone.py 셸 호출    ║
# ╚══════════════════════════════════════════════════════════════════════╝
def stage2_select_rescue_zone(victim_gps, grid_bundle, heli_size=HELI_SIZE,
                              radius_m=RESCUE_RADIUS_M):
    """
    500 m 반경 격자를 추출해 Stage 2(select_rescue_zone)에 전달하고,
    목적지 노드(착륙/호이스트 지점)를 돌려받는다.

    Args:
        heli_size : "light" | "heavy" — A* 컨벤션. select_rescue_zone(XGBoost)은
                    "small"/"large"를 받으므로 아래에서 매핑한다.
    """
    print("\n[Stage 2] AI 구조구역 선정 (XGBoost 추론)")
    mask = _radius_mask(victim_gps, grid_bundle["dem_lats"], grid_bundle["dem_lons"], radius_m)
    grid_data = {
        "radius_m": radius_m,
        "radius_mask": mask,
        "dem_lats": grid_bundle["dem_lats"],
        "dem_lons": grid_bundle["dem_lons"],
        "dem_array": grid_bundle["dem_array"],
        "terrain": grid_bundle["terrain"],
    }
    # A* 컨벤션(light/heavy) → XGBoost 모델 컨벤션(small/large)
    ml_size = {"light": "small", "heavy": "large"}.get(heli_size, "small")
    return select_rescue_zone(victim_gps, grid_data, heli_size=ml_size)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  [Stage 3] 동적 경로 모델링 (기상 융합 + Tobler A*)                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
def stage3_path_modeling(victim_gps, destination, grid_bundle, heli_size=HELI_SIZE):
    """
    KMA 실시간 풍황을 지형 인지형 보간으로 융합하고, 구조대 베이스 → 목적지 노드까지
    Tobler 보행함수 + 풍속 페널티 A* 로 최적 우회 경로와 예상 소요시간(ETA)을 산출.

    Args:
        heli_size : "light" | "heavy" — 비행 A*/보행 A* 비용에 반영할 기종.

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

    # ── 3-2. 베이스 → 착륙지점: '헬기 비행' A* (풍속·풍향·지형추종 비용) ──────
    base_grid = latlon_to_grid(FIRE_STATION["latitude"], FIRE_STATION["longitude"],
                               dem_lats, dem_lons)
    dest_grid = (destination["row"], destination["col"])
    landing_lat = float(dem_lats[dest_grid[0], dest_grid[1]])
    landing_lon = float(dem_lons[dest_grid[0], dest_grid[1]])
    print(f"  • 베이스={FIRE_STATION['name']} → 착륙지점 헬기 비행 A* ({heli_size})")
    heli_grid = A_star_helicopter(base_grid, dest_grid, terrain, wind_field,
                                  dem_lats, dem_lons,
                                  margin_m=FLIGHT_MARGIN_M, size=heli_size)
    if heli_grid:
        flight_path = flight_path_from_grid(heli_grid, dem_lats, dem_lons, dem_array,
                                            margin_m=FLIGHT_MARGIN_M,
                                            wind_field=wind_field, terrain=terrain)
    else:
        print("  [폴백] 비행 A* 실패 → 직선 비행경로 사용")
        flight_path = make_flight_path(
            start_latlon=(FIRE_STATION["latitude"], FIRE_STATION["longitude"]),
            end_latlon=(landing_lat, landing_lon),
            dem_lats=dem_lats, dem_lons=dem_lons, dem_array=dem_array,
            margin_m=FLIGHT_MARGIN_M,
        )
    print(f"    - 순항고도 ≈ {max(p['alt_m'] for p in flight_path):.0f} m, "
          f"비행거리 ≈ {flight_path[-1]['dist_m']:.0f} m, 포인트 {len(flight_path)}")

    # ── 3-3. 착륙지점 → 조난자: 지상 보행 (Tobler A*) ───────────────────────
    victim_grid = latlon_to_grid(victim_gps["latitude"], victim_gps["longitude"],
                                 dem_lats, dem_lons)
    print(f"  • 착륙지점 → 조난자 도보 경로 탐색")
    path_to_victim = _route_walk(dest_grid, victim_grid, terrain, dem_lats, dem_lons,
                                 size=heli_size)
    # full_path: 기존 시각화 호환을 위해 '도보 구간' 만 격자 경로로 유지
    # (비행 구간은 별도 flight_path 키로 전달)
    full_path = path_to_victim

    # ── 3-4. 경로 패널티 + Tobler ETA ───────────────────────────────────────
    path_penalties = []
    for r, c in full_path:
        slope = terrain["slope"][r, c]
        ws = wind_field["ws"][r, c]
        # 풍향(불어오는 방향, deg): wind_field의 흐름벡터 u,v로 역산
        wd = float(np.degrees(np.arctan2(-wind_field["u"][r, c], -wind_field["v"][r, c])) % 360)
        is_forest = terrain["is_open"][r, c] == 0
        is_ridge = terrain["is_ridge"][r, c]
        penalty  = (slope / 45.0) ** 1.5 * 0.6
        penalty += min(ws, 15.0) / 15.0 * 0.3
        penalty += (0.5 if is_forest else 0)
        penalty += (0.3 if is_ridge else 0)
        path_penalties.append({"row": r, "col": c, "slope": float(slope), "wind_speed": float(ws),
                               "wind_dir": wd, "is_forest": bool(is_forest), "is_ridge": bool(is_ridge),
                               "penalty": float(penalty)})

    eta_min = estimate_path_time(full_path, terrain, dem_lats, dem_lons)
    from config import HELI_CRUISE_SPEED_MS
    flight_eta_min = estimate_flight_time(flight_path, HELI_CRUISE_SPEED_MS)
    transit_eta_min = flight_eta_min + eta_min  # 호이스트 제외 이동 ETA 합계
    print(f"  • 도보 노드 수: {len(full_path)}, 도보 ETA: {eta_min:.1f}분")
    print(f"  • 비행 ETA: {flight_eta_min:.1f}분 (순항 {HELI_CRUISE_SPEED_MS:.0f}m/s 가정)")
    print(f"  • 이동 합계 ETA(비행+도보, 호이스트 제외): {transit_eta_min:.1f}분")

    return {"full_path": full_path, "path_penalties": path_penalties,
            "wind_field": wind_field, "eta_min": eta_min,
            "flight_eta_min": flight_eta_min, "transit_eta_min": transit_eta_min,
            "flight_path": flight_path}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  [Stage 4] 3D 관제 시각화 (VWorld WebGL)                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
def stage4_visualization(victim_gps, destination, route, grid_bundle):
    """생성된 3D 경로/마커를 VWorld 3D 지형 뷰어(+보조 지도)로 스트리밍 시각화."""
    print("\n[Stage 4] 3D 관제 시각화 (VWorld WebGL)")
    dem_lats, dem_lons = grid_bundle["dem_lats"], grid_bundle["dem_lons"]
    dem_array, terrain = grid_bundle["dem_array"], grid_bundle["terrain"]
    full_path, path_penalties = route["full_path"], route["path_penalties"]

    flight_path = route.get("flight_path")
    print("  • VWorld 3D 미션 맵 생성 중...")
    create_vworld_3d_mission_map(full_path, dem_lats, dem_lons, dem_array, path_penalties,
                                 FIRE_STATION, destination, victim_gps, terrain,
                                 flight_path=flight_path)
    print("  • Folium 헬기 미션 지도 생성 중...")
    create_helicopter_mission_folium_map(full_path, dem_lats, dem_lons, dem_array, path_penalties,
                                         FIRE_STATION, destination, victim_gps, terrain,
                                         flight_path=flight_path,
                                         flight_eta_min=route["flight_eta_min"],
                                         walk_eta_min=route["eta_min"],
                                         transit_eta_min=route["transit_eta_min"])
    # Plotly 3D / 2D 는 '보조' 지도 — 미설치/오류 시 핵심 산출물을 막지 않도록 graceful skip
    print("  • Plotly 3D / 2D 보조 지도 생성 중...")
    try:
        create_3d_visualization(dem_lats, dem_lons, dem_array, terrain, full_path, path_penalties,
                                FIRE_STATION, destination, victim_gps, flight_path=flight_path)
    except ImportError as e:
        print(f"  [건너뜀] Plotly 3D 보조 지도 — {e} (pip install plotly 시 생성됨)")
    try:
        create_2d_visualization(full_path, victim_gps, destination, route["wind_field"],
                                dem_lats, dem_lons, path_penalties, flight_path=flight_path)
    except ImportError as e:
        print(f"  [건너뜀] 2D 보조 지도 — {e}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                     4단계 파이프라인 오케스트레이터                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def run_pipeline(victim_csv_path: str, heli_size: str = HELI_SIZE):
    """Stage 1 → 2(XGBoost) → 3 → 4 데이터 흐름 실행.

    Args:
        heli_size : "light" | "heavy" — Stage 2(모델 선택)·Stage 3(A* 비용)에 일관 적용.
    """
    # Stage 1
    victim_gps = stage1_patient_intake(victim_csv_path)

    # 공통 지형 격자 준비 (Stage 2 입력)
    grid_bundle = prepare_terrain_grid(victim_gps)
    if grid_bundle is None:
        return

    # Stage 2 (XGBoost 추론) → 목적지 노드
    destination = stage2_select_rescue_zone(victim_gps, grid_bundle, heli_size=heli_size)

    # Stage 3 → 경로 + ETA
    route = stage3_path_modeling(victim_gps, destination, grid_bundle, heli_size=heli_size)
    if not route["full_path"]:
        print("경로를 생성할 수 없습니다.")
        return

    # Stage 4 → 시각화
    stage4_visualization(victim_gps, destination, route, grid_bundle)

    # 경로 평가(참고)
    print("\n=== 경로 평가 ===")
    evaluate_path(route["full_path"], grid_bundle["terrain"], route["wind_field"],
                  grid_bundle["dem_lats"], grid_bundle["dem_lons"], size=heli_size)


# 하위호환 별칭 (기존 호출부 보호)
main = run_pipeline

if __name__ == "__main__":
    run_pipeline("data/victim_gps.csv")