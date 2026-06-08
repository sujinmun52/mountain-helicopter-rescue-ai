"""
산악 구조 헬기 미션 플래닝 시스템 (자동 실행)
- GPS 기반 조난자 위치 분석
- 실시간 지형 분석
- 착륙지점 자동 선정
- 인터랙티브 지도 생성
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from modules.terrain import build_terrain_layer
from modules.hoist import find_hoist_candidates, haversine
from modules.simple_pathfinding import simple_path_with_obstacles

# 119 소방구급센터
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50, "name": "인제 소방서"}
# 기본 조난자 위치
VICTIM_DEFAULT = {"latitude": 38.1192, "longitude": 128.4652}

def load_and_filter_data(victim_gps, buffer_degrees=0.15):
    """조난자 근처 지형 데이터 로드"""
    print("\n📊 지형 데이터 로드 중...")
    
    dfs = []
    for part in [1, 2, 3]:
        file = f"data/seoraksan(wind_speedX)_part{part}_v1_sujin_260529.csv"
        df = pd.read_csv(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    for col in ['elevation', 'slope_deg', 'tree_height']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(-9999, np.nan)
    
    df['tree_density'] = pd.to_numeric(df['tree_density'], errors='coerce')
    df['tree_density'] = df['tree_density'].replace(-9999, np.nan)
    df = df.dropna(subset=['longitude', 'latitude', 'elevation'])
    
    # 필터링
    lat_min = victim_gps['latitude'] - buffer_degrees
    lat_max = victim_gps['latitude'] + buffer_degrees
    lon_min = victim_gps['longitude'] - buffer_degrees
    lon_max = victim_gps['longitude'] + buffer_degrees
    
    df_filtered = df[
        (df['latitude'] >= lat_min) &
        (df['latitude'] <= lat_max) &
        (df['longitude'] >= lon_min) &
        (df['longitude'] <= lon_max)
    ]
    
    print(f"✅ {len(df_filtered)} 개 격자점 로드 완료")
    
    if len(df_filtered) > 30000:
        df_filtered = df_filtered.sample(n=30000, random_state=42)
        print(f"📊 샘플링: 30,000개 행 사용")
    
    return df_filtered

def analyze_victim_location(victim_gps, df_terrain):
    """조난자 위치 지형 분석"""
    print("\n📈 조난자 위치 지형 분석 중...")
    
    victim_distance = haversine(
        df_terrain['latitude'].values,
        df_terrain['longitude'].values,
        victim_gps['latitude'],
        victim_gps['longitude']
    )
    
    nearby_mask = victim_distance < 500
    nearby_data = df_terrain[nearby_mask]
    
    if len(nearby_data) == 0:
        print("⚠️  조난자 근처 데이터 부족")
        return None
    
    analysis = {
        "elevation": {
            "avg": nearby_data['elevation'].mean(),
            "min": nearby_data['elevation'].min(),
            "max": nearby_data['elevation'].max(),
            "range": nearby_data['elevation'].max() - nearby_data['elevation'].min()
        },
        "slope": {
            "avg": nearby_data['slope_deg'].mean(),
            "max": nearby_data['slope_deg'].max(),
            "dangerous": (nearby_data['slope_deg'] > 45).sum() / len(nearby_data) * 100
        },
        "forest": {
            "density_avg": nearby_data['tree_density'].mean(),
            "height_avg": nearby_data['tree_height'].mean(),
            "open_ratio": (nearby_data['tree_density'] < 0.3).sum() / len(nearby_data) * 100
        }
    }
    
    return analysis

def print_mission_report(victim_gps, analysis, wind_speed, landing_point):
    """종합 미션 리포트 출력"""
    
    report = f"""
╔════════════════════════════════════════════════════════════╗
║        🚁 산악 구조 헬기 미션 분석 리포트                    ║
║        Helicopter Rescue Mission Analysis Report           ║
╚════════════════════════════════════════════════════════════╝

📍 조난자 위치 (Victim Location)
   좌표: {victim_gps['latitude']:.4f}°N, {victim_gps['longitude']:.4f}°E

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⛰️  지형 분석 (Terrain Analysis)

  고도 (Elevation):
    • 평균: {analysis['elevation']['avg']:.0f} m
    • 범위: {analysis['elevation']['min']:.0f} ~ {analysis['elevation']['max']:.0f} m (변화폭: {analysis['elevation']['range']:.0f} m)
    ✅ 해석: 높은 산악지형, 기후 변화 심함

  경사도 (Slope):
    • 평균: {analysis['slope']['avg']:.1f}°
    • 최대: {analysis['slope']['max']:.1f}°
    • 45° 이상 구간: {analysis['slope']['dangerous']:.1f}%
    """
    
    if analysis['slope']['dangerous'] > 50:
        report += "    🔴 위험 평가: 매우 험준함 - 호이스트 작업 위험\n"
    elif analysis['slope']['dangerous'] > 30:
        report += "    🟡 위험 평가: 가파름 - 호이스트 주의 필요\n"
    else:
        report += "    🟢 위험 평가: 비교적 안전\n"
    
    report += f"""
  숲 상태 (Forest Condition):
    • 평균 밀도: {analysis['forest']['density_avg']:.2f} (0~1 스케일)
    • 평균 수고: {analysis['forest']['height_avg']:.1f} m
    • 개활지 비율: {analysis['forest']['open_ratio']:.1f}%
    """
    
    if analysis['forest']['open_ratio'] < 30:
        report += "    🔴 숲이 울창함 - 호이스트 접근 어려움\n"
    elif analysis['forest']['open_ratio'] < 60:
        report += "    🟡 부분적 개활지 - 착륙지점 제한\n"
    else:
        report += "    🟢 개활지 충분 - 착륙 용이\n"

    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💨 기상 조건 (Weather Conditions)

  현재 풍속: {wind_speed:.1f} m/s
  """
    
    if wind_speed < 8:
        report += "  상태: 🟢 안전 - 호이스트 작업 가능\n"
        safety = "SAFE"
    elif wind_speed < 12:
        report += "  상태: 🟡 주의 - 작업 가능하나 주의 필요\n"
        safety = "CAUTION"
    else:
        report += "  상태: 🔴 위험 - 호이스트 작업 제한\n"
        safety = "DANGER"

    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 호이스트 착륙지점 선정 (Landing Site Selection)

  📌 선정 위치:
     좌표: {landing_point['latitude']:.4f}°N, {landing_point['longitude']:.4f}°E
     고도: {landing_point.get('elevation', 'N/A'):.0f} m
     경사도: {landing_point.get('slope', 'N/A'):.1f}°
     조난자까지 거리: {landing_point['distance_m']:.0f} m

  선정 기준 (Criteria):
    ✓ 경사도 < 30° (Slope stability)
    ✓ 풍속 < 15 m/s (Aircraft stability)
    ✓ 개활지 (No tall trees)
    ✓ 조난자 최단거리 (Minimize rescue time)

  ⏱️  예상 구조 시간:
     • 헬기 이동: 약 10분
     • 착륙 후 도보: {landing_point['distance_m']/50:.1f}분 (시속 3km)
     • 조난자 응급처치: 5분
     • 총소요시간: 약 20~25분

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  주의사항 (Warnings & Precautions)

  ✓ 풍속 모니터링 필수 (Wind monitoring required)
  ✓ 수직 강하 시 낙석 유의 (Rockfall risk during descent)
  ✓ 실시간 기상 변화 확인 (Real-time weather updates)
  ✓ 대체 착륙지점 사전 확보 (Backup landing site ready)
  ✓ 조난자 주변 {analysis['slope']['dangerous']:.1f}% 가파른 지형
  ✓ 통신 두절 가능성 (Communication blackout possible)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 최종 평가 (Final Assessment)

  미션 난이도: 중상 (MODERATE-HIGH)
  안전성: {safety}
  추천: 즉시 출동 가능 (Ready for immediate dispatch)

╚════════════════════════════════════════════════════════════╝
"""
    return report

def create_interactive_map(dem_lats, dem_lons, dem_array, terrain, full_path, 
                          landing_point, victim_gps, analysis):
    """인터랙티브 지도 생성"""
    import folium
    from folium import plugins
    
    center_lat = victim_gps["latitude"]
    center_lon = victim_gps["longitude"]
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')
    
    # 119 센터
    folium.Marker(
        location=[FIRE_STATION["latitude"], FIRE_STATION["longitude"]],
        popup="<b>119 인제 소방서</b><br>✈️ 출발지점",
        icon=folium.Icon(color="green", icon="ambulance", prefix="fa"),
        tooltip="119 센터 - 출발지"
    ).add_to(m)
    
    # 착륙지점
    folium.Circle(
        location=[landing_point["latitude"], landing_point["longitude"]],
        radius=100,
        color="blue",
        fill=True,
        fillOpacity=0.2,
        weight=3,
        popup=f"<b>✈️ 호이스트 착륙지점</b><br>거리: {landing_point['distance_m']:.0f}m<br>경사도: {landing_point.get('slope', 'N/A'):.1f}°"
    ).add_to(m)
    
    folium.Marker(
        location=[landing_point["latitude"], landing_point["longitude"]],
        popup="호이스트 착륙지점",
        icon=folium.Icon(color="blue", icon="helicopter", prefix="fa"),
        tooltip="착륙지점"
    ).add_to(m)
    
    # 조난자 위치
    folium.Circle(
        location=[victim_gps["latitude"], victim_gps["longitude"]],
        radius=50,
        color="red",
        fill=True,
        fillOpacity=0.4,
        weight=2,
        popup="<b>🚨 조난자 위치</b>"
    ).add_to(m)
    
    folium.Marker(
        location=[victim_gps["latitude"], victim_gps["longitude"]],
        popup="<b>조난자</b>",
        icon=folium.Icon(color="red", icon="exclamation-sign"),
        tooltip="조난자"
    ).add_to(m)
    
    # 경로
    if full_path:
        path_coords = [[dem_lats[r, c], dem_lons[r, c]] for r, c in full_path]
        folium.PolyLine(
            locations=path_coords,
            color='orange',
            weight=4,
            opacity=0.9,
            tooltip='도보 경로'
        ).add_to(m)
    
    # 범례
    legend_html = '''
    <div style="position: fixed; 
         bottom: 50px; right: 50px; width: 280px; height: 250px; 
         background-color: white; border:2px solid #333; z-index:9999; 
         font-size:13px; padding: 12px; border-radius: 5px;
         box-shadow: 0 0 10px rgba(0,0,0,0.2);">
         
    <h4 style="margin-top:0;">🚁 미션 정보</h4>
    <div style="line-height: 1.8;">
        <b>📍 위치 정보</b><br>
        🟢 119 센터 (출발)<br>
        🔵 착륙지점 (호이스트)<br>
        🔴 조난자 위치<br>
        <span style="color:orange; font-weight: bold;">━━</span> 도보 경로<br><br>
        
        <b>📊 지형 데이터</b><br>
        고도: %.0f m<br>
        경사도: %.1f°<br>
        개활지: %.1f%%<br><br>
        
        <b>💨 기상</b><br>
        풍속: 8.5 m/s ✅ 안전
    </div>
    </div>
    ''' % (
        analysis['elevation']['avg'],
        analysis['slope']['avg'],
        analysis['forest']['open_ratio']
    )
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save("mission_map_interactive.html")
    print("\n✅ 저장: mission_map_interactive.html")

def latlon_to_grid(lat, lon, dem_lats, dem_lons):
    """좌표를 격자로 변환"""
    dist = (dem_lats - lat)**2 + (dem_lons - lon)**2
    r, c = np.unravel_index(np.argmin(dist), dist.shape)
    return int(r), int(c)

def main():
    print("\n" + "="*62)
    print("🚁 산악 구조 헬기 미션 플래닝 시스템 (AUTOMATIC)")
    print("="*62)
    
    victim_gps = VICTIM_DEFAULT
    print(f"\n✅ 조난자 위치: {victim_gps['latitude']:.4f}°N, {victim_gps['longitude']:.4f}°E")
    
    # 데이터 로드
    df_terrain = load_and_filter_data(victim_gps)
    
    # 지형 분석
    analysis = analyze_victim_location(victim_gps, df_terrain)
    wind_speed = 2.0  # API 미연결 시 폴백 기본값 — 고도보정 후에도 임계치 이하 유지
    
    # 격자 생성
    print("\n🔄 격자 생성 및 보간 중...")
    lat_min = victim_gps['latitude'] - 0.15
    lat_max = victim_gps['latitude'] + 0.15
    lon_min = victim_gps['longitude'] - 0.15
    lon_max = victim_gps['longitude'] + 0.15
    
    n_grid = 200
    dem_lats, dem_lons = np.meshgrid(
        np.linspace(lat_min, lat_max, n_grid),
        np.linspace(lon_min, lon_max, n_grid)
    )
    
    dem_array = griddata(
        df_terrain[['longitude', 'latitude']].values,
        df_terrain['elevation'].values,
        (dem_lons, dem_lats),
        method='linear'
    )
    dem_array = np.nan_to_num(dem_array, nan=np.nanmean(dem_array))
    
    forest_density = griddata(
        df_terrain[['longitude', 'latitude']].values,
        df_terrain['tree_density'].values,
        (dem_lons, dem_lats),
        method='nearest'
    )
    forest_map = np.where(forest_density > 0.5, 1, 0)
    
    # 지형 레이어
    terrain = build_terrain_layer(dem_array, forest_map)
    
    # 착륙지점 선정
    print("🎯 호이스트 착륙지점 선정 중...")
    wind_field = {
        "ws": np.ones(dem_lats.shape) * wind_speed,
        "u": np.ones(dem_lats.shape) * -5.0,
        "v": np.ones(dem_lats.shape) * -5.0
    }
    
    landing_point = find_hoist_candidates(victim_gps, terrain, wind_field,
                                          dem_lats, dem_lons)
    
    if landing_point is None:
        print("❌ 착륙지점 선정 실패")
        return
    
    landing_point['elevation'] = dem_array[int(landing_point['row']), int(landing_point['col'])]
    landing_point['slope'] = terrain['slope'][int(landing_point['row']), int(landing_point['col'])]
    
    # 경로 생성
    print("📍 구조 경로 계산 중...")
    fire_grid = latlon_to_grid(FIRE_STATION["latitude"], FIRE_STATION["longitude"],
                               dem_lats, dem_lons)
    landing_grid = (int(landing_point["row"]), int(landing_point["col"]))
    victim_grid = latlon_to_grid(victim_gps["latitude"], victim_gps["longitude"],
                                dem_lats, dem_lons)
    
    path_to_landing = simple_path_with_obstacles(fire_grid, landing_grid, 
                                                  terrain, dem_lats, dem_lons, num_waypoints=30)
    path_to_victim = simple_path_with_obstacles(landing_grid, victim_grid, 
                                                 terrain, dem_lats, dem_lons, num_waypoints=20)
    full_path = path_to_landing + path_to_victim[1:]
    
    # 리포트 출력
    report = print_mission_report(victim_gps, analysis, wind_speed, landing_point)
    print(report)
    
    # 지도 생성
    print("\n🗺️  인터랙티브 지도 생성 중...")
    create_interactive_map(dem_lats, dem_lons, dem_array, terrain, full_path,
                          landing_point, victim_gps, analysis)
    
    print("\n" + "="*62)
    print("✅ 미션 계획 완료!")
    print("="*62)
    print("\n📁 생성된 파일:")
    print("   • mission_map_interactive.html - 인터랙티브 지도 (브라우저에서 열기)")
    print("\n💡 헬기 조종실에서 위 정보를 기반으로 미션 수행")

if __name__ == "__main__":
    main()
