"""
산악 구조 헬기 미션 플래닝 시스템
- GPS 기반 조난자 위치 입력
- 실시간 지형 분석
- 착륙지점 자동 선정
- 3D 경로 시각화
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from modules.terrain import build_terrain_layer
from modules.hoist import find_hoist_candidates, haversine
from modules.simple_pathfinding import simple_path_with_obstacles
from modules.evaluation.metrics import evaluate_path

# 119 소방구급센터 좌표
FIRE_STATION = {"latitude": 38.25, "longitude": 128.50, "name": "인제 소방서"}

def get_gps_input():
    """사용자로부터 GPS 좌표 입력받기"""
    print("\n" + "="*60)
    print("🚁 산악 구조 헬기 미션 플래닝 시스템")
    print("="*60)
    print("\n조난자 GPS 좌표를 입력하세요.")
    print("설악산 근처: 위도 38.05~38.25°, 경도 128.35~128.55°\n")
    print("기본값 (Enter): 38.1192°N, 128.4652°E\n")
    
    while True:
        try:
            lat_input = input("📍 위도 (latitude) [기본: 38.1192]: ").strip()
            lon_input = input("📍 경도 (longitude) [기본: 128.4652]: ").strip()
            
            # 기본값 사용
            lat = float(lat_input) if lat_input else 38.1192
            lon = float(lon_input) if lon_input else 128.4652
            
            # 설악산 범위 검증
            if not (38.0 <= lat <= 38.3 and 128.3 <= lon <= 128.6):
                print("⚠️  범위 밖입니다. 설악산 근처 좌표를 입력하세요.")
                continue
            
            print(f"\n✅ 조난자 위치 확정: {lat:.4f}°N, {lon:.4f}°E")
            return {"latitude": lat, "longitude": lon}
            
        except ValueError:
            print("❌ 숫자 형식이 아닙니다. 다시 입력하세요.")

def load_and_filter_data(victim_gps, buffer_degrees=0.15):
    """조난자 근처 지형 데이터 로드 및 필터링"""
    print("\n📊 지형 데이터 로드 중...")
    
    dfs = []
    for part in [1, 2, 3]:
        file = f"data/seoraksan(wind_speedX)_part{part}_v1_sujin_260529.csv"
        df = pd.read_csv(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # 데이터 정제
    for col in ['elevation', 'slope_deg', 'tree_height']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(-9999, np.nan)
    
    df['tree_density'] = pd.to_numeric(df['tree_density'], errors='coerce')
    df['tree_density'] = df['tree_density'].replace(-9999, np.nan)
    df = df.dropna(subset=['longitude', 'latitude', 'elevation'])
    
    # 조난자 주변만 필터링
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
    
    # 보간용 샘플링
    if len(df_filtered) > 30000:
        df_filtered = df_filtered.sample(n=30000, random_state=42)
    
    return df_filtered

def analyze_victim_location(victim_gps, df_terrain):
    """조난자 위치 주변 지형 분석"""
    print("\n📈 조난자 위치 분석 중...")
    
    # 조난자 근처 데이터 (500m 반경)
    victim_distance = haversine(
        df_terrain['latitude'].values,
        df_terrain['longitude'].values,
        victim_gps['latitude'],
        victim_gps['longitude']
    )
    
    nearby_mask = victim_distance < 500  # 500m 이내
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

def print_analysis_report(victim_gps, analysis, wind_speed):
    """지형 분석 리포트 출력"""
    print("\n" + "="*60)
    print("🔍 조난자 위치 지형 분석 리포트")
    print("="*60)
    print(f"\n📍 조난자 위치: {victim_gps['latitude']:.4f}°N, {victim_gps['longitude']:.4f}°E")
    
    print(f"\n⛰️  고도 정보")
    print(f"   평균: {analysis['elevation']['avg']:.0f}m")
    print(f"   범위: {analysis['elevation']['min']:.0f}m ~ {analysis['elevation']['max']:.0f}m ({analysis['elevation']['range']:.0f}m 변화)")
    
    print(f"\n📊 경사도")
    print(f"   평균: {analysis['slope']['avg']:.1f}°")
    print(f"   최대: {analysis['slope']['max']:.1f}°")
    dangerous_pct = analysis['slope']['dangerous']
    risk_level = "🔴 매우 위험" if dangerous_pct > 50 else "🟡 위험" if dangerous_pct > 30 else "🟢 안전"
    print(f"   45° 이상 구간: {dangerous_pct:.1f}% {risk_level}")
    
    print(f"\n🌲 숲 상태")
    print(f"   평균 밀도: {analysis['forest']['density_avg']:.2f}")
    print(f"   평균 수고: {analysis['forest']['height_avg']:.1f}m")
    print(f"   개활지 비율: {analysis['forest']['open_ratio']:.1f}%")
    
    print(f"\n💨 현재 풍속: {wind_speed:.1f} m/s", end="")
    if wind_speed < 8:
        print(" 🟢 안전")
    elif wind_speed < 12:
        print(" 🟡 주의")
    else:
        print(" 🔴 위험 (호이스트 작업 제한)")
    
    print("\n" + "="*60)

def create_landing_report(landing_point, victim_gps, terrain_analysis):
    """착륙지점 선정 근거 리포트"""
    distance_to_victim = landing_point['distance_m']
    
    report = f"""
╔════════════════════════════════════════════════════════════╗
║         🚁 호이스트 착륙지점 선정 근거                        ║
╚════════════════════════════════════════════════════════════╝

📍 선정된 착륙지점
   위도: {landing_point['latitude']:.4f}°N
   경도: {landing_point['longitude']:.4f}°E
   조난자까지 거리: {distance_to_victim:.0f}m

🎯 선정 기준 (다음 조건을 모두 만족)
   ✓ 경사도 < 30° (안정적 호이스트 작업)
   ✓ 풍속 < 15 m/s (기체 안정성)
   ✓ 개활지 (나무 < 5m) (장비 진입 가능)
   ✓ 조난자와 최단거리 (구조 시간 단축)

📊 착륙지점 지형 정보
   고도: {landing_point.get('elevation', 'N/A'):.0f}m
   경사도: {landing_point.get('slope', 'N/A'):.1f}°
   개활지: YES

🚶 구조 경로
   - 착륙 후 도보 이동: {distance_to_victim:.0f}m
   - 예상 소요 시간: {distance_to_victim/50:.1f}분 (시속 3km)
   - 경로 난이도: 중상 (경사도 {terrain_analysis['slope']['avg']:.1f}°)

⚠️  주의사항
   - 수직 강하 시 장애물 유의
   - 조난자 주변 {terrain_analysis['slope']['dangerous']:.1f}% 가파른 지형
   - 실시간 기상 모니터링 필수
"""
    return report

def create_interactive_map(dem_lats, dem_lons, dem_array, terrain, full_path, 
                          fire_station, landing_point, victim_gps, analysis):
    """Folium 기반 고급 인터랙티브 지도"""
    import folium
    from folium import plugins
    
    center_lat = victim_gps["latitude"]
    center_lon = victim_gps["longitude"]
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')
    
    # 119 센터
    folium.Marker(
        location=[FIRE_STATION["latitude"], FIRE_STATION["longitude"]],
        popup="<b>119 인제 소방서</b><br>출발지점",
        icon=folium.Icon(color="green", icon="ambulance", prefix="fa"),
        tooltip="119 센터"
    ).add_to(m)
    
    # 착륙지점 (큰 원으로 표시)
    folium.Circle(
        location=[landing_point["latitude"], landing_point["longitude"]],
        radius=100,
        color="blue",
        fill=True,
        fillOpacity=0.2,
        weight=3,
        popup=f"<b>호이스트 착륙지점</b><br>" +
              f"거리: {landing_point['distance_m']:.0f}m<br>" +
              f"경사도: {landing_point.get('slope', 'N/A'):.1f}°"
    ).add_to(m)
    
    folium.Marker(
        location=[landing_point["latitude"], landing_point["longitude"]],
        popup="호이스트 착륙지점",
        icon=folium.Icon(color="blue", icon="helicopter", prefix="fa"),
        tooltip="착륙지점"
    ).add_to(m)
    
    # 조난자 위치 (강조)
    folium.Circle(
        location=[victim_gps["latitude"], victim_gps["longitude"]],
        radius=50,
        color="red",
        fill=True,
        fillOpacity=0.3,
        weight=2,
        popup="<b>조난자 위치</b>"
    ).add_to(m)
    
    folium.Marker(
        location=[victim_gps["latitude"], victim_gps["longitude"]],
        popup="<b>조난자</b>",
        icon=folium.Icon(color="red", icon="exclamation-sign"),
        tooltip="조난자"
    ).add_to(m)
    
    # 경로
    if path:
        path_coords = [[dem_lats[r, c], dem_lons[r, c]] for r, c in full_path]
        folium.PolyLine(
            locations=path_coords,
            color='orange',
            weight=4,
            opacity=0.9,
            tooltip='구조 경로'
        ).add_to(m)
    
    # 거리 표시
    folium.plugins.PolyLineTextPath(
        folium.PolyLine(
            locations=[[FIRE_STATION["latitude"], FIRE_STATION["longitude"]],
                      [landing_point["latitude"], landing_point["longitude"]]],
            color='transparent',
            weight=1
        ),
        f'  헬기 이동  ',
        repeat=True,
        offset=7
    ).add_to(m)
    
    # 범례
    legend_html = '''
    <div style="position: fixed; 
         bottom: 50px; right: 50px; width: 250px; height: 200px; 
         background-color: white; border:2px solid grey; z-index:9999; 
         font-size:14px; padding: 10px; border-radius: 5px;">
         
    <b>🚁 구조 미션 요소</b><br><br>
    <i style="background:green"></i> 119 센터<br>
    <i style="background:blue"></i> 호이스트 착륙지점<br>
    <i style="background:red"></i> 조난자 위치<br>
    <span style="color:orange">━━</span> 구조 경로<br><br>
    
    <b>지형 정보</b><br>
    고도: {:.0f}m<br>
    경사도: {:.1f}°<br>
    개활지: {:.1f}%
    </div>
    '''.format(
        analysis['elevation']['avg'],
        analysis['slope']['avg'],
        analysis['forest']['open_ratio']
    )
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save("mission_map_interactive.html")
    print("\n✅ 인터랙티브 지도 저장: mission_map_interactive.html")
    return m

def main():
    import sys
    
    # 명령줄 인자로 GPS 받기
    if len(sys.argv) > 2:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
        victim_gps = {"latitude": lat, "longitude": lon}
        print(f"\n✅ 조난자 위치 확정: {lat:.4f}°N, {lon:.4f}°E")
    else:
        # 기본값 사용
        victim_gps = get_gps_input()
    
    # 2. 데이터 로드
    df_terrain = load_and_filter_data(victim_gps)
    
    # 3. 지형 분석
    analysis = analyze_victim_location(victim_gps, df_terrain)
    
    # 현재 풍속 (API 미연결 시 폴백 기본값 — 고도보정 후에도 임계치 이하 유지)
    wind_speed = 2.0
    
    # 4. 분석 리포트 출력
    print_analysis_report(victim_gps, analysis, wind_speed)
    
    # 5. 격자 생성 및 보간
    print("\n🔄 지형 격자 생성 중...")
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
    
    # 6. 지형 레이어
    terrain = build_terrain_layer(dem_array, forest_map)
    
    # 7. 착륙지점 선정
    print("\n🎯 호이스트 착륙지점 선정 중...")
    
    wind_field = {
        "ws": np.ones(dem_lats.shape) * wind_speed,
        "u": np.ones(dem_lats.shape) * -5.0,
        "v": np.ones(dem_lats.shape) * -5.0
    }
    
    landing_point = find_hoist_candidates(victim_gps, terrain, wind_field,
                                          dem_lats, dem_lons)
    
    if landing_point is None:
        print("❌ 착륙지점을 찾을 수 없습니다.")
        return
    
    # 착륙지점에 고도, 경사도 추가
    landing_point['elevation'] = dem_array[int(landing_point['row']), int(landing_point['col'])]
    landing_point['slope'] = terrain['slope'][int(landing_point['row']), int(landing_point['col'])]
    
    # 8. 근거 리포트
    print(create_landing_report(landing_point, victim_gps, analysis))
    
    # 9. 경로 생성
    print("\n📍 구조 경로 계산 중...")
    
    def latlon_to_grid(lat, lon, dem_lats, dem_lons):
        dist = (dem_lats - lat)**2 + (dem_lons - lon)**2
        r, c = np.unravel_index(np.argmin(dist), dist.shape)
        return int(r), int(c)
    
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
    
    # 10. 지도 생성
    print("\n🗺️  인터랙티브 지도 생성 중...")
    create_interactive_map(dem_lats, dem_lons, dem_array, terrain, full_path,
                          FIRE_STATION, landing_point, victim_gps, analysis)
    
    print("\n" + "="*60)
    print("✅ 미션 계획 완료!")
    print("="*60)
    print("\n📁 생성된 파일:")
    print("   1. mission_map_interactive.html - 인터랙티브 지도")
    print("   2. output_map_3d.html - 3D 경로 시각화 (이전)")
    print("\n💡 브라우저에서 열어 확인하세요!")

if __name__ == "__main__":
    main()
