"""
V-World 3D API를 활용한 실제 지형 기반 시각화
"""
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()
VWORLD_3D_KEY = os.getenv("3D_API_KEY")

def haversine(lat1, lon1, lat2, lon2):
    """두 위경도 간 거리 계산"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def latlon_to_grid(lat, lon, dem_lats, dem_lons):
    """위경도를 가장 가까운 격자 (row, col)로 변환"""
    dist = (dem_lats - lat)**2 + (dem_lons - lon)**2
    r, c = np.unravel_index(np.argmin(dist), dist.shape)
    return int(r), int(c)

def create_vworld_3d_mission_map(full_path, dem_lats, dem_lons, dem_array, path_penalties,
                                  fire_station, landing_point, victim_gps, terrain,
                                  flight_path=None):
    """
    V-World 3D 지도 위에 구조 미션 정보를 표시
    Cesium.js 기반 HTML 생성

    Args:
        full_path: 착륙지점 → 조난자 도보 격자 경로 [(r, c), ...]
        flight_path: 119 → 착륙지점 헬기 비행 경로 [{lat, lon, alt_m, ...}, ...]
                     None 이면 (구버전 호환) 시작점/끝점만 직선으로 표시.
    """

    # 경로 좌표 변환 — 지면 보행 경로 (착륙지점 → 조난자)
    path_coords = []
    for r, c in full_path:
        lat = dem_lats[r, c]
        lon = dem_lons[r, c]
        height = dem_array[r, c]
        path_coords.append({
            "lat": float(lat),
            "lon": float(lon),
            "height": float(height)
        })

    # 비행 경로 좌표 (z 포함) — 시각화 시 별도 폴리라인으로 그림
    if flight_path:
        flight_coords = [
            {"lat": p["lat"], "lon": p["lon"], "height": p["alt_m"]}
            for p in flight_path
        ]
        flight_dist_m = flight_path[-1]["dist_m"]
        cruise_alt_m = max(p["alt_m"] for p in flight_path)
    else:
        flight_dist_m = 0.0
        cruise_alt_m = 0.0
    if not flight_path:
        flight_coords = [
            {"lat": float(fire_station["latitude"]), "lon": float(fire_station["longitude"]),
             "height": 1500.0},
            {"lat": float(landing_point["latitude"]), "lon": float(landing_point["longitude"]),
             "height": float(dem_array[int(landing_point["row"]), int(landing_point["col"])])},
        ]
    
    # 착륙지점 고도
    landing_height = dem_array[int(landing_point["row"]), int(landing_point["col"])]
    
    # 조난자 고도
    victim_row, victim_col = latlon_to_grid(victim_gps["latitude"], victim_gps["longitude"],
                                             dem_lats, dem_lons)
    victim_height = dem_array[victim_row, victim_col]
    
    # 총 경로 거리 계산
    total_distance = 0
    for i in range(len(full_path)-1):
        r1, c1 = full_path[i]
        r2, c2 = full_path[i+1]
        dist = haversine(dem_lats[r1, c1], dem_lons[r1, c1],
                        dem_lats[r2, c2], dem_lons[r2, c2])
        total_distance += dist
    
    # 경로상 패널티 정보 수집
    route_info = []
    for i, (r, c) in enumerate(full_path):
        penalty = path_penalties[i]["penalty"] if i < len(path_penalties) else 0
        slope = path_penalties[i]["slope"] if i < len(path_penalties) else 0
        wind_speed = path_penalties[i]["wind_speed"] if i < len(path_penalties) else 0
        is_forest = path_penalties[i]["is_forest"] if i < len(path_penalties) else False
        
        route_info.append({
            "index": i,
            "lat": float(dem_lats[r, c]),
            "lon": float(dem_lons[r, c]),
            "height": float(dem_array[r, c]),
            "slope": float(slope),
            "wind_speed": float(wind_speed),
            "is_forest": bool(is_forest),
            "penalty": float(penalty)
        })
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V-World 3D 산악 구조 미션 플래너</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/v-world-api@latest/css/vworld.css">
    <script src="https://cdn.jsdelivr.net/npm/v-world-api@latest/js/vworld.js"></script>
    <script src="https://cesiumjs.org/releases/1.104/Cesium.js"></script>
    <link rel="stylesheet" href="https://cesiumjs.org/releases/1.104/Widgets/widgets.css">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #fff;
        }}
        
        .container {{
            display: flex;
            height: 100vh;
            gap: 10px;
            padding: 10px;
            background: #0d0d0d;
        }}
        
        #map {{
            flex: 3;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            position: relative;
        }}
        
        .info-panel {{
            flex: 1;
            background: #1e1e1e;
            border-radius: 8px;
            padding: 15px;
            overflow-y: auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 1px solid #333;
        }}
        
        .mission-header {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ff6b35;
            color: #ff6b35;
        }}
        
        .location-card {{
            background: #252525;
            border-left: 4px solid #ff6b35;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        
        .location-label {{
            font-weight: bold;
            color: #ffd700;
            font-size: 12px;
            text-transform: uppercase;
        }}
        
        .location-data {{
            margin: 5px 0;
            font-size: 12px;
            line-height: 1.4;
        }}
        
        .status-good {{
            color: #4ade80;
        }}
        
        .status-warning {{
            color: #facc15;
        }}
        
        .status-critical {{
            color: #ef4444;
        }}
        
        .route-summary {{
            background: #252525;
            border-left: 4px solid #3b82f6;
            padding: 12px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        
        .stat {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 12px;
        }}
        
        .stat-value {{
            font-weight: bold;
            color: #3b82f6;
        }}
        
        .waypoint-list {{
            background: #252525;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
            max-height: 300px;
            overflow-y: auto;
        }}
        
        .waypoint {{
            background: #1a1a1a;
            padding: 8px;
            margin: 5px 0;
            border-radius: 3px;
            font-size: 11px;
            border-left: 3px solid #888;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .waypoint:hover {{
            border-left-color: #ff6b35;
            background: #2a2a2a;
        }}
        
        .waypoint.danger {{
            border-left-color: #ef4444;
        }}
        
        .waypoint.warning {{
            border-left-color: #facc15;
        }}
        
        .waypoint.safe {{
            border-left-color: #4ade80;
        }}
        
        .analysis {{
            background: #252525;
            border-left: 4px solid #a78bfa;
            padding: 12px;
            margin: 15px 0;
            border-radius: 4px;
            font-size: 12px;
            line-height: 1.5;
        }}
        
        .analysis-title {{
            font-weight: bold;
            color: #a78bfa;
            margin-bottom: 8px;
        }}
        
        .legend {{
            background: #252525;
            border-radius: 4px;
            padding: 10px;
            margin: 15px 0;
            font-size: 11px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div id="map"></div>
        
        <div class="info-panel">
            <div class="mission-header">🚁 미션 브리핑</div>
            
            <!-- 조난자 위치 -->
            <div class="location-card">
                <div class="location-label">🔴 조난자 위치</div>
                <div class="location-data">
                    위도: {victim_gps['latitude']:.4f}°<br>
                    경도: {victim_gps['longitude']:.4f}°<br>
                    고도: <span class="status-critical">{victim_height:.0f}m</span>
                </div>
            </div>
            
            <!-- 119 센터 -->
            <div class="location-card">
                <div class="location-label">🟢 119 센터 (출발지)</div>
                <div class="location-data">
                    위도: {fire_station['latitude']:.4f}°<br>
                    경도: {fire_station['longitude']:.4f}°<br>
                    고도: 1500m (기준)
                </div>
            </div>
            
            <!-- 착륙지점 -->
            <div class="location-card">
                <div class="location-label">🔵 호이스트 착륙지점</div>
                <div class="location-data">
                    위도: {landing_point['latitude']:.4f}°<br>
                    경도: {landing_point['longitude']:.4f}°<br>
                    고도: <span class="status-good">{landing_height:.0f}m</span><br>
                    조난자까지: <span class="status-good">{landing_point['distance_m']:.0f}m</span>
                </div>
            </div>
            
            <!-- 경로 요약 -->
            <div class="route-summary">
                <div class="location-label">📊 경로 분석</div>
                <div class="stat">
                    <span>비행 거리 (119→착륙):</span>
                    <span class="stat-value">{flight_dist_m:.0f}m</span>
                </div>
                <div class="stat">
                    <span>도보 거리 (착륙→조난자):</span>
                    <span class="stat-value">{sum([haversine(dem_lats[full_path[i][0], full_path[i][1]], dem_lons[full_path[i][0], full_path[i][1]], dem_lats[full_path[i+1][0], full_path[i+1][1]], dem_lons[full_path[i+1][0], full_path[i+1][1]]) for i in range(len(full_path)-1)]):.0f}m</span>
                </div>
                <div class="stat">
                    <span>순항고도:</span>
                    <span class="stat-value">{cruise_alt_m:.0f}m</span>
                </div>
                <div class="stat">
                    <span>도보 포인트:</span>
                    <span class="stat-value">{len(full_path)}</span>
                </div>
                <div class="stat">
                    <span>평균 경사도:</span>
                    <span class="stat-value">{sum([p['slope'] for p in path_penalties])/max(len(path_penalties), 1):.1f}°</span>
                </div>
                <div class="stat">
                    <span>최대 풍속:</span>
                    <span class="stat-value">{max([p['wind_speed'] for p in path_penalties]):.1f} m/s</span>
                </div>
            </div>
            
            <!-- 지형 분석 -->
            <div class="analysis">
                <div class="analysis-title">📍 조난자 위치 지형 분석</div>
                <div>
                    • 고도: {victim_height:.0f}m<br>
                    • 경사도: {terrain['slope'][victim_row, victim_col]:.1f}°<br>
                    • 임목 상태: {"밀생" if not terrain['is_open'][victim_row, victim_col] else "개활지"}<br>
                    • 능선상: {"예" if terrain['is_ridge'][victim_row, victim_col] else "아니오"}<br>
                    <br>
                    ✓ 착륙지점 선정 사유:<br>
                    조난자 근처의 개활지에서 경사도<br>
                    30° 이하, 수고 5m 이하 지역을<br>
                    우선 선정. 풍속 15m/s 이상<br>
                    구간 회피.
                </div>
            </div>
            
            <!-- 범례 -->
            <div class="legend">
                <div class="location-label">범례</div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4ade80;"></div>
                    <span>안전 (패널티 &lt;0.5)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #facc15;"></div>
                    <span>주의 (패널티 0.5-1.5)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ef4444;"></div>
                    <span>위험 (패널티 &gt;1.5)</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // V-World 3D 지도 초기화
        var map = new ol.Map({{
            target: 'map',
            layers: [
                new ol.layer.Tile({{
                    source: new ol.source.XYZ({{
                        url: 'https://xdworld.vworld.kr:8080/3d/rest/data/lidar?f=image&z={{z}}&x={{x}}&y={{y}}&apiKey={VWORLD_3D_KEY}'
                    }})
                }})
            ],
            view: new ol.View({{
                center: ol.proj.fromLonLat([{victim_gps['longitude']}, {victim_gps['latitude']}]),
                zoom: 14
            }})
        }});
        
        // 경로 표시 — (1) 도보 (착륙→조난자) (2) 비행 (119→착륙)
        var walkCoordinates = {path_coords};
        var flightCoordinates = {flight_coords};

        var walkPoints = walkCoordinates.map(p => ol.proj.fromLonLat([p.lon, p.lat]));
        var flightPoints = flightCoordinates.map(p => ol.proj.fromLonLat([p.lon, p.lat]));

        var walkFeature = new ol.Feature({{ geometry: new ol.geom.LineString(walkPoints) }});
        walkFeature.setStyle(new ol.style.Style({{
            stroke: new ol.style.Stroke({{ color: '#3b82f6', width: 3 }})
        }}));

        var flightFeature = new ol.Feature({{ geometry: new ol.geom.LineString(flightPoints) }});
        flightFeature.setStyle(new ol.style.Style({{
            stroke: new ol.style.Stroke({{
                color: '#ff6b35', width: 3, lineDash: [8, 6]
            }})
        }}));

        var vectorSource = new ol.source.Vector({{
            features: [walkFeature, flightFeature]
        }});

        var vectorLayer = new ol.layer.Vector({{ source: vectorSource }});
        map.addLayer(vectorLayer);
        
        // 마커 추가
        // 119 센터
        addMarker({fire_station['longitude']}, {fire_station['latitude']}, 
                  '119 센터', '#22c55e', map);
        
        // 착륙지점
        addMarker({landing_point['longitude']}, {landing_point['latitude']}, 
                  '착륙지점', '#3b82f6', map);
        
        // 조난자
        addMarker({victim_gps['longitude']}, {victim_gps['latitude']}, 
                  '조난자', '#ef4444', map);
        
        function addMarker(lon, lat, label, color, map) {{
            var iconFeature = new ol.Feature({{
                geometry: new ol.geom.Point(ol.proj.fromLonLat([lon, lat])),
                name: label
            }});
            
            var iconStyle = new ol.style.Style({{
                image: new ol.style.Circle({{
                    fill: new ol.style.Fill({{ color: color }}),
                    stroke: new ol.style.Stroke({{
                        color: '#fff',
                        width: 2
                    }}),
                    radius: 8
                }})
            }});
            
            iconFeature.setStyle(iconStyle);
            vectorSource.addFeature(iconFeature);
        }}
    </script>
</body>
</html>
"""
    
    import os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/helicopter_mission_3d.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ V-World 기반 3D 미션 맵 저장: helicopter_mission_3d.html")
    return "helicopter_mission_3d.html"

def haversine(lat1, lon1, lat2, lon2):
    """두 위경도 간 거리 계산"""
    import numpy as np
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
