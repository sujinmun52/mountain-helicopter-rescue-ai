"""
Folium 기반 헬기 미션 플래닝 지도
"""
import folium
from folium import plugins
import numpy as np
from modules.hoist import haversine

def create_helicopter_mission_folium_map(full_path, dem_lats, dem_lons, dem_array, path_penalties,
                                        fire_station, landing_point, victim_gps, terrain,
                                        flight_path=None,
                                        flight_eta_min=None, walk_eta_min=None,
                                        transit_eta_min=None):
    """
    Folium을 사용한 헬기 미션 플래닝 지도
    실제 지형 기반 패널티 시각화

    flight_path: 119 → 착륙지점 비행 경로 [{lat, lon, alt_m, ...}]. 주황 점선으로 표시.
    flight_eta_min / walk_eta_min / transit_eta_min: 미션 ETA(분). None이면 범례 ETA 박스 생략.
    """

    # 중심점 설정
    center_lat = victim_gps["latitude"]
    center_lon = victim_gps["longitude"]

    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )

    # 119 → 착륙지점 비행 경로 (세그먼트별: 풍속 위험에 따라 색·우회이유 tooltip)
    if flight_path:
        for i in range(len(flight_path) - 1):
            p = flight_path[i]
            ws = p.get("wind_speed", 0)
            rs = []
            if ws > 8:               rs.append(f"강풍 {ws:.0f}m/s")
            if p.get("is_ridge"):    rs.append("능선 난류")
            reason = " · ".join(rs) if rs else "양호(순항)"
            fcolor = "#ef4444" if ws > 12 else ("#f59e0b" if ws > 8 else "#ff6b35")
            tip = (f"<b>비행 · {reason}</b><br>풍속 {ws:.1f}m/s · "
                   f"풍향 {p.get('wind_dir', 0):.0f}° · 고도 {p['alt_m']:.0f}m")
            folium.PolyLine(
                locations=[[flight_path[i]["lat"], flight_path[i]["lon"]],
                           [flight_path[i + 1]["lat"], flight_path[i + 1]["lon"]]],
                color=fcolor, weight=3, opacity=0.85, dash_array='8, 6',
                tooltip=tip, popup=tip,
            ).add_to(m)

    # ===== 경로 구성요소 =====
    path_coords = [[dem_lats[r, c], dem_lons[r, c]] for r, c in full_path]
    
    # 패널티 구간(절대값)별 경로 세그먼트 색상 + '우회 이유' tooltip(hover)
    def _risk_reason(p):
        rs = []
        if p["slope"] > 30:               rs.append(f"급경사 {p['slope']:.0f}°")
        if p["wind_speed"] > 8:           rs.append(f"강풍 {p['wind_speed']:.0f}m/s")
        if p.get("is_forest"):            rs.append("밀림")
        if p.get("is_ridge"):             rs.append("능선 난류")
        return " · ".join(rs) if rs else "양호(저위험)"

    for i in range(len(path_coords) - 1):
        if i < len(path_penalties):
            p = path_penalties[i]
            pen = p["penalty"]
            if pen < 0.5:
                color, weight = "#22c55e", 3   # 안전
            elif pen < 0.8:
                color, weight = "#eab308", 4   # 주의
            else:
                color, weight = "#ef4444", 5   # 위험

            tip = (f"<b>⚠ {_risk_reason(p)}</b><br>"
                   f"패널티 {pen:.2f}<br>"
                   f"경사 {p['slope']:.0f}° · 풍속 {p['wind_speed']:.1f}m/s · "
                   f"풍향 {p.get('wind_dir', 0):.0f}°")
            folium.PolyLine(
                locations=[path_coords[i], path_coords[i + 1]],
                color=color, weight=weight, opacity=0.85,
                tooltip=tip, popup=tip,
            ).add_to(m)
    
    # ===== 마커 추가 =====
    
    # 119 센터 (출발점)
    folium.Marker(
        location=[fire_station["latitude"], fire_station["longitude"]],
        popup=f"""
        <b>🚒 119 센터 (출발점)</b><br>
        위도: {fire_station['latitude']:.4f}°<br>
        경도: {fire_station['longitude']:.4f}°<br>
        <hr>
        <b>거리</b> (착륙지점까지): {haversine(fire_station['latitude'], fire_station['longitude'], landing_point['latitude'], landing_point['longitude'])/1000:.2f}km
        """,
        icon=folium.Icon(color='green', icon='info-sign', prefix='glyphicon'),
        tooltip="119 센터"
    ).add_to(m)
    
    # 구조 지점 — 전술(mode)에 따라 라벨 구분 (landing=착륙 / hoist=호이스트)
    landing_height = dem_array[int(landing_point["row"]), int(landing_point["col"])]
    _spot = "호이스트 지점" if "hoist" in str(landing_point.get("mode", "")) else "착륙 지점"
    folium.Marker(
        location=[landing_point["latitude"], landing_point["longitude"]],
        popup=f"""
        <b>🚁 {_spot}</b><br>
        위도: {landing_point['latitude']:.4f}°<br>
        경도: {landing_point['longitude']:.4f}°<br>
        고도: {landing_height:.0f}m<br>
        <hr>
        <b>조난자까지 거리</b>: {landing_point['distance_m']:.0f}m<br>
        <b>지형 조건</b>: 개활지, 경사도 {terrain['slope'][int(landing_point['row']), int(landing_point['col'])]:.1f}°
        """,
        icon=folium.Icon(color='blue', icon='arrow-up', prefix='glyphicon'),
        tooltip="착륙지점"
    ).add_to(m)
    
    # 조난자 위치
    victim_row = int((victim_gps["latitude"] - dem_lats.min()) / (dem_lats.max() - dem_lats.min()) * dem_lats.shape[0])
    victim_col = int((victim_gps["longitude"] - dem_lons.min()) / (dem_lons.max() - dem_lons.min()) * dem_lons.shape[1])
    victim_height = dem_array[victim_row, victim_col]
    
    victim_slope = terrain['slope'][victim_row, victim_col] if 0 <= victim_row < terrain['slope'].shape[0] and 0 <= victim_col < terrain['slope'].shape[1] else 0
    victim_is_ridge = terrain['is_ridge'][victim_row, victim_col] if 0 <= victim_row < terrain['is_ridge'].shape[0] and 0 <= victim_col < terrain['is_ridge'].shape[1] else False
    victim_is_open = terrain['is_open'][victim_row, victim_col] if 0 <= victim_row < terrain['is_open'].shape[0] and 0 <= victim_col < terrain['is_open'].shape[1] else True
    
    folium.Marker(
        location=[victim_gps["latitude"], victim_gps["longitude"]],
        popup=f"""
        <b>🆘 조난자 위치</b><br>
        위도: {victim_gps['latitude']:.4f}°<br>
        경도: {victim_gps['longitude']:.4f}°<br>
        고도: {victim_height:.0f}m<br>
        <hr>
        <b>지형 상황</b><br>
        • 경사도: {victim_slope:.1f}° {"⚠️ 매우 가파름" if victim_slope > 45 else "⚠️ 가파름" if victim_slope > 30 else "✓ 보통"}<br>
        • 능선상: {"예 (강풍주의)" if victim_is_ridge else "아니오"}<br>
        • 임목: {"밀생 (호이스트 어려움)" if not victim_is_open else "개활지 (호이스트 용이)"}<br>
        """,
        icon=folium.Icon(color='red', icon='exclamation-sign', prefix='glyphicon'),
        tooltip="조난자"
    ).add_to(m)
    
    # ===== 거리 원 추가 =====
    # 119센터 기준 활동 반경 표시
    folium.Circle(
        location=[fire_station["latitude"], fire_station["longitude"]],
        radius=30000,  # 30km
        color='green',
        fill=False,
        weight=1,
        opacity=0.3,
        popup='30km 반경',
        dash_array='5, 5'
    ).add_to(m)
    
    # ===== 경로 통계 =====
    total_distance = sum([haversine(path_coords[i][0], path_coords[i][1], 
                                    path_coords[i+1][0], path_coords[i+1][1])
                         for i in range(len(path_coords)-1)])
    
    avg_slope = np.mean([p["slope"] for p in path_penalties]) if path_penalties else 0
    max_wind = max([p["wind_speed"] for p in path_penalties]) if path_penalties else 0
    
    # 범례 및 통계 추가
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 320px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 15px; border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <div style="font-weight: bold; margin-bottom: 10px; color: #ff6b35; font-size: 16px;">
            🚁 미션 브리핑
        </div>
        
        <div style="margin: 8px 0; padding: 8px; background: #f0f0f0; border-radius: 4px; font-size: 12px;">
            <b>경로 통계</b><br>
            • 총 거리: {total_distance/1000:.2f}km<br>
            • 포인트 수: {len(path_coords)}<br>
            • 평균 경사도: {avg_slope:.1f}°<br>
            • 최대 풍속: {max_wind:.1f}m/s<br>
        </div>
        
        <div style="margin: 8px 0; padding: 8px; background: #f0f0f0; border-radius: 4px; font-size: 12px;">
            <b>거리</b><br>
            • 119→착륙: {haversine(fire_station['latitude'], fire_station['longitude'], landing_point['latitude'], landing_point['longitude'])/1000:.2f}km<br>
            • 착륙→조난자: {landing_point['distance_m']/1000:.2f}km<br>
        </div>
        {("<div style='margin: 8px 0; padding: 8px; background: #e8f5e9; border-left: 4px solid #2e7d32; border-radius: 4px; font-size: 12px;'>"
          "<b>⏱ 이동 ETA (호이스트 제외)</b><br>"
          f"• 비행 (119→착륙): {flight_eta_min:.1f}분<br>"
          f"• 도보 (착륙→조난자): {walk_eta_min:.1f}분<br>"
          f"• <b>합계: {transit_eta_min:.1f}분</b>"
          "</div>") if (flight_eta_min is not None and walk_eta_min is not None and transit_eta_min is not None) else ""}
        <div style="margin: 8px 0; padding: 8px; background: #f0f0f0; border-radius: 4px; font-size: 12px;">
            <b>고도</b><br>
            • 조난자: {victim_height:.0f}m<br>
            • 착륙지: {landing_height:.0f}m<br>
            • 고도 차: {abs(victim_height - landing_height):.0f}m<br>
        </div>
        
        <div style="margin: 8px 0; padding: 8px; background: #fff8e1; border-left: 4px solid #ff6b35; border-radius: 4px; font-size: 11px;">
            <b>⚠️ 주의사항</b><br>
            • 조난자 위치 경사도: {victim_slope:.1f}°<br>
            • 능선상 위치 (강풍)<br>
            • 호이스트 시간: ~15분
        </div>
        
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 11px; color: #666;">
            <b>색상 범례</b><br>
            <span style="display: inline-block; width: 12px; height: 12px; background: #22c55e; border-radius: 2px;"></span> 안전 (낮은 패널티)<br>
            <span style="display: inline-block; width: 12px; height: 12px; background: #eab308; border-radius: 2px;"></span> 주의 (중간 패널티)<br>
            <span style="display: inline-block; width: 12px; height: 12px; background: #ef4444; border-radius: 2px;"></span> 위험 (높은 패널티)
        </div>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))

    # 🛰️ 실시간 기상청 AWS 연동 배지 (상단) — 실시간 API 기반임을 명시
    _avg_ws = (sum(p["wind_speed"] for p in path_penalties) / len(path_penalties)
               if path_penalties else 0.0)
    kma_badge = f"""
    <div style="position:fixed; top:12px; left:50%; transform:translateX(-50%);
         z-index:9999; background:rgba(16,41,26,0.92); color:#9be7a0;
         padding:8px 18px; border-radius:20px; font-size:13px; font-weight:bold;
         box-shadow:0 2px 8px rgba(0,0,0,0.35); white-space:nowrap;">
      🛰️ 실시간 기상청(KMA) AWS 관측 연동 · 경로 평균 풍속 {_avg_ws:.1f} m/s
    </div>"""
    m.get_root().html.add_child(folium.Element(kma_badge))

    # 정보 패널(범례·배지·통계)을 마우스로 드래그해 옮길 수 있게 (경로 가림 방지)
    drag_js = """
    <script>
    window.addEventListener('load', function () {
      setTimeout(function () {
        document.querySelectorAll('body > div[style*="fixed"]').forEach(function (el) {
          el.style.cursor = 'move';
          var drag = false, sx, sy, sl, st;
          el.addEventListener('mousedown', function (e) {
            drag = true; sx = e.clientX; sy = e.clientY;
            var r = el.getBoundingClientRect(); sl = r.left; st = r.top;
            el.style.transform = 'none'; el.style.right = 'auto'; el.style.bottom = 'auto';
            el.style.left = sl + 'px'; el.style.top = st + 'px'; e.preventDefault();
          });
          document.addEventListener('mousemove', function (e) {
            if (drag) { el.style.left = (sl + e.clientX - sx) + 'px';
                        el.style.top = (st + e.clientY - sy) + 'px'; }
          });
          document.addEventListener('mouseup', function () { drag = false; });
        });
      }, 600);
    });
    </script>"""
    m.get_root().html.add_child(folium.Element(drag_js))

    # 지도 저장
    import os
    os.makedirs("outputs", exist_ok=True)
    m.save("outputs/helicopter_mission_folium.html")
    print("✅ Folium 기반 헬기 미션 지도 저장: outputs/helicopter_mission_folium.html")

    return "outputs/helicopter_mission_folium.html"
