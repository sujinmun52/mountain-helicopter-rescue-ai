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

    # 119 → 착륙지점 비행 경로 (주황 점선)
    if flight_path:
        folium.PolyLine(
            locations=[[p["lat"], p["lon"]] for p in flight_path],
            color='#ff6b35', weight=3, opacity=0.85, dash_array='8, 6',
            tooltip='비행 경로 (119→착륙)'
        ).add_to(m)

    # ===== 경로 구성요소 =====
    path_coords = [[dem_lats[r, c], dem_lons[r, c]] for r, c in full_path]
    
    # 패널티에 따른 경로 세그먼트 색상화
    max_penalty = max([p["penalty"] for p in path_penalties]) if path_penalties else 1
    
    for i in range(len(path_coords) - 1):
        if i < len(path_penalties):
            penalty = path_penalties[i]["penalty"]
            penalty_ratio = penalty / max_penalty if max_penalty > 0 else 0
            
            # 패널티에 따른 색상 선택
            if penalty_ratio < 0.3:
                color = '#22c55e'  # 녹색
                weight = 2
            elif penalty_ratio < 0.6:
                color = '#eab308'  # 노란색
                weight = 3
            else:
                color = '#ef4444'  # 빨간색
                weight = 4
            
            # 경로 세그먼트
            folium.PolyLine(
                locations=[path_coords[i], path_coords[i+1]],
                color=color,
                weight=weight,
                opacity=0.8,
                popup=f"지점 {i+1}<br>경사도: {path_penalties[i]['slope']:.1f}°<br>풍속: {path_penalties[i]['wind_speed']:.1f}m/s"
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
    
    # 호이스트 착륙지점
    landing_height = dem_array[int(landing_point["row"]), int(landing_point["col"])]
    folium.Marker(
        location=[landing_point["latitude"], landing_point["longitude"]],
        popup=f"""
        <b>🚁 호이스트 착륙지점</b><br>
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
    
    # 지도 저장
    import os
    os.makedirs("outputs", exist_ok=True)
    m.save("outputs/helicopter_mission_folium.html")
    print("✅ Folium 기반 헬기 미션 지도 저장: outputs/helicopter_mission_folium.html")

    return "outputs/helicopter_mission_folium.html"
