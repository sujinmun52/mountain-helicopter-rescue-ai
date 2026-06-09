import folium
import numpy as np

def render_map(path, victim_gps, landing_point, wind_field, dem_lats, dem_lons):
    """
    Folium 지도 시각화
    - 조난자 위치 마커
    - 착륙지점 마커
    - A* 경로 폴리라인
    - 풍속 히트맵 (선택)
    """
    center_lat = victim_gps["latitude"]
    center_lon = victim_gps["longitude"]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

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

    # A* 경로 폴리라인
    if path:
        path_coords = [
            [dem_lats[r, c], dem_lons[r, c]] for r, c in path
        ]
        folium.PolyLine(
            locations=path_coords,
            color="orange",
            weight=3,
            opacity=0.8,
            tooltip="최적 구조 경로"
        ).add_to(m)

    # 풍속 위험 격자 표시 (15m/s 이상)
    danger_mask = wind_field["ws"] > 15.0
    danger_indices = np.argwhere(danger_mask)
    for idx in danger_indices[::50]:  # 성능상 50개 간격 샘플링
        r, c = idx
        folium.CircleMarker(
            location=[dem_lats[r, c], dem_lons[r, c]],
            radius=3,
            color="red",
            fill=True,
            fill_opacity=0.4,
            tooltip=f"위험풍속: {wind_field['ws'][r,c]:.1f}m/s"
        ).add_to(m)

    import os
    os.makedirs("outputs", exist_ok=True)
    m.save("outputs/output_map.html")
    print("지도 저장 완료: outputs/output_map.html")
    return m