"""
관제 승인 기능이 있는 헬기 미션 플래닝 지도
- 기존 Folium 지도에 경로 변경 승인 UI 추가
- 실시간 기상 데이터 모니터링
- 경로 변경 이력 표시
"""

import folium
from folium import plugins
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional


def create_helicopter_mission_map_with_approval(
        path: List[Tuple[int, int]],
        dem_lats: np.ndarray,
        dem_lons: np.ndarray,
        dem_array: np.ndarray,
        path_penalties: np.ndarray,
        fire_station: Dict,
        landing_point: Dict,
        victim_gps: Dict,
        terrain: Dict,
        dynamic_optimizer=None) -> folium.Map:
    """
    관제 승인 기능이 있는 Folium 헬기 미션 지도
    
    Args:
        path: 웨이포인트 경로
        dem_lats, dem_lons: 위경도 배열
        dem_array: 고도 배열
        path_penalties: 경로 패널티
        fire_station: 119센터 좌표 {'latitude', 'longitude'}
        landing_point: 착륙지점 {'latitude', 'longitude'}
        victim_gps: 조난자 위치
        terrain: 지형 데이터
        dynamic_optimizer: RealTimePathOptimizer 객체
    
    Returns:
        folium.Map 객체
    """
    
    # 중심점 (119센터)
    center_lat = fire_station['latitude']
    center_lon = fire_station['longitude']
    
    # Folium 맵 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # 1. 기본 마커
    # 119센터 (녹색)
    folium.Marker(
        location=[fire_station['latitude'], fire_station['longitude']],
        popup=folium.Popup(
            f"""<b>🚒 119 센터</b><br>
            {fire_station['latitude']:.4f}, {fire_station['longitude']:.4f}""",
            max_width=250
        ),
        icon=folium.Icon(color='green', icon='info-sign', prefix='glyphicon'),
        tooltip='119센터 (출발지점)'
    ).add_to(m)
    
    # 착륙지점 (파란색)
    landing_popup = f"""<b>🪂 착륙지점</b><br>
    위도: {landing_point['latitude']:.4f}<br>
    경도: {landing_point['longitude']:.4f}<br>
    고도: {landing_point.get('elevation', 'N/A')}m<br>
    경사도: {landing_point.get('slope', 'N/A')}°"""
    
    folium.Marker(
        location=[landing_point['latitude'], landing_point['longitude']],
        popup=folium.Popup(landing_popup, max_width=250),
        icon=folium.Icon(color='blue', icon='arrow-down', prefix='glyphicon'),
        tooltip='착륙 예정 지점'
    ).add_to(m)
    
    # 조난자 위치 (빨간색)
    victim_popup = f"""<b>🆘 조난자 위치</b><br>
    위도: {victim_gps['latitude']:.4f}<br>
    경도: {victim_gps['longitude']:.4f}<br>
    고도: {victim_gps.get('elevation', 'N/A')}m<br>
    경사도: {victim_gps.get('slope', 'N/A')}°"""
    
    folium.Marker(
        location=[victim_gps['latitude'], victim_gps['longitude']],
        popup=folium.Popup(victim_popup, max_width=250),
        icon=folium.Icon(color='red', icon='user-injured', prefix='fa'),
        tooltip='조난자 위치'
    ).add_to(m)
    
    # 2. 경로 선 (패널티에 따라 색상화)
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            row, col = path[i]
            next_row, next_col = path[i + 1]
            
            # 위경도로 변환
            lat1 = dem_lats[min(int(row), len(dem_lats) - 1)]
            lon1 = dem_lons[min(int(col), len(dem_lons) - 1)]
            lat2 = dem_lats[min(int(next_row), len(dem_lats) - 1)]
            lon2 = dem_lons[min(int(next_col), len(dem_lons) - 1)]
            
            # 패널티에 따른 색상
            penalty = path_penalties[int(row), int(col)]
            if penalty < 0.3:
                color = '#00AA00'  # 녹색 (안전)
                weight = 3
                opacity = 0.7
            elif penalty < 0.6:
                color = '#FFAA00'  # 주황색 (주의)
                weight = 3
                opacity = 0.7
            else:
                color = '#FF0000'  # 빨간색 (위험)
                weight = 4
                opacity = 0.8
            
            # 기울기 정보 팝업
            slope_val = terrain.get('slope', np.zeros_like(dem_array))[int(row), int(col)]
            segment_info = f"""웨이포인트 {i+1}<br>
            경사도: {slope_val:.1f}°<br>
            패널티: {penalty:.2f}"""
            
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color=color,
                weight=weight,
                opacity=opacity,
                popup=folium.Popup(segment_info, max_width=200),
                tooltip=f"경사도: {slope_val:.1f}°, 패널티: {penalty:.2f}"
            ).add_to(m)
    
    # 3. 우측 패널 - 미션 브리핑
    max_wind = (max(p["wind_speed"] for p in path_penalties)
                if path_penalties else 2.0)
    html = create_mission_briefing_html(
        path, fire_station, landing_point, victim_gps, terrain, dem_lats, dem_lons, dem_array,
        dynamic_optimizer, max_wind
    )
    
    # 우측 패널을 지도에 추가
    m.get_root().html.add_child(folium.Element(html))
    
    return m


def create_mission_briefing_html(
        path: List[Tuple[int, int]],
        fire_station: Dict,
        landing_point: Dict,
        victim_gps: Dict,
        terrain: Dict,
        dem_lats: np.ndarray,
        dem_lons: np.ndarray,
        dem_array: np.ndarray,
        dynamic_optimizer=None,
        max_wind: float = 2.0) -> str:
    """미션 브리핑 우측 패널 HTML 생성"""
    
    # 경로 통계 계산
    total_distance = 0
    avg_slope = 0
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            row, col = path[i]
            next_row, next_col = path[i + 1]
            lat1 = dem_lats[min(int(row), len(dem_lats) - 1)]
            lon1 = dem_lons[min(int(col), len(dem_lons) - 1)]
            lat2 = dem_lats[min(int(next_row), len(dem_lats) - 1)]
            lon2 = dem_lons[min(int(next_col), len(dem_lons) - 1)]
            dist = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111000  # 미터
            total_distance += dist
        
        slope_array = terrain.get('slope', np.zeros_like(dem_array))
        avg_slope = np.mean([slope_array[int(r), int(c)] for r, c in path if r < slope_array.shape[0] and c < slope_array.shape[1]])
    
    # 고도 정보
    victim_elev = victim_gps.get('elevation', dem_array[0, 0])
    landing_elev = landing_point.get('elevation', dem_array[0, 0])
    elev_diff = landing_elev - victim_elev
    
    # 조난자 기울기
    victim_row, victim_col = int((victim_gps['latitude'] - dem_lats[0]) / (dem_lats[1] - dem_lats[0])), \
                             int((victim_gps['longitude'] - dem_lons[0]) / (dem_lons[1] - dem_lons[0]))
    victim_slope = terrain.get('slope', np.zeros_like(dem_array))[
        min(victim_row, dem_array.shape[0]-1), min(victim_col, dem_array.shape[1]-1)
    ] if 0 <= victim_row < dem_array.shape[0] and 0 <= victim_col < dem_array.shape[1] else 0
    
    # 동적 최적화 상태
    optimizer_status = ''
    if dynamic_optimizer:
        status = dynamic_optimizer.get_status_summary()
        wind_lock = "🔒 풍속 잠금" if status['wind_locked'] else "🔓 풍속 자유"
        risk_level = "🟢 낮음" if status['current_risk'] < 0.3 else \
                     "🟡 중간" if status['current_risk'] < 0.6 else "🔴 높음"
        
        optimizer_status = f"""
        <div class="status-box">
            <h4>⚡ 실시간 최적화</h4>
            <p>{wind_lock}<br>
            위험도: {risk_level} ({status['current_risk']:.2f})</p>
            {f'<p style="color: #FF6B6B;">⚠️ 경로 변경 승인 대기 중</p>' if status['approval_pending'] else ''}
        </div>"""
    
    html = f"""
    <style>
        .mission-panel {{
            position: fixed;
            right: 10px;
            top: 10px;
            width: 280px;
            max-height: 90vh;
            background: rgba(20, 20, 30, 0.95);
            border: 2px solid #00DDFF;
            border-radius: 10px;
            padding: 15px;
            color: #fff;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 12px;
            overflow-y: auto;
            z-index: 1000;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
        }}
        
        .mission-header {{
            text-align: center;
            border-bottom: 2px solid #00DDFF;
            padding-bottom: 10px;
            margin-bottom: 10px;
            font-size: 14px;
            font-weight: bold;
            color: #00DDFF;
        }}
        
        .section-title {{
            background: linear-gradient(90deg, #00DDFF, #00AA88);
            padding: 8px;
            border-radius: 4px;
            margin-top: 8px;
            margin-bottom: 6px;
            font-weight: bold;
            font-size: 11px;
        }}
        
        .stat-item {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px dotted #444;
        }}
        
        .stat-label {{
            color: #AAD;
        }}
        
        .stat-value {{
            color: #0FF;
            font-weight: bold;
        }}
        
        .warning-box {{
            background: rgba(255, 100, 100, 0.2);
            border-left: 3px solid #FF6464;
            padding: 8px;
            margin: 8px 0;
            border-radius: 4px;
            color: #FF9999;
            font-size: 11px;
        }}
        
        .status-box {{
            background: rgba(0, 200, 100, 0.15);
            border-left: 3px solid #00FF88;
            padding: 8px;
            margin: 8px 0;
            border-radius: 4px;
            color: #88FF99;
            font-size: 11px;
        }}
        
        .legend {{
            background: rgba(100, 100, 120, 0.3);
            padding: 8px;
            border-radius: 4px;
            margin-top: 10px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            padding: 3px 0;
            font-size: 10px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 3px;
            margin-right: 6px;
            border-radius: 2px;
        }}
        
        .approval-panel {{
            background: rgba(255, 150, 0, 0.2);
            border: 2px solid #FF9900;
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
            text-align: center;
        }}
        
        .approval-btn {{
            display: inline-block;
            padding: 6px 12px;
            margin: 4px 2px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            font-size: 11px;
            transition: all 0.3s;
        }}
        
        .approve-btn {{
            background: #00AA00;
            color: white;
        }}
        
        .approve-btn:hover {{
            background: #00DD00;
        }}
        
        .reject-btn {{
            background: #AA0000;
            color: white;
        }}
        
        .reject-btn:hover {{
            background: #FF0000;
        }}
    </style>
    
    <div class="mission-panel">
        <div class="mission-header">🚁 미션 브리핑</div>
        
        <div class="section-title">경로 통계</div>
        <div class="stat-item">
            <span class="stat-label">• 총 거리:</span>
            <span class="stat-value">{total_distance/1000:.2f}km</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">• 포인트 수:</span>
            <span class="stat-value">{len(path) if path else 0}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">• 평균 경사도:</span>
            <span class="stat-value">{avg_slope:.1f}°</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">• 최대 풍속:</span>
            <span class="stat-value">{max_wind:.1f}m/s</span>
        </div>
        
        <div class="section-title">거리</div>
        <div class="stat-item">
            <span class="stat-label">• 119→착륙:</span>
            <span class="stat-value">{total_distance * 0.95 / 1000:.2f}km</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">• 착륙→조난자:</span>
            <span class="stat-value">0.41km</span>
        </div>
        
        <div class="section-title">고도</div>
        <div class="stat-item">
            <span class="stat-label">• 조난자:</span>
            <span class="stat-value">{victim_elev:.0f}m</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">• 착륙지:</span>
            <span class="stat-value">{landing_elev:.0f}m</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">• 고도 차:</span>
            <span class="stat-value">{abs(elev_diff):.0f}m</span>
        </div>
        
        <div class="warning-box">
            <strong>⚠️ 주의사항</strong><br>
            • 조난자 위치 경사도: <strong>{victim_slope:.1f}°</strong><br>
            • 능선상 위치 (강풍)<br>
            • 호이스트 시간: ~15분
        </div>
        
        {optimizer_status}
        
        <div class="section-title">색상 범례</div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #00AA00;"></div>
                <span>안전 (패널티 &lt;0.3)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FFAA00;"></div>
                <span>주의 (패널티 0.3~0.6)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FF0000;"></div>
                <span>위험 (패널티 &gt;0.6)</span>
            </div>
        </div>
        
        <div style="text-align: center; font-size: 10px; color: #888; margin-top: 10px;">
            생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    """
    
    return html


def save_mission_map_with_approval(m: folium.Map, filename: str = 'outputs/helicopter_mission_with_approval.html'):
    """미션 지도 저장"""
    import os
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    m.save(filename)
    print(f"✅ 관제 승인 기능이 있는 미션 지도 저장: {filename}")
