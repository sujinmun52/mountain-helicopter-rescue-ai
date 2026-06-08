"""[진단] 보행 A* 실패 원인 추적 — 격자 간격/경사 분포/차단율 점검."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import importlib.util

# main.py 동적 로드 (루트 진입점 함수 재사용)
spec = importlib.util.spec_from_file_location(
    "rootmain", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from modules.hoist import haversine
from modules.pathfinding import get_bounding_box, A_star_rescuer

victim_gps = m.stage1_patient_intake("data/victim_gps.csv")
bundle = m.prepare_terrain_grid(victim_gps)

dem_lats = bundle["dem_lats"]; dem_lons = bundle["dem_lons"]
terrain = bundle["terrain"]
slope = terrain["slope"]

# 실제 셀 간격(m)
mid = dem_lats.shape[0] // 2
cell_lat_m = haversine(dem_lats[mid, mid], dem_lons[mid, mid],
                       dem_lats[mid+1, mid], dem_lons[mid+1, mid])
cell_lon_m = haversine(dem_lats[mid, mid], dem_lons[mid, mid],
                       dem_lats[mid, mid+1], dem_lons[mid, mid+1])
print(f"\n[격자] shape={dem_lats.shape}")
print(f"[격자] 실제 셀 간격: 행방향≈{cell_lat_m:.1f}m, 열방향≈{cell_lon_m:.1f}m  (slope 계산은 resolution=15m 가정)")

# 착륙지/조난자 grid
dest = m.select_rescue_zone(victim_gps, {
    "radius_m": m.RESCUE_RADIUS_M,
    "radius_mask": m._radius_mask(victim_gps, dem_lats, dem_lons, m.RESCUE_RADIUS_M),
    "dem_lats": dem_lats, "dem_lons": dem_lons,
    "dem_array": bundle["dem_array"], "terrain": terrain,
}, heli_size={"light": "small", "heavy": "large"}.get(m.HELI_SIZE, "small"))
dest_grid = (dest["row"], dest["col"])
victim_grid = m.latlon_to_grid(victim_gps["latitude"], victim_gps["longitude"], dem_lats, dem_lons)
print(f"\n[노드] 착륙지 grid={dest_grid}, 조난자 grid={victim_grid}")

# 바운딩 박스(보행 마진 20) 내 경사 분포
bounds = get_bounding_box(dest_grid, victim_grid, slope.shape, 20)
min_r, max_r, min_c, max_c = bounds
sub = slope[min_r:max_r+1, min_c:max_c+1]
print(f"\n[바운딩박스] {bounds}, 셀 {sub.size}개")
print(f"  경사 min={np.nanmin(sub):.1f}° / median={np.nanmedian(sub):.1f}° / mean={np.nanmean(sub):.1f}° / max={np.nanmax(sub):.1f}°")
for thr in (30, 45, 60):
    pct = 100.0 * np.mean(sub > thr)
    print(f"  경사>{thr}° 비율: {pct:.1f}%")

# 45° 하드차단 가정에서 통행 가능 셀 비율
walkable = np.mean(sub <= 45.0) * 100.0
print(f"  → 보행 가능(≤45°) 셀: {walkable:.1f}%")

path = A_star_rescuer(dest_grid, victim_grid, terrain, size=m.HELI_SIZE)
print(f"\n[A* 결과] 경로 길이: {len(path)}  ({'성공' if path else '실패→폴백'})")
