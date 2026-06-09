"""[A* 가치 시연] 험준지에서 '단순 직선' vs 'A* 안전경로' 비교.
직선은 절벽을 가로질러 위험/도달불가, A*는 안전 우회 → A*의 핵심 가치 입증.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rootmain", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

from modules.pathfinding import A_star_rescuer, estimate_path_time
from modules.hoist import haversine

victim_gps = m.stage1_patient_intake("data/victim_gps.csv")
bundle = m.prepare_terrain_grid(victim_gps)
dem_lats, dem_lons, terrain = bundle["dem_lats"], bundle["dem_lons"], bundle["terrain"]
slope = terrain["slope"]
victim = m.latlon_to_grid(victim_gps["latitude"], victim_gps["longitude"], dem_lats, dem_lons)


def straight(a, b):
    """두 격자점을 잇는 직선 경로(격자 셀들)."""
    n = max(abs(b[0]-a[0]), abs(b[1]-a[1])) + 1
    return [(int(round(a[0]+(b[0]-a[0])*t/(n-1))),
             int(round(a[1]+(b[1]-a[1])*t/(n-1)))) for t in range(n)] if n > 1 else [a]


def path_stats(path):
    if not path:
        return None
    sl = [slope[r, c] for r, c in path]
    dist = sum(haversine(dem_lats[path[i][0], path[i][1]], dem_lons[path[i][0], path[i][1]],
                         dem_lats[path[i+1][0], path[i+1][1]], dem_lons[path[i+1][0], path[i+1][1]])
               for i in range(len(path)-1))
    eta = estimate_path_time(path, terrain, dem_lats, dem_lons)
    cliff = sum(1 for s in sl if s > 45.0)
    return dict(거리=dist, 평균경사=np.mean(sl), 최대경사=np.max(sl), 절벽셀=cliff, ETA=eta, 노드=len(path))


# 조난자 주변에서 '직선이 가장 험준한' 착륙지 후보 자동 탐색 (대비 효과 큰 케이스)
best = None
for dr in range(-35, 36, 5):
    for dc in range(-35, 36, 5):
        if abs(dr) < 15 and abs(dc) < 15:
            continue
        s = (victim[0]+dr, victim[1]+dc)
        if not (0 <= s[0] < slope.shape[0] and 0 <= s[1] < slope.shape[1]):
            continue
        line = straight(s, victim)
        sev = np.mean([slope[r, c] for r, c in line])
        if best is None or sev > best[0]:
            best = (sev, s)
start = best[1]
d0 = haversine(dem_lats[start[0], start[1]], dem_lons[start[0], start[1]],
               victim_gps["latitude"], victim_gps["longitude"])
print(f"시나리오: 착륙지{start} → 조난자{victim} (직선거리 {d0:.0f}m)\n")

st = path_stats(straight(start, victim))
ap = path_stats(A_star_rescuer(start, victim, terrain, size=m.HELI_SIZE))

print(f"{'':<14}{'단순 직선':>14}{'A* 안전경로':>16}")
print("-"*46)
def row(k, unit="", f="{:.0f}"):
    sv = st[k] if st else None; av = ap[k] if ap else None
    print(f"{k:<14}{(f.format(sv)+unit):>14}{((f.format(av)+unit) if av is not None else '도달실패'):>16}")
row("거리", "m"); row("평균경사", "°", "{:.1f}"); row("최대경사", "°", "{:.1f}")
row("절벽셀", "개"); row("ETA", "분", "{:.1f}")
print("-"*46)

print("\n[해석]")
if st and st["절벽셀"] > 0:
    print(f"  • 직선: 절벽(>45°) {st['절벽셀']}개 셀 통과 → 보행 위험/불가, 최대경사 {st['최대경사']:.0f}°")
if ap:
    print(f"  • A*: 절벽 {ap['절벽셀']}개 회피, 최대경사 {ap['최대경사']:.0f}° → 안전 우회 도달")
    if st:
        print(f"  • A*는 거리 {ap['거리']-st['거리']:+.0f}m 우회하지만 ETA {st['ETA']-ap['ETA']:+.1f}분 (안전+현실적 시간)")
else:
    print("  • A*도 경로 실패 (영역/지형 제약)")
