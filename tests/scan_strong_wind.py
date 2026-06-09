"""[재학습 1단계] KMA 과거 관측에서 '강풍 시점'을 찾는다.
현 학습데이터(2025-10, 약풍 0~4.9 m/s)를 보강할 강풍 기간 후보 탐색.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.rescue_zone_inference import _fetch_kma_realtime, _SEORAK_STATIONS

# 강풍 후보 시기: 봄철 양간지풍(3~4월)·겨울 북서계절풍(12~2월) 위주
# (각 날짜의 오전9시/오후3시 = 일변화상 바람 강한 시간대)
CAND_DATES = []
for ym in ["202601", "202602", "202603", "202604", "202512", "202511"]:
    for dd in ["05", "12", "19", "26"]:
        for hhmm in ["0900", "1500"]:
            CAND_DATES.append(f"{ym}{dd}{hhmm}")

print(f"스캔 시점 {len(CAND_DATES)}개 (관측소 {len(_SEORAK_STATIONS)}개)\n")

results = []  # (max_ws, time, station_name, detail)
for t in CAND_DATES:
    try:
        live = _fetch_kma_realtime(target_time=t)
    except Exception as e:
        continue
    if not live:
        continue
    # 이 시점의 관측소별 풍속 중 최대
    best_ws, best_name = -1.0, "?"
    detail = []
    for stn_id, d in live.items():
        ws = d["ws"]
        name = _SEORAK_STATIONS.get(int(stn_id), {}).get("name", stn_id)
        detail.append(f"{name}={ws:.1f}")
        if ws > best_ws:
            best_ws, best_name = ws, name
    results.append((best_ws, t, best_name, " ".join(detail)))

# 최대 풍속 기준 내림차순
results.sort(reverse=True)
print("=== 강풍 시점 TOP 15 (최대풍속 내림차순) ===")
print(f"{'시각':<14}{'최대풍속':<10}{'관측소':<8}상세")
for ws, t, name, detail in results[:15]:
    ts = f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:{t[10:12]}"
    print(f"{ts:<16}{ws:>5.1f}m/s  {name:<8}{detail}")

if results:
    print(f"\n조회 성공 시점: {len(results)}개 / 최대풍속 전체: {results[0][0]:.1f} m/s")
else:
    print("\n[경고] 과거 관측 조회 실패 — API 보존기간/권한 확인 필요")
