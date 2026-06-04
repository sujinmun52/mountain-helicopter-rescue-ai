"""
[Stage 2] AI 구조구역 선정 — MOCK / PLACEHOLDER 모듈
=====================================================

⚠️  이 모듈은 팀원이 개발 중인 ML 분류 코드가 통합되기 전까지 사용하는 임시 셸이다.
    파이프라인(Stage 1→4)이 끊기지 않도록, 실제 AI 출력 형식을 똑같이 흉내 내는
    더미(mock) 결과를 반환한다.

────────────────────────────────────────────────────────────────────────────
[코드 격리 원칙 — 나중에 실제 ML 로 교체하는 방법]
────────────────────────────────────────────────────────────────────────────
  • select_rescue_zone() 의 "입력 계약(IN)"과 "출력 계약(OUT)"은 고정이다.
  • 팀의 ML 추론 코드가 준비되면, 아래 표시된 [MOCK 로직 시작]~[MOCK 로직 끝]
    블록의 '내부'만 교체하면 된다. Stage 3/4 등 나머지 파이프라인은 절대 손대지 않는다.

  ── 입력 계약(IN) ──
    gps_coord : {"latitude": float, "longitude": float}   # 조난자 GPS (Stage 1)
    grid_data : {                                          # 500 m 반경 격자 묶음
        "radius_m":     float,                  # 추출 반경(m)
        "radius_mask":  np.ndarray(bool, 2D),   # 반경 내 셀 마스크
        "dem_lats":     np.ndarray(2D),         # 위도 격자
        "dem_lons":     np.ndarray(2D),         # 경도 격자
        "dem_array":    np.ndarray(2D),         # DEM 표고(m)
        "terrain":      dict,                   # slope/TRI/is_open/canopy_height 등
    }

  ── 출력 계약(OUT) : Stage 3(경로)·Stage 4(시각화)가 그대로 소비 ──
    {
        "row":       int,    "col":      int,      # DEM 격자 인덱스(목적지 노드)
        "latitude":  float,  "longitude": float,   # 목적지 좌표
        "distance_m": float,                        # 조난자까지 직선거리(m)
        "mode":      str,    "score":    float,     # 선정 모드/점수(ML 가상 출력)
        "is_mock":   bool,                          # True = 더미 결과(검증용 플래그)
    }
"""

import numpy as np
from modules.hoist import haversine


def select_rescue_zone(gps_coord, grid_data):
    """
    [Stage 2 셸] 500 m 반경 격자를 받아 '최적 착륙/호이스트 지점'(목적지 노드)을 반환.

    현재는 ML 미통합 상태이므로 더미 결과를 돌려준다(is_mock=True).
    실제 ML 추론으로 교체할 때는 아래 [MOCK 로직] 블록 내부만 바꾸면 된다.
    """
    victim_lat = gps_coord["latitude"]
    victim_lon = gps_coord["longitude"]

    dem_lats = grid_data["dem_lats"]
    dem_lons = grid_data["dem_lons"]
    terrain  = grid_data["terrain"]
    mask     = grid_data["radius_mask"]
    radius_m = grid_data.get("radius_m", 500.0)

    # ── 수신 로깅: 500 m 격자 데이터를 정상 수신했는지 확인 ──────────────────
    n_cells = int(np.count_nonzero(mask))
    print(f"[Stage 2/MOCK] 구조구역 선정 셸 호출")
    print(f"  • 수신: 조난자 GPS=({victim_lat:.5f}, {victim_lon:.5f}), 반경={radius_m:.0f}m")
    print(f"  • 수신: 반경 내 지형 격자 셀 {n_cells}개 (slope/TRI/임상 피처 포함)")
    if n_cells == 0:
        print("  • [경고] 반경 내 격자가 없어 조난자 위치를 목적지로 폴백합니다.")

    # ══════════════════════════════════════════════════════════════════════
    # [MOCK 로직 시작]  ⚠ 팀 ML 추론 코드 준비 시 이 블록 '내부'만 교체
    #   - 실제 ML: grid_data 의 피처(slope, TRI, tree_density/height, land_cover)를
    #     입력으로 4모드 분류·회귀를 수행해 최적 노드를 반환할 예정.
    #   - 현재 더미: 반경 내 '개활지 & 최소 경사' 셀 1개를 고르는 비-ML 단순 휴리스틱.
    #     (파이프라인 연결 검증용일 뿐, AI 판단이 아님)
    # ══════════════════════════════════════════════════════════════════════
    slope = terrain["slope"]
    is_open = terrain.get("is_open", np.ones_like(slope, dtype=bool))

    candidate_mask = mask & is_open
    if not np.any(candidate_mask):
        candidate_mask = mask if n_cells > 0 else np.ones_like(slope, dtype=bool)

    # 후보 중 경사 최소 셀 선택 (큰 값으로 마스킹 후 argmin)
    masked_slope = np.where(candidate_mask, slope, np.inf)
    r, c = np.unravel_index(np.argmin(masked_slope), masked_slope.shape)
    r, c = int(r), int(c)

    mock_mode  = "H_Light"   # 더미: 호이스트-소형 가정 (ML 이 실제 모드 결정 예정)
    mock_score = 0.5         # 더미: 중립 위험점수 (ML 이 실제 점수 산출 예정)
    # ══════════════════════════════════════════════════════════════════════
    # [MOCK 로직 끝]
    # ══════════════════════════════════════════════════════════════════════

    dest_lat = float(dem_lats[r, c])
    dest_lon = float(dem_lons[r, c])
    dist_m = float(haversine(victim_lat, victim_lon, dest_lat, dest_lon))

    destination = {
        "row": r, "col": c,
        "latitude": dest_lat, "longitude": dest_lon,
        "distance_m": dist_m,
        "mode": mock_mode, "score": mock_score,
        "is_mock": True,
    }
    print(f"  • 반환(더미 목적지 노드): ({dest_lat:.5f}, {dest_lon:.5f}) "
          f"grid=({r},{c}), 경사={slope[r, c]:.1f}°, 조난자까지 {dist_m:.0f}m")
    return destination
