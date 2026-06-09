"""
착륙지 위험도 스코어링 모듈 (Dual-Mode × Helicopter Size = 4 모드)
====================================================================

프로젝트 개요의 4모드 체계(L_Light, L_Heavy, H_Light, H_Heavy)에 대해
가중치 합(Weighted Sum) 기반 Risk Score 를 산출한다. 각 피처를 물리적 한계로
[0,1] '위험 기여도'로 정규화한 뒤 모드별 가중치를 곱해 합산하므로, 최종 점수도
자연스럽게 [0,1] 범위에 놓인다(가중치 합 = 1.0).

────────────────────────────────────────────────────────────────────────────
[가중치 재검토 및 재제안 — 항공 구조/기상 도메인 근거]
────────────────────────────────────────────────────────────────────────────
초기 가중치는 방향성(착륙=경사 절대, 호이스트=풍속 민감, 소형=풍 민감)이 대체로
타당했다. 다만 항공 산악구조 원칙에 비추어 두 가지 핵심 누락/왜곡을 보정했다.

  ① [신규 변수] altitude(밀도고도) 추가 — "High, Hot, Heavy" 원칙
     설악산은 대청봉 1,708 m 의 고지대다. 고도가 높고 기온이 높을수록 공기밀도가
     낮아져(밀도고도 상승) 로터 양력·호버 성능이 급감한다. 특히 지면효과 밖(OGE)
     호버가 필수인 호이스트와, 중량이 큰 대형기에서 치명적이다. 초기 표에는 이
     변수가 빠져 있었으므로 신규 추가하고, H_Heavy(0.18)·L_Heavy(0.12)에 큰 비중을 둔다.

  ② tree_height 비중 상향 (호이스트) — 수직 장애물 클리어런스
     호이스트는 기체가 착지하지 않고 호버하며 케이블을 내린다. 이때 결정적 제약은
     '평지 여부(경사)'가 아니라 '로터/케이블이 수목 상단을 안전하게 넘는가'이다.
     따라서 호이스트 모드에서 tree_height 를 0.10 → 0.18~0.20 으로 상향하고,
     경사도(slope)는 0.18 → 0.10~0.12 로 하향했다.

  그 외: 소형기가 대형기보다 풍 민감(저관성), 호버가 가장 풍속 민감이라는
  초기 설계의 방향성은 유지·정련했다.

  ── 재제안 가중치 표 (각 열 합 = 1.00) ──
   변수            L_Light  L_Heavy  H_Light  H_Heavy
   slope_deg        0.42     0.46     0.12     0.10
   tree_density     0.10     0.14     0.15     0.24
   tree_height      0.08     0.08     0.20     0.18
   wind_speed       0.20     0.12     0.30     0.18
   wind_dir         0.12     0.08     0.15     0.12
   altitude         0.08     0.12     0.08     0.18
   ─────────────────────────────────────────────────
   합계             1.00     1.00     1.00     1.00
"""

import numpy as np
import pandas as pd

from config import (
    SLOPE_RISK_REF, WS_RISK_REF, DENSITY_RISK_REF,
    CANOPY_RISK_REF, ELEV_BASE, ELEV_MAX,
)

# ── 재제안 모드별 가중치 (열 합 = 1.0) ──────────────────────────────────────
REVISED_WEIGHTS = {
    #              slope  density  height  w_speed  w_dir  altitude
    "L_Light": {"slope_deg": 0.42, "tree_density": 0.10, "tree_height": 0.08,
                "wind_speed": 0.20, "wind_dir": 0.12, "altitude": 0.08},
    "L_Heavy": {"slope_deg": 0.46, "tree_density": 0.14, "tree_height": 0.08,
                "wind_speed": 0.12, "wind_dir": 0.08, "altitude": 0.12},
    "H_Light": {"slope_deg": 0.12, "tree_density": 0.15, "tree_height": 0.20,
                "wind_speed": 0.30, "wind_dir": 0.15, "altitude": 0.08},
    "H_Heavy": {"slope_deg": 0.10, "tree_density": 0.24, "tree_height": 0.18,
                "wind_speed": 0.18, "wind_dir": 0.12, "altitude": 0.18},
}


def _assert_weights_normalized(weights=REVISED_WEIGHTS, tol=1e-9):
    """각 모드 가중치 합이 1.0 인지 검증 (설정 오류 조기 발견)."""
    for mode, w in weights.items():
        total = sum(w.values())
        if abs(total - 1.0) > tol:
            raise ValueError(f"가중치 합 오류: {mode} = {total:.4f} (1.0 이어야 함)")


# ──────────────────────────────────────────────────────────────────────────
# 피처별 정규화: 각 변수를 [0,1] 위험 기여도로 변환 (1 = 가장 위험)
# ──────────────────────────────────────────────────────────────────────────
def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def normalize_slope(slope_deg):
    """경사도 위험: 기준각(SLOPE_RISK_REF)에서 포화. 착륙 가능 한계각 부근에서 1."""
    return _clip01(np.asarray(slope_deg, dtype=float) / SLOPE_RISK_REF)


def normalize_wind_speed(wind_speed):
    """풍속 위험: 산악 호이스트 상한(WS_RISK_REF)에서 포화."""
    return _clip01(np.asarray(wind_speed, dtype=float) / WS_RISK_REF)


def normalize_tree_density(tree_density):
    """임목 밀도 위험: 데이터 스케일(DENSITY_RISK_REF) 기준 [0,1]."""
    return _clip01(np.asarray(tree_density, dtype=float) / DENSITY_RISK_REF)


def normalize_tree_height(tree_height):
    """수고 위험: 호이스트 수직 클리어런스 기준(CANOPY_RISK_REF)에서 포화."""
    return _clip01(np.asarray(tree_height, dtype=float) / CANOPY_RISK_REF)


def normalize_altitude(elevation, temperature_c=None):
    """
    밀도고도(Density Altitude) 위험 정규화.

    기온이 주어지면 근사 밀도고도 = 기하고도 + 120·(OAT - ISA온도) 로 환산해
    더위·고도 복합 효과를 반영(High-Hot). 기온이 없으면 기하고도만 사용.
    """
    elev = np.asarray(elevation, dtype=float)
    if temperature_c is not None:
        oat = np.asarray(temperature_c, dtype=float)
        isa_temp = 15.0 - 0.0065 * elev          # 표준대기 기온(°C)
        density_alt = elev + 120.0 * (oat - isa_temp)
    else:
        density_alt = elev
    return _clip01((density_alt - ELEV_BASE) / (ELEV_MAX - ELEV_BASE))


def normalize_wind_dir(wind_dir_deg, aspect_deg=None):
    """
    풍향 위험 정규화 — 원형(circular) 데이터 타입 핸들링의 핵심.

    ⚠ 방위각(0~360°)을 절대 min-max/선형 정규화하면 안 된다. 359°와 1°는
       물리적으로 인접하지만 선형 스케일에서는 양 극단이 된다.

    해결: 풍향 자체가 아니라 '지형 대비 상대 위험'으로 변환한다.
      - 사면 방향(aspect)이 주어지면: 바람이 능선/사면을 가로지르는(cross-slope)
        성분이 클수록 풍하측 와류·난류가 강해져 위험. 위험 = |sin(Δ)|,
        Δ = 풍향과 사면방향의 각도차. (정풍/배풍 0, 측풍 1)
      - aspect 가 없으면 풍향 단독으로는 위험을 정의할 수 없으므로 0(중립) 반환.
        (풍속 항이 강도를 이미 담당)
    """
    wd = np.asarray(wind_dir_deg, dtype=float)
    if aspect_deg is None:
        return np.zeros_like(wd)
    asp = np.asarray(aspect_deg, dtype=float)
    delta = np.radians(wd - asp)
    return _clip01(np.abs(np.sin(delta)))


# ──────────────────────────────────────────────────────────────────────────
# 4모드 Risk Score 산출
# ──────────────────────────────────────────────────────────────────────────
def compute_risk_scores(
    df,
    weights=REVISED_WEIGHTS,
    col_slope="slope_deg",
    col_density="tree_density",
    col_height="tree_height",
    col_wind_speed="wind_speed",
    col_wind_dir="wind_direction",
    col_elev="elevation",
    col_aspect="aspect",
    col_temp=None,
    scale_0_100=False,
):
    """
    착륙지 후보 DataFrame 에 4개 모드 Risk Score 컬럼을 추가한다.

    파이프라인:
      1. 각 피처를 [0,1] 위험 기여도로 정규화 (풍향은 지형 상대 위험으로 변환)
      2. 모드별 가중치 가중합 → risk_<mode> 컬럼 생성
      3. (옵션) 0~100 스케일링

    Returns:
        risk_L_Light, risk_L_Heavy, risk_H_Light, risk_H_Heavy 컬럼이 추가된 복사본 df.
    """
    _assert_weights_normalized(weights)
    out = df.copy()

    aspect = out[col_aspect].values if col_aspect in out.columns else None
    temp   = out[col_temp].values if (col_temp and col_temp in out.columns) else None

    # 1. 정규화된 위험 기여도 (결측은 0.5=중립 위험으로 보수적 대체)
    risk = {
        "slope_deg":    normalize_slope(out[col_slope].values),
        "tree_density": normalize_tree_density(out[col_density].values),
        "tree_height":  normalize_tree_height(out[col_height].values),
        "wind_speed":   normalize_wind_speed(out[col_wind_speed].values),
        "wind_dir":     normalize_wind_dir(out[col_wind_dir].values, aspect),
        "altitude":     normalize_altitude(out[col_elev].values, temp),
    }
    for k in risk:
        risk[k] = np.nan_to_num(risk[k], nan=0.5)

    # 2. 모드별 가중합
    for mode, w in weights.items():
        score = np.zeros(len(out))
        for feat, weight in w.items():
            score = score + weight * risk[feat]
        out[f"risk_{mode}"] = score * 100.0 if scale_0_100 else score

    return out


# ──────────────────────────────────────────────────────────────────────────
# PDF 표 기반 계단식(Step) Risk 함수 — A* 셀별 비용 산출용
#   기존 normalize_* (선형) 은 배치/DataFrame 점수화용으로 보존하고,
#   여기서는 PDF "도메인 점수화 - 이거 보세요" 표를 셀단위 계단식으로 코드화한다.
#   mode: "landing" | "hoist", size: "light" | "heavy"
# ──────────────────────────────────────────────────────────────────────────
PDF_BLOCK_THRESHOLD = 1.0  # 어느 변수든 1.0 이상이면 운용 불가 셀


def step_slope_risk(slope_deg: float, mode: str) -> float:
    if mode == "landing":
        if slope_deg < 7:  return 0.00
        if slope_deg < 15: return 0.60
        return 1.00
    # hoist
    if slope_deg < 7:  return 0.00
    if slope_deg < 15: return 0.15
    if slope_deg < 25: return 0.30
    if slope_deg < 40: return 0.70
    return 1.00


def step_density_risk(density_0to1: float, size: str) -> float:
    """density_0to1: 임관피도 비율 (0.0~1.0). PDF 의 % 기준 매핑."""
    pct = max(0.0, min(1.0, float(density_0to1))) * 100.0
    if size == "light":
        if pct < 10: return 0.00
        if pct < 30: return 0.10
        if pct < 70: return 0.60
        return 1.00
    if pct < 10: return 0.00
    if pct < 30: return 0.30
    if pct < 70: return 0.85
    return 1.00


def step_height_risk(canopy_m: float, size: str) -> float:
    """PDF 0~2m 의 소형 1.00 표기는 설명('장애물 없음') 과 모순 → 0.00 으로 보정."""
    if canopy_m < 2:
        return 0.00
    if size == "light":
        if canopy_m < 8:  return 0.30
        if canopy_m < 17: return 0.60
        return 0.85
    if canopy_m < 8:  return 0.45
    if canopy_m < 17: return 0.75
    return 0.95


def step_wind_speed_risk(ws_mps: float, size: str) -> float:
    if ws_mps < 5:
        return 0.00
    if size == "light":
        if ws_mps < 10: return 0.40
        if ws_mps < 15: return 0.80
        return 1.00
    if ws_mps < 10: return 0.20
    if ws_mps < 15: return 0.50
    return 0.90


def step_wind_dir_risk(angle_offset_deg: float, size: str) -> float:
    """angle_offset_deg: 헬기 진행방향 대비 풍향 각도 (0=정풍, 180=배풍)."""
    a = abs(angle_offset_deg) % 360.0
    if a > 180:
        a = 360.0 - a
    if a <= 45:
        return 0.00  # 정풍
    if a < 135:
        return 0.70 if size == "light" else 0.50  # 측풍
    return 0.95 if size == "light" else 0.885  # 배풍


def step_altitude_risk(elev_m: float) -> float:
    """밀도고도 위험 (선형 근사: ELEV_BASE 이하 0, ELEV_MAX 이상 1)."""
    if elev_m <= ELEV_BASE:
        return 0.0
    if elev_m >= ELEV_MAX:
        return 1.0
    return float((elev_m - ELEV_BASE) / (ELEV_MAX - ELEV_BASE))


# mode/size 짧은 이름 → 가중치 키 매핑
_MODE_KEY = {("landing", "light"): "L_Light", ("landing", "heavy"): "L_Heavy",
             ("hoist",   "light"): "H_Light", ("hoist",   "heavy"): "H_Heavy"}


def cell_risk_step(*, slope_deg, density_0to1, canopy_m, ws_mps,
                   wd_offset_deg=None, elev_m=0.0,
                   mode="hoist", size="heavy"):
    """
    A* 셀별 종합 risk 산출 (PDF 계단식).
    Returns:
        (risk: float in [0,1], blocked: bool)
    blocked: 경사/임관/풍속 중 하나라도 1.0 이상이면 True.
    """
    w = REVISED_WEIGHTS[_MODE_KEY[(mode, size)]]
    rs = step_slope_risk(slope_deg, mode)
    rd = step_density_risk(density_0to1, size)
    rh = step_height_risk(canopy_m, size)
    rws = step_wind_speed_risk(ws_mps, size)
    ralt = step_altitude_risk(elev_m)
    blocked = (rs >= PDF_BLOCK_THRESHOLD or rd >= PDF_BLOCK_THRESHOLD
               or rws >= PDF_BLOCK_THRESHOLD)
    if wd_offset_deg is not None:
        rwd = step_wind_dir_risk(wd_offset_deg, size)
        score = (w["slope_deg"]*rs + w["tree_density"]*rd + w["tree_height"]*rh
                 + w["wind_speed"]*rws + w["wind_dir"]*rwd + w["altitude"]*ralt)
    else:
        denom = 1.0 - w["wind_dir"]
        score = (w["slope_deg"]*rs + w["tree_density"]*rd + w["tree_height"]*rh
                 + w["wind_speed"]*rws + w["altitude"]*ralt) / denom
    return float(min(1.0, max(0.0, score))), bool(blocked)


def build_risk_lookup(df, mode, id_cols=("row", "col")):
    """
    A* 배치 추론용 Lookup Table 생성 (프로젝트 개요의 'Batch Inference → O(1) 조회').

    A* 탐색 직전에 모든 노드의 Risk Score 를 한 번에 계산해 dict 로 저장해 두면,
    탐색 루프에서는 모델/수식을 재호출하지 않고 O(1) 로 위험도를 조회할 수 있다.

    Args:
        df:     compute_risk_scores() 결과 DataFrame
        mode:   "L_Light" | "L_Heavy" | "H_Light" | "H_Heavy"
        id_cols: 노드 식별 키 컬럼들 (예: (row, col) 격자 인덱스)

    Returns:
        {(id...): risk_score, ...} 딕셔너리
    """
    col = f"risk_{mode}"
    if col not in df.columns:
        raise KeyError(f"{col} 컬럼이 없습니다. 먼저 compute_risk_scores() 를 실행하세요.")
    keys = list(zip(*[df[c].values for c in id_cols]))
    return dict(zip(keys, df[col].values))
