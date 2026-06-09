"""
[Stage 2] XGBoost 기반 실제 구조구역 추론 엔진
=====================================================

rescue_zone.py(MOCK 셸)가 호출하는 '실 추론 백엔드'.

설계 배경 (Option A):
  • XGBoost 4모드 모델은 `terrain_base.parquet`의 **학습과 동일한 인코딩**
    (이산 등급 slope_deg/tree_*, land 원핫 land_0/1/2)으로 학습됐다.
  • 따라서 루트 파이프라인의 연속값 격자(grid_data)를 모델에 직접 넣지 않고,
    학습 때와 똑같은 parquet을 GPS+반경으로 필터해 추론한다(학습=추론 정합성).
  • 결과 좌표(lat/lon)는 호출측(rescue_zone.py)이 dem 격자 (row,col)로 역매핑한다.

가중치/피처 정의는 단일 출처(XGBoost_Model/preprocess/score_utils.py)를 재사용한다.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from dotenv import load_dotenv

# ── 경로 해석: modules/ → repo root → XGBoost_Model/... ─────────────────────
_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_MODULES_DIR)
_XGB_DIR     = os.path.join(_REPO_ROOT, "XGBoost_Model")
_PARQUET     = os.path.join(_XGB_DIR, "dataset", "terrain_base.parquet")
_MODELS_DIR  = os.path.join(_XGB_DIR, "models")

# 가중치/피처 단일 출처 재사용 (학습 코드와 동일 정의)
sys.path.insert(0, os.path.join(_XGB_DIR, "preprocess"))
from score_utils import (                       # noqa: E402
    FEATURE_COLUMNS, TACTIC_WEIGHTS,
    compute_wind_score, compute_wind_dir_score,
)

# 학습(2.xgb_train)에서 추가한 물리변수 상호작용 피처 — 모델 입력은 13개.
EXTENDED_FEATURES = FEATURE_COLUMNS + ["tree_risk", "aero_risk", "slope_wind_risk"]

# KMA API 키 로드 (.env → config → 환경변수 순 폴백)
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)
try:
    from config import KMA_API_KEY            # 루트 config.py
except Exception:
    KMA_API_KEY = os.getenv("KMA_API_KEY", "")

# 설악산 인근 기상청 AWS 관측소 (3.xgb_inference.py와 동일)
_SEORAK_STATIONS = {
    90:  {"name": "속초",   "lat": 38.2506, "lon": 128.5644},
    100: {"name": "대관령", "lat": 37.6764, "lon": 128.7183},
    105: {"name": "강릉",   "lat": 37.7514, "lon": 128.8908},
    211: {"name": "인제",   "lat": 38.0606, "lon": 128.1717},
    212: {"name": "홍천",   "lat": 37.6863, "lon": 127.8883},
}

# 무거운 자원(149만 행 parquet, 부스터)은 1회 로드 후 캐시
_MASTER_DF = None
_MODEL_CACHE = {}


# ══════════════════════════════════════════════════════════════════════════
# 자원 로더 (캐시)
# ══════════════════════════════════════════════════════════════════════════
def _load_master() -> pd.DataFrame:
    global _MASTER_DF
    if _MASTER_DF is None:
        _MASTER_DF = pd.read_parquet(_PARQUET)
    return _MASTER_DF


def _load_models(heli_size: str):
    """heli_size('small'/'large')의 landing/hoist 부스터 2개 로드(CPU 강제)."""
    if heli_size not in _MODEL_CACHE:
        models = {}
        for tactic in ("landing", "hoist"):
            path = os.path.join(_MODELS_DIR, f"xgb_{heli_size}_{tactic}_feature_model.ubj")
            booster = xgb.Booster()
            booster.load_model(path)
            booster.set_param({"device": "cpu"})   # GPU 의존 제거(파이프라인 안정성)
            models[tactic] = booster
        _MODEL_CACHE[heli_size] = models
    return _MODEL_CACHE[heli_size]


# ══════════════════════════════════════════════════════════════════════════
# 실시간 기상 수집 + 고도 보정 (학습/검증 스크립트와 동일 로직)
# ══════════════════════════════════════════════════════════════════════════
def _fetch_kma_realtime(target_time: str = None) -> dict:
    """기상청 API허브 지상 AWS 관측 호출 → {stn_id: {ws, wd, lat, lon}}."""
    import requests
    if target_time is not None:
        base_dt = datetime.strptime(target_time, "%Y%m%d%H%M")
        tm2 = base_dt.strftime("%Y%m%d%H%M")
        tm1 = (base_dt - timedelta(minutes=10)).strftime("%Y%m%d%H%M")
    else:
        now = datetime.now()
        tm2 = (now - timedelta(minutes=10)).strftime("%Y%m%d%H%M")
        tm1 = (now - timedelta(minutes=20)).strftime("%Y%m%d%H%M")

    base_url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
    wind_data = {}
    for stn_id, stn_info in _SEORAK_STATIONS.items():
        url = (f"{base_url}?tm1={tm1}&tm2={tm2}&stn={stn_id}"
               f"&disp=0&help=2&authKey={KMA_API_KEY}")
        try:
            res = requests.get(url, timeout=5)
            res.raise_for_status()
        except requests.exceptions.RequestException:
            continue
        try:
            raw_lines = [line.split() for line in res.text.splitlines()
                         if line.strip() and not line.startswith("#")]
            valid = None
            for row in reversed(raw_lines):
                try:
                    wd_c, ws_c = float(row[2]), float(row[3])
                except (IndexError, ValueError):
                    continue
                if ws_c < -50 or wd_c < -50 or ws_c > 100 or wd_c > 360:
                    continue
                valid = (ws_c, wd_c)
                break
            if valid is None:
                continue
            ws, wd = valid
            wind_data[str(stn_id)] = {"ws": ws, "wd": wd,
                                      "lat": stn_info["lat"], "lon": stn_info["lon"]}
        except (IndexError, ValueError):
            continue
    return wind_data


def _apply_elevation_wind_correction(grid_ws, dem, work_altitude=60.0):
    """대기 경계층 멱법칙(α=0.27): 지상 10m 풍속 → 헬기 운항 고도 풍속 보정."""
    alpha, z_ref = 0.27, 10.0
    return grid_ws * ((dem + work_altitude) / z_ref) ** alpha


def _fuse_realtime_wind(cand: pd.DataFrame) -> pd.DataFrame:
    """후보 격자에 실시간 풍속/풍향을 공간 보간·고도 보정해 주입.
    실패 시 보수적 고정 기상(6.5 m/s, 270°)으로 폴백(모델 폭사 방지)."""
    try:
        live = _fetch_kma_realtime()
        if not live or len(live) < 2:
            raise ValueError("유효 관측소 수 부족")

        pts = np.array([[v["lon"], v["lat"]] for v in live.values()])
        ws  = np.array([v["ws"] for v in live.values()])
        wd_rad = np.radians(np.array([v["wd"] for v in live.values()]))
        u, v = -np.sin(wd_rad), -np.cos(wd_rad)

        tgt = (cand["longitude"].values, cand["latitude"].values)
        g_ws = griddata(pts, ws, tgt, method="linear")
        g_u  = griddata(pts, u,  tgt, method="linear")
        g_v  = griddata(pts, v,  tgt, method="linear")
        nan = np.isnan(g_ws)
        if np.any(nan):
            g_ws[nan] = griddata(pts, ws, tgt, method="nearest")[nan]
            g_u[nan]  = griddata(pts, u,  tgt, method="nearest")[nan]
            g_v[nan]  = griddata(pts, v,  tgt, method="nearest")[nan]

        cand["wind_direction"] = np.degrees(np.arctan2(-g_u, -g_v)) % 360.0
        cand["wind_speed"] = _apply_elevation_wind_correction(g_ws, cand["elevation"].values)
        print("  • [Stage 2/ML] 실시간 AWS 기상 융합 완료(선형 보간 + 고도 멱법칙 보정)")
    except Exception as e:
        print(f"  • [Stage 2/ML] 기상 융합 실패({e}) → 고정 기상(6.5 m/s, 270°)으로 폴백")
        cand["wind_speed"] = 6.5
        cand["wind_direction"] = 270.0
    return cand


# ══════════════════════════════════════════════════════════════════════════
# 메인 추론 진입점
# ══════════════════════════════════════════════════════════════════════════
def _series_to_zone(s) -> dict:
    """best(pandas Series) → 표준 zone dict (IN/OUT 계약 고정)."""
    return {
        "latitude":   float(s["latitude"]),
        "longitude":  float(s["longitude"]),
        "mode":       str(s["mode"]),
        "score":      float(s["risk_score"]),
        "risk_class": int(s["risk_class"]),
        "wind_speed": float(s["wind_speed"]),
        "distance_m": float(s["dist_to_rescue_m"]),
    }


def _compute_best_per_tactic(gps_coord: dict, heli_size: str = "small",
                             radius_m: float = 500.0):
    """반경 내 후보로 landing/hoist 각 best(Series)를 산출. 후보 없으면 {} 반환."""
    if heli_size not in ("small", "large"):
        raise ValueError("heli_size must be 'small' or 'large'")

    v_lat, v_lon = gps_coord["latitude"], gps_coord["longitude"]
    master = _load_master()

    # ── 1) 반경 필터 (bbox 1차 차단 → 정밀 거리) ────────────────────────────
    deg = radius_m / 111_000.0
    box = master[
        (master["latitude"].between(v_lat - deg, v_lat + deg)) &
        (master["longitude"].between(v_lon - deg, v_lon + deg))
    ].copy()
    if box.empty:
        return {}

    # 위경도 근사 평면거리(m): 위도 1°≈111km, 경도는 cos(위도) 보정
    dlat = (box["latitude"] - v_lat) * 111_000.0
    dlon = (box["longitude"] - v_lon) * 111_000.0 * np.cos(np.radians(v_lat))
    box["dist_to_rescue_m"] = np.sqrt(dlat**2 + dlon**2)
    cand = box[box["dist_to_rescue_m"] <= radius_m].copy()

    # 하천(land_2) 안전 차단
    if "land_2" in cand.columns:
        cand = cand[cand["land_2"] != 1].copy()
    if cand.empty:
        return {}

    # ── 2) 실시간 기상 융합 + 모델 입력 피처 구성 ───────────────────────────
    cand = _fuse_realtime_wind(cand)
    wd_rad = np.radians(cand["wind_direction"].values)
    cand["wind_dir_sin"] = np.sin(wd_rad).astype("float32")
    cand["wind_dir_cos"] = np.cos(wd_rad).astype("float32")

    # 동적 점수(풍속/풍향) → 실제 Risk Score 합산용
    #   wind_score는 3.xgb_inference와 정합되도록 '기종별' 임계 적용
    #   (항공안전법 별표24 기반: 소형 10 / 대형 20 m/s 한계. 학습 타겟용
    #    score_utils.compute_wind_score(기종무관)와 달리 추론 점수에만 기종별 적용)
    _ws = cand["wind_speed"].values
    if heli_size == "small":
        cand["wind_score"] = np.select(
            [_ws < 5.0, (_ws >= 5.0) & (_ws < 8.0), (_ws >= 8.0) & (_ws < 10.0)],
            [1.0, 0.6, 0.2], default=0.0).astype("float32")
    else:  # large
        cand["wind_score"] = np.select(
            [_ws < 5.0, (_ws >= 5.0) & (_ws < 12.0), (_ws >= 12.0) & (_ws < 20.0)],
            [1.0, 0.6, 0.2], default=0.0).astype("float32")
    cand["wind_dir_score"] = compute_wind_dir_score(
        cand["latitude"].values, cand["longitude"].values, cand["wind_direction"].values)

    # 피처 엔지니어링 (학습(2.xgb_train)과 동일 — 물리변수 비선형 상호작용)
    #   aero_risk·slope_wind_risk에 wind_speed가 들어가 모델의 '풍속 반응'을 살림.
    cand["tree_risk"]       = (cand["tree_density"] * cand["tree_height"]).astype("float32")
    cand["aero_risk"]       = (cand["elevation"] * cand["wind_speed"]).astype("float32")
    cand["slope_wind_risk"] = (cand["slope_deg"] * cand["wind_speed"]).astype("float32")

    # ── 3) XGBoost 추론 (landing/hoist) + Risk Score 산출 ───────────────────
    models = _load_models(heli_size)

    best_per_tactic = {}
    for tactic in ("landing", "hoist"):
        key = f"{heli_size}_{tactic}"
        score_col = f"static_score_{key}"

        # [nan 방어] static_score(지형 기본점수)가 비어있는(nan) 후보 제외.
        #   terrain_base.parquet의 static_score는 일부(약 45%)가 nan이라,
        #   미처리 시 risk_score=nan이 best로 섞여 결과가 오염될 수 있음.
        if score_col in cand.columns:
            valid = cand[cand[score_col].notna()].copy()
        else:
            valid = cand.copy()
        if valid.empty:
            continue  # 이 전술은 유효(점수 있는) 착륙 후보 없음 → 스킵

        dmat = xgb.DMatrix(valid[EXTENDED_FEATURES].astype("float32"))
        pred = models[tactic].predict(dmat)
        risk_class = pred.argmax(axis=1) if pred.ndim == 2 else pred.astype(int)

        d = TACTIC_WEIGHTS[key]["dynamic"]
        risk_score = (valid[score_col].values
                      + valid["wind_score"].values * d["wind_score"]
                      + valid["wind_dir_score"].values * d["wind_dir_score"])

        sub = pd.DataFrame({
            "latitude": valid["latitude"].values,
            "longitude": valid["longitude"].values,
            "dist_to_rescue_m": valid["dist_to_rescue_m"].values,
            "wind_speed": valid["wind_speed"].values,
            "risk_class": risk_class,
            "risk_score": risk_score,
            "mode": key,
        })
        # 안전등급(오름차순) → Risk Score(내림차순) → 거리(오름차순)
        sub = sub.sort_values(["risk_class", "risk_score", "dist_to_rescue_m"],
                              ascending=[True, False, True])
        best_per_tactic[tactic] = sub.iloc[0]

    return best_per_tactic


def infer_best_zone(gps_coord: dict, heli_size: str = "small",
                    radius_m: float = 500.0) -> dict | None:
    """단일 best 구조 지점 (자동 파이프라인용).

    '착륙 우선' 규칙: landing이 가능(위험 등급2 아님)하면 착륙, 불가 시 호이스트.
    (실제 산악구조: 착륙 가능하면 빠른 착륙, 착륙 불가 험지에서만 호이스트)

    Returns: {latitude, longitude, mode, score, risk_class, wind_speed, distance_m} 또는 None.
    """
    bpt = _compute_best_per_tactic(gps_coord, heli_size, radius_m)
    if not bpt:
        return None
    land = bpt.get("landing")
    hoist = bpt.get("hoist")
    if land is not None and int(land["risk_class"]) < 2:
        best = land            # 착륙 가능 → 착륙 우선
    elif hoist is not None:
        best = hoist           # 착륙 불가 → 호이스트
    else:
        best = land
    return _series_to_zone(best)


def infer_tactics(gps_coord: dict, heli_size: str = "small",
                  radius_m: float = 500.0):
    """착륙(A안)·호이스트(B안) 둘 다의 best를 반환 — 3.xgb_inference 방식(기장 판단용).

    Returns: {"landing": zone|None, "hoist": zone|None} 또는 None(후보 전무).
    """
    bpt = _compute_best_per_tactic(gps_coord, heli_size, radius_m)
    if not bpt:
        return None
    return {t: _series_to_zone(s) for t, s in bpt.items()}
