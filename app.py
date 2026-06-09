# -*- coding: utf-8 -*-
"""
산악구조 최적화 AI 시스템 — Gradio 배포 앱.
조난자 GPS + 기종 → ML 최적 착륙지 + A* 안전 경로 + 3D 미션맵.

로컬 실행:  python app.py
배포:       Hugging Face Spaces (app.py + requirements.txt + 모델/데이터)
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import io
import html
import contextlib
import gradio as gr
import main as m
from modules.rescue_zone_inference import _load_master, infer_tactics

_GRADE = {0: "🟢 안전", 1: "🟡 주의", 2: "🔴 위험"}

_SYSTEM_DESC = """
### 🧠 이 시스템은 어떻게 동작하나요?

**입력** 조난자 GPS + 투입 기종(소형/대형) → **출력** 최적 구조점 + 안전 경로 + ETA

**① ML 위험 평가 (XGBoost 4모드)**
- `소형/대형 × 착륙/호이스트` 4개 전용 모델이 **149만 격자 후보**를 평가
- 13개 피처: 지형(경사·고도·임목밀도·수고·토지) + 실시간 기상(풍속·풍향) + 물리 상호작용(`aero_risk`·`slope_wind_risk`)
- 전술별 가중치가 다름 — **착륙=경사 중시**, **호이스트=임목(밀도·수고) 중시**
- 안전등급: 🟢안전(score≥0.80) · 🟡주의(≥0.55) · 🔴위험(<0.55)

**② 기상 반영**
- 기상청 AWS 실시간 풍속·풍향을 고도 보정(멱법칙)해 위험·경로에 반영
- 기종별 풍속 제한: 소형 10 / 대형 20 m/s (항공안전법 별표24)

**③ A\\* 경로**
- 비행(119→구조점): 순항고도 + 풍속·풍향·능선 회피
- 도보(구조점→조난자): Tobler 보행함수(경사) + 임상 저항

**④ 전술 선택**
- 착륙(A안)·호이스트(B안)를 **둘 다 추천** → 기장이 판단 (모델링 코드와 동일)
- 권장: 착륙 가능하면 착륙, 불가 시 호이스트
"""


def random_victim():
    """모델링 방식(3.xgb_inference)과 동일하게 마스터 데이터에서 실제 좌표 무작위 샘플."""
    s = _load_master().sample(1)
    return round(float(s["latitude"].values[0]), 5), round(float(s["longitude"].values[0]), 5)


def _card(name, z):
    if z is None:
        return f"- **{name}**: 적합 후보 없음"
    return (f"- **{name}** (`{z['mode']}`): {_GRADE.get(z['risk_class'], '—')} · "
            f"Risk {z['score']:.3f} · 조난자까지 {z['distance_m']:.0f}m · "
            f"({z['latitude']:.5f}, {z['longitude']:.5f})")


def run_rescue(lat, lon, heli_kor, tactic_kor):
    """조난자 GPS+기종 → 착륙/호이스트 둘 다 추천 + 선택 전술 경로·3D맵 + 처리 로그."""
    heli = {"소형 헬기": "light", "대형 헬기": "heavy"}.get(heli_kor, "heavy")
    ml_size = {"light": "small", "heavy": "large"}[heli]
    victim = {"latitude": float(lat), "longitude": float(lon)}
    buf = io.StringIO()
    report, fig, fol_path = "", None, None
    try:
        with contextlib.redirect_stdout(buf):   # 파이프라인 터미널 출력을 로그 창으로 캡처
            bundle = m.prepare_terrain_grid(victim)
            if bundle is None:
                report = "⚠️ 해당 좌표 주변 지형 데이터가 부족합니다."
            else:
                # 모델링 방식(3.xgb): 착륙·호이스트 둘 다 추천
                tactics = infer_tactics(victim, heli_size=ml_size, radius_m=m.RESCUE_RADIUS_M)
                if not tactics:
                    report = "⚠️ 반경 내 적합한 구조 지점이 없습니다."
                else:
                    land, hoist = tactics.get("landing"), tactics.get("hoist")
                    land_ok = land is not None and int(land["risk_class"]) < 2
                    recommend = "🛬 착륙 (가능)" if land_ok else "🪢 호이스트 (착륙 불가)"

                    sel = {"🛬 착륙": "landing", "🪢 호이스트": "hoist"}.get(tactic_kor, "landing")
                    chosen = tactics.get(sel) or land or hoist
                    r, c = m.latlon_to_grid(chosen["latitude"], chosen["longitude"],
                                            bundle["dem_lats"], bundle["dem_lons"])
                    dest = {**chosen, "row": r, "col": c}
                    route = m.stage3_path_modeling(victim, dest, bundle, heli_size=heli)

                    fig = m.create_3d_visualization(
                        bundle["dem_lats"], bundle["dem_lons"], bundle["dem_array"], bundle["terrain"],
                        route["full_path"], route["path_penalties"],
                        m.FIRE_STATION, dest, victim, flight_path=route.get("flight_path"))

                    # 2D Folium(실제 지도 위 착륙지·경로) — HTML 저장 후 iframe으로 표시
                    fol_path = m.create_helicopter_mission_folium_map(
                        route["full_path"], bundle["dem_lats"], bundle["dem_lons"], bundle["dem_array"],
                        route["path_penalties"], m.FIRE_STATION, dest, victim, bundle["terrain"],
                        flight_path=route.get("flight_path"),
                        flight_eta_min=route["flight_eta_min"], walk_eta_min=route["eta_min"],
                        transit_eta_min=route["transit_eta_min"])

                    # 착륙지·조난자 상세 (격자 지형 조회)
                    _tr, _da = bundle["terrain"], bundle["dem_array"]
                    _lr, _lc = dest["row"], dest["col"]
                    _vr, _vc = m.latlon_to_grid(victim["latitude"], victim["longitude"],
                                                bundle["dem_lats"], bundle["dem_lons"])
                    _land_open = "개활지" if _tr["is_open"][_lr, _lc] else "수림"
                    _vic_open = "개활지" if _tr["is_open"][_vr, _vc] else "수림(밀생)"

                    report = f"""## 🚁 구조 작전 추천 (기종: {heli_kor})

### AI 추천 (기장 판단용 — 착륙/호이스트 둘 다 제시)
{_card("🛬 착륙(A안)", land)}
{_card("🪢 호이스트(B안)", hoist)}

> **권장 전술: {recommend}** — 착륙 가능하면 착륙, 불가 시 호이스트

### 선택 전술({tactic_kor})로 경로 산출 → `{dest['mode']}`, {_GRADE.get(dest['risk_class'], '—')}
- 비행(119→구조점): **{route['flight_eta_min']:.1f}분**
- 도보(구조점→조난자): **{route['eta_min']:.1f}분**
- **합계: {route['transit_eta_min']:.1f}분**

### 📍 구조점(착륙지) 상세
- 좌표 ({dest['latitude']:.5f}, {dest['longitude']:.5f}) · 고도 **{_da[_lr, _lc]:.0f}m**
- 경사 **{_tr['slope'][_lr, _lc]:.0f}°** · 지형 **{_land_open}** · 풍속 **{dest['wind_speed']:.1f}m/s**
- AI 안전등급 {_GRADE.get(dest['risk_class'], '—')} · Risk {dest['score']:.3f}

### 🆘 조난자 위치
- GPS ({victim['latitude']:.5f}, {victim['longitude']:.5f}) · 고도 **{_da[_vr, _vc]:.0f}m**
- 경사 **{_tr['slope'][_vr, _vc]:.0f}°** · 지형 **{_vic_open}** · 구조점까지 {dest['distance_m']:.0f}m
"""
    except Exception as e:
        report = f"❌ 처리 중 오류: {e}"

    # Folium HTML → iframe(srcdoc)으로 감싸 gr.HTML에 안전하게 표시
    fol_html = ""
    if fol_path:
        try:
            with open(fol_path, encoding="utf-8") as f:
                raw = f.read()
            fol_html = (f'<iframe srcdoc="{html.escape(raw)}" '
                        f'style="width:100%;height:520px;border:none;"></iframe>')
        except Exception:
            fol_html = "<p>2D 지도 생성 실패</p>"

    return report, fig, (buf.getvalue() or "(로그 없음)"), fol_html


with gr.Blocks(title="산악구조 최적화 AI") as demo:
    gr.Markdown(
        "# 🏔️ 산악구조 최적화 AI 시스템\n"
        "조난자 GPS와 투입 기종을 입력하면, **ML이 최적 착륙지**를 선정하고 "
        "**A\\*가 안전 경로**를 산출합니다. (설악산 영역)")
    with gr.Accordion("ℹ️ 시스템 설명 (모델·기상·경로가 어떻게 동작하나)", open=False):
        gr.Markdown(_SYSTEM_DESC)
    with gr.Row():
        with gr.Column(scale=1):
            lat = gr.Number(label="조난자 위도 (38.00~38.25)", value=38.1192)
            lon = gr.Number(label="조난자 경도 (128.25~128.50)", value=128.4652)
            heli = gr.Radio(["소형 헬기", "대형 헬기"], value="대형 헬기", label="투입 기종")
            tactic = gr.Radio(["🛬 착륙", "🪢 호이스트"], value="🛬 착륙",
                              label="경로 산출 전술 (둘 다 추천되며, 경로는 선택한 전술로)")
            with gr.Row():
                btn_rand = gr.Button("🎲 랜덤 조난자 (실데이터)")
                btn = gr.Button("🚁 구조 작전 분석", variant="primary")
        with gr.Column(scale=2):
            out_md = gr.Markdown(label="작전 결과")
            with gr.Tabs():
                with gr.Tab("🗺️ 2D 미션맵 (실제 지도)"):
                    out_folium = gr.HTML(label="2D 미션맵")
                with gr.Tab("🏔️ 3D 지형뷰 (회전/확대)"):
                    out_plot = gr.Plot(label="3D 미션맵 (드래그=회전 · 스크롤=확대)")
    with gr.Accordion("🖥️ 처리 로그 (모델 추론·기상 수집·경로 탐색 과정)", open=False):
        out_log = gr.Textbox(label="파이프라인 로그", lines=16, max_lines=30,
                             interactive=False)
    btn_rand.click(random_victim, outputs=[lat, lon])
    btn.click(run_rescue, inputs=[lat, lon, heli, tactic],
              outputs=[out_md, out_plot, out_log, out_folium])
    gr.Examples(
        examples=[[38.1192, 128.4652, "대형 헬기", "🛬 착륙"],
                  [38.1389, 128.3079, "소형 헬기", "🪢 호이스트"]],
        inputs=[lat, lon, heli, tactic])


if __name__ == "__main__":
    demo.launch()
