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
from modules.rescue_zone_inference import _load_master, infer_tactics_ranked

_GRADE = {0: "🟢 안전", 1: "🟡 주의", 2: "🔴 위험"}

# 3D 위에 띄우는 '고정·드래그' 정보 패널을 옮길 수 있게 하는 스크립트(헤드 주입).
#   .drag-panel 요소를 마우스로 드래그 → 스크롤과 무관하게 위치 고정/이동.
_DRAG_JS = """
<script>
(function () {
  var cur=null, ox=0, oy=0, sl=0, st=0, moved=false;
  document.addEventListener('mousedown', function (e) {
    // summary(펼침)·버튼·링크 클릭은 드래그로 가로채지 않음
    if (e.target.closest('summary, button, a, input')) return;
    var p = e.target.closest && e.target.closest('.drag-panel');
    if (!p) return;
    cur=p; ox=e.clientX; oy=e.clientY; moved=false;
    var pr=(p.offsetParent||document.body).getBoundingClientRect();
    var r=p.getBoundingClientRect();
    sl=r.left-pr.left; st=r.top-pr.top;
    p.style.right='auto'; p.style.left=sl+'px'; p.style.top=st+'px';
  });
  document.addEventListener('mousemove', function (e) {
    if (!cur) return;
    moved=true;
    cur.style.left=(sl+e.clientX-ox)+'px';
    cur.style.top=(st+e.clientY-oy)+'px';
    e.preventDefault();
  });
  document.addEventListener('mouseup', function () { cur=null; });
})();
</script>
"""


def _rank_inline(zones):
    """순위 목록을 '1순위 215m · 2순위 246m …' 한 줄로."""
    if not zones:
        return "<span style='color:#999'>후보 없음</span>"
    return " · ".join(f"{z.get('rank', i)}순위 <b>{z['distance_m']:.0f}m</b>"
                      for i, z in enumerate(zones, 1))


def _why_html(sel_list, sel_tactic_kor):
    """2·3순위가 '왜' 그 순위인지 — 1순위 대비 차이를 펼침(details)으로."""
    if not sel_list or len(sel_list) < 2:
        return ""
    top = sel_list[0]
    rows = []
    for z in sel_list[1:]:
        ds, dd = z["score"] - top["score"], z["distance_m"] - top["distance_m"]
        rows.append(
            f"<div style='margin-top:3px'><b>{z.get('rank')}순위</b> "
            f"{_GRADE.get(z['risk_class'], '—')} · 적합도 {z['score']:.3f} "
            f"<span style='color:#999'>({ds:+.3f})</span> · "
            f"거리 {z['distance_m']:.0f}m <span style='color:#999'>({dd:+.0f}m)</span> · "
            f"풍속 {z['wind_speed']:.1f}m/s</div>")
    return (
        "<details style='margin-top:5px'>"
        "<summary style='cursor:pointer; color:#7c3aed; font-weight:600'>▸ 2·3순위인 이유</summary>"
        "<div style='margin-top:4px; padding:6px 8px; background:#f6f1ff; border-radius:6px; "
        "font-size:11px; line-height:1.5'>"
        "정렬 기준: <b>안전등급 → AI 안전점수 → 거리</b>. 1순위와 50m+ 떨어진 "
        f"<b>대체 {('호이스트' if 'hoist' in str(top.get('mode','')) else '착륙')}</b> 지점입니다.<br>"
        f"<span style='color:#666'>(괄호=1순위 대비 차이)</span>"
        + "".join(rows) + "</div></details>")


def _info_panel_html(land_list, hoist_list, victim, sel_list, sel_tactic_kor):
    """3D 위에 고정·드래그로 띄우는 '조난자까지 거리 + 2·3순위 이유' 패널 HTML."""
    return (
        "<div class='drag-panel' style='position:absolute; top:8px; right:8px; "
        "z-index:60; cursor:move; user-select:none; background:rgba(255,255,255,0.96); "
        "border:1px solid #7c3aed; border-radius:8px; padding:8px 12px; "
        "font-size:12px; line-height:1.6; max-width:320px; "
        "box-shadow:0 2px 10px rgba(0,0,0,0.25);'>"
        "<b style='color:#7c3aed'>🚁 구조 후보 · 조난자까지 거리</b> "
        "<span style='color:#999; font-size:10px'>(드래그해서 이동)</span><br>"
        f"🛬 <b>착륙</b> {_rank_inline(land_list)}<br>"
        f"🪢 <b>호이스트</b> {_rank_inline(hoist_list)}<br>"
        f"<span style='color:#666'>🆘 조난자 GPS "
        f"({victim['latitude']:.4f}, {victim['longitude']:.4f}) · 경로 전술: {sel_tactic_kor}</span>"
        + _why_html(sel_list, sel_tactic_kor) + "</div>")

_SYSTEM_DESC = """
### 🧠 이 시스템은 어떻게 동작하나요?

**입력** 조난자 GPS + 투입 기종(소형/대형) → **출력** 최적 구조점 + 안전 경로 + ETA

**① ML 위험 평가 (XGBoost 4모드)**
- `소형/대형 × 착륙/호이스트` 4개 전용 모델이 **149만 격자 후보**를 평가
- 13개 피처: 지형(경사·고도·임목밀도·수고·토지) + 실시간 기상(풍속·풍향) + 물리 상호작용(`aero_risk`·`slope_wind_risk`)
- 전술별 가중치가 다름 — **착륙=경사 중시**, **호이스트=임목(밀도·수고) 중시**
- 안전등급(risk scoring): 🟢안전(적합도≥0.80) · 🟡주의(≥0.55) · 🔴위험(<0.55)

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


def _rank_lines(name, zones):
    """전술별 1·2·3순위 목록을 마크다운 라인들로 (모델링 3.xgb의 '최대 3순위' 출력 반영)."""
    if not zones:
        return f"- **{name}**: 적합 후보 없음"
    head = zones[0]
    lines = [f"- **{name}** (`{head['mode']}`)"]
    for z in zones:
        lines.append(
            f"    - **{z.get('rank', 1)}순위** {_GRADE.get(z['risk_class'], '—')} · "
            f"적합도 {z['score']:.3f} · 조난자까지 {z['distance_m']:.0f}m · "
            f"({z['latitude']:.5f}, {z['longitude']:.5f})")
    return "\n".join(lines)


def run_rescue(lat, lon, heli_kor, tactic_kor):
    """조난자 GPS+기종 → 착륙/호이스트 둘 다 추천 + 선택 전술 경로·3D맵 + 처리 로그."""
    heli = {"소형 헬기": "light", "대형 헬기": "heavy"}.get(heli_kor, "heavy")
    ml_size = {"light": "small", "heavy": "large"}[heli]
    victim = {"latitude": float(lat), "longitude": float(lon)}
    buf = io.StringIO()
    report, fig, fol_path, info_html = "", None, None, ""
    try:
        with contextlib.redirect_stdout(buf):   # 파이프라인 터미널 출력을 로그 창으로 캡처
            bundle = m.prepare_terrain_grid(victim)
            if bundle is None:
                report = "⚠️ 해당 좌표 주변 지형 데이터가 부족합니다."
            else:
                # 모델링 방식(3.xgb): 착륙·호이스트 각 1·2·3순위 추천
                tactics = infer_tactics_ranked(victim, heli_size=ml_size,
                                               radius_m=m.RESCUE_RADIUS_M, top_n=3)
                if not tactics:
                    report = "⚠️ 반경 내 적합한 구조 지점이 없습니다."
                else:
                    land_list = tactics.get("landing") or []
                    hoist_list = tactics.get("hoist") or []
                    land = land_list[0] if land_list else None
                    hoist = hoist_list[0] if hoist_list else None
                    land_ok = land is not None and int(land["risk_class"]) < 2
                    recommend = "🛬 착륙 (가능)" if land_ok else "🪢 호이스트 (착륙 불가)"

                    sel = {"🛬 착륙": "landing", "🪢 호이스트": "hoist"}.get(tactic_kor, "landing")
                    sel_list = tactics.get(sel) or land_list or hoist_list
                    chosen = sel_list[0]
                    r, c = m.latlon_to_grid(chosen["latitude"], chosen["longitude"],
                                            bundle["dem_lats"], bundle["dem_lons"])
                    dest = {**chosen, "row": r, "col": c}

                    # 선택 전술의 2·3순위 = 지도 토글용 대체 후보(격자 좌표 부여)
                    alt_points = []
                    for z in sel_list[1:]:
                        ar, ac = m.latlon_to_grid(z["latitude"], z["longitude"],
                                                  bundle["dem_lats"], bundle["dem_lons"])
                        alt_points.append({**z, "row": ar, "col": ac})

                    route = m.stage3_path_modeling(victim, dest, bundle, heli_size=heli)

                    fig = m.create_3d_visualization(
                        bundle["dem_lats"], bundle["dem_lons"], bundle["dem_array"], bundle["terrain"],
                        route["full_path"], route["path_penalties"],
                        m.FIRE_STATION, dest, victim, flight_path=route.get("flight_path"),
                        alt_points=alt_points)

                    # 2D Folium(실제 지도 위 착륙지·경로) — HTML 저장 후 iframe으로 표시
                    fol_path = m.create_helicopter_mission_folium_map(
                        route["full_path"], bundle["dem_lats"], bundle["dem_lons"], bundle["dem_array"],
                        route["path_penalties"], m.FIRE_STATION, dest, victim, bundle["terrain"],
                        flight_path=route.get("flight_path"),
                        flight_eta_min=route["flight_eta_min"], walk_eta_min=route["eta_min"],
                        transit_eta_min=route["transit_eta_min"], alt_points=alt_points)

                    # 3D 위 고정·드래그 정보 패널(거리 1·2·3순위 + 선택 전술의 '이유')
                    info_html = _info_panel_html(land_list, hoist_list, victim,
                                                 sel_list, tactic_kor)

                    # 착륙지·조난자 상세 (격자 지형 조회)
                    _tr, _da = bundle["terrain"], bundle["dem_array"]
                    _lr, _lc = dest["row"], dest["col"]
                    _vr, _vc = m.latlon_to_grid(victim["latitude"], victim["longitude"],
                                                bundle["dem_lats"], bundle["dem_lons"])
                    _land_open = "개활지" if _tr["is_open"][_lr, _lc] else "수림"
                    _vic_open = "개활지" if _tr["is_open"][_vr, _vc] else "수림(밀생)"

                    report = f"""## 🚁 구조 작전 추천 (기종: {heli_kor})

### AI 추천 (기장 판단용 — 착륙/호이스트 각 최대 3순위)
{_rank_lines("🛬 착륙(A안)", land_list)}
{_rank_lines("🪢 호이스트(B안)", hoist_list)}

> **권장 전술: {recommend}** — 착륙 가능하면 착륙, 불가 시 호이스트
> 지도의 **2·3순위 대체 후보**는 토글(2D: 우상단 레이어 / 3D: 상단 버튼)로 표시됩니다.

### 선택 전술({tactic_kor})로 경로 산출 → `{dest['mode']}`, {_GRADE.get(dest['risk_class'], '—')}
- 비행(119→구조점): **{route['flight_eta_min']:.1f}분**
- 도보(구조점→조난자): **{route['eta_min']:.1f}분**
- **합계: {route['transit_eta_min']:.1f}분**

### 📍 구조점(착륙지) 상세
- 좌표 ({dest['latitude']:.5f}, {dest['longitude']:.5f}) · 고도 **{_da[_lr, _lc]:.0f}m**
- 경사 **{_tr['slope'][_lr, _lc]:.0f}°** · 지형 **{_land_open}** · 풍속 **{dest['wind_speed']:.1f}m/s**
- AI 안전등급 {_GRADE.get(dest['risk_class'], '—')} · 적합도 {dest['score']:.3f} <sub>(높을수록 적합)</sub>

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

    return report, fig, (buf.getvalue() or "(로그 없음)"), fol_html, info_html


# 3D 토글은 '클라이언트 JS'로 처리 — 서버 왕복/재렌더 없이 Plotly를 직접 조작해
#   사용자가 돌려둔 카메라(시점)를 초기화하지 않는다.
_ALT_TOGGLE_JS = """
() => {
  const box = document.querySelector('#plot3d-box .js-plotly-plot');
  if (!box || !box.data || !window.Plotly) return;
  let idx = -1;
  for (let i = 0; i < box.data.length; i++) {
    const nm = box.data[i].name || '';
    if (nm.indexOf('대체') >= 0) { idx = i; break; }
  }
  if (idx < 0) return;
  const cur = box.data[idx].visible;
  const next = (cur === false) ? true : false;
  window.Plotly.restyle(box, {visible: next}, [idx]);
  // 표시(ON)=채워진 색, 숨김(OFF)=연한 색 — 누르면 색이 바뀐다
  const btn = document.querySelector('#btn-alt3d button');
  if (btn) btn.classList.toggle('btn-off', next === false);
}
"""

_DRAG_TOGGLE_JS = """
() => {
  const box = document.querySelector('#plot3d-box .js-plotly-plot');
  if (!box || !window.Plotly) return;
  const sc = (box.layout && box.layout.scene) || {};
  const cur = sc.dragmode || 'turntable';
  const next = (cur === 'pan') ? 'turntable' : 'pan';
  window.Plotly.relayout(box, {'scene.dragmode': next});
  const btn = document.querySelector('#btn-drag3d button');
  if (btn) {
    btn.textContent = (next === 'pan')
        ? '🔄 드래그=회전으로 전환'
        : '🖐 드래그=지도 이동으로 전환';
    // 지도이동(pan, ON)=채워진 색, 회전(OFF)=연한 색
    btn.classList.toggle('btn-on', next === 'pan');
  }
}
"""

# 새 분석 실행 후 버튼 상태를 기본값으로 동기화(2·3순위=표시 ON, 드래그=회전 OFF).
_RESET_BTN_JS = """
() => {
  const a = document.querySelector('#btn-alt3d button');
  if (a) a.classList.remove('btn-off');
  const d = document.querySelector('#btn-drag3d button');
  if (d) { d.classList.remove('btn-on'); d.textContent = '🖐 드래그=지도 이동으로 전환'; }
}
"""


# 3D 지도를 스크롤 가능한 박스에 — 내부 plot을 컨테이너보다 크게 강제해
#   하단(가로)·우측(세로) 스크롤바가 생기게 한다(잘림 없이 전체 탐색).
_CSS = """
/* 3D 탭: 정보 패널(absolute)이 이 영역 기준으로 떠 있도록 relative */
#plot3d-tab { position: relative; }
/* 패널 호스트는 자리(높이)를 차지하지 않게 — 절대배치 패널만 위에 떠 있음 */
#info3d-host { height: 0; min-height: 0; padding: 0; margin: 0; overflow: visible; }
#plot3d-box { overflow: auto !important; max-height: 600px;
              border: 1px solid #ddd; border-radius: 6px; }
#plot3d-box .js-plotly-plot, #plot3d-box .plot-container {
              min-width: 1100px !important; min-height: 780px !important; }
/* 토글 버튼: 누르면 색이 바뀜(활성=채워진 색 / 비활성=연한 색) */
#btn-alt3d button, #btn-drag3d button {
              font-weight:700 !important; border-radius:8px !important; }
/* 2·3순위 토글 — 기본=표시(ON, 채워진 보라). .btn-off=숨김(연한 보라) */
#btn-alt3d button { background:#7c3aed !important; border:2px solid #7c3aed !important;
              color:#fff !important; }
#btn-alt3d button.btn-off { background:#f3e8ff !important; color:#5b21b6 !important; }
/* 드래그 모드 — 기본=회전(OFF, 연한 청록). .btn-on=지도이동(채워진 청록) */
#btn-drag3d button { background:#e0f2fe !important; border:2px solid #0284c7 !important;
              color:#075985 !important; }
#btn-drag3d button.btn-on { background:#0284c7 !important; color:#fff !important; }
"""


with gr.Blocks(title="산악구조 최적화 AI", css=_CSS, head=_DRAG_JS) as demo:
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
                    gr.Markdown("기본 드래그=회전 · 스크롤=확대 · 박스 하단/우측 바=영역 이동 · "
                                "**지도 이동** 버튼으로 2D처럼 끌어 이동")
                    # 토글 버튼들은 스크롤 박스 '바깥' → 영역 스크롤 중에도 화면에 고정
                    with gr.Row():
                        btn_alt = gr.Button("🔶 2·3순위 대체후보 숨김 / 표시",
                                            elem_id="btn-alt3d", variant="secondary")
                        btn_drag = gr.Button("🖐 드래그=지도 이동으로 전환",
                                             elem_id="btn-drag3d", variant="secondary")
                    with gr.Column(elem_id="plot3d-tab"):
                        # 고정·드래그 정보 패널(거리) — 스크롤 박스 밖이라 스크롤해도 안 밀림
                        out_info3d = gr.HTML(elem_id="info3d-host")
                        out_plot = gr.Plot(elem_id="plot3d-box",
                                           label="3D 미션맵 (드래그=회전 · 스크롤=확대)")
    with gr.Accordion("🖥️ 처리 로그 (모델 추론·기상 수집·경로 탐색 과정)", open=False):
        out_log = gr.Textbox(label="파이프라인 로그", lines=16, max_lines=30,
                             interactive=False)
    btn_rand.click(random_victim, outputs=[lat, lon])
    btn.click(run_rescue, inputs=[lat, lon, heli, tactic],
              outputs=[out_md, out_plot, out_log, out_folium, out_info3d]
              ).then(None, None, None, js=_RESET_BTN_JS)   # 실행 후 버튼 상태 기본값 동기화
    # 토글은 클라이언트 JS만 실행 → 서버 재렌더 없음 → 카메라(시점) 유지
    btn_alt.click(None, None, None, js=_ALT_TOGGLE_JS)
    btn_drag.click(None, None, None, js=_DRAG_TOGGLE_JS)
    gr.Examples(
        examples=[[38.1192, 128.4652, "대형 헬기", "🛬 착륙"],
                  [38.1389, 128.3079, "소형 헬기", "🪢 호이스트"]],
        inputs=[lat, lon, heli, tactic])


if __name__ == "__main__":
    demo.launch()
