# -*- coding: utf-8 -*-
"""산악구조 최적화 AI 시스템 — 발표 PPT 생성 (python-pptx)."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── 팔레트 (Forest & Moss — 산악구조) ──
DARK   = RGBColor.from_string("16291A")
FOREST = RGBColor.from_string("2C5F2D")
MOSS   = RGBColor.from_string("97BC62")
RED    = RGBColor.from_string("C0392B")
SAFE   = RGBColor.from_string("27AE60")
WHITE  = RGBColor.from_string("FFFFFF")
INK    = RGBColor.from_string("2B2B2B")
GRAY   = RGBColor.from_string("6B7280")
LIGHTB = RGBColor.from_string("EEF3EA")
HF = "맑은 고딕"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333


def slide(bg=WHITE):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    return s


def box(s, x, y, w, h, color, line=None, rounded=False):
    shp = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = color
    if line: shp.line.color.rgb = line; shp.line.width = Pt(1)
    else: shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def txt(s, text, x, y, w, h, size, color=INK, bold=False, align=PP_ALIGN.LEFT,
        font=HF, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = ln
        r.font.name = font; r.font.size = Pt(size); r.font.bold = bold
        r.font.italic = italic; r.font.color.rgb = color
    return tb


def bullets(s, items, x, y, w, h, size=15, color=INK, gap=6):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (lvl, t, c) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl; p.space_after = Pt(gap)
        r = p.add_run(); r.text = ("• " if lvl == 0 else "– ") + t
        r.font.name = HF; r.font.size = Pt(size - lvl); r.font.bold = (lvl == 0)
        r.font.color.rgb = c or (INK if lvl == 0 else GRAY)
    return tb


def header(s, n, title, tag=""):
    txt(s, title, 0.6, 0.35, 11.0, 0.9, 30, FOREST, bold=True)
    if tag:
        txt(s, tag, 11.3, 0.45, 1.5, 0.5, 11, MOSS, bold=True, align=PP_ALIGN.RIGHT)
    txt(s, f"{n:02d}", 12.5, 6.9, 0.7, 0.4, 11, GRAY, align=PP_ALIGN.RIGHT)


def table(s, data, x, y, w, colw, hrow=True, fs=13):
    rows, cols = len(data), len(data[0])
    gt = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w),
                            Inches(0.5*rows)).table
    for ci, cw in enumerate(colw): gt.columns[ci].width = Inches(cw)
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = gt.cell(ri, ci)
            cell.margin_left = Inches(0.08); cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            tf = cell.text_frame; p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT
            r = p.add_run(); r.text = str(val); r.font.name = HF; r.font.size = Pt(fs)
            if ri == 0 and hrow:
                cell.fill.solid(); cell.fill.fore_color.rgb = FOREST
                r.font.color.rgb = WHITE; r.font.bold = True
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHTB
                r.font.color.rgb = INK
    return gt


def callout(s, num, label, x, y, w=2.6, h=1.5, color=FOREST):
    box(s, x, y, w, h, color, rounded=True)
    txt(s, num, x, y+0.18, w, 0.8, 40, WHITE, bold=True, align=PP_ALIGN.CENTER)
    txt(s, label, x, y+h-0.5, w, 0.45, 12, WHITE, align=PP_ALIGN.CENTER)


# ════════ S1 표지 ════════
s = slide(DARK)
box(s, 0, 0, W, 0.12, MOSS)
txt(s, "산악구조 최적화 AI 시스템", 0.9, 2.5, 11.5, 1.2, 48, WHITE, bold=True)
txt(s, "설악산 지형·실시간 기상 기반 헬기 최적 착륙지·경로 산출", 0.9, 3.7, 11.5, 0.7, 20, MOSS)
txt(s, "ML(안전지 평가) + A*(안전 경로)  ·  2026.06.11", 0.9, 4.5, 11.5, 0.5, 14, RGBColor.from_string("AAB8A0"))

# ════════ S2 문제 ════════
s = slide(); header(s, 2, "문제: 왜 필요한가", "통찰")
bullets(s, [
    (0, "산악 조난 구조의 두 가지 난제", FOREST),
    (1, "어디 내릴까 — 잘못 내리면 2차 사고", None),
    (1, "어떻게 갈까 — 골든타임, 험준지형 접근", None),
    (0, "사람의 한계: 험준한 산에서 안전지를 육안으로 즉시 판단 불가", INK),
], 0.6, 1.5, 7.0, 3.5, 17)
callout(s, "3.5%", "설악산 안전 착륙지 비율", 8.6, 1.7, 3.8, 1.8, RED)
txt(s, "→ 데이터·AI로 희소 안전지를 정밀 탐색하는\n   의사결정 지원이 필요", 8.6, 3.8, 4.0, 1.3, 16, FOREST, bold=True)

# ════════ S3 시스템 개요 ════════
s = slide(); header(s, 3, "시스템 개요", "완성도")
txt(s, "입력  →  ML(의사결정)  →  A*(실행)  →  출력", 0.6, 1.4, 12, 0.6, 18, FOREST, bold=True)
cards = [("입력", "조난자 GPS\n지형(DEM·임상)\n실시간 기상(KMA)", MOSS),
         ("ML 의사결정", "149만 후보 평가\n최적 안전 착륙지·전술 선정", FOREST),
         ("A* 실행", "보행=지형 회피\n비행=풍속 회피", FOREST),
         ("출력", "착륙지 + 경로 + ETA\n미션 지도", MOSS)]
for i, (t, d, c) in enumerate(cards):
    x = 0.6 + i*3.05
    box(s, x, 2.3, 2.8, 2.6, LIGHTB, rounded=True)
    box(s, x, 2.3, 2.8, 0.6, c, rounded=False)
    txt(s, t, x, 2.36, 2.8, 0.5, 15, WHITE, bold=True, align=PP_ALIGN.CENTER)
    txt(s, d, x+0.15, 3.1, 2.5, 1.7, 13, INK)
txt(s, "Stage 1→4 end-to-end 동작 (python main.py)", 0.6, 5.3, 12, 0.5, 14, GRAY, italic=True)

# ════════ S4 데이터 ════════
s = slide(); header(s, 4, "데이터", "완성도")
bullets(s, [
    (0, "지형: 설악산 15m DEM + 임상도(수관밀도·수고) → 149만 격자", FOREST),
    (0, "기상: 기상청 AWS 5개소 실시간 관측", FOREST),
    (1, "고도보정(멱법칙 α=0.27): 지상 10m 풍속 → 헬기 운용고도 환산", None),
    (1, "저지대 잔잔해도 고지대는 강풍 — 고도 효과 반영", None),
    (0, "피처 10개 / 타겟 3등급 (안전 ≥0.80 / 주의 ≥0.55 / 위험 <0.55)", FOREST),
], 0.6, 1.6, 12, 3.5, 17)

# ════════ S5 ML 핵심 ════════
s = slide(); header(s, 5, "ML: 핵심 의사결정 (착륙지 선정)", "난이도")
bullets(s, [
    (0, "XGBoost 4모드: small/large × landing/hoist (기종·전술별)", FOREST),
    (0, "149만 후보 → 실시간 위험 평가 → 안전지 선정", FOREST),
    (1, "사람이 못 하는 규모·속도의 정밀 탐색", None),
    (0, "모드별 가중치가 도메인 물리 반영", FOREST),
    (1, "landing → 경사 중심 (slope 0.46)", None),
    (1, "hoist → 임목밀도·수고 중심 (density 0.24)", None),
], 0.6, 1.6, 12, 3.8, 17)

# ════════ S6 검증 성능지표 ════════
s = slide(); header(s, 6, "ML 검증 — 성능지표", "완성도")
txt(s, "test셋 848만 행 평가", 0.6, 1.4, 6, 0.5, 15, GRAY)
table(s, [["모드", "Accuracy", "F1(macro)"],
          ["small_landing", "0.951", "0.910"],
          ["large_landing", "0.991", "0.955"],
          ["large_hoist", "0.950", "0.932"]],
      0.6, 2.0, 6.2, [2.6, 1.8, 1.8])
box(s, 7.3, 2.0, 5.4, 2.2, LIGHTB, rounded=True)
txt(s, "Confusion Matrix 핵심", 7.55, 2.2, 5, 0.5, 16, FOREST, bold=True)
txt(s, "위험을 '안전'으로 오분류 = 0건", 7.55, 2.8, 5, 0.6, 20, RED, bold=True)
txt(s, "→ 구조 안전 관점에서 가장 중요한\n   '위험 누락 없음' 달성", 7.55, 3.4, 5, 0.8, 14, INK)

# ════════ S7 인사이트① 희소성 ════════
s = slide(); header(s, 7, "인사이트 ① 안전지 희소성", "통찰")
table(s, [["모드", "안전", "주의", "위험"],
          ["small_landing", "10.2%", "34.9%", "55.0%"],
          ["large_hoist", "4.4%", "24.4%", "71.3%"]],
      0.6, 1.7, 7.0, [2.8, 1.4, 1.4, 1.4])
callout(s, "3.5~10%", "안전 착륙지", 8.2, 1.7, 4.4, 1.6, RED)
txt(s, "사람이 못 찾으니 AI가 필요\n→ 희소성이 곧 프로젝트의 존재 이유", 8.2, 3.5, 4.4, 1.2, 16, FOREST, bold=True)

# ════════ S8 인사이트② 기종별 위험 ════════
s = slide(); header(s, 8, "인사이트 ② 기종·전술별 위험 차이", "통찰·난이도")
bullets(s, [
    (0, "large_hoist가 가장 보수적(위험 71%)인 이유", FOREST),
    (1, "호이스트는 임목밀도·수고 가중 (density 0.24)", None),
    (1, "설악산 72.6%가 수관밀도 70%+ (밀)", None),
    (0, "근거 (산림청 임상도): 소(≤50%) / 중(51~70%) / 밀(70%+)", FOREST),
    (1, "70%+ 울폐림은 직접 호이스트 불가 → 환자 이송 후 인양", None),
    (0, "→ 모델이 실제 산악구조 물리 제약을 정확히 반영", SAFE),
], 0.6, 1.6, 12, 4.0, 16)

# ════════ S9 인사이트③ 풍속 무반응 → 해결 ════════
s = slide(); header(s, 9, "인사이트 ③ 풍속 무반응 → 피처 엔지니어링으로 해결", "통찰·난이도")
bullets(s, [
    (0, "Before: 초기 모델은 풍속 0→25 바꿔도 예측 동일 (무반응)", RED),
    (1, "원인: 학습 데이터가 약풍(0~4.9 m/s)에 치우침", None),
    (0, "해결: 물리변수 비선형 상호작용 피처 추가 (룰 점수 배제)", FOREST),
    (1, "aero_risk = 고도×풍속,  slope_wind_risk = 경사×풍속", None),
    (0, "After: 풍속 ↑ → 위험도 ↑ (반응 확인)", SAFE),
], 0.6, 1.55, 7.2, 3.3, 16)
# before/after 표
table(s, [["풍속", "위험(2) 비율"],
          ["2 m/s", "67%"],
          ["8 m/s", "68%"],
          ["16 m/s", "82% ↑"]],
      8.1, 1.7, 4.3, [2.0, 2.3])
txt(s, "교훈: 학습 데이터 분포·피처 설계가\n모델의 기상 반영을 결정", 8.1, 4.2, 4.4, 1.0, 14, FOREST, bold=True)
txt(s, "+ 고도보정(멱법칙 α=0.27) · 기종별 풍속제한(소형10/대형20, 항공안전법 별표24)",
    0.6, 5.15, 12, 0.5, 13, GRAY, italic=True)

# ════════ S10 A* 실행 ════════
s = slide(); header(s, 10, "A*: 안전 경로 실행", "난이도")
bullets(s, [
    (0, "보행 A*: Tobler 보행시간(경사) + 임상저항 → 위험지형 회피", FOREST),
    (0, "비행 A*: 풍속·풍향·능선난류 회피 (순항직선 + 위험구간 우회)", FOREST),
    (1, "근거: 이동=순항직선(최속), 산악접근=지형·바람 회피 (EASA/SAR)", None),
    (0, "결과 예시: 비행 6.3분 + 도보 5.4분 = 총 11.6분", SAFE),
], 0.6, 1.6, 12, 3.5, 17)

# ════════ S11 메인결과 ════════
s = slide(); header(s, 11, "메인 결과: A*가 구조시간을 줄인다", "완성도·통찰")
txt(s, "\"최단거리 ≠ 최단시간\"  —  직선 vs A* (험준 구간)", 0.6, 1.35, 12, 0.5, 17, FOREST, bold=True)
table(s, [["지표", "단순 직선", "A* 안전경로", "개선"],
          ["평균 경사", "30.5°", "15.5°", "경사 ½"],
          ["최대 경사", "37.7°", "27.2°", "완화"],
          ["도보 소요시간", "기준", "—", "약 58% ↓"]],
      0.6, 1.95, 7.6, [2.2, 1.8, 1.9, 1.7])
callout(s, "58% ↓", "구조시간 단축", 8.7, 1.95, 3.9, 1.5, SAFE)
txt(s, "직선은 급경사로 위험·비효율\nA*는 +33% 우회하지만 완경사 선택", 8.7, 3.6, 4.0, 1.0, 14, INK)
txt(s, "⚠ 절대 분(min)은 현재 격자(셀 81~140m)에서 보수적 → 비율이 핵심 지표 (15m DEM 적용 시 정밀화)",
    0.6, 5.5, 12, 0.6, 12, GRAY, italic=True)

# ════════ S12 E2E 데모 ════════
s = slide(); header(s, 12, "결과 — E2E 데모", "완성도")
bullets(s, [
    (0, "조난자 GPS → ML 착륙지 선정 → A* 경로 → 3D/Folium 미션맵", FOREST),
    (0, "Stage 2: large_landing, 안전등급 0, RiskScore 0.87", INK),
    (0, "Stage 3: 비행 6.3분 + 도보 5.4분 = 총 11.6분", INK),
], 0.6, 1.6, 12, 2.2, 17)
box(s, 0.6, 4.0, 12.1, 2.6, LIGHTB, rounded=True)
txt(s, "[ outputs/ 미션 지도 스크린샷 삽입 위치 ]", 0.6, 5.0, 12.1, 0.6, 16, GRAY,
    align=PP_ALIGN.CENTER, italic=True)

# ════════ S13 디버깅 통찰 ════════
s = slide(); header(s, 13, "개발 과정의 발견 (디버깅 통찰)", "통찰")
bullets(s, [
    (0, "경사도 버그: 격자 셀(81~140m)을 15m로 계산 → 경사 폭증 → 보행불가", FOREST),
    (1, "수정: 실제 셀 간격 반영 → 경사 73°→24°, 구조 ETA 66분→12분", None),
    (1, "교훈: 격자 해상도와 물리 계산의 정합성이 결과를 좌우", None),
    (0, "외부 검증 시도: 국립공원 위험지역 → 변별력 한계 발견", FOREST),
    (1, "설악산 전역이 위험 분류 + '고립위험 ≠ 착륙위험'", None),
    (1, "교훈: 외부 검증의 어려움 + 개념 정의의 중요성", None),
], 0.6, 1.6, 12, 4.2, 16)

# ════════ S14 한계 ════════
s = slide(); header(s, 14, "ML의 정직한 위치 & 한계", "통찰")
bullets(s, [
    (0, "현 ML = 룰 대리모델(surrogate) — 실측 사고데이터 부재로 룰로 정답 정의", FOREST),
    (1, "가치: 149만 후보 고속·일관 평가, 일반화, 검증 가능", SAFE),
    (1, "한계: 룰을 못 넘음 → 실측 라벨 확보 시 진짜 지도학습으로 고도화", None),
    (0, "풍속: 약풍 학습데이터 한계 → 강풍 포함 재학습 (향후)", None),
    (0, "격자 해상도: 200격자 → 15m DEM 원본 활용 시 정밀화 (향후)", None),
], 0.6, 1.6, 12, 4.0, 16)

# ════════ S15 결론 ════════
s = slide(DARK)
box(s, 0, 0, W, 0.12, MOSS)
txt(s, "결론", 0.9, 0.9, 11, 0.9, 34, WHITE, bold=True)
bullets(s, [
    (0, "설악산 안전 착륙지 3.5% → 육안으로 못 찾는 안전지를 ML로 정밀 탐색", WHITE),
    (0, "ML = 핵심 의사결정(착륙지),  A* = 안전 실행(경로) — 두 축", MOSS),
    (0, "도메인 분석(기종별 위험·임상밀도)으로 현실 부합성 확인", WHITE),
    (0, "한계를 데이터로 진단 → '문제를 제대로 이해한 시스템'", WHITE),
], 0.9, 2.2, 11.5, 3.5, 19, gap=14)
txt(s, "안전하게, 구조 시간을 줄인다.", 0.9, 6.1, 11.5, 0.7, 22, MOSS, bold=True, italic=True)

out = "outputs/산악구조_발표.pptx"
os.makedirs("outputs", exist_ok=True)
prs.save(out)
print("저장 완료:", out, "| 슬라이드", len(prs.slides._sldIdLst))
