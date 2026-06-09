# Mountain Helicopter Rescue AI System

설악산 지형·기상 데이터 기반 산악구조 헬기 최적 경로 및 착륙지 산출 시뮬레이션 시스템.

조난자의 GPS 좌표가 입력되면 수치표고모델(DEM)·임상도·기상청 실시간 풍속 데이터를 종합 분석하여,
머신러닝(XGBoost)으로 안전한 헬기 착륙 구역을 선정하고 A* 알고리즘으로 최적 비행 경로를 산출하는
엔드투엔드(E2E) 시뮬레이션 프로젝트입니다.

---

## 시스템 흐름

```
조난자 GPS 입력
  ① Static   DEM·임상도 → 가용 착륙지 후보 계산 (지형 기반)
  ② Dynamic  기상청 실시간 API(현재 −20분) → 풍속·풍향으로 Risk Score 재평가
  ③ ML       XGBoost 4모드 분류 → 착륙지별 위험도(안전/주의/위험) 산출
  ④ A*       h(n) = f(Distance) + λ × RiskScore(n) → 최적 경로 산출
  → 착륙지 · 비행/도보 경로 · ETA · 미션 지도(2D/3D)
```

| 단계 | 입력 | 처리 | 출력 |
|------|------|------|------|
| Static | DEM, 임상도 | 고도·경사·임목 기반 가용 착륙지 | 후보 격자 |
| Dynamic | 기상청 실시간 API | 풍속·풍향 + 고도보정 | 동적 Risk Score |
| ML | 13개 피처(지형+기상+상호작용) | XGBoost 4모드 추론 | 안전등급·착륙지 |
| A* | 위험도·지형·바람 | 비행=풍속 회피, 도보=지형 회피 | 최적 경로·ETA |

---

## 머신러닝 모델

- 모델 비교: Random Forest, LightGBM, XGBoost → 성능·추론속도 종합 우위로 **XGBoost 채택**
- 타겟: 4가지 전술 모드별 분류 — `small_landing`, `small_hoist`, `large_landing`, `large_hoist` (기종 × 착륙/호이스트)
- 라벨 (3등급)

  | 라벨 | 등급 | 기준 |
  |------|------|------|
  | 0 | 안전 | Risk Score ≥ 0.80 |
  | 1 | 주의 | Risk Score ≥ 0.55 |
  | 2 | 위험 | Risk Score < 0.55 |

- 피처(13개): 지형(고도·경사·임목밀도·수고·토지) + 실시간 기상(풍속·풍향 sin/cos) + 물리 상호작용(`tree_risk`·`aero_risk`·`slope_wind_risk`)
- 모드별 가중치가 도메인 물리 반영: 착륙=경사 중심 / 호이스트=임목밀도·수고 중심

---

## 데이터

| 데이터 | 출처 | 용도 |
|--------|------|------|
| DEM (수치표고모델) | 국토지리정보원 | 고도·경사도 계산 |
| 임상도 | 산림청 | 수목밀도(임관피도)·수목높이 |
| 기상 과거 데이터 | 기상청 | 모델 학습용 |
| 기상 실시간 API | 기상청 AWS | 추론용 (현재 −20분 관측) |

---

## 가중치 설계 근거

- 풍속 임계값: 산림청 산림항공기 운항 규정 3.3.3 기반 (기종별 운용 한계 — 소형 10 m/s / 대형 20 m/s, 항공안전법 시행규칙 별표24 준용)
- 경사도·수목 임계값: 항공 운용 매뉴얼 기반

임계치·가중치는 임의값이 아닌 운항 규정·매뉴얼 근거에 따라 설정되었습니다.

---

## 프로젝트 구조

```text
AISystem/
├── main.py                       # 4단계 파이프라인 오케스트레이터 (CLI)
├── app.py                        # Gradio 웹앱 (GPS+기종 → 착륙지·경로·2D/3D 지도)
├── config.py                     # 전역 설정 (API키, 임계치, 가중치)
├── requirements.txt
│
├── modules/                      # 핵심 모듈
│   ├── terrain.py                #   지형 레이어 (경사/곡률/능선/개활지)
│   ├── weather.py                #   기상청 AWS 실시간 관측 수집
│   ├── weather_interpolation.py  #   지형 인지형 기상 보간 (고도보정 IDW)
│   ├── data_preprocessing.py     #   좌표 변환·기상-지형 융합
│   ├── rescue_zone_inference.py  #   [Stage 2] XGBoost 추론 — 착륙/호이스트 선정
│   ├── rescue_zone.py            #   [Stage 2] 구조구역 선정 셸
│   ├── risk_scoring.py           #   헬기 4모드 위험도 스코어링
│   ├── pathfinding.py            #   [Stage 3] A* + Tobler 보행함수 (비용/ETA)
│   ├── flight_path.py            #   비행 경로 (순항고도 + 풍속·능선 정보)
│   ├── hoist.py                  #   호이스트 후보지 분석
│   ├── simple_pathfinding.py     #   웨이포인트 경로 (폴백)
│   ├── dynamic_pathfinding.py    #   동적 경로 (진행방향 가중)
│   ├── helicopter_mission_map.py #   [Stage 4] Folium 2D 미션 지도
│   ├── vworld_3d.py              #   VWorld 3D 시각화
│   ├── visualize.py              #   2D 패널티 지도
│   └── evaluation/metrics.py     #   경로/착륙지 평가 지표
│
├── XGBoost_Model/                # ML 학습·추론
│   ├── 1.xgb_preprocess.py       #   전처리 (지형·기상 → 학습셋)
│   ├── 2.xgb_train.py            #   4모드 학습 (피처 엔지니어링)
│   ├── 3.xgb_inference.py        #   추론·검증 (stress test)
│   ├── models/*.ubj              #   학습된 4모드 모델 (feature_model)
│   └── dataset/terrain_base.parquet
│
├── data/                         # 입력 데이터 (gitignore — 로컬 전용)
│   ├── Final_seoraksan_part1~3_*.csv
│   ├── victim_gps.csv
│   └── 설악산15mDEM_*.tif
│
├── outputs/                      # 생성 결과 HTML/이미지 (gitignore)
└── docs/                         # 발표 정리·이슈 노트·배포 가이드
```

---

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정
레포 루트에 `.env` 파일을 생성하고 기상청 API 키를 입력합니다.
```
KMA_API_KEY=발급받은_기상청_API_키
```
> `.env`는 커밋하지 않습니다(`.gitignore`). 키는 [기상청 API 허브](https://apihub.kma.go.kr)에서 발급.

### 3. 실행
```bash
python app.py     # 웹앱 → 브라우저에서 http://127.0.0.1:7860 접속
python main.py    # CLI — 4단계 파이프라인 (착륙지·경로·시각화)
```
> 데이터 경로가 상대경로이므로 반드시 레포 루트에서 실행합니다.

### 웹앱(`app.py`) 기능
- 조난자 GPS + 기종 입력 → 착륙(A안)·호이스트(B안) 둘 다 추천 (기장 판단)
- 전술 선택 → A* 경로 산출 + 착륙지·조난자 상세 + ETA
- 2D 미션맵(실제 지도 + 경로 우회 이유 + 실시간 기상 연동) / 3D 지형뷰(회전·확대) 탭
- 랜덤 조난자(실데이터) 입력, 처리 로그(추론·기상·경로 과정) 확인

---

## 현재 상태

- XGBoost 4모드 학습·추론 완료 — `feature_model`(13피처), 학습=추론 정합
- 실시간 기상 융합 — 기상청 AWS 관측 + 고도보정으로 동적 Risk Score
- A* 경로 — 비행(풍속·능선 회피) + 도보(Tobler 경사·임상 저항)
- 시각화 — 2D Folium(실제 지도) / 3D Plotly(인터랙티브) / Gradio 웹앱
- 풍속 반영 — 물리변수 상호작용 피처로 풍속 민감도 확보

---

## 향후 보완 계획

- 실측 라벨 확보 — 현재 ML은 도메인 룰 기반 타겟(surrogate). 실제 착륙·사고 데이터 확보 시 진짜 지도학습으로 고도화
- 격자 해상도 정밀화 — 현재 다운샘플 격자(셀 81~140m) → 15m DEM 원본 활용 시 경로·시간 정밀화
- A* 휴리스틱 RiskScore 완전 통합 — λ 가중치 튜닝으로 위험도-거리 균형 최적화
- 배포 — Hugging Face Spaces (가이드: `docs/HF배포_가이드.md`)

---

## 팀 구성원 및 역할

| 이름 | GitHub | 역할 |
|------|--------|------|
| 문수진 | [@sujinmun52](https://github.com/sujinmun52) | 기획 및 시스템 통합 |
| 김수인 | [@suin25](https://github.com/suin25) | 데이터 수집 및 AI 모델링 |
| 이다원 | [@many-one-22](https://github.com/many-one-22) | 데이터 엔지니어링 및 AI 모델링 |
| 김민서 | [@seonimk](https://github.com/seonimk) | 알고리즘 설계 |
| 이채윤 | [@leechaiyun](https://github.com/leechaiyun) | 검증 및 성능 최적화 |

---

## Tech Stack

- Language: Python 3.11
- ML: XGBoost (vs. Random Forest, LightGBM)
- Pathfinding: A* + Tobler hiking function
- App / Visualization: Gradio, Folium, Plotly
- Data: 국토지리정보원 DEM, 산림청 임상도, 기상청 AWS
