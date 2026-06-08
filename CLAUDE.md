# 산악구조최적화 AI 시스템 — Project Guidelines

설악산 지형·기상 데이터 기반 헬기 최적 경로/착륙지 산출 시뮬레이션.
조난자 GPS → (DEM + 임상도 + 기상청 AWS) → ML Risk Scoring → A* 경로 → 시각화.

---

## Core Rules (최우선 — 항상 준수)
1. **대규모 수정 전 먼저 질문**: 코드 상당 부분 수정이나 구조 변경이 필요하면 임의로 진행하지 말고, 먼저 나에게 묻고 승인을 받은 뒤 진행한다.
2. **모호하면 가정하지 말고 확인**: 요청이 모호하거나 불명확하면 추측으로 코드를 작성/실행하지 말고, 내가 요청한 바에 대한 너의 이해를 먼저 설명하고 의도를 질문으로 확인한다.

## Architecture & Style
- **모듈화·관심사 분리**: 모든 코드를 한 파일에 넣지 말 것. 기능·역할별로 철저히 모듈화하여 별도 파일로 분리한다 (`modules/` 패키지 규칙 준수).
- **읽기 쉬운 코드**: 깔끔하고 가독성 높은 코드를 유지하고, 새로 만들거나 분리한 모듈 간 의존성을 명확히 관리한다.
- **출력 인코딩**: Windows 콘솔(cp949)에서 유니코드(°, ✅ 등) 출력 시 깨짐 → 진입 스크립트 상단에서 `sys.stdout.reconfigure(encoding="utf-8")` 패턴을 유지한다.

---

## 실행 및 환경
- **OS/Shell**: Windows 11 / PowerShell (`$null`, `$env:VAR`, 백틱 줄바꿈 사용)
- **의존성 설치**: `pip install -r requirements.txt`
  - ⚠️ `requirements.txt`에 현재 `numpy scipy pandas requests folium python-dotenv scikit-learn`만 있음.
    XGBoost/LightGBM 모델 추론이 필요하면 `xgboost`, `lightgbm`을 별도 설치/추가해야 한다.
- **API 키**: 기상청 KMA API 키는 `.env`(`KMA_API_KEY=...`)로 로드. `.env`는 **읽기/수정/커밋 금지**.
- **실행 명령**: `python main.py` (4단계 파이프라인: 지형→구조구역→경로→시각화 오케스트레이터).
  - 데이터 경로가 `data/...` 상대경로이므로 **반드시 레포 루트(cwd=AISystem)에서 실행**한다.

## 활성 코드 트리 (중요)
- **실제 개발/실행 대상은 레포 루트**: `modules/`, `config.py`, `data/`, `outputs/`, `tests/`.
  - `rescue_zone.py`, `weather_interpolation.py`는 **루트 `modules/`에만** 존재 → 루트가 최신 활성본이다.
- `XGBoost_Model/`, `RF_Model/`, `LightGBM_Model/` 하위의 `weather/modules/`는 **오래된 복사본**이다.
  수정은 루트 `modules/`에 하고, 중복 트리는 임의로 건드리지 않는다.
- ⚠️ `RF_Model/`·`LightGBM_Model/` 트리는 **향후 정리(삭제) 예정**이나, **내가 명시적으로 지시하기 전까지 삭제하지 말 것**.

## 프로젝트 구조 (루트 기준)
```
config.py                  # 전역 설정 (임계치, 보간/스코어링 파라미터)
modules/
  terrain.py               # 지형 레이어 (경사/곡률/능선/개활지)
  weather.py               # KMA AWS 실시간 관측 수집
  weather_interpolation.py # 지형 인지형 기상 보간 (고도보정 다변량 IDW)
  data_preprocessing.py    # 좌표 변환·기상-지형 융합
  rescue_zone.py           # [Stage 2] AI 구조구역 선정 — 현재 MOCK 셸 (아래 참조)
  hoist.py                 # 호이스트 후보지 분석, haversine
  pathfinding.py           # A* + Tobler 보행함수 (ETA), compute_cost/heuristic
  simple_pathfinding.py    # 웨이포인트 폴백 경로
  dynamic_pathfinding.py   # 실시간 동적 경로 (진행방향 가중)
  vworld_3d.py             # [Stage 4] VWorld 3D 시각화
  helicopter_mission_map.py / *_approval.py  # Folium 미션 지도
  visualize.py             # 2D 패널티 지도
  evaluation/metrics.py    # 경로/착륙지/보정 평가 지표
data/                      # 입력 데이터 (gitignore, 로컬 전용)
outputs/                   # 생성 HTML 결과 (gitignore)
legacy/                    # 미사용 보관 스크립트 — 수정 대상 아님
```

## 도메인 규약 (모델·피처)
- **Feature 순서 고정** (학습=추론 반드시 일치):
  `elevation, slope_deg, tree_density, tree_height, wind_speed, wind_dir_sin, wind_dir_cos, land_0, land_1, land_2`
- **풍향은 sin/cos 분해** 후 입력 (raw 각도 직접 입력 금지).
- **4 전술 모드**: `small_landing`, `small_hoist`, `large_landing`, `large_hoist`.
- **타겟 라벨**: 0=안전(score≥0.80), 1=주의(score≥0.55), 2=위험(score<0.55).
- **Risk Score** = static_score(지형) + wind_score·w1 + wind_dir_score·w2 (동적 기상 반영).
- 결측치는 `-9999`를 `np.nan`으로 치환 후 처리한다.

## 현재 상태 / 알려진 갭 (작업 시 인지)
- ✅ XGBoost 4모드 모델 학습 완료 (`XGBoost_Model/models/*.ubj`), 기상 API 수집 구현됨.
- ❌ **`modules/rescue_zone.py`는 아직 MOCK 셸** — 항상 더미값(`score=0.5`, `mode="H_Light"`) 반환.
  실제 ML 추론으로 교체 시 `[MOCK 로직 시작]~[끝]` 블록 **내부만** 바꾸고, IN/OUT 계약과 Stage 3/4는 건드리지 않는다.
- ❌ **A* 휴리스틱에 Risk Score 미반영** — `pathfinding.py`의 `heuristic()`은 거리만, `compute_cost()`는 지형/바람만 사용.
  통합 시 `λ` 가중치 근거(매뉴얼/논문)를 함께 남긴다.

## Git / 워크플로우
- 커밋·푸시는 **내가 요청할 때만** 수행한다.
- 기본 브랜치는 `main`, 작업 브랜치는 `sujin`.
- `data/`, `outputs/`, `.env`, `*.html`, `__pycache__`는 커밋하지 않는다 (gitignore 준수).

## 금지 사항
- `.env` 읽기/수정/커밋 금지.
- `legacy/` 및 비활성 모델 트리(`*_Model/weather/`) 임의 수정 금지. (`RF_Model/`·`LightGBM_Model/`는 지시 전 삭제 금지)
- 데이터 파일(CSV/DEM/parquet) 임의 변경 금지.
- **Feature 컬럼의 순서·이름 임의 변경 금지** (학습↔추론 불일치를 유발).
- **학습된 모델 파일(`*.ubj`/`*.joblib`) 덮어쓰기·재학습 금지** — 재학습은 명시적 지시가 있을 때만.
- **`rescue_zone.py`의 IN/OUT 계약 변경 금지** — MOCK 블록 내부만 교체하고 Stage 3/4 연결을 깨지 않는다.
- **임계치·가중치(슬로프 한계, 풍속 구간, A*의 `λ` 등) 근거 없이 수정 금지** — 변경 시 출처(매뉴얼/논문)를 함께 남긴다.
