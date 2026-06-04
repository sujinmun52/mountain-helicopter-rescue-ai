# 🛸 산악구조최적화AI시스템 (Mountain Helicopter Rescue AI System)
> **설악산 지형·기상 데이터 기반 산악구조 헬기 최적 경로 및 착륙지 산출 시뮬레이션 시스템**

## 📌 프로젝트 개요
조난자의 GPS 좌표가 입력되면 수치표고모델(DEM), 임상도, 기상청 풍속 데이터를 종합적으로 분석하여 머신러닝 기반으로 안전한 헬기 착륙 구역을 선정하고, A* 알고리즘을 통해 최적 비행 경로를 산출하는 엔드투엔드(E2E) 시뮬레이션 프로젝트입니다.

## 프로젝트 폴더 구조
```text
AISystem/
│
├── main.py                         # 🚀 진입점 — 4단계 파이프라인 오케스트레이터
├── config.py                       # 전역 설정 (API키, 임계치, 보간/스코어링 파라미터)
├── requirements.txt
├── .gitignore                      # .env, __pycache__, .venv, data/, *.html 무시
│
├── modules/                        # 핵심 모듈 패키지
│   ├── terrain.py                  #  지형 레이어 (경사/곡률/능선/개활지)
│   ├── weather.py                  #  KMA API허브 실시간 AWS 관측 수집
│   ├── weather_interpolation.py    #  🆕 지형 인지형 기상 보간 (고도보정 다변량 IDW)
│   ├── data_preprocessing.py       #  좌표 변환·기상-지형 융합
│   ├── rescue_zone.py              #  🆕 [Stage 2] AI 구조구역 선정 (MOCK 셸)
│   ├── risk_scoring.py             #  🆕 헬기 4모드 위험도 스코어링
│   ├── hoist.py                    #  호이스트 후보지 분석
│   ├── pathfinding.py              #  A* + Tobler 보행함수 (ETA 산출)
│   ├── simple_pathfinding.py       #  웨이포인트 경로 (폴백)
│   ├── dynamic_pathfinding.py      #  동적 경로(진행방향 가중)
│   ├── vworld_3d.py                #  [Stage 4] VWorld 3D WebGL 시각화
│   ├── helicopter_mission_map.py   #  Folium 미션 지도
│   ├── helicopter_mission_map_approval.py  # 관제 승인형 지도
│   ├── visualize.py                #  2D 패널티 지도
│   └── evaluation/
│       └── metrics.py              #  경로/착륙지/보정 평가 지표
│
├── data/                           # 📂 입력 데이터 (gitignore — 로컬 전용)
│   ├── Final_seoraksan_part1~3_*.csv
│   ├── victim_gps.csv
│   └── 설악산15mDEM_*.tif
│
├── outputs/                    🆕  # 📊 생성 결과 HTML (gitignore)
│   ├── helicopter_mission_3d.html      # ⭐ VWorld 3D 관제 뷰
│   ├── helicopter_mission_folium.html
│   ├── output_map_3d.html              # Plotly 3D
│   ├── output_map_2d_penalty.html
│   └── output_map.html
│
├── tests/                      🆕
│   └── test_dynamic_pathfinding.py
│
├── legacy/                     🆕  # 미사용 독립 스크립트 (보관용)
│   ├── helicopter_mission.py
│   ├── helicopter_mission_auto.py
│   └── weather_processor.py
│
└── .vscode/
    └── launch.json
```

## 🛠️ 담당 역할 (문수진)
- **비용함수(Cost Function) 설계 및 구현**: 헬기 운항 매뉴얼 기준값을 활용하여 지형 고도, 경사도 및 고도 구간별 풍속 패널티 가중치를 유기적으로 결합한 A* 알고리즘 비용함수 최적화
- **Folium 기반 지도 시나리오 시각화**: GPS 입력에 따른 헬기 착륙 후보지와 최적 비행 경로를 동적으로 지도상에 매핑하는 가상 시뮬레이션 파이프라인 구축
- **프로젝트 데이터 규격 정의 및 성과 보고서 작성**

## 📅 Tech Stack & Data
- **Language & Library**: Python, Folium, A* Algorithm, Random Forest
- **Data Source**: 국토지리정보원 DEM(수치표고모델), 산림청 임상도, 기상청 AWS(방재기상관측) 데이터
