# Hugging Face Spaces 배포 가이드

산악구조 최적화 AI 시스템(`app.py`)을 HF Spaces에 배포하는 절차.

> ⚠️ 우리 시스템은 **data(63MB)·parquet(35MB)·모델(6MB) 의존**이 있어,
> 일반 이미지분류 예제와 달리 **데이터 자산을 함께 올려야** 합니다.

---

## 0. 사전 준비
- Hugging Face 계정 (https://huggingface.co)
- `pip install huggingface_hub` + `git` (대용량은 git-lfs 자동)
- 준비된 키: `KMA_API_KEY`, `DATA_API_KEY` (`.env`에 있는 값 — 절대 커밋 금지, Secrets로만)

---

## 1. Space 생성 (웹)
1. https://huggingface.co/new-space 접속
2. 설정:
   - **Owner / Space name**: 예) `sujinmun/mountain-rescue-ai`
   - **SDK**: **Gradio** 선택
   - **Hardware**: CPU basic (무료) — 첫 요청은 격자 로딩으로 수십 초 걸릴 수 있음
   - Visibility: Public 또는 Private

---

## 2. README.md 헤더 (HF Space는 이걸 읽음)
Space repo 루트 `README.md` **맨 위**에 아래 YAML 헤더를 넣는다:

```yaml
---
title: 산악구조 최적화 AI
emoji: 🏔️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
---
```

(GitHub README와 별개. HF Space repo 전용)

---

## 3. 올려야 할 파일
| 분류 | 파일/폴더 |
|------|-----------|
| 앱 | `app.py`, `main.py`, `config.py`, `requirements.txt`, `README.md`(헤더) |
| 모듈 | `modules/` 전체 |
| 추론 | `XGBoost_Model/models/*.ubj`, `XGBoost_Model/dataset/terrain_base.parquet`, `XGBoost_Model/preprocess/score_utils.py` |
| 데이터 | `data/Final_seoraksan_part*.csv`, `data/victim_gps.csv`, `data/*.tif` |

> `data/`·`*.csv`·`*.ubj`는 우리 `.gitignore`에 있으므로, **HF repo에 올릴 땐 `git add -f`로 강제 추가**한다.

---

## 4. Secrets 설정 (API 키 — Settings → Variables and secrets)
- `KMA_API_KEY` = (기상청 키)
- `DATA_API_KEY` = (공공데이터 키)

→ `config.py`의 `os.getenv(...)`가 자동으로 읽음. **`.env`는 올리지 않는다.**

---

## 5. 업로드 (git)
```bash
# Space repo clone
git clone https://huggingface.co/spaces/<owner>/<space-name>
cd <space-name>

# 프로젝트 파일 복사 (data·모델 포함)
#   .gitignore 무시하고 강제 추가
git add -f app.py main.py config.py requirements.txt README.md
git add -f modules XGBoost_Model/models XGBoost_Model/dataset/terrain_base.parquet
git add -f XGBoost_Model/preprocess/score_utils.py
git add -f data/Final_seoraksan_part*.csv data/victim_gps.csv data/*.tif

git commit -m "deploy: 산악구조 AI Gradio 앱"
git push
```

→ push 후 HF가 자동 빌드. **Build logs**에서 `requirements.txt` 설치 + 앱 기동 확인.

---

## 6. 동작 확인 / 주의
- 첫 실행은 `prepare_terrain_grid`가 CSV(63MB) 로딩 → **수십 초** 소요(정상)
- 기상 API 실패 시 폴백 동작 (로그 창에서 확인)
- 메모리 부족하면 Hardware 업그레이드 또는 **경량화**(아래) 고려

### (선택) 경량 배포 — 데이터 없이
전체 격자(CSV) 대신 **착륙지 추천만**(parquet 기반)으로 줄이면 data 업로드 불필요.
단 A* 경로·시각화는 빠지므로, **풀데모는 전체 배포 권장**.

---

## 체크리스트
- [ ] Space 생성 (Gradio SDK)
- [ ] README.md YAML 헤더
- [ ] 코드 + modules + XGBoost_Model(models·parquet·score_utils)
- [ ] data CSV·tif (`git add -f`)
- [ ] Secrets: KMA_API_KEY, DATA_API_KEY
- [ ] push → Build logs 확인 → 앱 접속
