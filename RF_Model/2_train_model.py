import os
import sys
from dotenv import load_dotenv

# 1. 인프라 및 상대 경로 설정 (XGBoost_Model 인프라와 완벽 동기화)
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, '.env'), override=True)
project_root = os.path.dirname(current_dir)

import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import joblib
import matplotlib.pyplot as plt
import platform

# 🎯 맷플롯립 한글 깨짐 방지 글로벌 패치
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 소스를 XGBoost_Model 패키지 내부의 배치 디렉토리로 조준 타격
XGB_BATCH_DIR = os.path.join(project_root, 'XGBoost_Model', 'dataset', 'batches')
MODEL_DIR     = os.path.join(current_dir, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================================================================
# [ENGINE] ROC & PR 통합 플로팅 시각화 대시보드 생성 함수
# ==============================================================================
def plot_evaluation_curves(y_true, y_prob, model_key):
    n_classes = 3
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    class_labels = {0: "Class 0: Safe (안전)", 1: "Class 1: Caution (주의)", 2: "Class 2: Danger (위험)"}
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # Left Panel: ROC
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i], lw=2.5, label=f"{class_labels[i]} (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.5, label='Random Guess (AUC = 0.50)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (TPR / Sensitivity)', fontsize=11)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # Right Panel: PR
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax2.plot(recall, precision, color=colors[i], lw=2.5, label=f"{class_labels[i]} (AP = {ap_score:.4f})")
        baseline = np.sum(y_true_bin[:, i]) / len(y_true)
        ax2.axhline(y=baseline, color=colors[i], linestyle='--', alpha=0.35)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (재현율)', fontsize=11)
    ax2.set_ylabel('Precision (정밀도)', fontsize=11)
    ax2.set_title('Precision-Recall (PR) Curve', fontsize=12, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.suptitle(f'[RF_{model_key.upper()}] Performance Evaluation Dashboard', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(MODEL_DIR, f"curves_{model_key}.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[시각화 완료] 대시보드 저장 성공 ➔ {output_path}")
    plt.close()


# ==============================================================================
# [STEP 1] 모델 인터랙티브 인터페이스 
# ==============================================================================
print("\n" + "="*60)
print("[산악 구조 AI - Random Forest 베이스라인] 통합 훈련 엔진")
print("="*60)
print(" 1. 소형 안착 착륙 모델 (Small Helicopter Landing)")
print(" 2. 소형 강하 호이스트 모델 (Small Helicopter Hoist)")
print(" 3. 대형 안착 착륙 모델 (Large Helicopter Landing)")
print(" 4. 대형 강하 호이스트 모델 (Large Helicopter Hoist)")
print("="*60)

user_input = input("학습을 진행할 모델의 번호를 입력하세요 (1 ~ 4): ").strip()
tactics_map = {
    "1": ("small_landing", "target_small_landing", "소형 착륙 모델"),
    "2": ("small_hoist",   "target_small_hoist",   "소형 호이스트 모델"),
    "3": ("large_landing", "target_large_landing", "대형 착륙 모델"),
    "4": ("large_hoist",   "target_large_hoist",   "대형 호이스트 모델")
}
if user_input not in tactics_map:
    sys.exit("올바른 번호를 입력하세요.")

model_key, target_column, model_kor_name = tactics_map[user_input]
print(f"\n[데이터 파이프라인 미러링] XGBoost의 분기 Parquet 배치를 직접 스캔합니다.")
print(f"대상 기체 전술: 【 {model_kor_name} 】\n")

# --- [STEP 2] 공유 피처 로드 (Polars LazyFrame 가속 스캔) ---
feature_columns = [
    'elevation', 'slope_deg', 'tree_density', 'tree_height',
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos',
    'land_0', 'land_1', 'land_2'
]
load_cols = feature_columns + [target_column, 'is_train_final', 'is_test']

lazy_all = pl.scan_parquet(os.path.join(XGB_BATCH_DIR, "*.parquet")).select(load_cols)

print("공용 검증셋(Test Set) 수집 중...")
test_pd = lazy_all.filter(pl.col('is_test')).collect(engine="streaming").to_pandas()
X_test, y_test = test_pd[feature_columns].values.astype(np.float32), test_pd[target_column].values.astype(np.int32)
print(f" ➔  검증셋 매트릭스 확보 완료: {len(test_pd):,}행")
del test_pd

print("공용 훈련셋(Train Set) 수집 및 다운캐스팅 중...")
train_pd = lazy_all.filter(pl.col('is_train_final')).collect(engine="streaming").to_pandas()

# 🎯 [버그 해결]: 꼬여있던 타이포 구문을 풀고 정상적인 분리형 넘파이 할당문으로 교정 완료
X_train = train_pd[feature_columns].values.astype(np.float32)
y_train = train_pd[target_column].values.astype(np.int32)
print(f" ➔  훈련셋 매트릭스 확보 완료: {len(train_pd):,}행")
del train_pd

# --- [STEP 3] 코어 엔진 트레이닝 ---
print(f"\n[RF 코어 피팅 개시] CPU 멀티 프로세싱 하이브리드 가동 중...")
rf_model = RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_leaf=4,
    class_weight='balanced', n_jobs=6, random_state=42, max_samples=4000000
)
rf_model.fit(X_train, y_train)

# --- [STEP 4] 검증 리포트 및 성능 곡선 플로팅 ---
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test) 

print(f"\n[검증 결과 분석 - RF_{model_key.upper()}]")
print(classification_report(y_test, y_pred, target_names=["안전(0)", "주의(1)", "위험(2)"]))

# 대시보드 시각화 파일 내보내기
plot_evaluation_curves(y_test, y_prob, model_key)

# 직렬화 저장
output_model_path = os.path.join(MODEL_DIR, f'rf_{model_key}_model.joblib')
joblib.dump(rf_model, output_model_path)
print(f"\n[완료] 하이브리드 아키텍처 모델 저장 완료 ➔ {output_model_path}")