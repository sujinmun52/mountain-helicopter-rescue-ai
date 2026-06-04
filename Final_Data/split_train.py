# =====================================================================
# 🔪 train_set.csv 분할 도구 (4가지 방식 통합)
# ---------------------------------------------------------------------
# 사용법: BASE_DIR에 train_set.csv 두고 실행
# =====================================================================

import os
import numpy as np
import pandas as pd

# ✅ 경로 자동 인식
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📂 작업 폴더: {BASE_DIR}\n")

# 입력 파일
INPUT_FILE = os.path.join(BASE_DIR, 'train_set.csv')

# 출력 폴더 (분할본 따로 보관)
OUT_DIR = os.path.join(BASE_DIR, 'train_splits')
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# 0. 데이터 로드 + 기본 정보
# =====================================================================
print("=" * 70)
print("STEP 0. train_set.csv 로드")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ train_set.csv를 찾을 수 없습니다: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
print(f"  ✅ 로드 완료: {len(df):,}행 × {len(df.columns)}컬럼")
print(f"  📊 컬럼: {list(df.columns)}")
print(f"  📐 좌표 범위:")
print(f"     Longitude: {df['longitude'].min():.5f} ~ {df['longitude'].max():.5f}")
print(f"     Latitude : {df['latitude'].min():.5f} ~ {df['latitude'].max():.5f}")


# =====================================================================
# ① 행 수 기준 분할 (50만 행씩)
# =====================================================================
def split_by_rows(df, rows_per_file=500_000, prefix='train_rows'):
    """단순히 N행씩 잘라서 저장"""
    print("\n" + "=" * 70)
    print(f"① 행 수 기준 분할 ({rows_per_file:,}행씩)")
    print("=" * 70)

    n_files = int(np.ceil(len(df) / rows_per_file))
    saved = []
    for i in range(n_files):
        start = i * rows_per_file
        end   = min(start + rows_per_file, len(df))
        chunk = df.iloc[start:end]
        out   = os.path.join(OUT_DIR, f'{prefix}_{i+1:02d}.csv')
        chunk.to_csv(out, index=False)
        saved.append(out)
        print(f"  💾 {os.path.basename(out)}: {len(chunk):,}행")
    return saved


# =====================================================================
# ② Excel 한계 기준 분할 (1,000,000행 - 헤더 여유)
# =====================================================================
def split_for_excel(df, prefix='train_excel'):
    """Excel(1,048,576행)에서 열 수 있도록 100만 행씩 분할"""
    print("\n" + "=" * 70)
    print("② Excel 호환 분할 (1,000,000행씩)")
    print("=" * 70)
    return split_by_rows(df, rows_per_file=1_000_000, prefix=prefix)


# =====================================================================
# ③ 블록 ID 기준 분할 (공간적 의미 보존)
# =====================================================================
def split_by_block(df, grid_size=10, prefix='train_block'):
    """좌표를 격자로 나눠 블록별 CSV 생성 (행/열 인덱스 기반)"""
    print("\n" + "=" * 70)
    print(f"③ 블록 ID 기준 분할 ({grid_size}x{grid_size} 격자)")
    print("=" * 70)

    lon_edges = np.linspace(df['longitude'].min(), df['longitude'].max(), grid_size + 1)
    lat_edges = np.linspace(df['latitude'].min(),  df['latitude'].max(),  grid_size + 1)

    df = df.copy()
    df['col_idx'] = np.clip(np.digitize(df['longitude'], lon_edges) - 1, 0, grid_size - 1)
    df['row_idx'] = np.clip(np.digitize(df['latitude'],  lat_edges) - 1, 0, grid_size - 1)
    df['block_id'] = df['row_idx'].astype(str) + '_' + df['col_idx'].astype(str)

    # 블록 단위로 그룹화 후 저장
    saved = []
    for block_id, group in df.groupby('block_id'):
        out = os.path.join(OUT_DIR, f'{prefix}_r{group["row_idx"].iloc[0]:02d}_c{group["col_idx"].iloc[0]:02d}.csv')
        group.drop(columns=['col_idx', 'row_idx', 'block_id']).to_csv(out, index=False)
        saved.append(out)
    print(f"  💾 총 {len(saved)}개 블록 파일 저장 → {OUT_DIR}")
    print(f"  📊 블록당 평균 {len(df)//len(saved):,}행")
    return saved


# =====================================================================
# ④ 권역 기준 분할 (NW / NE / Central / S) — 논문용 4-Fold CV
# =====================================================================
def split_by_region(df, prefix='train_region'):
    """설악산을 4개 권역으로 분할 (4-Fold Spatial CV용)"""
    print("\n" + "=" * 70)
    print("④ 권역 기준 분할 (NW / NE / Central / S)")
    print("=" * 70)

    lon_mid = (df['longitude'].min() + df['longitude'].max()) / 2
    lat_mid = (df['latitude'].min()  + df['latitude'].max())  / 2

    print(f"  📐 분할 기준점: longitude={lon_mid:.5f}, latitude={lat_mid:.5f}")

    # 권역 정의 (설악산 지리적 특성 반영)
    regions = {
        'NW_Baekdamsa' : (df['longitude'] <  lon_mid) & (df['latitude'] >= lat_mid),
        'NE_Ulsanbawi' : (df['longitude'] >= lon_mid) & (df['latitude'] >= lat_mid),
        'SW_Central'   : (df['longitude'] <  lon_mid) & (df['latitude'] <  lat_mid),
        'SE_Osaek'     : (df['longitude'] >= lon_mid) & (df['latitude'] <  lat_mid),
    }

    saved = []
    for region_name, mask in regions.items():
        sub = df[mask]
        out = os.path.join(OUT_DIR, f'{prefix}_{region_name}.csv')
        sub.to_csv(out, index=False)
        saved.append(out)
        print(f"  💾 {region_name:18s}: {len(sub):>8,}행  ({len(sub)/len(df)*100:>5.1f}%)")
    return saved


# =====================================================================
# 실행
# =====================================================================
if __name__ == "__main__":

    # 🎯 사용할 방식을 True/False로 선택하세요
    USE_METHOD_1 = False  # 행 수 기준 (50만씩)
    USE_METHOD_2 = True   # Excel 호환 (100만씩) ← 사용자 요청
    USE_METHOD_3 = False  # 블록 ID 기준
    USE_METHOD_4 = True   # 권역 기준 (4-Fold CV용)

    if USE_METHOD_1: split_by_rows(df, rows_per_file=500_000)
    if USE_METHOD_2: split_for_excel(df)
    if USE_METHOD_3: split_by_block(df, grid_size=10)
    if USE_METHOD_4: split_by_region(df)

    print("\n" + "=" * 70)
    print(f"🎉 완료! 결과 폴더: {OUT_DIR}")
    print("=" * 70)
