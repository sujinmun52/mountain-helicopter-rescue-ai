# =====================================================================
# 🏔️ 설악산 데이터 통합 파이프라인 (병합 + Spatial Block Split)
# ---------------------------------------------------------------------
# [STEP 1] 3개 CSV 안전 병합  (스키마 검증 + 중복 좌표 제거)
# [STEP 2] 격자(Grid) 기반 Spatial Block Splitter
# [STEP 3] KDTree 누수 진단 (Test→Train 최근접 거리)
# [STEP 4] Train/Test 산점도 + 블록 컬러맵 시각화
# [STEP 5] 결과 CSV 저장
#
# ✅ 경로 자동 인식: 스크립트 파일이 있는 폴더를 기준으로 동작
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# [STEP 1] CSV 병합 함수
# =====================================================================
def merge_csv_files(file_paths, dedup_keys=('longitude', 'latitude'),
                    verbose=True):
    """
    분할된 CSV 파일들을 하나의 DataFrame으로 안전하게 병합한다.

    Parameters
    ----------
    file_paths : list[str]
        병합할 CSV 파일 경로 리스트
    dedup_keys : tuple
        중복 제거 기준 컬럼 (기본: 위경도)
    verbose : bool
        진행 상황 로그 출력 여부

    Returns
    -------
    pd.DataFrame
        병합 및 정제된 데이터프레임
    """
    if not file_paths:
        raise ValueError("file_paths가 비어 있습니다.")

    dfs = []
    for i, path in enumerate(file_paths, 1):
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일을 찾을 수 없음: {path}")
        d = pd.read_csv(path)
        if verbose:
            print(f"  [{i}/{len(file_paths)}] {os.path.basename(path):55s} | shape={d.shape}")
        dfs.append(d)

    # 컬럼 스키마 일치 여부 검증
    base_cols = set(dfs[0].columns)
    for i, d in enumerate(dfs[1:], 2):
        if set(d.columns) != base_cols:
            diff = set(d.columns).symmetric_difference(base_cols)
            raise ValueError(f"파일 {i}의 컬럼 스키마가 다릅니다. 차이: {diff}")

    # 행 단위 concat
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    before = len(merged)

    # 결측 좌표 제거
    merged = merged.dropna(subset=list(dedup_keys)).reset_index(drop=True)

    # 위경도 중복 제거 (인접 부분 파일에서 같은 셀이 중복될 가능성 차단)
    merged = merged.drop_duplicates(subset=list(dedup_keys)).reset_index(drop=True)
    after = len(merged)

    if verbose:
        print(f"\n  ✅ 병합 완료: {before:,}행 → 중복/결측 제거 후 {after:,}행 "
              f"({before - after:,}개 제거)")
        print(f"  📊 컬럼({len(merged.columns)}): {list(merged.columns)}")
        print(f"  📐 좌표 범위:")
        print(f"     Longitude: {merged['longitude'].min():.5f} ~ {merged['longitude'].max():.5f}")
        print(f"     Latitude : {merged['latitude'].min():.5f} ~ {merged['latitude'].max():.5f}")

    return merged


# =====================================================================
# [STEP 2] Spatial Block Splitter 클래스
# =====================================================================
class SpatialBlockSplitter:
    """
    격자(Grid) 기반 공간 블록 분리기.

    Parameters
    ----------
    grid_size : int
        격자의 한 변 블록 수 (예: 10 → 최대 100 블록)
    test_ratio : float
        Test 블록 비율 (0~1)
    buffer_deg : float
        Train-Test 경계 버퍼 거리 (degree 단위, 0.005 ≈ 약 500m)
    random_state : int
        블록 셔플 시드
    """

    def __init__(self, grid_size=10, test_ratio=0.20,
                 buffer_deg=0.005, random_state=42):
        self.grid_size    = grid_size
        self.test_ratio   = test_ratio
        self.buffer_deg   = buffer_deg
        self.random_state = random_state
        # fit 후 채워질 속성들
        self.lon_edges_ = self.lat_edges_ = None
        self.lon_min_ = self.lon_max_ = None
        self.lat_min_ = self.lat_max_ = None
        self.train_blocks_ = self.test_blocks_ = None

    # -----------------------------------------------------------------
    def _assign_block_id(self, df):
        """각 좌표를 (row, col) 블록 ID로 할당"""
        col_idx = np.clip(np.digitize(df['longitude'], self.lon_edges_) - 1,
                          0, self.grid_size - 1)
        row_idx = np.clip(np.digitize(df['latitude'],  self.lat_edges_) - 1,
                          0, self.grid_size - 1)
        return list(zip(row_idx, col_idx))

    # -----------------------------------------------------------------
    def _in_buffer_mask(self, sub_df, test_block_set):
        """Train 좌표 중 Test 블록과 buffer_deg 이내 인접한 점 식별"""
        if self.buffer_deg <= 0:
            return np.zeros(len(sub_df), dtype=bool)
        in_buf = np.zeros(len(sub_df), dtype=bool)
        lons = sub_df['longitude'].values
        lats = sub_df['latitude'].values
        for (r, c) in test_block_set:
            lon_lo = self.lon_edges_[c]   - self.buffer_deg
            lon_hi = self.lon_edges_[c+1] + self.buffer_deg
            lat_lo = self.lat_edges_[r]   - self.buffer_deg
            lat_hi = self.lat_edges_[r+1] + self.buffer_deg
            in_buf |= ((lons >= lon_lo) & (lons <= lon_hi) &
                       (lats >= lat_lo) & (lats <= lat_hi))
        return in_buf

    # -----------------------------------------------------------------
    def fit_transform(self, df):
        """격자 적합 + Train/Test/Buffer 분리"""
        # (1) Bounding box 계산
        self.lon_min_, self.lon_max_ = df['longitude'].min(), df['longitude'].max()
        self.lat_min_, self.lat_max_ = df['latitude'].min(),  df['latitude'].max()

        # (2) 격자 경계선 생성
        self.lon_edges_ = np.linspace(self.lon_min_, self.lon_max_, self.grid_size + 1)
        self.lat_edges_ = np.linspace(self.lat_min_, self.lat_max_, self.grid_size + 1)

        # (3) 블록 ID 할당
        df = df.copy()
        df['block_id'] = self._assign_block_id(df)

        # (4) 블록 단위 무작위 셔플 → Test 블록 선정
        unique_blocks = list(set(df['block_id']))
        n_test  = max(1, int(round(len(unique_blocks) * self.test_ratio)))
        rng     = np.random.default_rng(self.random_state)
        shuffle = rng.permutation(len(unique_blocks))
        self.test_blocks_  = set(unique_blocks[i] for i in shuffle[:n_test])
        self.train_blocks_ = set(unique_blocks[i] for i in shuffle[n_test:])

        # (5) Train/Test 분리
        is_test  = df['block_id'].isin(self.test_blocks_)
        train_df = df[~is_test].copy()
        test_df  = df[is_test].copy()

        # (6) Buffer 적용 (경계 누수 차단)
        in_buf    = self._in_buffer_mask(train_df, self.test_blocks_)
        buffer_df = train_df[in_buf].copy()
        train_df  = train_df[~in_buf].copy()

        # (7) 로그 출력
        print(f"\n  🗂️  Grid: {self.grid_size}x{self.grid_size} "
              f"(활성 블록 {len(unique_blocks)}개)")
        print(f"  🎯 Test 블록 : {len(self.test_blocks_):>3d}개")
        print(f"  🎯 Train 블록: {len(self.train_blocks_):>3d}개")
        print(f"  📈 Train 샘플: {len(train_df):>9,}")
        print(f"  📉 Test  샘플: {len(test_df):>9,}")
        print(f"  🚧 Buffer 제외: {len(buffer_df):>9,} (buffer_deg={self.buffer_deg})")
        return train_df, test_df, buffer_df

    # -----------------------------------------------------------------
    def diagnose_leakage(self, train_df, test_df):
        """Test→Train 최근접 거리 분포 진단"""
        tree = cKDTree(train_df[['longitude', 'latitude']].values)
        dists, _ = tree.query(test_df[['longitude', 'latitude']].values, k=1)
        dists_m  = dists * 111_000  # degree → meter 근사 (위도 기준)
        grade = ("✅ 등급 A (우수)" if dists_m.min() >= 500 else
                 "🟡 등급 B (양호)" if dists_m.min() >= 100 else
                 "❌ 등급 C (누수 의심)")
        print(f"\n  [누수 진단] Test→Train 최근접 거리")
        print(f"    min  = {dists_m.min():>8.1f} m")
        print(f"    p5   = {np.percentile(dists_m, 5):>8.1f} m")
        print(f"    med  = {np.median(dists_m):>8.1f} m")
        print(f"    mean = {dists_m.mean():>8.1f} m")
        print(f"    100m 이내 비율 = {(dists_m < 100).mean()*100:.2f}%")
        print(f"    {grade}")
        return dists_m

    # -----------------------------------------------------------------
    def plot(self, train_df, test_df, buffer_df=None,
             sample_size=15000, save_path=None):
        """Train/Test 산점도 + 블록 컬러맵"""
        def _sample(d, n):
            return d.sample(n=n, random_state=0) if len(d) > n else d
        tr = _sample(train_df, sample_size)
        te = _sample(test_df,  sample_size // 3)
        bf = _sample(buffer_df, sample_size // 5) \
             if buffer_df is not None and len(buffer_df) else None

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # (a) 산점도 + 격자
        ax = axes[0]
        ax.scatter(tr['longitude'], tr['latitude'], s=2, c='#2E86DE',
                   alpha=0.45, label=f'Train (n={len(train_df):,})')
        ax.scatter(te['longitude'], te['latitude'], s=2, c='#EE5253',
                   alpha=0.65, label=f'Test (n={len(test_df):,})')
        if bf is not None:
            ax.scatter(bf['longitude'], bf['latitude'], s=2, c='#A4B0BE',
                       alpha=0.5, label=f'Buffer excluded (n={len(buffer_df):,})')
        for x in self.lon_edges_: ax.axvline(x, color='black', lw=0.4, alpha=0.4)
        for y in self.lat_edges_: ax.axhline(y, color='black', lw=0.4, alpha=0.4)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title(f'Spatial Block Split (grid {self.grid_size}x{self.grid_size}, '
                     f'test={self.test_ratio:.0%}, buffer={self.buffer_deg} deg)',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', markerscale=4)
        ax.set_aspect('equal', adjustable='datalim')

        # (b) 블록 컬러맵
        ax2 = axes[1]
        for (r, c) in self.train_blocks_:
            ax2.add_patch(Rectangle(
                (self.lon_edges_[c], self.lat_edges_[r]),
                self.lon_edges_[c+1] - self.lon_edges_[c],
                self.lat_edges_[r+1] - self.lat_edges_[r],
                facecolor='#2E86DE', alpha=0.5, edgecolor='white', lw=0.5))
        for (r, c) in self.test_blocks_:
            ax2.add_patch(Rectangle(
                (self.lon_edges_[c], self.lat_edges_[r]),
                self.lon_edges_[c+1] - self.lon_edges_[c],
                self.lat_edges_[r+1] - self.lat_edges_[r],
                facecolor='#EE5253', alpha=0.75, edgecolor='white', lw=0.5))
        ax2.set_xlim(self.lon_min_, self.lon_max_)
        ax2.set_ylim(self.lat_min_, self.lat_max_)
        ax2.set_xlabel('Longitude'); ax2.set_ylabel('Latitude')
        ax2.set_title('Block-level Assignment Map', fontsize=13, fontweight='bold')
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.legend(handles=[
            Patch(facecolor='#2E86DE', alpha=0.5, label='Train block'),
            Patch(facecolor='#EE5253', alpha=0.75, label='Test block')],
            loc='upper right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=140, bbox_inches='tight')
            print(f"  💾 시각화 저장: {save_path}")
        plt.show()
        return fig


# =====================================================================
# [STEP 3] 메인 파이프라인
# =====================================================================
if __name__ == "__main__":

    # =================================================================
    # ✅ 경로 자동 인식: 스크립트 파일이 있는 폴더를 기준으로 동작
    # =================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"📂 작업 폴더(BASE_DIR): {BASE_DIR}\n")

    # ---------- (1) CSV 병합 ----------
    print("=" * 70)
    print("STEP 1. 3개 CSV 파일 병합")
    print("=" * 70)

    csv_files = [
        os.path.join(BASE_DIR, 'Final_seoraksan_part1_v2_sujin_260602.csv'),
        os.path.join(BASE_DIR, 'Final_seoraksan_part2_v2_sujin_260602.csv'),
        os.path.join(BASE_DIR, 'Final_seoraksan_part3_v2_sujin_260602.csv'),
    ]

    # 사전 점검 — 어디서 실행해도 친절하게 알려줌
    missing = [p for p in csv_files if not os.path.exists(p)]
    if missing:
        print("\n❌ 다음 파일을 찾을 수 없습니다:")
        for p in missing:
            print(f"   - {p}")
        print(f"\n💡 위 3개 CSV를 다음 폴더에 두세요:\n   {BASE_DIR}")
        raise SystemExit(1)

    df = merge_csv_files(csv_files, verbose=True)

    # 병합 결과 즉시 저장 (재사용 편의)
    merged_path = os.path.join(BASE_DIR, 'merged_seoraksan.csv')
    df.to_csv(merged_path, index=False)
    print(f"\n  💾 병합본 저장: {merged_path} ({len(df):,}행)")
    print(f"\n  미리보기:")
    print(df.head().to_string(index=False))

    # ---------- (2) Spatial Block Split ----------
    print("\n" + "=" * 70)
    print("STEP 2. Spatial Block Split")
    print("=" * 70)

    splitter = SpatialBlockSplitter(
        grid_size    = 10,     # 10×10 = 최대 100 블록
        test_ratio   = 0.20,   # 20% 블록을 Test로
        buffer_deg   = 0.005,  # 약 500m 버퍼
        random_state = 42,
    )
    train_df, test_df, buffer_df = splitter.fit_transform(df)

    # ---------- (3) 누수 진단 ----------
    print("\n" + "=" * 70)
    print("STEP 3. 분리 품질 진단")
    print("=" * 70)
    splitter.diagnose_leakage(train_df, test_df)

    # ---------- (4) 시각화 ----------
    print("\n" + "=" * 70)
    print("STEP 4. 시각화")
    print("=" * 70)
    plot_path = os.path.join(BASE_DIR, 'spatial_block_split.png')
    splitter.plot(train_df, test_df, buffer_df,
                  sample_size=15000,
                  save_path=plot_path)

    # ---------- (5) 결과 저장 ----------
    train_path = os.path.join(BASE_DIR, 'train_set.csv')
    test_path  = os.path.join(BASE_DIR, 'test_set.csv')
    train_df.drop(columns=['block_id']).to_csv(train_path, index=False)
    test_df.drop(columns=['block_id']).to_csv(test_path,  index=False)
    print(f"\n  💾 저장 완료:")
    print(f"     - {train_path}")
    print(f"     - {test_path}")
    print("\n  🎉 전체 파이프라인 완료!")
