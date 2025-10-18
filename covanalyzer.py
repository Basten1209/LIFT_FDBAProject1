import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from collections import Counter
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform

# ======================================================
# 1️⃣ Step 1: Load Covariance Data
# ======================================================

file_paths = {
    'Integrated': 'daily_integrated_model_covariances_2022_2024.pkl',
    'LASSO-only': 'daily_lasso_only_covariances_2022_2024.pkl',
    'POET-only': 'daily_poet_only_covariances_2022_2024.pkl',
    'OLS (Shrinkage)': 'daily_ols_shrinkage_covariances_2022_2024.pkl'
}

model_covariances = {}
for name, path in file_paths.items():
    with open(path, 'rb') as f:
        model_covariances[name] = pickle.load(f)
    print(f"✅ Loaded: {name} ({len(model_covariances[name])} days)")


# ======================================================
# 2️⃣ Step 2: Eigenvalue Spectrum by Year (Fixed)
# ======================================================

years = [2022, 2023, 2024]
plt.style.use('seaborn-v0_8-whitegrid')

for year in years:
    plt.figure(figsize=(9,6))

    for model_name, cov_dict in model_covariances.items():
        yearly_covs = [cov for date, cov in cov_dict.items() if date.year == year]
        if not yearly_covs:
            continue

        # --- Integrated 모델만: 가장 많이 등장한 상위 50개 자산 기준 ---
        if model_name == 'Integrated':
            asset_counts = Counter([asset for cov in yearly_covs for asset in cov.columns])
            top_assets = [a for a, _ in asset_counts.most_common(50)]
            aligned_covs = [
                cov.reindex(index=top_assets, columns=top_assets, fill_value=0).values
                for cov in yearly_covs
            ]

        # --- 나머지 모델: 공통 자산 교집합 기준 ---
        else:
            common_assets = set.intersection(*[set(cov.index) for cov in yearly_covs])
            if len(common_assets) < 2:
                continue
            aligned_covs = [cov.loc[list(common_assets), list(common_assets)].values for cov in yearly_covs]

        # 평균 공분산 행렬
        avg_cov = np.mean(np.stack(aligned_covs), axis=0)

        # 고유값 스펙트럼 계산
        eigvals = np.linalg.eigvalsh(avg_cov)
        plt.plot(
            sorted(eigvals, reverse=True),
            label=f"{model_name} ({'Top 50' if model_name=='Integrated' else f'n={len(common_assets)}'})"
        )

    # === Plot Formatting ===
    plt.title(f"Eigenvalue Spectrum Comparison ({year})", fontsize=14, fontweight='bold')
    plt.xlabel("Eigenvalue Rank")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ======================================================
# 3️⃣ Step 3: Frobenius / KL / Risk Gap Stability by Year
# ======================================================

def calculate_frobenius(A, B):
    return np.linalg.norm(A - B, 'fro') / A.shape[0]

def calculate_kl_divergence(A, B):
    # PSD 변환 + 안정화
    eigvals_a, eigvecs_a = np.linalg.eigh(A)
    eigvals_b, eigvecs_b = np.linalg.eigh(B)
    eigvals_a[eigvals_a < 1e-8] = 1e-8
    eigvals_b[eigvals_b < 1e-8] = 1e-8
    A_psd = eigvecs_a @ np.diag(eigvals_a) @ eigvecs_a.T
    B_psd = eigvecs_b @ np.diag(eigvals_b) @ eigvecs_b.T
    inv_B = np.linalg.inv(B_psd)
    term1 = np.trace(inv_B @ A_psd)
    logdet_ratio = np.linalg.slogdet(B_psd)[1] - np.linalg.slogdet(A_psd)[1]
    kl = 0.5 * (term1 - A.shape[0] + logdet_ratio)
    return kl

def calculate_risk_gap(realized, forecast, w=None):
    if w is None:
        w = np.ones(realized.shape[0]) / realized.shape[0]
    ex_ante = np.sqrt(w.T @ forecast @ w)
    ex_post = np.sqrt(w.T @ realized @ w)
    return abs(ex_post - ex_ante) / (ex_post + 1e-8)

# 실제로는 rolling pair로 근사 (t vs t+1)
stability_records = []

for model_name, cov_dict in tqdm(model_covariances.items(), desc="Computing stability"):
    sorted_dates = sorted(cov_dict.keys())
    for i in range(1, len(sorted_dates)):
        d1, d2 = sorted_dates[i-1], sorted_dates[i]
        if d1.year != d2.year:  # 연도별 분리
            continue
        cov1, cov2 = cov_dict[d1].values, cov_dict[d2].values
        if cov1.shape != cov2.shape:
            continue
        frob = calculate_frobenius(cov1, cov2)
        kl = calculate_kl_divergence(cov1, cov2)
        rg = calculate_risk_gap(cov1, cov2)
        stability_records.append({'Model': model_name, 'Year': d2.year,
                                  'Frobenius': frob, 'KL Divergence': kl, 'Risk Gap': rg})

stability_df = pd.DataFrame(stability_records)
summary_stability = stability_df.groupby(['Model', 'Year']).agg(['mean', 'std'])
print("\n--- Stability Summary ---")
print(summary_stability.round(6))

# 시각화 (Boxplot)
for metric in ['Frobenius', 'KL Divergence', 'Risk Gap']:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=stability_df, x='Year', y=metric, hue='Model')
    plt.title(f'{metric} Distribution by Year', fontsize=13, fontweight='bold')
    plt.ylabel(metric)
    plt.xlabel('Year')
    plt.legend()
    plt.grid(True)
    plt.show()


# ======================================================
# 4️⃣ Step 4: Metric Correlation Analysis
# ======================================================

corr_summary = stability_df.groupby('Model')[['Frobenius', 'KL Divergence', 'Risk Gap']].mean().corr()
plt.figure(figsize=(5,4))
sns.heatmap(corr_summary, annot=True, cmap='coolwarm', fmt=".3f")
plt.title("Correlation among Predictive Metrics (Mean over Years)", fontsize=13, fontweight='bold')
plt.show()


# ======================================================
# 5️⃣ Step 5: Covariance Heatmap (각 모델별 + 자산명 6글자 제한)
# ======================================================

example_year = 2023
example_date = random.choice([d for d in model_covariances['Integrated'].keys() if d.year == example_year])

for model_name, cov_dict in model_covariances.items():
    cov_df = cov_dict[example_date]
    assets = cov_df.columns.tolist()

    # 자산 이름을 최대 6글자로 제한 (예: "NASDAQ100" → "NASDA…")
    truncated_names = [
        name[:6] + "…" if len(name) > 6 else name for name in assets
    ]
    cov_df.columns = truncated_names
    cov_df.index = truncated_names

    # 자산이 많으면 일부만 샘플링
    if len(assets) > 30:
        selected_assets = np.random.choice(truncated_names, size=30, replace=False)
        cov_df = cov_df.loc[selected_assets, selected_assets]

    # Cluster-based reordering
    try:
        linkage_matrix = linkage(cov_df, method='ward')
        ordered_idx = leaves_list(linkage_matrix)
        cov_df = cov_df.iloc[ordered_idx, ordered_idx]
    except Exception:
        pass

    # === 시각화 ===
    plt.figure(figsize=(7, 6))
    sns.heatmap(cov_df, cmap='coolwarm', center=0, cbar=True)
    plt.title(f'{model_name} Covariance Structure ({example_date.date()})', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

# ======================================================
# 6️⃣ Step 6: Correlation Structure & Cluster Dendrogram
# ======================================================

example_year = 2023
example_date = random.choice([d for d in model_covariances['Integrated'].keys() if d.year == example_year])

for model_name, cov_dict in model_covariances.items():
    cov_df = cov_dict[example_date]
    assets = cov_df.columns.tolist()

    # --- 자산 이름 Truncate (최대 6글자) ---
    truncated_names = [name[:6] + "…" if len(name) > 6 else name for name in assets]
    cov_df.columns = truncated_names
    cov_df.index = truncated_names

    # --- 자산이 많을 경우 일부 샘플링 ---
    if len(assets) > 30:
        selected_assets = np.random.choice(truncated_names, size=30, replace=False)
        cov_df = cov_df.loc[selected_assets, selected_assets]

    # --- 1️⃣ Correlation Matrix 계산 ---
    diag_std = np.sqrt(np.diag(cov_df))
    corr_df = cov_df / np.outer(diag_std, diag_std)
    corr_df = pd.DataFrame(corr_df, index=cov_df.index, columns=cov_df.columns)
    corr_df = corr_df.clip(-1, 1)  # 수치 안정화

    # --- 2️⃣ Cluster Linkage 계산 ---
    # corr_df는 -1~1 범위의 similarity이므로 distance로 바꿔야 함
    dist_matrix = 1 - corr_df
    condensed_dist = squareform(dist_matrix, checks=False)  # ✅ condensed로 변환
    linkage_matrix = linkage(condensed_dist, method='ward')

    # === (A) Correlation Heatmap ===
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_df, cmap='coolwarm', center=0, cbar=True)
    plt.title(f'{model_name} Correlation Structure ({example_date.date()})', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

    # === (B) Cluster Dendrogram ===
    plt.figure(figsize=(8, 5))
    dendrogram(
        linkage_matrix,
        labels=corr_df.columns,
        leaf_rotation=45,
        leaf_font_size=8,
        color_threshold=0.7 * max(linkage_matrix[:, 2])
    )
    plt.title(f'{model_name} Asset Cluster Dendrogram ({example_date.date()})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
