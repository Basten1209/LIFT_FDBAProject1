import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import entropy
from scipy.linalg import eigh
from collections import Counter
from tqdm import tqdm

plt.style.use("seaborn-v0_8-whitegrid")

# === 0. Load covariance data ===
file_map = {
    "Integrated": "daily_integrated_model_covariances_2022_2024.pkl",
    "LASSO-only": "daily_lasso_only_covariances_2022_2024.pkl",
    "POET-only": "daily_poet_only_covariances_2022_2024.pkl",
    "OLS (Shrinkage)": "daily_ols_shrinkage_covariances_2022_2024.pkl",
}

model_covs = {}
for name, path in file_map.items():
    with open(path, "rb") as f:
        model_covs[name] = pickle.load(f)
    print(f"✅ Loaded {name}: {len(model_covs[name])} daily matrices")

years = [2022, 2023, 2024]


# === 1️⃣ Cumulative Variance Explained (CVE) ===
print("\n[1] Cumulative Variance Explained (CVE)")

for year in years:
    plt.figure(figsize=(8, 6))
    for model, cov_dict in model_covs.items():
        yearly_covs = [cov for d, cov in cov_dict.items() if d.year == year]
        if not yearly_covs:
            continue

        # 평균 공분산 계산
        common_assets = list(set.intersection(*[set(cov.columns) for cov in yearly_covs]))
        if len(common_assets) < 5:
            continue
        avg_cov = np.mean([cov.loc[common_assets, common_assets].values for cov in yearly_covs], axis=0)
        eigvals = np.sort(np.linalg.eigvalsh(avg_cov))[::-1]
        eigvals = eigvals / eigvals.sum()  # normalize
        cve = np.cumsum(eigvals)

        plt.plot(np.arange(1, len(cve)+1), cve, label=f"{model} (n={len(common_assets)})")

    plt.title(f"Cumulative Variance Explained ({year})", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Factors")
    plt.ylabel("Cumulative Variance Explained")
    plt.legend()
    plt.grid(True)
    plt.show()


# === 2️⃣ Rolling Stability Test (Frobenius distance) ===
print("\n[2] Rolling Covariance Stability Test")

window = 20
for model, cov_dict in model_covs.items():
    fro_dists = []
    dates = sorted(cov_dict.keys())
    for i in range(window, len(dates)):
        cov_prev = cov_dict[dates[i-window]]
        cov_curr = cov_dict[dates[i]]
        common_assets = cov_prev.columns.intersection(cov_curr.columns)
        if len(common_assets) < 5:
            continue
        c1 = cov_prev.loc[common_assets, common_assets].values
        c2 = cov_curr.loc[common_assets, common_assets].values
        dist = np.linalg.norm(c1 - c2, "fro")
        fro_dists.append(dist)
    plt.plot(fro_dists, label=model)

plt.title("Rolling Frobenius Distance (20-day window)", fontsize=14, fontweight="bold")
plt.xlabel("Time Step")
plt.ylabel("Distance")
plt.legend()
plt.show()


# === 3️⃣ Principal Portfolio Analysis ===
print("\n[3] Principal Portfolio Analysis (Top Eigenvectors)")
example_year = 2023
for model, cov_dict in model_covs.items():
    yearly_covs = [cov for d, cov in cov_dict.items() if d.year == example_year]
    if not yearly_covs:
        continue
    common_assets = list(set.intersection(*[set(cov.columns) for cov in yearly_covs]))
    avg_cov = np.mean([cov.loc[common_assets, common_assets].values for cov in yearly_covs], axis=0)
    eigvals, eigvecs = np.linalg.eigh(avg_cov)
    eigvecs = eigvecs[:, -3:]  # 상위 3개 요인

    # === 상위 20개 자산만 시각화 (기여도 높은 순)
    # 각 자산별로 3개 요인의 절댓값 합 계산 → 가장 많이 기여한 자산만 표시
    importance = np.sum(np.abs(eigvecs), axis=1)
    top_idx = np.argsort(importance)[-20:][::-1]

    top_assets = [a[:6] for a in np.array(common_assets)[top_idx]]  # 6글자만 표시
    eigvecs_top = eigvecs[top_idx, :]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        np.abs(eigvecs_top),
        cmap="viridis",
        annot=False,
        yticklabels=top_assets,
        xticklabels=["Factor 1", "Factor 2", "Factor 3"]
    )
    plt.title(f"{model}: Top 3 Eigenvector Loadings (2023) — Top 20 assets", fontsize=13, fontweight="bold")
    plt.xlabel("Principal Factors")
    plt.ylabel("Assets (Top 20 by loading magnitude)")
    plt.tight_layout()
    plt.show()


# === 4️⃣ Shrinkage Intensity Comparison (trace of Σ) ===
print("\n[4] Shrinkage Intensity Comparison")

trace_df = []
for model, cov_dict in model_covs.items():
    for d, cov in cov_dict.items():
        trace_df.append({"Model": model, "Date": d, "Trace": np.trace(cov.values)})

trace_df = pd.DataFrame(trace_df)
sns.boxplot(data=trace_df, x="Model", y="Trace")
plt.title("Trace(Σ) Distribution Comparison (Variance Level)")
plt.ylabel("Total Variance (tr(Σ))")
plt.show()


# === 5️⃣ Stress Regime Analysis (using simple volatility proxy) ===
print("\n[5] Stress Regime Analysis")

vol_threshold = trace_df["Trace"].quantile(0.9)
trace_df["Regime"] = np.where(trace_df["Trace"] > vol_threshold, "Stress", "Normal")

sns.boxplot(data=trace_df, x="Model", y="Trace", hue="Regime")
plt.title("Variance under Stress vs Normal Regimes")
plt.show()


# === 7️⃣ Entropy-based Information Efficiency ===
print("\n[7] Entropy-based Information Efficiency")

entropy_results = []
for model, cov_dict in model_covs.items():
    for d, cov in cov_dict.items():
        eigvals = np.linalg.eigvalsh(cov.values)
        eigvals = np.clip(eigvals, 1e-10, None)
        H = 0.5 * np.log((2 * np.pi * np.e) ** len(eigvals) * np.prod(eigvals))
        entropy_results.append({"Model": model, "Entropy": H})

entropy_df = pd.DataFrame(entropy_results)
sns.boxplot(data=entropy_df, x="Model", y="Entropy")
plt.title("Differential Entropy Comparison")
plt.show()

print("\n✅ All analyses complete.")
