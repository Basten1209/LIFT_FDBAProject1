import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("최종_성과_비교.csv")

# 스타일 설정
sns.set(style="whitegrid")
plt.rcParams.update({
    "axes.unicode_minus": False,  # 음수 깨짐 방지
    "figure.dpi": 120
})

# === 지표 구분 ===
line_metrics = ["Annualized Risk (%)", "Annualized Return (%)", "Sharpe Ratio"]
bar_metrics = ["Frobenius", "KL Divergence", "Risk Gap"]

# -------------------------------------------------------------
# 📈 1️⃣ 라인그래프 (Risk / Return / Sharpe)
# -------------------------------------------------------------
for metric in line_metrics:
    g = sns.relplot(
        data=df,
        x="Gross Exposure",
        y=metric,
        hue="Model",
        col="Year",
        kind="line",
        marker="o",
        linewidth=2.5,
        height=5,
        aspect=0.7,
        facet_kws={'sharey': False}
    )

    # 제목 설정
    g.fig.suptitle(
        f"{metric} by Gross Exposure Constraint (Yearly)",
        fontsize=16,
        fontweight="bold"
    )

    g.set_axis_labels("Gross Exposure Limit", metric)
    g.set_titles("Year: {col_name}")

    # 제목 안잘리게 여백 조정
    g.fig.subplots_adjust(top=0.88, wspace=0.25)

    plt.show()

# -------------------------------------------------------------
# 📊 2️⃣ 막대그래프 (Frobenius / KL Divergence / Risk Gap)
# -------------------------------------------------------------
# Gross Exposure 평균 계산
summary = (
    df.groupby(["Year", "Model"])[bar_metrics]
    .mean()
    .reset_index()
)

for metric in bar_metrics:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=summary,
        x="Year",
        y=metric,
        hue="Model",
        palette="Set2",
        edgecolor="black"
    )

    plt.title(
        f"{metric} (Mean across Gross Exposure Levels)",
        fontsize=16,
        fontweight="bold",
        pad=15
    )
    plt.xlabel("Year")
    plt.ylabel(f"Average {metric}")
    plt.legend(title="Model", loc="upper right", frameon=True)
    plt.tight_layout()
    plt.show()
