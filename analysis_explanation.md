# Analysis in resultmaker.py

**Purpose**

Evaluates covariance stability and forecast accuracy of each model across years.

**Metric Interpretation**

| Metric            | Meaning                                                               | Desirable Direction         |
| ----------------- | --------------------------------------------------------------------- | --------------------------- |
| **Frobenius**     | Average day-to-day change in covariance matrices                      | ↓ Lower = smoother dynamics |
| **KL Divergence** | Distributional difference between forecasted and realized covariances | ↓ Lower = more consistent   |
| **Risk Gap**      | Relative deviation between ex-ante and realized portfolio risk        | ↓ Lower = more accurate     |

**How to Read the Bar Charts**

Lower bars indicate stronger model robustness.

If a model consistently holds lower KL and Risk Gap values, it implies better forward-looking reliability.

Slightly higher Frobenius can mean adaptive reactivity (responding to market regime shifts).

| Category                     | Metric                     | Ideal Trend                     | Economic Meaning |
| ---------------------------- | -------------------------- | ------------------------------- | ---------------- |
| **Performance**              | Risk ↓, Return ↑, Sharpe ↑ | Efficient risk–return tradeoff  |                  |
| **Structural Stability**     | Frobenius ↓                | Smooth covariance evolution     |                  |
| **Distributional Stability** | KL ↓                       | Probabilistic robustness        |                  |
| **Forecast Accuracy**        | Risk Gap ↓                 | Precise ex-ante risk prediction |                  |


# Covariance Structure and Stability Diagnostics in covanalyzer1.py

This report analyzes the covariance structure and temporal stability of four model specifications — **Integrated**, **LASSO-only**, **POET-only**, and **OLS (Shrinkage)** — across the years 2022–2024.  
Each diagnostic step aims to evaluate complementary aspects of how reliable and interpretable the estimated covariance matrices are.

---

## 1️⃣ Data Loading

Each model’s daily covariance matrices are loaded from `.pkl` files covering 2022–2024.  
The data represent daily estimates of asset-level covariance structures under each modeling method.

> ✅ Expected output:  
> `✅ Loaded: Integrated (752 days)`  
> etc.

---

## 2️⃣ Eigenvalue Spectrum by Year

**Purpose:**  
To assess the **dimensionality** and **factor structure** of each model’s average covariance matrix for each year.

**Method:**  
- Compute eigenvalues of the average annual covariance matrix.  
- Sort them in descending order and plot the eigenvalue spectrum.

**Interpretation:**
- **Steeper drop-off** in the spectrum → few dominant factors explain most variance (clear structure).  
- **Flatter tail** → more diffuse or noisy covariance structure.  
- **Integrated (Top 50)** uses the 50 most frequently observed assets to ensure comparability.  
- Other models use the intersection of common assets per year.

**Desirable pattern:**  
A model that captures market-wide risk efficiently will exhibit a spectrum where the first few eigenvalues are large and quickly decay to zero.

---

## 3️⃣ Frobenius, KL-Divergence, and Risk-Gap Stability

**Purpose:**  
To evaluate **temporal stability** and **forecast consistency** of covariance estimates on a day-to-day basis.

### • Frobenius Distance

$$
D_F = \frac{\|\Sigma_t - \Sigma_{t-1}\|_F}{n}
$$

- Measures the absolute structural change between consecutive covariances.  
- Smaller values → smoother and more stable estimates.

### • Kullback–Leibler (KL) Divergence

$$
D_{\text{KL}}(\Sigma_1 || \Sigma_2)
= \frac{1}{2}[\text{tr}(\Sigma_2^{-1}\Sigma_1) - n + \log(\frac{|\Sigma_2|}{|\Sigma_1|})]
$$

- 
- Quantifies how different two Gaussian covariance structures are.  
- Lower KL → closer probabilistic structure, hence higher stability.

### • Risk Gap

$$
RG = \frac{|\,\sqrt{w^\top \Sigma_{\text{real}} w} - \sqrt{w^\top \Sigma_{\text{forecast}} w}\,|}{\sqrt{w^\top \Sigma_{\text{real}} w}}
$$
- Measures the difference between ex-post and ex-ante portfolio risk.  
- Lower values → more accurate forward-looking risk prediction.

**Visualization:**  
Boxplots by year show the distributions of Frobenius, KL, and Risk Gap across models.

**Interpretation:**

| Metric | Lower Value Means | Interpretation |
|---------|------------------|----------------|
| Frobenius | Smaller day-to-day covariance drift | Temporal stability |
| KL Divergence | Higher similarity of covariance distributions | Probabilistic stability |
| Risk Gap | Ex-ante risk matches ex-post risk | Forecast accuracy |

**Desirable model:**  

- Consistently lower medians for all three metrics  
- Narrow interquartile range (IQR) indicating robustness.

---

## 4️⃣ Metric Correlation Analysis

**Purpose:**  
To explore how the three stability measures (Frobenius, KL, Risk Gap) relate to each other.

**Method:**  
Compute mean values per model and then their Pearson correlation matrix.

**Interpretation:**
- High positive correlation → metrics capture similar dynamics.  
- Low or negative correlation → each metric reflects different stability aspects (e.g., numerical vs. predictive).

**Desirable outcome:**  
Moderate correlation (0.3–0.6) suggests complementary diagnostic power rather than redundancy.

---

## 5️⃣ Covariance Structure Visualization

**Purpose:**  

To directly visualize each model’s covariance structure on a specific sample date (randomly drawn from 2023).

**Procedure:**

1. Select one representative date (e.g., 2023-04-05).  
2. Truncate long asset names to ≤ 6 characters for readability.  
3. Randomly sample 30 assets if too many are available.  
4. Apply hierarchical clustering (Ward linkage) to reorder rows/columns.  
5. Plot the covariance matrix as a heatmap.

**Interpretation:**

- **Strong diagonal dominance** → large idiosyncratic variances.  
- **Clear block patterns** → sectoral or group correlations captured.  
- **Noisy, patchy heatmap** → unstable estimation.  
- Comparing across models shows whether shrinkage or regularization sharpens the block structure.

**Desirable behavior:**

- Integrated and POET models often show clear block clusters.  
- OLS may appear noisier, while LASSO could be over-sparse.

---

## 6️⃣ Correlation Structure and Cluster Dendrogram

**Purpose:**  

To analyze the **relational structure** among assets implied by each covariance matrix.

**Steps:**

1. Convert covariance to correlation matrix:
   $$
   \rho_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}}
   $$
2. Transform similarity to distance:  
   $$d_{ij} = 1 - \rho_{ij}$$
3. Compute hierarchical linkage using Ward’s method.  
4. Visualize both the correlation heatmap and the dendrogram.

**Interpretation:**
- **Correlation Heatmap:**  
  - Red = strong positive relation, Blue = negative.  
  - Clustered red blocks → coherent groups (sectors, markets).
- **Dendrogram:**  
  - Height indicates dissimilarity between clusters.  
  - Branches joined at low heights = highly correlated groups.  
  - Separate tall branches = independent asset clusters.

**Desirable pattern:**
- Distinct clusters (short vertical joins within groups, tall joins between groups).  
- Integrated model should produce a **balanced hierarchical tree**, where major asset groups merge at moderate distances — showing structured but not over-compressed relationships.

---

## 7️⃣ Practical Reading Guide

| Diagnostic | Good Model Behavior | Analytical Meaning |
|-------------|--------------------|--------------------|
| **Eigenvalue Spectrum** | Rapid eigenvalue decay | Strong low-dimensional factor structure |
| **Frobenius Stability** | Small mean, low volatility | Smooth time evolution |
| **KL Divergence** | Low and consistent | Robust probabilistic structure |
| **Risk Gap** | Small | Accurate risk forecasts |
| **Covariance Heatmap** | Clear block patterns | Captures sectoral dependencies |
| **Correlation Matrix** | Strong but localized clusters | Realistic cross-asset structure |
| **Dendrogram** | Well-defined clusters with moderate separation | Balanced hierarchical relationships |

---



# Covariance Model Diagnostics and Interpretation in covanalyzer2.py

This report summarizes the results of multiple covariance model diagnostics — including factor structure, temporal stability, shrinkage level, and information efficiency — for **Integrated**, **LASSO-only**, **POET-only**, and **OLS (Shrinkage)** models.  
Each section below explains the metric, what indicates a desirable behavior, and how to interpret the corresponding plot.

---

## 1️⃣ Cumulative Variance Explained (CVE)

**Concept:**  
The cumulative proportion of total variance explained by the top eigenvalues of the covariance matrix.  
It shows how many factors are needed to summarize the main market structure.

$$
\text{CVE}(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^n \lambda_i}
$$

**What’s desirable:**
| Pattern | Interpretation |
|----------|----------------|
| CVE curve rises steeply (e.g., >90% within top 10–20 factors) | Strong factor structure, efficient risk representation |
| CVE increases slowly | Noise-dominated, weak factor structure |
| Integrated has steepest CVE curve | Combines submodels effectively, captures dominant risk sources |

**How to read the plot:**
- X-axis = number of principal factors  
- Y-axis = cumulative variance explained  
- Models that reach higher CVE with fewer factors are more efficient.

---

## 2️⃣ Rolling Covariance Stability (Frobenius Distance)

**Concept:**  
Measures temporal stability of covariance estimates using rolling Frobenius norms:

$$
d_t = \|\Sigma_t - \Sigma_{t-20}\|_F
$$

**What’s desirable:**

| Behavior | Interpretation |
|-----------|----------------|
| Smooth, low variation | Stable estimation across time |
| Spiky or volatile | Over-reactive or unstable covariance estimates |
| Integrated shows the lowest, smoothest pattern | Most time-consistent model |

**How to read the plot:**

- The Y-axis shows distance between adjacent windows (e.g., 20-day apart).  
- Smaller values = more consistent structure; abrupt spikes = structural breaks.

---

## 3️⃣ Principal Portfolio Analysis (Top Eigenvectors)

**Concept:**  
Visualizes the top 3 eigenvectors (principal factors) of the average covariance matrix.  
Each cell in the heatmap represents a factor loading magnitude for each asset.

**What’s desirable:**

| Pattern | Interpretation |
|----------|----------------|
| Distinct color blocks by asset groups | Strong sectoral or factor structure |
| Blurry / uniform colors | Weak or noisy factor structure |
| Balanced, interpretable loadings | Good multi-factor representation |

**How to read the plot:**

- Y-axis = top 20 assets contributing most to principal factors  
- X-axis = top 3 factors  
- Darker color = stronger contribution  
- Clear clustering across columns implies interpretable risk factors.

---

## 4️⃣ Shrinkage Intensity Comparison (Trace of Σ)

**Concept:**  
The trace of the covariance matrix (sum of variances) indicates the overall volatility level.

$$
\text{Trace}(\Sigma) = \sum_i \sigma_i^2
$$

**What’s desirable:**

| Behavior | Interpretation |
|-----------|----------------|
| Trace too low | Over-shrinkage → variance underestimation |
| Trace too high | Overfitting or excessive noise |
| Stable median, small IQR | Consistent variance level |

**How to read the plot:**

- Boxplots compare total variance distributions across models.  
- Balanced trace with few outliers indicates robust estimation.

---

## 5️⃣ Stress Regime Analysis

**Concept:**  

Splits periods into *Normal* and *Stress* regimes using the 90th percentile of total variance.  
Compares how each model reacts to volatility surges.

**What’s desirable:**

| Pattern | Interpretation |
|----------|----------------|
| Trace rises during stress regimes | Model is sensitive to market volatility |
| Overreaction (Trace always high) | Instability or noise |
| Distinct separation between regimes | Proper regime responsiveness |

**How to read the plot:**

- Two distributions (Normal vs. Stress) per model.  
- Clear vertical separation = model differentiates regimes effectively.

---

## 7️⃣ Entropy-Based Information Efficiency

**Concept:**  
Differential entropy measures the information content of the Gaussian distribution defined by Σ:

$$
H = \frac{1}{2}\log\!\left[(2\pi e)^n \det(\Sigma)\right]
$$

- Higher entropy → richer, more diverse covariance structure  
- Lower entropy → over-shrunk, information-lossy estimates

**What’s desirable:**

| Behavior | Interpretation |
|-----------|----------------|
| Very low entropy | Over-regularized, under-dispersed model |
| Very high entropy | Noisy, unstable structure |
| Moderate-high entropy with low variance | Balanced and information-efficient |

**How to read the plot:**

- Boxplot median shows typical information level; IQR width shows consistency.  
- Integrated with medium-high entropy and low dispersion indicates best balance.

---

## 📊 Overall Diagnostic Summary in covanalyzer2.py

| Metric | Good Model Behavior | Interpretation |
|---------|--------------------|----------------|
| **CVE** | High variance explained with few factors | Efficient factor capture |
| **Frobenius** | Low and smooth | Temporal stability |
| **Eigenvectors** | Clear clusters | Distinct factor structure |
| **Trace(Σ)** | Stable, moderate | Balanced shrinkage |
| **Stress Regime** | Distinct stress reaction | Adaptive to volatility |
| **Entropy** | Medium-high, narrow spread | Informational efficiency |


