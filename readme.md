# Integrated High-Dimensional Methods for Portfolio Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project provides an R implementation of an integrated framework for high-dimensional portfolio optimization. In modern financial markets, constructing a stable and robust portfolio is challenging due to the high dimensionality of assets and economic factors. This project addresses this by combining three powerful methodologies into a single, cohesive process:

1.  **LASSO Regression**: To select a sparse and meaningful set of macroeconomic factors from a vast number of potential predictors.
2.  **POET (Principal Orthogonal Complement Thresholding)**: To estimate a large covariance matrix by separating the effects of systematic factors from idiosyncratic, sparse residual risks.
3.  **Spatial Autoregression (SAR)**: To capture the network-based dependencies (e.g., financial contagion) that may still exist in the residuals, going beyond the simple sparsity assumption of POET.

The ultimate goal is to build more stable and well-diversified portfolios (e.g., Global Minimum Variance, Mean-Variance Optimal) by leveraging a more accurate and robustly estimated covariance matrix.

---

## Project Structure

The project is organized into modular R scripts orchestrated by a main script. This structure enhances readability, maintainability, and ease of execution.

```
.
+-- main.R
+-- scripts/
|   +-- 1_data_preprocessing.R
|   +-- 2_model_execution.R
|   \-- 3_portfolio_evaluation.R
+-- data/
|   +-- raw/
|   \-- processed/
\-- results/
    +-- model_outputs/
    \-- portfolio_reports/
```


---

## Methodology & Implementation Steps

The implementation follows a clear 5-step pipeline, as outlined below.

### **Step 1: Data Aggregation and Preprocessing**

* **File**: `scripts/1_data_preprocessing.R`
* **Goal**: The primary goal is to load, parse, and clean raw data from various sources with non-uniform formats, and to create a unified, analysis-ready dataset.
* **Key Challenges & Process**:
    1.  **Flexible Data Loading & Parsing**: The script must handle diverse `.csv` file structures, identifying key columns (e.g., 'Close' price) and parsing unstructured metadata.
    2.  **Date Alignment and Imputation**: All time series must be aligned to a single master date index (e.g., U.S. trading days), with missing values handled via methods like forward-fill.
    3.  **Data Transformation and Standardization**: The cleaned price data is used to calculate log returns, which are then tested for stationarity and standardized (mean 0, std 1).
    4.  **Output**: The final dataset is saved as a single file (e.g., `processed_returns.csv`) in `data/processed/`.

### **Step 2: Macro Factor Selection with LASSO**

* **File**: `scripts/2_model_execution.R`
* **Process**: Apply **LASSO regression** to a wide range of macroeconomic predictors (e.g., CPI, interest rates) to select a sparse subset that best explains asset returns. This forms the observable macro-factor block, $f_{t}^{Macro}$.

### **Step 3: Covariance Estimation with POET**

* **File**: `scripts/2_model_execution.R`
* **Process**: Model asset returns as a combination of the LASSO-selected macro factors and unobserved latent factors ($R_{t}=Bf_{t}+u_{t}$). Use PCA on the residuals to estimate latent factors and construct the **POET estimator**, which combines the factor covariance with a thresholded residual covariance matrix: $\hat{\Sigma}_{POET}=\hat{\Sigma}_{Factor}+\mathcal{T}_{\omega}(\hat{\Sigma}_{u})$.

### **Step 4: Residual Network Modeling with Spatial AR**

* **File**: `scripts/2_model_execution.R`
* **Process**: Apply a **Spatial AR model** ($u_{t}=\rho Wu_{t}+\epsilon_{t}$) to the idiosyncratic shocks ($u_t$) from the POET model. This captures structured network dependencies using a correlation-based weight matrix `W`, resulting in a more realistic, integrated covariance matrix $\hat{\Sigma}_{Integrated}$.

### **Step 5: Portfolio Optimization & Evaluation**

* **File**: `scripts/3_portfolio_evaluation.R`
* **Process**:
    1.  **Portfolio Construction**: Use the final integrated covariance estimator $\hat{\Sigma}_{Integrated}$ to construct several optimal portfolios: Global Minimum Variance (GMV), Mean-Variance Optimal, and Risk Budgeting.
    2.  **Performance Analysis**: Evaluate the portfolios on stability, tail-risk metrics (VaR, ES), and diversification benefits.

---

## Dataset

The analysis will use a cross-asset dataset from **2015.01.01 to 2024.12.31**, covering approximately 70-80 series. Data will be sourced from public APIs and financial data providers like **Yahoo Finance** and **Investing.com**.

* **Equities**: Global indices (S&P 500 and its sectoral indices, MSCI World) and major regional indices (Nikkei, KOSPI, etc.).
* **Fixed Income**: U.S. Treasury yields, corporate bond spreads.
* **Commodities**: Gold, Oil (Brent, WTI), industrial metals, agricultural futures.
* **Currencies**: Major FX pairs against KRW (USDKRW, JPYKRW, EURKRW, CNYKRW).
* **Cryptocurrencies**: Bitcoin (BTC).
* **Volatility Measures & Macro Proxies**: VIX Index, Consumer Price Index (CPI), etc.

---

## Expected Contributions

* **Methodological Integration**: The primary contribution is the novel integration of LASSO, POET, and SAR into a unified framework for portfolio construction, jointly addressing predictor selection, factor structure, and network dependence.
* **Financial Implications**: This framework is expected to yield more stable asset allocations, enhance tail-risk management (VaR/ES), and provide a clearer understanding of cross-asset diversification by capturing contagion effects.
* **Pedagogical Value**: The project serves as a practical application connecting key concepts in financial big data analysis, including high-dimensional regression, factor models, and network econometrics.

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install R Dependencies:**
    Ensure you have R installed. Then, open an R console and install the required packages.
    ```R
    install.packages(c("glmnet", "zoo", "xts", "spdep", "PortfolioAnalytics", "lubridate"))
    ```

3.  **Run the Pipeline:**
    Execute the main script from your terminal. This will run the entire process sequentially.
    ```bash
    Rscript main.R
    ```
    The script will generate processed data in `data/processed/`, model outputs in `results/model_outputs/`, and final portfolio reports in `results/portfolio_reports/`.
