# 🚀 Development Flow Helper

### ** Guiding Principles for the Agent**

**IMPORTANT**: This project is a direct implementation of a research proposal. The primary objective is to rigorously and accurately translate the proposed methodologies into code.

1.  **Strict Adherence to Formulas**: **DO NOT** modify, simplify, or approximate the mathematical formulas provided in this guide for any reason (e.g., convenience, computational efficiency, or perceived performance improvements). The goal is to implement the proposal as it is written.
2.  **Clarity and Readability**: Structure the script's flow logically so that a human can easily understand the process by reading the code. Add comments to explain the purpose of functions, complex logic, and important new variables.
3.  **Prioritize Clarity over Optimization**: When implementing, prioritize structuring the code in a way that is mathematically easy to follow and understand. Algorithmic optimization should be secondary to this clarity.

---

This document provides a step-by-step guide for implementing the R scripts for the "Integrated High-Dimensional Methods for Portfolio Optimization" project. It breaks down the development process into manageable phases, corresponding to each script defined in `readme.md`.

## Phase 0: Project Setup

Before writing any code, ensure the project directory structure is set up as follows:

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

Place all raw `.csv` files into the `data/raw/` directory.

## Phase 1: Data Preprocessing (`scripts/1_data_preprocessing.R`)

**Objective**: To create a single, clean, and unified time-series dataset of asset returns from multiple raw, unstructured files.

### 📋 Task List:

1.  **Load Libraries**:
    * Load necessary R packages like `readr`, `dplyr`, `lubridate`, `xts`, and `zoo`.
2.  **Define a Flexible CSV Loader Function**:
    * Create a function `load_and_clean_csv(file_path)` that takes a file path as input.
    * Inside the function, intelligently detect the header row and the start of the actual data.
    * Identify the 'Date' and 'Close' (or equivalent price) columns, even if their names are inconsistent (e.g., '날짜', '종가', 'Price').
    * Parse the date column into a standard `Date` format.
    * Return a standardized `xts` or `data.frame` object with two columns: `Date` and `Price`, with the asset's ticker as the column name for `Price`.
3.  **Process All Raw Files**:
    * Get a list of all `.csv` files in the `data/raw/` directory.
    * Loop through the file list, apply the `load_and_clean_csv` function to each file, and store the resulting time-series objects in a list.
4.  **Merge and Align Time Series**:
    * Merge all the individual time-series objects from the list into a single wide-format `xts` object, using the date as the common index. This will automatically handle date alignment.
5.  **Handle Missing Values**:
    * The merged object will have `NA` values for non-trading days. Use a forward-fill method (e.g., `na.locf` from the `zoo` package) to fill these gaps.
6.  **Calculate and Standardize Returns**:
    * Calculate daily log returns for all assets. Remove the first row of `NA`s that results from the calculation.
    * Standardize each return series (transform to have a mean of 0 and a standard deviation of 1).
7.  **Save the Final Output**:
    * Save the final, cleaned data frame of standardized returns to `data/processed/processed_returns.csv`.

## Phase 2: Model Execution (`scripts/2_model_execution.R`)

**Objective**: To implement the core econometric models (LASSO, POET, SAR) and compute the final integrated covariance matrix.

### 📋 Task List:

1.  **Load Processed Data**:
    * Load `processed_returns.csv` from the `data/processed/` directory. Also, load the preprocessed macroeconomic factor data.
2.  **LASSO Factor Selection**:
    * **Key Formula**: The LASSO estimator finds the coefficients $\beta$ that solve the following optimization problem:
        $$
        \hat{\beta} = \arg\min_{\beta} ||y - X\beta||_{2}^{2} + \lambda||\beta||_{1}
        $$
    * **Implementation**:
        * Use the `glmnet` package.
        * Define a response variable `y` (e.g., the return of a market-wide index) and a predictor matrix `X` (the macro factors).
        * Use cross-validation (`cv.glmnet`) to find the optimal regularization parameter ($\lambda$).
        * Extract the non-zero coefficients to identify the selected macro factors.
3.  **POET Covariance Estimation**:
    * **Key Formulas**: The asset returns $R_t$ are modeled as a factor structure, and the covariance matrix $\Sigma$ is decomposed and estimated.
        $$
        R_{t} = Bf_{t} + u_{t}
        $$
        $$
        \hat{\Sigma}_{POET} = \hat{B}\hat{\Sigma}_{f}\hat{B}^{\top} + \mathcal{T}_{\omega}(\hat{\Sigma}_{u})
        $$
        where $\mathcal{T}_{\omega}$ is an adaptive thresholding operator.
    * **Implementation**:
        * Regress each asset's returns on the selected macro factors to get the residuals.
        * Perform PCA on the matrix of residuals to find latent factors.
        * Construct the factor-based covariance matrix ($\hat{B}\hat{\Sigma}_{f}\hat{B}^{\top}$).
        * Estimate the sparse residual covariance matrix ($\mathcal{T}_{\omega}(\hat{\Sigma}_{u})$) by applying thresholding to the sample covariance of the final residuals.
        * Sum the two matrices to get the final $\hat{\Sigma}_{POET}$.
4.  **Spatial AR Residual Modeling**:
    * **Key Formula**: The idiosyncratic shocks $u_t$ from the POET model are modeled to have a network structure:
        $$
        u_{t} = \rho W u_{t} + \epsilon_{t}
        $$
        where $W$ is the spatial weight matrix and $\rho$ is the spatial autoregressive parameter.
    * **Implementation**:
        * Using the final residuals from the POET step, compute a correlation matrix to serve as `W`.
        * Use a package like `spdep` to estimate the SAR model and find the parameter `rho`.
        * Adjust the POET residual covariance to incorporate the network structure, creating the final $\hat{\Sigma}_{Integrated}$.
5.  **Save Model Outputs**:
    * Save the final integrated covariance matrix $\hat{\Sigma}_{Integrated}$ and the list of selected macro factors to the `results/model_outputs/` directory.

## Phase 3: Portfolio Evaluation (`scripts/3_portfolio_evaluation.R`)

**Objective**: To construct and evaluate portfolios using the integrated covariance matrix.

### 📋 Task List:

1.  **Load Inputs**:
    * Load the integrated covariance matrix from `results/model_outputs/`.
    * Load the processed asset returns from `data/processed/`.
2.  **Define Portfolio Construction Functions**:
    * Write a function to calculate **Global Minimum Variance (GMV)** portfolio weights.
    * Write a a function to calculate **Mean-Variance Optimal** portfolio weights (this will require an estimate for expected returns, $\hat{\mu}$, which can be the simple sample mean of the returns).
    * (Optional) Write a function for **Risk Budgeting** portfolios.
3.  **Calculate Portfolio Performance**:
    * For each portfolio strategy, calculate the out-of-sample performance. A simple approach is to use the full-sample returns, but a rolling-window backtest is a more robust evaluation method.
    * Compute key performance metrics: Average Return, Standard Deviation (Volatility), Sharpe Ratio, Value-at-Risk (VaR), and Expected Shortfall (ES).
4.  **Generate and Save Results**:
    * Create summary tables and plots (e.g., portfolio weights, cumulative return plots).
    * Save these reports and visualizations to the `results/portfolio_reports/` directory.

## Phase 4: Main Orchestrator (`main.R`)

**Objective**: To run the entire pipeline sequentially.

### 📋 Task List:

1.  **Setup Environment**:
    * Clear the workspace.
    * Define file paths and check if the required directories exist; if not, create them.
2.  **Execute Scripts in Order**:
    * Use the `source()` function to run the scripts one by one.
    * Add `print()` statements between each step to log progress to the console.
    ```R
    # Example main.R structure
    print("Pipeline Started.")

    source("scripts/1_data_preprocessing.R")
    print("Step 1: Data Preprocessing COMPLETED.")

    source("scripts/2_model_execution.R")
    print("Step 2: Model Execution COMPLETED.")

    source("scripts/3_portfolio_evaluation.R")
    print("Step 3: Portfolio Evaluation COMPLETED.")

    print("Pipeline Finished Successfully.")
    ```