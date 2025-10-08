# ============================================================================
# Integrated High-Dimensional Portfolio Optimization (Corrected Version)
# ============================================================================
# Framework: LASSO (Factor Selection) + Latent Factors (PCA) + Spatial AR
# Optimization: Mean-Variance Optimization using CVXR to handle gross
# exposure constraints effectively.
#
# *** 주요 변경 사항 ***
# 1. 사용자의 요청에 따라, 최적화 목표를 GMV에서 다시 평균-분산(Mean-Variance) 최적화로 롤백했습니다.
# 2. 이에 따라 기대수익률(mu) 계산 로직을 다시 활성화하고, 위험 회피 계수(RISK_AVERSION)를
#    최적화 목표 함수에 적용합니다.
# 3. 거시 팩터는 BTC, FTSE, MSCI_World, NASDAQ, SP500를 유지합니다.
# ============================================================================

# Load Required Libraries ===================================================
suppressPackageStartupMessages({
  library(glmnet)        # LASSO regression
  library(spdep)         # Spatial weights
  library(spatialreg)    # Spatial AR
  library(zoo)           # For na.locf (NA imputation)
  library(ggplot2)
  library(gridExtra)
  library(CVXR)          # Use modern convex optimization solver
  library(dplyr)
  library(Matrix)        # For nearPD function
  library(tidyr)
})

# Configuration ==============================================================
TRAIN_YEARS <- 8
ROLLING_WINDOW_SIZE <- TRAIN_YEARS * 250 # Assuming 250 trading days per year
NUM_LATENT_FACTORS <- 5 # Number of latent factors to extract from PCA
GROSS_EXPOSURE_LEVELS <- seq(1.0, 3.0, by = 0.5)
RISK_AVERSION <- 0.5 # 위험 회피 계수 (0에 가까울수록 수익률 극대화, 높을수록 위험 최소화)

# Define the total backtest period
BACKTEST_START_DATE <- "2023-01-01"
BACKTEST_END_DATE <- "2024-12-31"


# Setup Paths ================================================================
processed_data_file <- "processed_returns.csv"

# Load and Prepare Data ======================================================
cat("Loading and preparing data...\n")
all_data <- read.csv(processed_data_file)
all_data$Date <- as.Date(all_data$Date)
all_data <- all_data[order(all_data$Date), ] # 날짜순으로 데이터 정렬

# 거시 팩터를 5개의 주요 글로벌 지수로 확장
macro_factor_names <- c("BTC", "FTSE", "MSCI_World", "NASDAQ", "SP500")

# Date를 제외한 나머지 모든 변수를 자산으로 간주
asset_names <- setdiff(colnames(all_data), c("Date"))
# 실제 투자 대상 자산 목록 (새로운 거시 팩터 제외)
investable_asset_names <- setdiff(asset_names, macro_factor_names)

# 데이터 기간 필터링
backtest_data <- all_data[all_data$Date >= as.Date(BACKTEST_START_DATE) & all_data$Date <= as.Date(BACKTEST_END_DATE), ]
num_backtest_days <- nrow(backtest_data)
cat("Backtest period: ", as.character(backtest_data$Date[1]), " to ", as.character(backtest_data$Date[length(backtest_data$Date)]), "\n")
cat("Total backtest days: ", num_backtest_days, "\n")


# Backtesting Loop ===========================================================
results_list <- list()

for (t in 1:num_backtest_days) {
  
  # Define training and test periods
  today <- backtest_data$Date[t]
  train_end_date <- today - 1
  
  # 1. 학습 종료일 이전의 모든 데이터를 가져옵니다.
  available_train_data <- all_data[all_data$Date <= train_end_date, ]
  
  # 2. 롤링 윈도우를 구성할 만큼 충분한 데이터가 있는지 확인합니다.
  if(nrow(available_train_data) < ROLLING_WINDOW_SIZE) {
    cat(sprintf("Skipping %s: Not enough historical data. Required: %d, Available: %d\n", 
                today, ROLLING_WINDOW_SIZE, nrow(available_train_data)))
    next
  }
  
  # 3. 마지막 ROLLING_WINDOW_SIZE 만큼의 데이터를 실제 학습 데이터로 사용합니다.
  train_data <- tail(available_train_data, ROLLING_WINDOW_SIZE)
  
  train_returns <- as.matrix(train_data[, investable_asset_names])
  train_macros <- as.matrix(train_data[, macro_factor_names])
  
  # Impute NA using Last Observation Carried Forward
  train_returns <- na.locf(train_returns, na.rm = FALSE)
  train_macros <- na.locf(train_macros, na.rm = FALSE)
  train_returns[is.na(train_returns)] <- 0
  train_macros[is.na(train_macros)] <- 0
  
  
  cat(sprintf("Processing %s (%d/%d)\n", today, t, num_backtest_days))
  
  N <- ncol(train_returns) # Number of assets
  
  # Step 1: LASSO for Macro Factor Selection
  B_lasso <- matrix(0, nrow = N, ncol = ncol(train_macros))
  residuals_lasso <- matrix(0, nrow = nrow(train_returns), ncol = N)
  
  for (i in 1:N) {
    cv_fit <- cv.glmnet(train_macros, train_returns[, i], alpha = 1)
    coefs <- coef(cv_fit, s = "lambda.min")
    B_lasso[i, ] <- as.matrix(coefs)[-1, 1]
    residuals_lasso[, i] <- train_returns[, i] - predict(cv_fit, newx = train_macros, s = "lambda.min")
  }
  colnames(B_lasso) <- macro_factor_names
  rownames(B_lasso) <- investable_asset_names
  
  # Step 2: Latent Factors from PCA on LASSO residuals
  pca <- prcomp(residuals_lasso, scale. = TRUE)
  F_latent <- pca$x[, 1:NUM_LATENT_FACTORS]
  B_latent <- pca$rotation[, 1:NUM_LATENT_FACTORS]
  
  residuals_pca <- residuals_lasso - F_latent %*% t(B_latent)
  
  # Step 3: Spatial AR on PCA residuals
  cor_matrix <- cor(residuals_pca)
  dist_matrix <- as.dist(1 - abs(cor_matrix))
  
  # 1. 거리 행렬을 완전한 정방 행렬(square matrix)로 변환합니다.
  full_dist_matrix <- as.matrix(dist_matrix)
  
  # 2. 임계값(0.2)을 기준으로 인접 행렬(adjacency matrix)을 생성합니다.
  adjacency_matrix <- ifelse(full_dist_matrix <= 0.2 & full_dist_matrix > 0, 1, 0)
  
  # 3. 인접 행렬에서 수동으로 'nb' 객체 생성
  nb_list <- vector("list", N)
  for (i in 1:N) {
    neighbors <- which(adjacency_matrix[i, ] == 1)
    if (length(neighbors) == 0) {
      nb_list[[i]] <- 0L
    } else {
      nb_list[[i]] <- as.integer(neighbors)
    }
  }
  
  class(nb_list) <- "nb"
  attr(nb_list, "region.id") <- as.character(1:N)
  
  # 이웃이 없는 자산이 있을 수 있으므로 zero.policy=TRUE 사용
  W <- nb2listw(nb_list, style = "W", zero.policy = TRUE)
  
  residuals_sar <- matrix(0, nrow = nrow(residuals_pca), ncol = N)
  rho_vec <- numeric(N)
  
  for (i in 1:N) {
    sarlm_fit_attempt <- try(
      lagsarlm(residuals_pca[, i] ~ 1, listw = W, zero.policy = TRUE),
      silent = TRUE
    )
    
    if (inherits(sarlm_fit_attempt, "try-error")) {
      rho_vec[i] <- 0
      residuals_sar[, i] <- residuals_pca[, i]
    } else {
      sarlm_fit <- sarlm_fit_attempt
      rho_vec[i] <- sarlm_fit$rho
      residuals_sar[, i] <- residuals(sarlm_fit)
    }
  }
  
  
  # Step 4: Construct the Integrated Covariance Matrix
  cov_F_lasso <- cov(train_macros)
  cov_F_latent <- cov(F_latent)
  cov_E_s <- cov(residuals_sar)
  
  Sigma <- B_lasso %*% cov_F_lasso %*% t(B_lasso) +
    B_latent %*% cov_F_latent %*% t(B_latent) +
    cov_E_s
  
  Sigma <- as.matrix(nearPD(Sigma)$mat) # Ensure positive semi-definite
  
  # *** 롤백: 기대수익률(mu) 계산 로직 다시 활성화 ***
  mu_macro <- colMeans(train_macros)
  mu_latent <- colMeans(F_latent)
  mu <- (B_lasso %*% mu_macro) + (B_latent %*% mu_latent)
  mu <- as.vector(mu) # CVXR에 맞게 벡터 형태로 변환
  
  # Step 5: Portfolio Optimization for each Gross Exposure level
  for (ge in GROSS_EXPOSURE_LEVELS) {
    w <- Variable(N)
    
    # *** 롤백: 최적화 목표 함수를 평균-분산(Mean-Variance) 최적화로 변경 ***
    objective <- Minimize(RISK_AVERSION * quad_form(w, Sigma) - t(mu) %*% w)
    
    constraints <- list(
      sum(w) == 1,
      sum(abs(w)) <= ge
    )
    
    problem <- Problem(objective, constraints)
    
    result_solve <- try(solve(problem, solver = "ECOS"), silent = TRUE)
    
    if (inherits(result_solve, "try-error") || is.null(result_solve$getValue(w)) || result_solve$status != "optimal") {
      cat(sprintf("  - Optimization failed for GE=%.1f. Skipping.\n", ge))
      next
    }
    
    weights <- as.vector(result_solve$getValue(w))
    names(weights) <- investable_asset_names
    
    # Store results
    actual_return_today <- backtest_data[t, investable_asset_names]
    valid_returns <- !is.na(actual_return_today)
    portfolio_return <- sum(weights[valid_returns] * actual_return_today[valid_returns])
    
    results_list[[length(results_list) + 1]] <- data.frame(
      Date = today,
      GrossExposure = ge,
      PortfolioReturn = portfolio_return,
      Weights = I(list(weights))
    )
  }
}

# Process and Display Results ===============================================
if (length(results_list) > 0) {
  results_df <- do.call(rbind, results_list)
  
  # Define periods
  results_df$Period <- case_when(
    format(results_df$Date, "%Y") == "2023" ~ "2023",
    format(results_df$Date, "%Y") == "2024" ~ "2024",
    TRUE ~ "Other"
  )
  
  # Create a combined period
  results_2023_2024 <- results_df %>% filter(Period %in% c("2023", "2024"))
  if(nrow(results_2023_2024) > 0) {
    results_2023_2024$Period <- "2023-2024"
    results_df <- bind_rows(results_df, results_2023_2024)
  }
  
  results_df <- results_df %>% filter(Period != "Other")
  
  # Calculate performance metrics
  performance_summary <- results_df %>%
    group_by(Period, GrossExposure) %>%
    summarise(
      AnnualizedReturn = mean(PortfolioReturn, na.rm = TRUE) * 252,
      PortfolioRisk = sd(PortfolioReturn, na.rm = TRUE) * sqrt(252),
      SharpeRatio = AnnualizedReturn / PortfolioRisk,
      .groups = 'drop'
    ) %>%
    # NaN/Inf 값은 0으로 대체하여 그래프 오류 방지
    mutate(SharpeRatio = ifelse(is.finite(SharpeRatio), SharpeRatio, 0))
  
  cat("\n--- Portfolio Performance Summary ---\n")
  print(performance_summary)
  
  # Plotting the results
  results_plot <- performance_summary
  period_colors <- c("2023" = "#1f77b4", "2024" = "#ff7f0e", "2023-2024" = "#2ca02c")
  
  p1 <- ggplot(results_plot, aes(x = GrossExposure, y = AnnualizedReturn * 100, color = Period, group = Period)) +
    geom_line(linewidth = 1.2) + geom_point(size = 4, aes(shape = Period)) +
    scale_color_manual(values = period_colors) +
    labs(title = "Annualized Return", x = "Gross Exposure", y = "Annualized Return (%)") +
    theme_minimal(base_size = 14) + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, face = "bold"))
  
  p2 <- ggplot(results_plot, aes(x = GrossExposure, y = SharpeRatio, color = Period, group = Period)) +
    geom_line(linewidth = 1.2) + geom_point(size = 4, aes(shape = Period)) +
    scale_color_manual(values = period_colors) +
    labs(title = "Sharpe Ratio", x = "Gross Exposure", y = "Annualized Sharpe Ratio") +
    theme_minimal(base_size = 14) + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, face = "bold"))
  
  p3 <- ggplot(results_plot, aes(x = GrossExposure, y = PortfolioRisk * 100, color = Period, group = Period)) +
    geom_line(linewidth = 1.2) + geom_point(size = 4, aes(shape = Period)) +
    scale_color_manual(values = period_colors) +
    labs(title = "Annualized Portfolio Risk", x = "Gross Exposure", y = "Annualized Risk (Volatility, %)") +
    theme_minimal(base_size = 14) + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, face = "bold"))
  
  grid.arrange(p1, p2, p3, ncol = 1)
  
} else {
  cat("No results were generated. Please check the data and backtest period.\n")
}

