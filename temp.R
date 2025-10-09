# ============================================================================
# Integrated High-Dimensional Portfolio Optimization with Asset Selection
# ============================================================================
# Framework: Compares the performance of the main Integrated model
# (LASSO Asset Selection + POET + SAR) against three benchmark models that
# use the full asset universe.
#
# *** 주요 변경 사항 (v4) ***
# 1. (선택 기준 강화) LASSO 자산 선택 시, 계수의 절대값이 미리 설정된 임계값
#    (ASSET_SELECTION_THRESHOLD) 이상인 자산만 선택하도록 기준을 추가했습니다.
# 2. (에러 수정) SAR 모델 생성 시 이웃이 없는 자산(zero-neighbor)이 발생할 경우를
#    처리하는 로직을 추가하여 'zero length neighbour vector' 에러를 해결했습니다.
# 3. daily_selected_assets 리스트를 추가하여 매일 선택된 자산을 기록하고,
#    스크립트 마지막에 요약 결과를 출력합니다.
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
RISK_AVERSION <- 0.5 # 위험 회피 계수
ASSET_SELECTION_THRESHOLD <- 0.01 # NEW: Threshold for LASSO coefficient magnitude

# Define the total backtest period
BACKTEST_START_DATE <- "2023-01-01"
BACKTEST_END_DATE <- "2024-12-31"

# Setup Paths ================================================================
processed_data_file <- "processed_returns.csv"

# Load and Prepare Data ======================================================
cat("Loading and preparing data...\n")
all_data <- read.csv(processed_data_file)
all_data$Date <- as.Date(all_data$Date)
all_data <- all_data[order(all_data$Date), ]

macro_factor_names <- c("FTSE", "MSCI_World", "NASDAQ", "SP500")
asset_names <- setdiff(colnames(all_data), c("Date"))
investable_asset_names <- setdiff(asset_names, macro_factor_names)

backtest_data <- all_data[all_data$Date >= as.Date(BACKTEST_START_DATE) & all_data$Date <= as.Date(BACKTEST_END_DATE), ]
num_backtest_days <- nrow(backtest_data)
cat("Backtest period: ", as.character(backtest_data$Date[1]), " to ", as.character(backtest_data$Date[length(backtest_data$Date)]), "\n")
cat("Total backtest days: ", num_backtest_days, "\n")


# Backtesting Loop ===========================================================
results_list <- list()
daily_selected_assets <- list() # Store selected assets for each day

for (t in 1:num_backtest_days) {
  
  today <- backtest_data$Date[t]
  train_end_date <- today - 1
  
  available_train_data <- all_data[all_data$Date <= train_end_date, ]
  
  if(nrow(available_train_data) < ROLLING_WINDOW_SIZE) {
    cat(sprintf("Skipping %s: Not enough historical data. Required: %d, Available: %d\n", 
                today, ROLLING_WINDOW_SIZE, nrow(available_train_data)))
    next
  }
  
  train_data <- tail(available_train_data, ROLLING_WINDOW_SIZE)
  train_returns <- as.matrix(train_data[, investable_asset_names])
  train_macros <- as.matrix(train_data[, macro_factor_names])
  
  train_returns <- na.locf(train_returns, na.rm = FALSE)
  train_macros <- na.locf(train_macros, na.rm = FALSE)
  train_returns[is.na(train_returns)] <- 0
  train_macros[is.na(train_macros)] <- 0
  
  cat(sprintf("Processing %s (%d/%d)\n", today, t, num_backtest_days))
  
  # ==========================================================================
  # === STEP 1: Calculate Inputs (Sigma, mu) for ALL Models =================
  # ==========================================================================
  model_inputs <- list()
  
  # --- 1. INTEGRATED MODEL (with Asset Selection) ---
  # NEW: Use LASSO to select significant assets first
  y_target <- rowMeans(train_returns, na.rm = TRUE)
  asset_selection_fit <- cv.glmnet(train_returns, y_target, alpha = 1)
  asset_coefs <- coef(asset_selection_fit, s = "lambda.1se")
  
  # CHANGED: Added absolute coefficient threshold for stricter selection
  selected_assets_indices <- which(abs(asset_coefs[-1, 1]) > ASSET_SELECTION_THRESHOLD)
  selected_assets_names <- investable_asset_names[selected_assets_indices]
  
  # Store the selected assets for this date
  daily_selected_assets[[as.character(today)]] <- selected_assets_names
  
  # Proceed only if enough assets are selected
  if (length(selected_assets_names) >= (NUM_LATENT_FACTORS + 1)) {
    cat(sprintf("  - [Integrated] Selected %d assets.\n", length(selected_assets_names)))
    
    # Subset data to selected assets
    train_returns_selected <- train_returns[, selected_assets_names, drop = FALSE]
    N_selected <- ncol(train_returns_selected)
    
    # LASSO Factor Modeling on SELECTED assets
    B_lasso_sel <- matrix(0, nrow = N_selected, ncol = ncol(train_macros))
    residuals_lasso_sel <- matrix(0, nrow = nrow(train_returns_selected), ncol = N_selected)
    for (i in 1:N_selected) {
      cv_fit <- cv.glmnet(train_macros, train_returns_selected[, i], alpha = 1)
      coefs <- coef(cv_fit, s = "lambda.min")
      B_lasso_sel[i, ] <- as.matrix(coefs)[-1, 1]
      residuals_lasso_sel[, i] <- train_returns_selected[, i] - predict(cv_fit, newx = train_macros, s = "lambda.min")
    }
    
    # PCA (POET) Part on SELECTED assets' residuals
    pca <- prcomp(residuals_lasso_sel, scale. = TRUE)
    F_latent <- pca$x[, 1:NUM_LATENT_FACTORS]
    B_latent <- pca$rotation[, 1:NUM_LATENT_FACTORS]
    residuals_pca <- residuals_lasso_sel - F_latent %*% t(B_latent)
    
    # SAR Part on SELECTED assets' residuals
    cor_matrix <- cor(residuals_pca)
    dist_matrix <- as.matrix(as.dist(1 - abs(cor_matrix)))
    adj_matrix <- ifelse(dist_matrix <= 0.2 & dist_matrix > 0, 1, 0)
    # FIXED: Handle assets with no neighbors to prevent error
    nb_list <- lapply(1:N_selected, function(i) {
      neighbors <- which(adj_matrix[i, ] == 1)
      if (length(neighbors) == 0) return(0L) else return(as.integer(neighbors))
    })
    class(nb_list) <- "nb"
    W <- nb2listw(nb_list, style = "W", zero.policy = TRUE)
    residuals_sar <- sapply(1:N_selected, function(i) {
      fit <- try(lagsarlm(residuals_pca[, i] ~ 1, listw = W, zero.policy = TRUE), silent = TRUE)
      if (inherits(fit, "try-error")) return(residuals_pca[, i]) else return(residuals(fit))
    })
    
    # Integrated Sigma and mu for SELECTED assets
    cov_F_lasso <- cov(train_macros)
    mu_macro <- colMeans(train_macros)
    Sigma_integrated <- B_lasso_sel %*% cov_F_lasso %*% t(B_lasso_sel) +
      B_latent %*% cov(F_latent) %*% t(B_latent) +
      cov(residuals_sar)
    mu_integrated <- (B_lasso_sel %*% mu_macro) + (B_latent %*% colMeans(F_latent))
    
    model_inputs$Integrated <- list(
      Sigma = as.matrix(nearPD(Sigma_integrated)$mat), 
      mu = as.vector(mu_integrated),
      assets = selected_assets_names # Pass selected asset names to optimization step
    )
    
  } else {
    cat(sprintf("  - [Integrated] Skipping. Not enough assets selected (%d).\n", length(selected_assets_names)))
  }
  
  # --- BENCHMARK MODELS (on ALL assets) ---
  N <- ncol(train_returns)
  mu_macro <- colMeans(train_macros)
  cov_F_lasso <- cov(train_macros)
  
  # --- 2. LASSO MODEL ---
  B_lasso <- matrix(0, nrow = N, ncol = ncol(train_macros))
  residuals_lasso <- matrix(0, nrow = nrow(train_returns), ncol = N)
  for (i in 1:N) {
    cv_fit <- cv.glmnet(train_macros, train_returns[, i], alpha = 1)
    B_lasso[i, ] <- as.matrix(coef(cv_fit, s="lambda.min"))[-1, 1]
    residuals_lasso[, i] <- train_returns[, i] - predict(cv_fit, newx = train_macros, s = "lambda.min")
  }
  Sigma_lasso <- B_lasso %*% cov_F_lasso %*% t(B_lasso) + cov(residuals_lasso)
  mu_lasso <- B_lasso %*% mu_macro
  model_inputs$LASSO <- list(Sigma = as.matrix(nearPD(Sigma_lasso)$mat), mu = as.vector(mu_lasso), assets = investable_asset_names)
  
  # --- 3. POET MODEL ---
  pca_poet <- prcomp(train_returns, scale. = TRUE)
  F_poet <- pca_poet$x[, 1:NUM_LATENT_FACTORS]
  B_poet <- pca_poet$rotation[, 1:NUM_LATENT_FACTORS]
  residuals_poet <- train_returns - F_poet %*% t(B_poet)
  Sigma_poet <- B_poet %*% cov(F_poet) %*% t(B_poet) + cov(residuals_poet)
  mu_poet <- B_poet %*% colMeans(F_poet)
  model_inputs$POET <- list(Sigma = as.matrix(nearPD(Sigma_poet)$mat), mu = as.vector(mu_poet), assets = investable_asset_names)
  
  # --- 4. OLS MODEL ---
  B_ols <- matrix(0, nrow = N, ncol = ncol(train_macros))
  residuals_ols <- matrix(0, nrow = nrow(train_returns), ncol = N)
  for(i in 1:N){
    fit <- lm(train_returns[, i] ~ train_macros)
    B_ols[i,] <- coef(fit)[-1]
    residuals_ols[,i] <- residuals(fit)
  }
  B_ols[is.na(B_ols)] <- 0
  Sigma_ols <- B_ols %*% cov_F_lasso %*% t(B_ols) + cov(residuals_ols)
  mu_ols <- B_ols %*% mu_macro
  model_inputs$OLS <- list(Sigma = as.matrix(nearPD(Sigma_ols)$mat), mu = as.vector(mu_ols), assets = investable_asset_names)
  
  # ==========================================================================
  # === STEP 2: Loop Through Models and Perform Optimization =================
  # ==========================================================================
  for (model_name in names(model_inputs)) {
    current_Sigma <- model_inputs[[model_name]]$Sigma
    current_mu <- model_inputs[[model_name]]$mu
    assets_for_opt <- model_inputs[[model_name]]$assets
    N_for_opt <- length(assets_for_opt)
    
    for (ge in GROSS_EXPOSURE_LEVELS) {
      w <- Variable(N_for_opt)
      objective <- Minimize(RISK_AVERSION * quad_form(w, current_Sigma) - t(current_mu) %*% w)
      constraints <- list(sum(w) == 1, sum(abs(w)) <= ge)
      problem <- Problem(objective, constraints)
      result_solve <- try(solve(problem, solver = "ECOS"), silent = TRUE)
      
      if (inherits(result_solve, "try-error") || is.null(result_solve$getValue(w)) || result_solve$status != "optimal") {
        cat(sprintf("  - [%s] Optimization failed for GE=%.1f. Skipping.\n", model_name, ge))
        next
      }
      
      weights_vec <- as.vector(result_solve$getValue(w))
      
      # Create a full weight vector with zeros for non-selected assets
      final_weights <- numeric(length(investable_asset_names))
      names(final_weights) <- investable_asset_names
      final_weights[assets_for_opt] <- weights_vec
      
      actual_return_today <- backtest_data[t, investable_asset_names]
      valid_returns <- !is.na(actual_return_today)
      portfolio_return <- sum(final_weights[valid_returns] * actual_return_today[valid_returns])
      
      results_list[[length(results_list) + 1]] <- data.frame(
        Date = today,
        Model = model_name,
        GrossExposure = ge,
        PortfolioReturn = portfolio_return
      )
    }
  }
}

# Process and Display Results ===============================================
if (length(results_list) > 0) {
  results_df <- do.call(rbind, results_list)
  
  # ... (The rest of the result processing and plotting code remains the same) ...
  results_df$Period <- case_when(
    format(results_df$Date, "%Y") == "2023" ~ "2023",
    format(results_df$Date, "%Y") == "2024" ~ "2024",
    TRUE ~ "Other"
  )
  
  results_2023_2024 <- results_df %>% filter(Period %in% c("2023", "2024"))
  if(nrow(results_2023_2024) > 0) {
    results_2023_2024$Period <- "2023-2024"
    results_df <- bind_rows(results_df, results_2023_2024)
  }
  results_df <- results_df %>% filter(Period != "Other")
  
  performance_summary <- results_df %>%
    group_by(Period, GrossExposure, Model) %>%
    summarise(
      AnnualizedReturn = mean(PortfolioReturn, na.rm = TRUE) * 252,
      PortfolioRisk = sd(PortfolioReturn, na.rm = TRUE) * sqrt(252),
      SharpeRatio = AnnualizedReturn / PortfolioRisk,
      .groups = 'drop'
    ) %>%
    mutate(SharpeRatio = ifelse(is.finite(SharpeRatio), SharpeRatio, 0))
  
  cat("\n--- Portfolio Performance Summary ---\n")
  print(performance_summary, n=100)
  
  model_colors <- c("Integrated" = "#e41a1c", "LASSO" = "#377eb8", "POET" = "#4daf4a", "OLS" = "#984ea3")
  
  p1 <- ggplot(performance_summary, aes(x = GrossExposure, y = AnnualizedReturn * 100, color = Model, group = Model)) +
    geom_line(linewidth = 1) + geom_point(size = 2.5) +
    facet_wrap(~Period, scales = "free_y") +
    scale_color_manual(values = model_colors) +
    labs(title = "Annualized Return vs Gross Exposure", x = "Gross Exposure", y = "Annualized Return (%)") +
    theme_bw(base_size = 12) + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, face = "bold"),
                                     strip.background = element_rect(fill="lightblue"))
  
  p2 <- ggplot(performance_summary, aes(x = GrossExposure, y = SharpeRatio, color = Model, group = Model)) +
    geom_line(linewidth = 1) + geom_point(size = 2.5) +
    facet_wrap(~Period, scales = "free_y") +
    scale_color_manual(values = model_colors) +
    labs(title = "Sharpe Ratio vs Gross Exposure", x = "Gross Exposure", y = "Annualized Sharpe Ratio") +
    theme_bw(base_size = 12) + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, face = "bold"),
                                     strip.background = element_rect(fill="lightblue"))
  
  p3 <- ggplot(performance_summary, aes(x = GrossExposure, y = PortfolioRisk * 100, color = Model, group = Model)) +
    geom_line(linewidth = 1) + geom_point(size = 2.5) +
    facet_wrap(~Period, scales = "free_y") +
    scale_color_manual(values = model_colors) +
    labs(title = "Annualized Portfolio Risk vs Gross Exposure", x = "Gross Exposure", y = "Annualized Risk (Volatility, %)") +
    theme_bw(base_size = 12) + theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, face = "bold"),
                                     strip.background = element_rect(fill="lightblue"))
  
  grid.arrange(p1, p2, p3, ncol = 1)
  
} else {
  cat("No results were generated. Please check the data and backtest period.\n")
}

# NEW: Display summary of selected assets from the Integrated model
if (length(daily_selected_assets) > 0) {
  cat("\n--- Integrated Model Asset Selection Summary ---\n")
  
  # 1. Frequency of selection for each asset
  asset_selection_counts <- sort(table(unlist(daily_selected_assets)), decreasing = TRUE)
  cat("Frequency of asset selection during the backtest period:\n")
  print(as.data.frame(asset_selection_counts, responseName = "Count"))
  
  # 2. Show selected assets for the last 5 days
  cat("\nSelected assets for the last 5 trading days:\n")
  last_5_days <- tail(names(daily_selected_assets), 5)
  for (day in last_5_days) {
    assets <- daily_selected_assets[[day]]
    if (length(assets) > 0) {
      cat(sprintf(" - %s: %s\n", day, paste(assets, collapse = ", ")))
    } else {
      cat(sprintf(" - %s: No assets were selected.\n", day))
    }
  }
}

write.csv(as.data.frame(asset_selection_counts, responseName = "Count"), file = 'temp_count')
write.csv(performance_summary, file = 'temp_result')
