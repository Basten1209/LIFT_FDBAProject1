# ============================================================================
# Traditional Portfolio Optimization Models
# ============================================================================
# This script evaluates classical portfolio construction techniques using the
# same rolling-window backtest structure employed by the integrated model. The
# resulting performance_summary shares the same core columns as
# integrated_model_portfolio.R so the outputs can be compared side-by-side.
# ============================================================================

suppressPackageStartupMessages({
  library(zoo)           # For na.locf
  library(CVXR)          # Optimization under gross exposure constraints
  library(dplyr)
  library(Matrix)        # For nearPD
  library(tidyr)
  library(ggplot2)
  library(gridExtra)
})

# Configuration ===============================================================
TRAIN_YEARS <- 8
ROLLING_WINDOW_SIZE <- TRAIN_YEARS * 250
GROSS_EXPOSURE_LEVELS <- seq(1.0, 3.0, by = 0.5)
RISK_AVERSION <- 0.5

BACKTEST_START_DATE <- "2023-01-01"
BACKTEST_END_DATE <- "2024-12-31"

processed_data_file <- "processed_returns.csv"

# Helper Functions ===========================================================
project_equal_weight <- function(target_weights, ge_level, Sigma_pd = NULL) {
  N <- length(target_weights)
  w <- Variable(N)
  objective <- Minimize(sum_squares(w - target_weights))
  constraints <- list(sum(w) == 1, sum(abs(w)) <= ge_level)
  problem <- Problem(objective, constraints)
  result <- try(solve(problem), silent = TRUE)
  if (inherits(result, "try-error") || is.null(result$getValue(w))) {
    return(NULL)
  }
  as.numeric(result$getValue(w))
}

solve_gmv <- function(Sigma_pd, ge_level) {
  N <- nrow(Sigma_pd)
  w <- Variable(N)
  objective <- Minimize(quad_form(w, Sigma_pd))
  constraints <- list(sum(w) == 1, sum(abs(w)) <= ge_level)
  problem <- Problem(objective, constraints)
  result <- try(solve(problem), silent = TRUE)
  if (inherits(result, "try-error") || is.null(result$getValue(w))) {
    return(NULL)
  }
  as.numeric(result$getValue(w))
}

solve_mean_variance <- function(mu_vec, Sigma_pd, ge_level, risk_aversion) {
  N <- length(mu_vec)
  w <- Variable(N)
  objective <- Minimize(0.5 * risk_aversion * quad_form(w, Sigma_pd) - t(mu_vec) %*% w)
  constraints <- list(sum(w) == 1, sum(abs(w)) <= ge_level)
  problem <- Problem(objective, constraints)
  result <- try(solve(problem), silent = TRUE)
  if (inherits(result, "try-error") || is.null(result$getValue(w))) {
    return(NULL)
  }
  as.numeric(result$getValue(w))
}

# Load and Prepare Data ======================================================
cat("Loading and preparing data...\n")
all_data <- read.csv(processed_data_file)
all_data$Date <- as.Date(all_data$Date)
all_data <- all_data[order(all_data$Date), ]

macro_factor_names <- c("BTC", "FTSE", "MSCI_World", "NASDAQ", "SP500")
asset_names <- setdiff(colnames(all_data), c("Date"))
investable_asset_names <- setdiff(asset_names, macro_factor_names)

backtest_data <- all_data[all_data$Date >= as.Date(BACKTEST_START_DATE) &
                           all_data$Date <= as.Date(BACKTEST_END_DATE), ]
num_backtest_days <- nrow(backtest_data)
cat("Backtest period:", as.character(backtest_data$Date[1]), "to",
    as.character(backtest_data$Date[length(backtest_data$Date)]), "\n")
cat("Total backtest days:", num_backtest_days, "\n")

results_list <- list()

if (num_backtest_days == 0) {
  stop("No backtest data available for the specified date range.")
}

for (t in 1:num_backtest_days) {
  today <- backtest_data$Date[t]
  train_end_date <- today - 1
  available_train_data <- all_data[all_data$Date <= train_end_date, ]

  if (nrow(available_train_data) < ROLLING_WINDOW_SIZE) {
    cat(sprintf("Skipping %s: Not enough historical data. Required: %d, Available: %d\n",
                today, ROLLING_WINDOW_SIZE, nrow(available_train_data)))
    next
  }

  train_data <- tail(available_train_data, ROLLING_WINDOW_SIZE)

  train_returns <- as.matrix(train_data[, investable_asset_names])
  train_returns <- na.locf(train_returns, na.rm = FALSE)
  train_returns[is.na(train_returns)] <- 0

  mu_vec <- colMeans(train_returns, na.rm = TRUE)
  Sigma <- cov(train_returns, use = "pairwise.complete.obs")
  Sigma_pd <- as.matrix(nearPD(Sigma)$mat)

  if (any(!is.finite(mu_vec))) {
    mu_vec[!is.finite(mu_vec)] <- 0
  }

  N <- length(mu_vec)
  equal_weight <- rep(1 / N, N)

  actual_return_today <- backtest_data[t, investable_asset_names]
  valid_returns <- !is.na(actual_return_today)

  for (ge in GROSS_EXPOSURE_LEVELS) {
    strategies <- list(
      list(name = "EqualWeight", weights = project_equal_weight(equal_weight, ge)),
      list(name = "GlobalMinimumVariance", weights = solve_gmv(Sigma_pd, ge)),
      list(name = "MeanVariance", weights = solve_mean_variance(mu_vec, Sigma_pd, ge, RISK_AVERSION))
    )

    for (strategy in strategies) {
      if (is.null(strategy$weights)) {
        cat(sprintf("  - %s optimization failed for GE=%.1f on %s. Skipping.\n",
                    strategy$name, ge, today))
        next
      }

      weights <- strategy$weights
      names(weights) <- investable_asset_names
      portfolio_return <- sum(weights[valid_returns] * actual_return_today[valid_returns])

      results_list[[length(results_list) + 1]] <- data.frame(
        Date = today,
        GrossExposure = ge,
        Strategy = strategy$name,
        PortfolioReturn = portfolio_return,
        Weights = I(list(weights))
      )
    }
  }
}

# Process and Display Results ===============================================
if (length(results_list) == 0) {
  cat("No results were generated. Please check the data and backtest period.\n")
} else {
  results_df <- do.call(rbind, results_list)

  results_df$Period <- case_when(
    format(results_df$Date, "%Y") == "2023" ~ "2023",
    format(results_df$Date, "%Y") == "2024" ~ "2024",
    TRUE ~ "Other"
  )

  results_2023_2024 <- results_df %>% filter(Period %in% c("2023", "2024"))
  if (nrow(results_2023_2024) > 0) {
    results_2023_2024$Period <- "2023-2024"
    results_df <- bind_rows(results_df, results_2023_2024)
  }

  results_df <- results_df %>% filter(Period != "Other")

  performance_summary <- results_df %>%
    group_by(Strategy, Period, GrossExposure) %>%
    summarise(
      AnnualizedReturn = mean(PortfolioReturn, na.rm = TRUE) * 252,
      PortfolioRisk = sd(PortfolioReturn, na.rm = TRUE) * sqrt(252),
      SharpeRatio = AnnualizedReturn / PortfolioRisk,
      .groups = "drop"
    ) %>%
    mutate(SharpeRatio = ifelse(is.finite(SharpeRatio), SharpeRatio, 0))

  cat("\n--- Traditional Portfolio Performance Summary ---\n")
  print(performance_summary)

  period_colors <- c("2023" = "#1f77b4", "2024" = "#ff7f0e", "2023-2024" = "#2ca02c")

  p1 <- ggplot(performance_summary,
               aes(x = GrossExposure, y = AnnualizedReturn * 100,
                   color = Period, group = Period)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 3, aes(shape = Period)) +
    scale_color_manual(values = period_colors) +
    labs(title = "Annualized Return", x = "Gross Exposure",
         y = "Annualized Return (%)") +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom",
          plot.title = element_text(hjust = 0.5, face = "bold")) +
    facet_wrap(~ Strategy, ncol = 1)

  p2 <- ggplot(performance_summary,
               aes(x = GrossExposure, y = SharpeRatio,
                   color = Period, group = Period)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 3, aes(shape = Period)) +
    scale_color_manual(values = period_colors) +
    labs(title = "Sharpe Ratio", x = "Gross Exposure",
         y = "Annualized Sharpe Ratio") +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom",
          plot.title = element_text(hjust = 0.5, face = "bold")) +
    facet_wrap(~ Strategy, ncol = 1)

  p3 <- ggplot(performance_summary,
               aes(x = GrossExposure, y = PortfolioRisk * 100,
                   color = Period, group = Period)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 3, aes(shape = Period)) +
    scale_color_manual(values = period_colors) +
    labs(title = "Annualized Portfolio Risk", x = "Gross Exposure",
         y = "Annualized Risk (%)") +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom",
          plot.title = element_text(hjust = 0.5, face = "bold")) +
    facet_wrap(~ Strategy, ncol = 1)

  grid.arrange(p1, p2, p3, ncol = 1)
}
