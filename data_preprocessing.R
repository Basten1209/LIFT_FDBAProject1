# ============================================================================
# Script 1: Data Preprocessing
# ============================================================================
# Objective: Create a unified, clean time-series dataset of asset returns
# from multiple raw CSV files with non-uniform formats
# ============================================================================

# Load Required Libraries ====================================================
library(readr)
library(dplyr)
library(lubridate)
library(xts)
library(zoo)
library(stringr)

# Setup Paths ================================================================
raw_data_dir <- "data/raw/"
processed_data_dir <- "data/processed/"

# Create processed directory if it doesn't exist
if (!dir.exists(processed_data_dir)) {
  dir.create(processed_data_dir, recursive = TRUE)
}

# Configuration ==============================================================
# Use SPY (S&P 500 ETF) as the reference for US trading days
REFERENCE_ASSET <- "SPY"
# Minimum data requirement: assets must have at least this percentage of reference data
MIN_DATA_COVERAGE <- 0.90  # 90% of reference trading days (excludes newer assets like most cryptos)

cat("Starting data preprocessing...\n")
cat(sprintf("Reference asset for US trading days: %s\n", REFERENCE_ASSET))
cat(sprintf("Minimum data coverage required: %.0f%%\n\n", MIN_DATA_COVERAGE * 100))

# Define Flexible CSV Loader Function ========================================
# This function intelligently handles diverse CSV formats with:
# - Non-standard headers and metadata rows
# - Inconsistent column naming conventions
# - Various date formats

load_and_clean_csv <- function(file_path) {
  
  tryCatch({
    # Extract asset name from filename (remove .csv extension)
    asset_name <- tools::file_path_sans_ext(basename(file_path))
    
    # Read the entire file to inspect structure
    raw_lines <- readLines(file_path, warn = FALSE)
    
    # Strategy: Most files have the pattern:
    # Row 1: Column headers (Price, Close, High, Low, Open, Volume)
    # Row 2: Ticker information
    # Row 3: Date header row
    # Row 4+: Actual data
    
    # Read from row 4 onwards (skip first 3 rows)
    data <- read_csv(file_path, 
                     skip = 3, 
                     col_names = c("Date", "Close", "High", "Low", "Open", "Volume"),
                     show_col_types = FALSE)
    
    # Parse dates - handle various date formats
    data <- data %>%
      mutate(Date = as.Date(Date, format = "%Y-%m-%d"))
    
    # Filter out rows with invalid dates or missing Close prices
    data <- data %>%
      filter(!is.na(Date), !is.na(Close))
    
    # Select only Date and Close price
    data <- data %>%
      select(Date, Close)
    
    # Rename Close column to asset name
    colnames(data)[2] <- asset_name
    
    # Convert to xts object for time series operations
    data_xts <- xts(data[, 2], order.by = data$Date)
    
    return(data_xts)
    
  }, error = function(e) {
    cat(sprintf("Warning: Failed to process %s - %s\n", basename(file_path), e$message))
    return(NULL)
  })
}

# Process All Raw Files ======================================================
cat("Loading and cleaning raw CSV files...\n")

# Get list of all CSV files in raw directory
csv_files <- list.files(raw_data_dir, pattern = "\\.csv$", full.names = TRUE)

cat(sprintf("Found %d CSV files to process\n", length(csv_files)))

# Process each file and store in a list
price_series_list <- list()

for (file in csv_files) {
  asset_data <- load_and_clean_csv(file)
  
  if (!is.null(asset_data) && nrow(asset_data) > 0) {
    asset_name <- tools::file_path_sans_ext(basename(file))
    price_series_list[[asset_name]] <- asset_data
    cat(sprintf("  Loaded: %s (%d observations)\n", 
                asset_name, nrow(asset_data)))
  } else if (!is.null(asset_data)) {
    cat(sprintf("  Skipped: %s (0 observations)\n", 
                tools::file_path_sans_ext(basename(file))))
  }
}

cat(sprintf("\nSuccessfully loaded %d assets\n", length(price_series_list)))

# Extract Reference Asset (US Trading Days) ==================================
cat("\nExtracting reference asset for US trading calendar...\n")

if (!REFERENCE_ASSET %in% names(price_series_list)) {
  stop(sprintf("Reference asset '%s' not found in the data!", REFERENCE_ASSET))
}

reference_data <- price_series_list[[REFERENCE_ASSET]]
reference_dates <- index(reference_data)
n_reference_days <- length(reference_dates)

cat(sprintf("Reference asset: %s\n", REFERENCE_ASSET))
cat(sprintf("Reference trading days: %d\n", n_reference_days))
cat(sprintf("Date range: %s to %s\n", min(reference_dates), max(reference_dates)))

# Filter Assets by Data Coverage ============================================
cat("\nFiltering assets by data coverage...\n")

filtered_assets <- list()
rejected_assets <- c()

for (asset_name in names(price_series_list)) {
  asset_data <- price_series_list[[asset_name]]
  
  # Align asset to reference dates (keep only dates that exist in reference)
  aligned_data <- asset_data[reference_dates]
  
  # Count non-NA values after alignment
  valid_count <- sum(!is.na(aligned_data))
  coverage <- valid_count / n_reference_days
  
  if (coverage >= MIN_DATA_COVERAGE) {
    filtered_assets[[asset_name]] <- aligned_data
    cat(sprintf("  ✓ %s: %.1f%% coverage (%d/%d days)\n", 
                asset_name, coverage * 100, valid_count, n_reference_days))
  } else {
    rejected_assets <- c(rejected_assets, asset_name)
    cat(sprintf("  ✗ %s: %.1f%% coverage - REJECTED (insufficient data)\n", 
                asset_name, coverage * 100))
  }
}

cat(sprintf("\nAssets retained: %d\n", length(filtered_assets)))
cat(sprintf("Assets rejected: %d\n", length(rejected_assets)))

if (length(rejected_assets) > 0) {
  cat(sprintf("\nRejected assets: %s\n", paste(rejected_assets, collapse = ", ")))
}

# Merge and Align Time Series ================================================
cat("\nMerging and aligning time series to US trading days...\n")

# Merge all filtered time series into a single xts object
# All assets are already aligned to reference_dates
merged_prices <- do.call(merge.xts, filtered_assets)

cat(sprintf("Merged dataset dimensions: %d dates × %d assets\n", 
            nrow(merged_prices), ncol(merged_prices)))

# Handle Missing Values ======================================================
cat("\nHandling missing values using forward-fill method...\n")

# Count missing values before imputation
na_count_before <- sum(is.na(merged_prices))
cat(sprintf("  Missing values before imputation: %d (%.2f%%)\n", 
            na_count_before, 
            100 * na_count_before / (nrow(merged_prices) * ncol(merged_prices))))

# Apply forward-fill (Last Observation Carried Forward)
# na.locf carries forward the last non-NA value
merged_prices_filled <- na.locf(merged_prices, na.rm = FALSE)

# For any remaining NAs at the beginning, use backward fill
merged_prices_filled <- na.locf(merged_prices_filled, fromLast = TRUE, na.rm = FALSE)

# Count remaining missing values
na_count_after <- sum(is.na(merged_prices_filled))
cat(sprintf("  Missing values after imputation: %d\n", na_count_after))

# Remove any rows that still have all NAs (if any)
merged_prices_filled <- merged_prices_filled[complete.cases(merged_prices_filled), ]

cat(sprintf("Final dataset after cleaning: %d dates × %d assets\n", 
            nrow(merged_prices_filled), ncol(merged_prices_filled)))

# Calculate Log Returns ======================================================
cat("\nCalculating log returns...\n")

# Log return formula: r_t = log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
# Using diff.xts which computes differences: log(P_t) - log(P_{t-1})
log_returns <- diff(log(merged_prices_filled), lag = 1)

# Remove the first row of NAs (result of diff operation)
log_returns <- log_returns[-1, ]

cat(sprintf("Log returns dimensions: %d dates × %d assets\n", 
            nrow(log_returns), ncol(log_returns)))

# Check for infinite or extreme values
inf_count <- sum(is.infinite(as.matrix(log_returns)))
if (inf_count > 0) {
  cat(sprintf("Warning: Found %d infinite values in log returns\n", inf_count))
}

# Calculate basic statistics (no standardization)
cat("\nCalculating return statistics...\n")
means <- colMeans(log_returns, na.rm = TRUE)
sds <- apply(log_returns, 2, sd, na.rm = TRUE)

cat(sprintf("Return statistics:\n"))
cat(sprintf("  Mean of means: %.6f (%.4f%% daily)\n", mean(means), mean(means) * 100))
cat(sprintf("  Mean of std devs: %.6f (%.4f%% daily)\n", mean(sds), mean(sds) * 100))

# Save Final Output ==========================================================
cat("\nSaving processed returns to CSV...\n")

# Convert xts to data frame for saving (using original log returns, NOT standardized)
output_df <- data.frame(Date = index(log_returns), 
                        coredata(log_returns))

# Save to CSV
output_file <- file.path(processed_data_dir, "processed_returns.csv")
# Normalize path to remove any double slashes
output_file <- normalizePath(output_file, winslash = "/", mustWork = FALSE)

# Try to write, handle permission errors
tryCatch({
  write.csv(output_df, output_file, row.names = FALSE)
}, error = function(e) {
  if (grepl("Permission denied|cannot open", e$message)) {
    warning(sprintf("\nCannot write to %s - file may be open in another program.", output_file))
    # Try alternative filename
    alt_file <- file.path(processed_data_dir, 
                          sprintf("processed_returns_%s.csv", 
                                  format(Sys.time(), "%Y%m%d_%H%M%S")))
    cat(sprintf("\nAttempting to save to alternative file: %s\n", alt_file))
    write.csv(output_df, alt_file, row.names = FALSE)
    output_file <<- alt_file  # Update output_file in parent scope
  } else {
    stop(e)
  }
})

cat(sprintf("Saved: %s\n", output_file))
cat(sprintf("  Dimensions: %d rows × %d columns (including Date)\n", 
            nrow(output_df), ncol(output_df)))

# Summary Statistics =========================================================
cat(paste0("\n", strrep("=", 60), "\n"))
cat("DATA PREPROCESSING SUMMARY\n")
cat(paste0(strrep("=", 60), "\n"))
cat(sprintf("Total assets processed: %d\n", ncol(log_returns)))
cat(sprintf("Date range: %s to %s\n", 
            min(index(log_returns)), 
            max(index(log_returns))))
cat(sprintf("Total observations: %d\n", nrow(log_returns)))
cat(sprintf("Output file: %s\n", output_file))
cat(paste0(strrep("=", 60), "\n"))
cat("\nData preprocessing completed successfully!\n\n")

# Optional: Display first few rows
cat("First 5 rows of log returns:\n")
print(head(output_df, 5))

