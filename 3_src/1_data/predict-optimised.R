# Load the necessary libraries
library(fields)
library(splines2)
library(nimble)
library(vegan)
library(httpgd)
rm(list = ls())

# Load posterior samples from nimble MCMC
post_samples <- readRDS("mod1_panama_post_samples.rds")  # Assuming you've saved the posterior samples as RDS

# Load the predictor matrix (X_GDM) created earlier
X_GDM_predictors <- read.csv("X_GDM_predictors.csv")
X_GDM_predictors <- as.matrix(X_GDM_predictors)  # Convert to matrix for matrix multiplication

# Extract posterior samples for the log-transformed coefficients and intercept
log_beta_samples <- post_samples$samples[, grepl('log_beta', colnames(post_samples$samples))]  # Extract all log_beta samples
beta_0_samples <- post_samples$samples[, "beta_0"]  # Extract the intercept samples

# Get number of samples and predictors
n_samples <- nrow(log_beta_samples)
n_predictors <- ncol(X_GDM_predictors)
n_pairs <- nrow(X_GDM_predictors)
n_sites <- (-1 + sqrt(1 + 8 * n_pairs)) / 2
n_sites <- round(n_sites)

# Ensure that the number of predictors matches the number of coefficients in log_beta_samples
if (n_predictors != ncol(log_beta_samples)) {
  stop("Number of predictors in X_GDM_predictors does not match number of coefficients in log_beta_samples.")
}

# Set up the progress bar
progress_bar <- txtProgressBar(min = 1, max = n_samples, style = 3)

# Initialize prediction matrix to store the pairwise dissimilarity matrix
prediction_matrix <- matrix(NA, nrow = n_sites + 1, ncol = n_sites + 1)

# Adjust chunk size to ensure it's manageable for memory
chunk_size <- 10  # Adjust this depending on memory capacity
for (start_idx in seq(1, n_samples, by = chunk_size)) {
  end_idx <- min(start_idx + chunk_size - 1, n_samples)
  chunk_beta_samples <- log_beta_samples[start_idx:end_idx, ]
  chunk_beta_0_samples <- beta_0_samples[start_idx:end_idx]
  
  # Calculate predictions for the current chunk
  predictions_chunk <- matrix(NA, nrow = nrow(X_GDM_predictors), ncol = length(start_idx:end_idx))
  
  # Loop over each posterior sample chunk and calculate predictions
  for (i in start_idx:end_idx) {
    predictions_chunk[, i - start_idx + 1] <- chunk_beta_0_samples[i - start_idx + 1] + X_GDM_predictors %*% chunk_beta_samples[i - start_idx + 1, ]
  }

  # Compute summary statistics for the chunk
  mean_predictions_chunk <- rowMeans(predictions_chunk)
  
  # Fill the prediction matrix's upper triangle with predictions
  upper_tri_indices <- upper.tri(prediction_matrix)
  prediction_matrix[upper_tri_indices] <- mean_predictions_chunk
  
  # Fill the lower triangle by mirroring the upper triangle
  prediction_matrix[lower.tri(prediction_matrix)] <- t(prediction_matrix)[lower.tri(prediction_matrix)]

  # Update the progress bar
  setTxtProgressBar(progress_bar, end_idx)
}

# Close the progress bar
close(progress_bar)

# Save the prediction matrix to a CSV file
write.csv(prediction_matrix, "predicted_dissimilarity_matrix_50x50.csv", row.names = FALSE)

# View the resulting matrix (optional)
head(prediction_matrix)
