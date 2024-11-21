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

# Step 1: Extract posterior samples for the log-transformed coefficients and intercept
log_beta_samples <- post_samples$samples[, grepl('log_beta', colnames(post_samples$samples))]  # Extract all log_beta samples
beta_0_samples <- post_samples$samples[, "beta_0"]  # Extract the intercept samples

# Check the shape of log_beta_samples and X_GDM_predictors
print(paste("Shape of log_beta_samples:", paste(dim(log_beta_samples), collapse = "x")))
print(paste("Shape of X_GDM_predictors:", paste(dim(X_GDM_predictors), collapse = "x")))

# Ensure that the number of columns in X_GDM_predictors matches the number of coefficients in log_beta_samples
n_predictors <- ncol(X_GDM_predictors)
n_samples <- nrow(log_beta_samples)

if (n_predictors != ncol(log_beta_samples)) {
  stop("Number of predictors in X_GDM_predictors does not match number of coefficients in log_beta_samples.")
}

# Step 2: Calculate predictions for each posterior sample
# Initialize a matrix to store the predictions
predictions <- matrix(NA, nrow = nrow(X_GDM_predictors), ncol = n_samples)

# Loop over each posterior sample and calculate the prediction
for (i in 1:n_samples) {
  # Prediction = intercept + X_GDM * log(beta) (log-transformed coefficients)
  predictions[, i] <- beta_0_samples[i] + X_GDM_predictors %*% log_beta_samples[i, ]
}

# Step 3: Summarize the predictions
# Compute the mean prediction across all posterior samples
mean_predictions <- rowMeans(predictions)

# Compute 95% credible intervals (2.5th and 97.5th percentiles)
lower_95 <- apply(predictions, 1, function(x) quantile(x, 0.025))
upper_95 <- apply(predictions, 1, function(x) quantile(x, 0.975))

# Step 4: Store the results in a DataFrame
predictions_df <- data.frame(
  mean_prediction = mean_predictions,
  lower_95 = lower_95,
  upper_95 = upper_95
)

# Save the predictions DataFrame to a CSV file
write.csv(predictions_df, "predictions_summary.csv", row.names = FALSE)

# Assuming predictions_df is the data frame containing 'mean_prediction' in the first column

# Number of pairwise comparisons (i.e., upper triangle elements)
n_pairs <- length(predictions_df$mean_prediction)

# Calculate the number of sites (n) based on the number of pairwise comparisons
n_sites <- (-1 + sqrt(1 + 8 * n_pairs)) / 2
n_sites <- round(n_sites)

# Step 1: Create an empty square matrix of size n_sites x n_sites
prediction_matrix <- matrix(NA, nrow = n_sites+1, ncol = n_sites+1)

# Step 2: Fill the upper triangle of the matrix with predictions
upper_tri_indices <- upper.tri(prediction_matrix)
prediction_matrix[upper_tri_indices] <- predictions_df$mean_prediction

# Step 3: Fill the lower triangle by mirroring the upper triangle
prediction_matrix[lower.tri(prediction_matrix)] <- t(prediction_matrix)[lower.tri(prediction_matrix)]

# Now prediction_matrix contains the predicted pairwise dissimilarities in matrix form

# Step 4: (Optional) Save the matrix to a CSV file
write.csv(prediction_matrix, "predicted_dissimilarity_matrix.csv", row.names = FALSE)

# View the resulting matrix
head(prediction_matrix)
