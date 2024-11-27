# library(splines)
library(fields)
library(splines2)
library(nimble)
library(vegan)
library(httpgd)
rm(list = ls())

#----------------------------------------------------------------
# load in and parse data
#----------------------------------------------------------------

midnor_env_pred  = read.csv("/cluster/home/haroldh/spGDMM/nimble_code/SINMOD/subgrid_data.csv")
sampled_locations = read.csv("/cluster/home/haroldh/spGDMM/nimble_code/SINMOD/vec_distance_matrix.csv")

# Parse data into location, environmental variables, and cover/presence data

location_mat = sampled_locations
envr_use = midnor_env_pred

# save number of sites

ns <- nrow(location_mat)

#----------------------------------------------------------------
# Define covariates that will be warped
#----------------------------------------------------------------

# Calculate geographical distance in km

vec_distance = unlist(location_mat)
vec_distance = vec_distance * 800 / 1000
vec_distance = matrix(vec_distance, ncol = 1)

# Define X to be environmental variables or a subset of them.

# How many knots do you want? What is the degree of the spline?
# Remember that in the specification, of the iSpline that the degree is
# one higher that what you say. Integration of m-spline adds one degree.

X = envr_use

deg = 3
knots = 1
df_use = deg + knots

# Get indices with values of -30000 from X, the default value for missing data
indices_with_neg_30000 <- which(X == -30000, arr.ind = TRUE)

formula_use = as.formula(paste("~ 0 +",paste(
  paste("iSpline(`",colnames(X),"`,degree=",deg - 1 ,",df = ",df_use, 
        " ,intercept = TRUE)",sep = ""),collapse = "+")))

# combine distance and environmental I-spline bases

I_spline_bases = model.matrix(formula_use,data = X)

# Set I_spline bases indices to NaN
I_spline_bases[indices_with_neg_30000[, 1], ] <- NaN

X_GDM = cbind(sapply(1:ncol(I_spline_bases),function(i){
  
  dist_temp = rdist(I_spline_bases[,i])
  vec_dist = dist_temp[upper.tri(dist_temp)]
  vec_dist
  
}),
iSpline(vec_distance,degree = deg -1,
        df = df_use,intercept = TRUE)
)

p = ncol(X_GDM)

colnames(X_GDM) = c(
  paste(rep(colnames(X),each = df_use ),"I",rep(1:df_use,times = ncol(X)),sep = ""),
  paste("dist","I",1:df_use,sep = "")
)

### Associate each dissimilarity with two sites (row and col index)

tmp = matrix(rep(1:nrow(X),each = nrow(X)),nrow = nrow(X))
col_ind = tmp[upper.tri(tmp)]
tmp = matrix(rep(1:nrow(X),times = nrow(X)),nrow = nrow(X))
row_ind = tmp[upper.tri(tmp)]

# Save X_GDM to a CSV file
write.csv(X_GDM, "X_GDM_predictors.csv", row.names = FALSE)