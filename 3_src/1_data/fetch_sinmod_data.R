library(ncdf4)

fetch_sinmod_data <- function(sinmod_data, sampled_locations, env_variables) {
    # Get no_sites # nolint
    no_sites <- nrow(sampled_locations)

    # Create an empty data frame to store the SINMOD data for multiple variables
    sinmod_df <- data.frame(matrix(NA, nrow = no_sites, ncol = length(env_variables)))
    colnames(sinmod_df) <- env_variables # Assign variable names as column names

    # Fetch data for each variable one by one at the sampled locations (avoid loading entire dataset)
    for (j in 1:length(env_variables)) {
        var <- env_variables[j]

        # Get dimensions of the variable (3D or 4D)
        var_dims <- length(sinmod_data$var[['temperature']]$dim)

        if (var_dims == 3) {
            # Fetch data for 3D variables like "elevation"
            for (i in 1:no_sites) {
                x <- sampled_locations[i, 1] # x-coordinate
                y <- sampled_locations[i, 2] # y-coordinate
                # Fetch data at the specific location
                sinmod_df[i, j] <- ncvar_get(sinmod_data, var, start = c(x, y, 1), count = c(1, 1, 1))
            }
        } else {
            # Fetch data for 4D variables like "temperature"
            for (i in 1:no_sites) {
                x <- sampled_locations[i, 1] # x-coordinate
                y <- sampled_locations[i, 2] # y-coordinate
                # Fetch data at the specific location and time/depth slice (time=0, depth=0)
                sinmod_df[i, j] <- ncvar_get(sinmod_data, var, start = c(x, y, 1, 1), count = c(1, 1, 1, 1))
            }
        }
    }
    return(sinmod_df)
}