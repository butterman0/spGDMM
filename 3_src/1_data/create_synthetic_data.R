library(fields)
library(splines2)
library(nimble)
library(vegan)

#----------------------------------------------------------------
# Function to create synthetic eDNA data
#----------------------------------------------------------------

## TODO: Add documentation
## TODO: Make eDNA data more realistic i.e. many more rare species, some very abundant species, etc.

create_synthetic_rel_abundance_data <- function(no_sites = 10, no_species = 20, seed = 1) {
    set.seed(seed) # Set seed for reproducibility

    # Create a random OTU table with a standard abundance distribution
    otu_table <- matrix(rlnorm(no_sites * no_species, meanlog = 0, sdlog = 1), nrow = no_sites, ncol = no_species)

    # Normalize to get relative abundances
    otu_table <- otu_table / rowSums(otu_table)

    # Convert to data frame for better readability
    otu_table <- as.data.frame(otu_table)
    colnames(otu_table) <- paste0("Species_", 1:no_species)
    rownames(otu_table) <- paste0("Site_", 1:no_sites)

    # Sum the rows and columns
    row_sums <- rowSums(otu_table)

    return(as.matrix(otu_table))
}

# # # Example usage

# otu_table <- create_synthetic_rel_abundance_data(no_sites = 10, no_species = 20, seed = 1)

# # Plotting row
# row <- 4

# # Order species by reducing abundance
# sorted_otu_table <- t(apply(otu_table, 1, function(x) sort(x, decreasing = TRUE)))

# # Set the background color to white
# par(bg = "white")
# # Plot the specified row
# barplot(as.numeric(sorted_otu_table[row, ]), main = paste("Abundance of Species at Site", row), xlab = "Species", ylab = "Abundance", las = 2, col = "blue")

# write.csv(otu_table, "1_data/1_raw/synthetic_abundance/otu_table.csv")