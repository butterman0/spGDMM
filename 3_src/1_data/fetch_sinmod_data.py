import xarray as xr
import pandas as pd

def fetch_sinmod_data(sinmod_data_path, sampled_locations, env_variables):
    """
    Fetch SINMOD data for specific (x, y, time) combinations defined in sampled_locations.

    Parameters:
        sinmod_data_path (str): Path to the SINMOD NetCDF file.
        sampled_locations (pd.DataFrame): DataFrame with columns ['x', 'y', 'time'] for sample locations.
        env_variables (list): List of environmental variable names to extract.

    Returns:
        pd.DataFrame: DataFrame with SINMOD data for the specified variables and locations.
    """
    # Open the SINMOD NetCDF file
    sinmod_data = xr.open_dataset(sinmod_data_path)
    
    # Prepare the result DataFrame by copying sampled_locations
    result = sampled_locations.copy()

    # Iterate over each environmental variable
    for var in env_variables:
        if var not in sinmod_data.variables:
            raise ValueError(f"Variable '{var}' not found in the SINMOD dataset.")
        
        # Fetch data for each row in sampled_locations
        result[var] = sampled_locations.apply(
            lambda row: sinmod_data[var].sel(
                x=row['x'], 
                y=row['y'], 
                time=row['time'], 
                method="nearest"
            ).values.item(),
            axis=1
        )
    
    # Close the dataset
    sinmod_data.close()

    return result
