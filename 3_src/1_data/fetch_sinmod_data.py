import xarray as xr
import pandas as pd

def fetch_sinmod_data(sinmod_data_path, sampled_locations, env_variables, target=True, normalise=False):
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
        
        if target:
            # Fetch data for each row in sampled_locations
            result[var] = sampled_locations.apply(
                lambda row: sinmod_data[var].isel(time=int(row['time_idx'])).sel(
                    xc=row['x'], 
                    yc=row['y'], 
                    method="nearest"
                ).values.item(),
                axis=1
            )
        else:
            # Iterate over each `stat` dimension for the variable
            for stat in sinmod_data[var].stat.values:
                # Fetch data for each stat dimension and label column as `{var}_{stat}`
                result[f"{var}_{stat}"] = sampled_locations.apply(
                    lambda row: sinmod_data[var].sel(stat=stat).sel(xc=row['x'], yc=row['y'], method="nearest").values.item(),
                    axis=1
                )
    # Close the dataset
    sinmod_data.close()

    if target:
        result[env_variables] = result[env_variables].fillna(0)
        if normalise:
            result[env_variables] = result[env_variables].div(result[env_variables].sum(axis=1), axis=0)
            
    return result
