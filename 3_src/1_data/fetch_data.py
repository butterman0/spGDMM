import xarray as xr
import pandas as pd

def fetch_data(data_path, sampled_locations, type, no_modes=None, env_variables=None, normalise=False):
    """
    Fetch EOF data for specific (x, y, time) combinations defined in sampled_locations.

    Parameters:
        data_path (str): Path to the EOF NetCDF file.
        sampled_locations (pd.DataFrame): DataFrame with columns ['x', 'y', 'time_idx'] for sample locations.
        type (str): Type of data to fetch ('target', 'stat', 'EOF_time').
        no_modes (int, optional): Number of EOF modes to fetch. Required if type is 'EOF_time'.
        env_variables (list, optional): List of environmental variable names to extract. Required if type is 'target' or 'stat'.
        normalise (bool, optional): Whether to normalise the fetched data. Only applicable if type is 'target'.

    Returns:
        pd.DataFrame: DataFrame with EOF data for the specified variables and locations.
    """

    if no_modes and env_variables:
        raise ValueError("Cannot specify both 'no_modes' and 'env_variables'. Please choose one.")

    # Open the SINMOD NetCDF file
    data = xr.open_dataset(data_path)
    
    # Prepare the result DataFrame by copying sampled_locations
    result = sampled_locations.copy()

    # Iterate over each environmental variable
    if env_variables:
        for var in env_variables:
            
            if type == 'target':
                # Fetch data for each row in sampled_locations
                result[var] = sampled_locations.apply(
                    lambda row: data[var].isel(time=int(row['time_idx'])).sel(
                        xc=row['x'], 
                        yc=row['y'], 
                        method="nearest"
                    ).values.item(),
                    axis=1
                )
            elif type == 'stat':
                # Iterate over each `stat` dimension for the variable
                for stat in data[var].stat.values:
                    # Fetch data for each stat dimension and label column as `{var}_{stat}`
                    result[f"{var}_{stat}"] = sampled_locations.apply(
                        lambda row: data[var].sel(stat=stat).sel(xc=row['x'], yc=row['y'], method="nearest").values.item(),
                        axis=1
                    )
    
    elif no_modes:
        if type == 'EOF_time':
            
            data = data.to_dataarray()

            if no_modes > data.mode.values.max():
                raise ValueError("Number of modes to keep exceeds the maximum number of modes in the data")

            # Iterate over each `mode` dimension for the variable
            for mode in range(1, no_modes + 1):
                # Fetch data for each mode dimension and label column as `{var}_{stat}`
                result[f"Mode_{mode}"] = sampled_locations.apply(
                    lambda row: data.sel(mode=mode, xc=row['x'], yc=row['y'], method="nearest").isel(time=int(row['time_idx'])).values.item(),
                    axis=1
                )
        else:
            raise ValueError("Check type of data to fetch")
    # Close the dataset
    data.close()

    if type == 'target':
        result[env_variables] = result[env_variables].fillna(0)
        if normalise:
            result[env_variables] = result[env_variables].div(result[env_variables].sum(axis=1), axis=0)
            
    return result
