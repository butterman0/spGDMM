import time
import xarray as xr
import numpy as np
import dask
from scipy.spatial.distance import pdist

def process_features(
    file_path,
    variable_name,
    layer_range = (0,1),
    x_range = (0,-1),
    y_range = (0,-1),
    chunks={"time":-1, "zc": -1, "yc": 50, "xc": 50},
    output_path=None
):
    """
    Process layer data for a specified variable in a NetCDF file.
    
    Parameters:
    - file_path (str): Path to the NetCDF file.
    - variable_name (str): Name of the variable to process.
    - layer_range (tuple): Range of layers to process (start, end).
    - x_range (slice, optional): Range of x coordinates to process. If None, process all.
    - output_path (str): Path to save the processed file (optional). If None, the result is not saved.
    
    Returns:
    - xarray.DataArray: The time-averaged bottom layer data.
    """
    time_start = time.time()

    # Open the dataset
    if chunks:
        ds = xr.open_dataset(file_path, chunks=chunks)
    else:
        ds = xr.open_dataset(file_path)

    print(f"\nAccessed the dataset after {time.time() - time_start:.2f} seconds")
    
    # Extract the variable
    if variable_name == "current_speed":
        data_var = (ds["u_velocity"][:, layer_range[0]:layer_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]]**2 + \
                   ds["v_velocity"][:, layer_range[0]:layer_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]]**2)**0.5
    else:
        data_var = ds[variable_name][:, layer_range[0]:layer_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]]

    # Calculate pairwise distances between points
    lon, lat = np.meshgrid(data_var["xc"].values, data_var["yc"].values)

    coordinates = np.vstack([lon.ravel(), lat.ravel()]).T

    try:
        pairwise_distances = pdist(coordinates, metric='euclidean')
    except MemoryError:
        print("Memory Error: Too many points to calculate pairwise distances")
        pairwise_distances = None
    ds.close()

    print(f"\nExtracted the layer data after {time.time() - time_start:.2f} seconds.\n\nStarting computation of statistics...")

    # Step 4: Calculate statistics across time
    # TODO: Decide on desired features
    time_avg_bottom_layer = data_var.mean(dim="time", skipna=True)
    time_percentiles = data_var.quantile([0.1,0.9], dim="time", skipna=True)

    print(f"\nComputed statistics after {time.time() - time_start:.2f} seconds")

    # Create a new DataArray with the (mean, 10th, 90th) percentiles and explicitly define the 'stat' dimension
    # Concatenate mean and percentiles in one line, drop 'quantile' and concatenate all together
    stats_array = xr.concat([time_avg_bottom_layer, time_percentiles.sel(quantile=0.1).drop_vars("quantile"), time_percentiles.sel(quantile=0.9).drop_vars("quantile")], dim="stat").rename(f"{variable_name}_features")

    # Name each value of the first dimension
    stats_array = stats_array.assign_coords(stat=["mean", "10th_percentile", "90th_percentile"])

    # Save to output file if specified
    if output_path:
        stats_array.to_netcdf(output_path)
    
    return stats_array, pairwise_distances