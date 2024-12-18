import pandas as pd
import random
import os
import xarray as xr

def create_sample_locations(no_sites, time_index, output_path=None, random_seed=1):
    """
    Create a DataFrame of sampled locations by randomly selecting valid (xc, yc) coordinates and assigning a time index.

    Parameters:
        no_sites (int): Number of random locations to generate.
        time_index (int): The time index to assign to all locations.
        valid_coordinates (list or array): Array or list of valid (xc, yc) coordinate pairs to sample from.
        output_path (str, optional): Path to save the resulting DataFrame as a CSV file.
                                     If provided, no_sites and time_index are appended to the filename.
        random_seed (int): Seed for reproducibility of random sampling. Default is 1.

    Returns:
        pd.DataFrame: DataFrame containing sampled locations with columns ['xc', 'yc', 'time_index'].
    """
    random.seed(random_seed)

    sinmod_path = "/cluster/home/haroldh/spGDMM/1_data/1_raw/biostates_surface_normalised.nc"

    # Open the SINMOD dataset and get the valid mask
    ds = xr.open_dataset(sinmod_path)['diatoms'].isel(time=0)
    valid_mask = ds.notnull()

    # Stack dimensions into pairs and filter valid locations
    stacked = valid_mask.stack(z=('xc', 'yc'))
    valid_coordinates = stacked[stacked.values].z.values

    ds.close()

    # Ensure valid_coordinates is a list of tuples
    valid_coordinates = list(valid_coordinates)

    # Ensure no_sites does not exceed available coordinates
    if no_sites > len(valid_coordinates):
        raise ValueError("Number of sites exceeds the available valid coordinates.")

    # Randomly sample from the valid coordinates
    sampled_coords = random.sample(valid_coordinates, no_sites)

    # Create the DataFrame
    sampled_locations = pd.DataFrame(sampled_coords, columns=['x', 'y'])
    sampled_locations['time_idx'] = time_index  # Assign the same time index to all locations

    # Save the DataFrame to the specified path if provided
    if output_path:
        # Append no_sites and time_index to the filename
        base, ext = os.path.splitext(output_path)
        output_file = f"{base}_sites{no_sites}_time{time_index}{ext}"
        sampled_locations.to_csv(output_file, index=False)
        print(f"Sampled locations saved to {output_file}")

    return sampled_locations