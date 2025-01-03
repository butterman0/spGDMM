import xarray as xr
from xeofs.single import ExtendedEOF
from dask.distributed import Client
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import time


def main():
    total_start = time.time()  # Start the total timer

    # Initialize Dask client
    start = time.time()
    if 'client' not in globals():
        client = Client()
        print(f"Dask dashboard link: {client.dashboard_link}")
    else:
        print("Dask client already exists. Skipping Dask client creation.")
    print(f"Dask client setup time: {time.time() - start:.2f} seconds")

    # Define file paths
    input_path = "/cluster/projects/itk-SINMOD/coral-mapping/midnor/PhysStates_2019.nc"
    job_id = os.getenv('SLURM_JOB_ID', 'default_job_id')
    output_path = f"/cluster/home/haroldh/spGDMM/1_data/4_interim/EOFs/EEOF_basic_{job_id}.nc"

    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    # Load dataset and select variable
    start = time.time()
    try:
        ds = xr.open_dataset(input_path, chunks='auto').isel(zc=0).drop_vars('zc')
        ds = ds[['temperature', 'salinity', 'u_velocity', 'v_velocity', 'elevation']]
        ds = ds.sel(xc=slice(0, 100), yc=slice(0, 100), time=slice(0, 100))
        data = ds.assign_coords(time=('time', range(ds.sizes['time'])))
    except Exception as e:
        raise RuntimeError(f"Error loading or processing dataset: {e}")
    print(f"Dataset loading and processing time: {time.time() - start:.2f} seconds")

    # Perform Extended EOF analysis
    start = time.time()
    try:
        model = ExtendedEOF(
            n_modes=8,               # Number of modes to calculate
            tau=14,                  # Time delay between successive copies
            embedding=3,             # How many lagged dimensions to include
            n_pca_modes=50,          # How many PCA modes to retain (will run PCA before EEOF)
            center=True,             # Center the input data
            standardize=True,        # Standardize the data
            use_coslat=False,        # Do not apply cosine latitude weighting
            check_nans=True,         # Ignore NaNs in analysis
            solver='auto',           # Use the default solver
        )

        model.fit(data, dim=['time'])
        print("Extended EOF analysis completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Error during Extended EOF analysis: {e}")
    print(f"Extended EOF analysis time: {time.time() - start:.2f} seconds")

    # Save the result
    start = time.time()
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save(output_path)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving results: {e}")
    print(f"Saving results time: {time.time() - start:.2f} seconds")

    # Total time
    print(f"Total execution time: {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()

# auto, no pca, check_nansfalse