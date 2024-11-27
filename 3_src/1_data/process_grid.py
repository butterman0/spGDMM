import xarray as xr
import numpy as np

# List of environmental variables of interest
env_vars = ["temperature", "salinity", "u_velocity", "v_velocity"]

ds = xr.open_dataset("/cluster/projects/itk-SINMOD/coral-mapping/midnor/PhyStates_2019.nc")

sub_grid = True

if sub_grid:
    subgrid_data = {}

    for var in env_vars:
        subgrid_data[var] = ds.variables[var][0, 0, start:(start+subgrid_size), start:(start+subgrid_size)]
        subgrid_data[var] = subgrid_data[var].flatten()