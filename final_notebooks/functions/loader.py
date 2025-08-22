from __future__ import annotations
from typing import Optional, Union, Tuple, Dict, Any, Literal
import xarray as xr
import s3fs
import fsspec
import numpy as np
import pathlib
import zarr
from datetime import datetime
import shutil
import os
import pandas as pd
from pydantic import BaseModel, Field, confloat
from langchain.tools import Tool, StructuredTool

### helper functions to normalize coords

def _select_variable(ds: xr.Dataset, var: Union[str, Dict[str, str]]) -> str:
    """
    Pick a variable name from a Dataset.

    If var is a string, it's treated as a variable name.
    If cf_xarray is available and var is a dict, it's used for CF-aware selection
    (e.g., `{"standard_name": "sea_surface_temperature"}`).
    A fallback attempts to match attributes case-insensitively if CF-selection fails.
    """
    if isinstance(var, str):
        if var in ds.data_vars:
            return var
        for k in ds.data_vars:
            if k.lower() == var.lower():
                return k
        raise KeyError(f"Variable '{var}' not found. Available: {list(ds.data_vars)}")

    #if _HAS_CF:
    #    for key in ["standard_name", "long_name", "units"]:
    #        if key in var:
    #            matches = ds.cf.select_variables(**{key: var[key]})
    #            if matches:
    #                return list(matches)[0]

    key_order = ["standard_name", "long_name", "units"]
    for candidate in ds.data_vars:
        attrs = {k: str(ds[candidate].attrs.get(k, "")).lower() for k in key_order}
        if any((k in var) and (str(var[k]).lower() == attrs.get(k, "")) for k in key_order):
            return candidate
    raise KeyError(f"Could not locate variable from hints {var}. Variables: {list(ds.data_vars)}")


def _get_coord_names(ds: xr.Dataset) -> Tuple[str, str]:
    """
    Get the longitude and latitude coordinate names from the dataset.
    Supports both long ('longitude', 'latitude') and short ('lon', 'lat') names.
    
    Returns
    -------
    tuple of (lon_name, lat_name)
    """
    lon_name = next((name for name in ['longitude', 'lon'] if name in ds.coords), None)
    lat_name = next((name for name in ['latitude', 'lat'] if name in ds.coords), None)
    
    if not lon_name or not lat_name:
        raise ValueError(f"Could not find longitude/latitude coordinates. Found: {list(ds.coords)}")
    
    return lon_name, lat_name


def _infer_target_lon_frame(lon_min: float, lon_max: float) -> str:
    """
    Infers whether user-provided longitude bounds are in 0-360 or -180-180 frame.
    """
    return "0-360" if (lon_min >= 0 and lon_max <= 360) else "-180-180"


def _coerce_longitudes(ds: xr.Dataset, target_frame: str, assume_frame: Optional[str] = None) -> xr.Dataset:
    """
    Coerce dataset longitudes to a target frame ('0-360' or '-180-180').
    Works with either 'longitude' or 'lon' coordinate names.
    """
    lon_name, _ = _get_coord_names(ds)
    
    lon = ds[lon_name].values
    if assume_frame:
        current = assume_frame
    else:
        current = "0-360" if (np.nanmin(lon) >= 0 and np.nanmax(lon) <= 360) else "-180-180"

    if current == target_frame:
        return ds

    if target_frame == "0-360":
        lon_new = np.mod(lon, 360.0)
    else:  # target is -180-180
        lon_new = ((lon + 180) % 360) - 180
    
    ds = ds.assign_coords({lon_name: lon_new})
    return ds.sortby(lon_name)


def _ensure_lat_monotonic(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensures the latitude coordinate is monotonically increasing.
    Works with either 'latitude' or 'lat' coordinate names.
    """
    _, lat_name = _get_coord_names(ds)
    
    if ds[lat_name].ndim == 1 and ds[lat_name].values[0] > ds[lat_name].values[-1]:
        return ds.sortby(lat_name)
    return ds

def download_to_temp(
    ds: Union[xr.Dataset, xr.DataArray],
    *,
    max_size_gb: float = 1.0,
    temp_dir: Optional[str] = "./temp",
    filename: Optional[str] = None,
) -> str:
    """
    Download a dataset/array to a temporary directory with size checks.
    
    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The dataset or array to save
    max_size_gb : float, default 1.0
        Maximum allowed size in gigabytes
    temp_dir : str, optional
        Directory to save to. If None, uses system temp directory
    filename : str, optional
        Name for the saved file. If None, generates a unique name
        
    Returns
    -------
    str
        Path to the saved file
        
    Raises
    ------
    ValueError
        If estimated size exceeds max_size_gb
    """
    
    # Estimate size in bytes (assuming float32)
    bytes_per_value = 4  # float32
    if isinstance(ds, xr.DataArray):
        n_values = ds.size
    else:
        n_values = sum(var.size for var in ds.values())
    
    estimated_gb = (n_values * bytes_per_value) / (1024**3)
    
    if estimated_gb > max_size_gb:
        raise ValueError(
            f"Dataset too large to download safely. "
            f"Estimated size: {estimated_gb:.2f} GB, "
            f"Maximum allowed: {max_size_gb:.2f} GB"
        )
    
    # Set up temporary directory
    if temp_dir is None:
        temp_dir = os.path.join('..', '..', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(ds, xr.DataArray):
            prefix = ds.name or "data"
        else:
            prefix = "dataset"
        filename = f"{prefix}_{timestamp}.nc"
    
    # Ensure .nc extension
    if not filename.endswith('.nc'):
        filename += '.nc'
    
    # Create full path
    save_path = os.path.join(temp_dir, filename)
    
    # Prepare data and encoding for saving
    if isinstance(ds, xr.DataArray):
        # For DataArray, create encoding for the single variable
        var_name = ds.name or 'data'
        ds_to_save = ds.to_dataset(name=var_name)
        encoding = {var_name: {'zlib': True, 'complevel': 5}}
    else:
        # For Dataset, create encoding for all variables
        ds_to_save = ds
        encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.variables}
    
    # Save the data
    print(f"Saving to {save_path} (estimated size: {estimated_gb:.2f} GB)")

    #size_bytes = ds.to_array().nbytes
    #size_gb = size_bytes / (1024**3)
    #print(f"Size in memory: {size_gb}")
    
    #if actual_gb > max_size_gb:
        # Clean up and raise error if actual size exceeds limit
    #    save_path.unlink()
    #    raise ValueError(
    #        f"Actual file size ({actual_gb:.2f} GB) exceeded maximum "
    #        f"allowed size ({max_size_gb:.2f} GB). File was deleted."
    #    )
    
    ds_to_save["time"] = pd.to_datetime(ds_to_save["time"].values).tz_localize("UTC").tz_convert(None)
    ds_to_save.to_netcdf(save_path, encoding=encoding)
    
    
    return save_path

def load_climate_data(
    store: Union[str, s3fs.S3Map, fsspec.mapping.FSMap],
    variable: Optional[Union[str, Dict[str, str]]],
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    *,
    time_range: Optional[Tuple[str, str]] = None,
    resample_to: Optional[str] = None,
    chunks: Optional[Dict[str, int]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
):
    """
    Load climate data from cloud storage (S3 or GCS) with consistent processing.
    
    Parameters
    ----------
    store : str or s3fs.S3Map or fsspec.mapping.FSMap
        Either a URL string (e.g., "s3://..." or "gs://...") or an existing store object
    variable : str or dict
        Variable name or CF-style selector (e.g., {"standard_name": "air_temperature"})
    lon_range : tuple of float, optional
        (min_longitude, max_longitude) in dataset's native frame. If None, keeps all longitudes.
    lat_range : tuple of float, optional
        (min_latitude, max_latitude). If None, keeps all latitudes.
    time_range : tuple of str, optional
        (start_date, end_date) as ISO strings. If None, keeps all times.
    resample_to : str, optional
        If provided, resample time dimension (e.g., "MS" for month start)
    chunks : dict, optional
        Dask chunks specification (e.g., {"time": 1024})
    storage_options : dict, optional
        Only used if store is a string URL. Additional storage options for cloud access.
        
    Returns
    -------
    dict
        A structured summary including local path and dataset metadata.
    """
    
    # Open dataset WITHOUT chunks to avoid initial chunk warnings, then rechunk later
    if isinstance(store, str):
        ds = xr.open_dataset(
            store,
            engine="zarr",
            chunks=None,
            backend_kwargs={"storage_options": storage_options},
        )
    else:
        ds = xr.open_zarr(store, chunks=None)
    
    # Get coordinate names
    lon_name, lat_name = _get_coord_names(ds)
    
    # Subset space and time
    region = {}
    if lon_range is not None and lat_range is not None:
        region.update({
            lon_name: slice(*lon_range),
            lat_name: slice(*lat_range)
        })

    if "time" in ds.coords:
        ds["time"] = pd.to_datetime(ds["time"].values).tz_localize("UTC")
    
    if time_range is not None:
        region["time"] = slice(*time_range)
    
    # Only apply selection if we have regions to subset
    if region:
        ds = ds.sel(**region)
    
    # Handle longitude frame and monotonic latitude
    if lon_range is not None:
        target_frame = _infer_target_lon_frame(*lon_range)
        ds = _coerce_longitudes(ds, target_frame)
    #ds = _ensure_lat_monotonic(ds)
    
    # Optional time resampling
    if resample_to:
        ds = ds.resample(time=resample_to).mean()
    
    # Ensure consistent dimension order
    dims = list(ds.dims)
    core_dims = ["time", "latitude", "longitude"]
    core_dims = [d for d in core_dims if d in dims]
    other_dims = [d for d in dims if d not in core_dims]
    final_dims = core_dims + other_dims
    ds = ds.transpose(*final_dims)

    # Variable selection
    selected_variable: Optional[str] = None
    if variable:
        selected_variable = _select_variable(ds, variable)
        ds = ds[selected_variable]

    # Rechunk after loading/subsetting
    default_chunks = {k: v for k, v in [("time", 24), ("latitude", 256), ("longitude", 256)] if k in ds.dims}
    ds = ds.chunk(chunks or default_chunks)

    # Save to temp and build metadata
    path = download_to_temp(ds)

    # Build a small preview safely
    try:
        preview_stats = {
            "min": float(ds.min().compute().item()),
            "max": float(ds.max().compute().item()),
            "mean": float(ds.mean().compute().item()),
        }
    except Exception:
        preview_stats = {}

    metadata = {
        "local_path": path,
        "variable": selected_variable,
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "coords": {c: (str(ds[c].dtype), ds[c].ndim) for c in ds.coords},
        "chunks": {k: tuple(map(int, v)) for k, v in ds.chunks.items()} if hasattr(ds, "chunks") and ds.chunks else {},
        "time_coverage": {
            "start": str(pd.to_datetime(ds.time.values.min())) if "time" in ds.coords and ds.time.size > 0 else None,
            "end": str(pd.to_datetime(ds.time.values.max())) if "time" in ds.coords and ds.time.size > 0 else None,
        },
        "preview": preview_stats,
    }

    return metadata


class ClimateDataParams(BaseModel):
    store: Literal[
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
        "gcs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/IO.zarr"
    ] = Field(
        ...,
        description="Available stores to read from."
    ),
    variable: Optional[Union[str, Dict[str, str]]] = Field(
        None,
        description="Variable name or CF-style selector, e.g.'air_temperature'"
    )
    lon_range: Optional[Tuple[confloat(ge=-180, le=180), confloat(ge=-180, le=180)]] = Field(
        None, description="Longitude range (min_lon, max_lon) in degrees"
    )
    lat_range: Optional[Tuple[confloat(ge=-90, le=90), confloat(ge=-90, le=90)]] = Field(
        None, description="Latitude range (min_lat, max_lat) in degrees"
    )
    time_range: Optional[Tuple[str, str]] = Field(
        None, description="Time range as tuple of ISO strings, e.g., ('2000-01-01', '2000-01-31')"
    )
    resample_to: Optional[str] = Field(
        None, description="Resample frequency string for time dimension, e.g., 'MS' for month start"
    )
    chunks: Optional[Dict[str, int]] = Field(
        None, description="Dask chunks specification, e.g., {'time': 1024}"
    )
    storage_options: Optional[Dict[str, Any]] = Field(
        None, description="Extra options for cloud storage access if 'store' is a URL string"
    )


def create_loader_tool():
    return StructuredTool.from_function(
        func=load_climate_data,
        name="load_climate_data",
        description="A general use function for downloading datasets from various sources on the internet. When choosing the variable, only use the variable name and nothing else. Returns a JSON dict with local_path and metadata.",
        args_schema=ClimateDataParams
    )