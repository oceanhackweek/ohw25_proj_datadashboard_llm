from typing import Optional, List
from pydantic import BaseModel

class TemporalBounds(BaseModel):
    start_time: str
    end_time: str

class SpatialBounds(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

class Access(BaseModel):
    platform: str
    path: str
    access_function: Optional[str] = ""
    other_args: Optional[dict] = {}

class Variable(BaseModel):
    standard_name: str
    description: str
    units: str

class Variables(BaseModel):
    variables: List[Variable]


class Dataset(BaseModel):
    name: str
    description: str
    temporal_bounds: TemporalBounds
    spatial_bounds: SpatialBounds
    variables: Variables
    access: Access

class DatasetCollection(BaseModel):
    datasets: List[Dataset]

# --- Add these at the bottom of dataset.py ---
import xarray as xr

def load_mur(path: str, platform: str | None = None, meta: dict | None = None, **kwargs):
    """
    Open the public GHRSST MUR L4 SST Zarr on S3.
    Catalog path (from datasets.json): s3://mur-sst/zarr-v1/
    """
    # anon S3 read
    try:
        return xr.open_zarr(path, storage_options={"anon": True}, consolidated=True)
    except Exception:
        # some mirrors aren’t consolidated; try non-consolidated read
        return xr.open_zarr(path, storage_options={"anon": True}, consolidated=False)

def load_indian_ocean(path: str, platform: str | None = None, meta: dict | None = None, **kwargs):
    """
    Open your Indian Ocean IO.zarr on GCS.
    Catalog path (from datasets.json): gcs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/IO.zarr
    """
    # anon GCS read (works only if bucket is public)
    try:
        return xr.open_zarr(path, storage_options={"token": "anon"}, consolidated=False)
    except Exception:
        # If anon fails, you’ll need GCP auth or a different store.
        raise
