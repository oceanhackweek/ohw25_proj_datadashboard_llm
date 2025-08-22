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
# --- Generic catalog loader used by datasets.json ---
import xarray as xr

def load_climate_data(path: str, platform: str | None = None, meta: dict | None = None, **kwargs):
    """
    Generic loader that opens a Zarr store given a cloud path.
    - For AWS S3 paths, uses anonymous access.
    - For GCS paths, tries anonymous token (works only if bucket is public).
    Any extra **kwargs are passed through to xr.open_zarr.
    """
    # Allow platform hint or infer from path
    p = (platform or "").lower()
    if not p:
        if path.startswith("s3://"):
            p = "aws"
        elif path.startswith(("gs://", "gcs://")):
            p = "gcs"

    if p == "aws":
        # Try consolidated first, fall back to non-consolidated
        try:
            return xr.open_zarr(path, storage_options={"anon": True}, consolidated=True, **kwargs)
        except Exception:
            return xr.open_zarr(path, storage_options={"anon": True}, consolidated=False, **kwargs)

    if p == "gcs":
        # Many public GCS buckets accept token="anon"
        # Consolidation varies; try non-consolidated first for safety
        try:
            return xr.open_zarr(path, storage_options={"token": "anon"}, consolidated=False, **kwargs)
        except Exception:
            return xr.open_zarr(path, storage_options={"token": "anon"}, consolidated=True, **kwargs)

    # Fallback: let xarray decide, try consolidated then not
    try:
        return xr.open_zarr(path, consolidated=True, **kwargs)
    except Exception:
        return xr.open_zarr(path, consolidated=False, **kwargs)
