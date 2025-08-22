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