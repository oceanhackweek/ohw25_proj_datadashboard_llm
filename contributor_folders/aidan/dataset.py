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
<<<<<<< HEAD
=======
    access_function: Optional[str] = ""
    other_args: Optional[dict] = {}
>>>>>>> aa213b2d152fb29a697d7271cdf076f9f2c1a546

class Variable(BaseModel):
    standard_name: str
    description: str
<<<<<<< HEAD
=======
    units: str
>>>>>>> aa213b2d152fb29a697d7271cdf076f9f2c1a546

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