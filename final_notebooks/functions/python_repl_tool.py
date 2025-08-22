from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool

def create_python_repl():
    python_repl = PythonREPL()
    
    def enhanced_python_repl(code):
        import os
        import matplotlib.pyplot as plt
        from datetime import datetime
        """Enhanced Python REPL that automatically saves plots and returns code + path"""
        
        # Ensure figures_temp directory exists
        os.makedirs('figures_temp', exist_ok=True)
        
        # Store original plt.show function
        original_show = plt.show
        saved_paths = []
        
        def custom_show(*args, **kwargs):
            """Custom show function that saves figures before displaying"""
            # Save all current figures before showing
            if plt.get_fignums():
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    
                    # Generate unique filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                    filename = f"plot_{timestamp}_fig{fig_num}.png"
                    filepath = os.path.join('figures_temp', filename)
                    
                    # Save the figure before showing
                    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                    saved_paths.append(filepath)
                    print(f"Figure saved to: {filepath}")
            
            # Now call the original show function
            return original_show(*args, **kwargs)
        
        # Replace plt.show temporarily
        plt.show = custom_show
        
        try:
            # Execute the code
            result = python_repl.run(code)
            
            # If code didn't call plt.show() but figures exist, save them anyway
            if plt.get_fignums() and not saved_paths:
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"plot_{timestamp}_fig{fig_num}.png"
                    filepath = os.path.join('figures_temp', filename)
                    
                    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                    saved_paths.append(filepath)
                    print(f"Figure saved to: {filepath}")
        
        finally:
            # Always restore the original plt.show function
            plt.show = original_show
        
        # Format the output
        output = f"Execution Result:\n{result}\n"
        if saved_paths:
            output += f"\nSaved Plots:\n"
            for path in saved_paths:
                output += f"- {path}\n"
        
        output += f"\nExecuted Code:\n{code}\n"
        
        return output
    
    python_repl_tool = Tool(
        name="python_repl",
        func=enhanced_python_repl,
        description="""
            You receive:
                - A file path to the downloaded dataset. This path comes from the load_climate_data tool. Do not use any other path.
                - Optionally, an example analysis function from advisor Tool that may be relevant to the query.
                
                NOTE: do NOT download any new data from anywhere. You should directly access the data at the path that is given to you.
                
                Your job:
                1. If an example function is provided and fits the user's query, use it directly with the dataset. Make sure you load the data!
                2. If no suitable example function is provided, write your own analysis code using standard scientific Python packages such as:
                   - xarray (for handling datasets)
                   - matplotlib (for plotting, with clean labels and colorbars)
                   - numpy (for computations)
                   - cartopy or geopandas (if maps are needed)
                   - cmocean or matplotlib colormaps (for nice scientific colormaps)
                3. Keep your code clean, minimal, and runnable in a single Python cell.
                4. If user's request cannot be completed, return a helpful error message explaining why.
                5. After you are done showing your result, please give an explanation of what the user is seeing.
                
                Guidelines:
                - Always open the dataset using xarray from the provided file path.
                - Assume the dataset may be large: use efficient operations (`.sel`, `.isel`, `.mean`, `.plot`, etc.).
                - Include titles, axis labels, and legends where appropriate.
                - Prefer clarity and readability of code over cleverness.
                - Do not invent dataset fields—only use those provided in the dataset info.
                - If the user asks for a time range that you might think is too large, warn them and suggest re-execution.

                NOTE: here's an example loader function you might take some ideas from. It tells you how to read some files from the cloud. I also
                provide you with a schema:

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
                    xr.Dataset
                        Processed dataset with consistent dimensions
                
                    
                    # Open dataset
                    if isinstance(store, str):
                        # If store is a URL string, use storage_options
                        ds = xr.open_dataset(
                            store,
                            engine="zarr",
                            chunks=chunks,
                            backend_kwargs={"storage_options": storage_options},
                        )
                    else:
                        # If store is already an FSMap object, use it directly
                        ds = xr.open_zarr(store, chunks=chunks)
                    
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
                    # Get available dimensions
                    dims = list(ds.dims)
                    # Core dims we want first (if they exist)
                    core_dims = ["time", "latitude", "longitude"]
                    # Filter out core dims that actually exist
                    core_dims = [d for d in core_dims if d in dims]
                    # Add any remaining dims at the end
                    other_dims = [d for d in dims if d not in core_dims]
                    # Combine for final ordering
                    final_dims = core_dims + other_dims
                    
                    ds = ds.transpose(*final_dims)
                
                    if variable:
                        var = _select_variable(ds, variable)
                        ds = ds[var]
                
                    path = download_to_temp(
                        ds
                    )
                    
                    return path
                
                
                class ClimateDataParams(BaseModel):
                    
                    A Pydantic model to define and validate parameters for accessing climate data.
                    It specifies the data store and the exact variable to be retrieved.
                    
                    store: Literal[
                        "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
                        "gcs://nmfs_odp_nwfsc/CB/mind_the_chl_gap/IO.zarr"
                    ] = Field(
                        ...,
                        description="The specific cloud storage path (store) where the dataset is located."
                    )
                
                    variable: Literal[
                        # Variables from ERA5 Atmospheric Surface Analysis
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "2m_dewpoint_temperature",
                        "2m_temperature",
                        "angle_of_sub_gridscale_orography",
                        "anisotropy_of_sub_gridscale_orography",
                        "boundary_layer_height",
                        "geopotential",
                        "geopotential_at_surface",
                        "high_vegetation_cover",
                        "lake_cover",
                        "land_sea_mask",
                        "leaf_area_index_high_vegetation",
                        "leaf_area_index_low_vegetation",
                        "low_vegetation_cover",
                        "mean_sea_level_pressure",
                        "mean_surface_latent_heat_flux",
                        "mean_surface_net_long_wave_radiation_flux",
                        "mean_surface_net_short_wave_radiation_flux",
                        "mean_surface_sensible_heat_flux",
                        "mean_top_downward_short_wave_radiation_flux",
                        "mean_top_net_long_wave_radiation_flux",
                        "mean_top_net_short_wave_radiation_flux",
                        "mean_vertically_integrated_moisture_divergence",
                        "potential_vorticity",
                        "sea_ice_cover",
                        "sea_surface_temperature",
                        "slope_of_sub_gridscale_orography",
                        "snow_depth",
                        "soil_type",
                        "specific_humidity",
                        "standard_deviation_of_filtered_subgrid_orography",
                        "standard_deviation_of_orography",
                        "surface_pressure",
                        "temperature",
                        "total_cloud_cover",
                        "total_column_water",
                        "total_column_water_vapour",
                        "total_precipitation_6hr",
                        "type_of_high_vegetation",
                        "type_of_low_vegetation",
                        "u_component_of_wind",
                        "v_component_of_wind",
                        "vertical_velocity",
                        "volumetric_soil_water_layer_1",
                        "volumetric_soil_water_layer_2",
                        "volumetric_soil_water_layer_3",
                        "volumetric_soil_water_layer_4",
                
                        # Variables from Indian Ocean grid
                        "adt",
                        "air_temp",
                        "mlotst",
                        "sla",
                        "so",
                        "sst",
                        "topo",
                        "u_curr",
                        "v_curr",
                        "ug_curr",
                        "vg_curr",
                        "u_wind",
                        "v_wind",
                        "curr_speed",
                        "curr_dir",
                        "wind_speed",
                        "wind_dir",
                        "CHL_cmes-level3",
                        "CHL_cmes_flags-level3",
                        "CHL_cmes_uncertainty-level3",
                        "CHL_cmes-gapfree",
                        "CHL_cmes_flags-gapfree",
                        "CHL_cmes_uncertainty-gapfree",
                        "CHL_cci",
                        "CHL_cci_uncertainty",
                        "CHL_dinoef",
                        "CHL_dinoef_uncertainty",
                        "CHL_dinoef_flag"
                    ] = Field(
                        ...,
                        description="The specific variable name to be selected from the chosen data store."
                    )
                    lon_range: Optional[Tuple[float, float]] = Field(
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
                                
            """,
    )
    return python_repl_tool