import os
import pandas as pd
import numpy as np
import xarray as xr
from src.pipeline.dicra_fetcher import get_district_coordinates, MP_DISTRICTS

def get_nearest_grid_index(lat_arr, lon_arr, target_lat, target_lon):
    """
    Finds the 2D grid index (lat_idx, lon_idx) nearest to target coordinates.
    Works for both 1D and 2D latitude/longitude coordinate grids.
    """
    # If 1D coordinate arrays
    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        lat_idx = int(np.abs(lat_arr - target_lat).argmin())
        lon_idx = int(np.abs(lon_arr - target_lon).argmin())
        return lat_idx, lon_idx
        
    # If 2D arrays (e.g. curvilinear grids in regional climate models)
    dist = (lat_arr - target_lat)**2 + (lon_arr - target_lon)**2
    flat_idx = dist.argmin()
    lat_idx, lon_idx = np.unravel_index(flat_idx, dist.shape)
    return int(lat_idx), int(lon_idx)


def parse_and_process_netcdf(
    netcdf_dir: str = "data/raw/netcdf",
    output_dir: str = "data/processed",
    scenarios: list = ["ssp245", "ssp585"]
):
    """
    Slices CORDEX South Asia NetCDF files down to Madhya Pradesh coordinates,
    extracts daily variables, maps grid cells to the 52 MP districts,
    and saves processed CSV caches in the format expected by the forecaster.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(netcdf_dir):
        print(f"⚠️ NetCDF raw directory not found: {netcdf_dir}. Download datasets first.")
        return False

    files = [f for f in os.listdir(netcdf_dir) if f.endswith(".nc")]
    if not files:
        print(f"⚠️ No NetCDF (.nc) files found in {netcdf_dir}.")
        return False

    print(f"📂 Processing {len(files)} NetCDF files...")

    # Load all coordinate centroids for the 52 MP districts
    district_centroids = {}
    for dist in MP_DISTRICTS:
        lat, lon = get_district_coordinates(dist)
        district_centroids[dist] = (lat, lon)

    # Scenarios mapping: CORDEX uses lowercase standard (ssp245) or CMIP5 RCP counterpart (rcp45)
    # The main app expects files with "SSP2_4.5" and "SSP5_8.5" scenarios
    scenario_map = {
        "ssp245": "SSP2_4.5",
        "ssp585": "SSP5_8.5",
        "rcp45": "SSP2_4.5",
        "rcp85": "SSP5_8.5"
    }

    # Group files by scenario and variable for unified extraction
    for scenario_code, scenario_app in scenario_map.items():
        print(f"\n🌍 Processing Scenario: {scenario_app} ({scenario_code})")
        
        # Filter files for this scenario
        scenario_files = [f for f in files if scenario_code in f.lower() or scenario_code.replace("ssp", "ssp_") in f.lower()]
        if not scenario_files:
            continue
        
        # We need to build a combined timeseries dataframe for each district.
        # Key: district, Value: dict of date -> {tmax, soil_moisture, precipitation}
        district_timeseries = {dist: {} for dist in MP_DISTRICTS}

        for nc_file in scenario_files:
            file_path = os.path.join(netcdf_dir, nc_file)
            print(f"   📖 Reading {nc_file} ...")
            
            try:
                ds = xr.open_dataset(file_path)
            except Exception as e:
                print(f"   ❌ Failed to open NetCDF file: {e}")
                continue

            # Identify coordinate and time variable names
            lat_var = [v for v in ["lat", "latitude", "nav_lat"] if v in ds.variables or v in ds.coords]
            lon_var = [v for v in ["lon", "longitude", "nav_lon"] if v in ds.variables or v in ds.coords]
            time_var = [v for v in ["time", "Time"] if v in ds.variables or v in ds.coords]
            
            if not lat_var or not lon_var or not time_var:
                print("   ❌ Error: Missing lat/lon/time coordinate dimensions in NetCDF.")
                ds.close()
                continue
                
            lat_name = lat_var[0]
            lon_name = lon_var[0]
            time_name = time_var[0]

            lats = ds[lat_name].values
            lons = ds[lon_name].values
            times = ds[time_name].values
            
            # Map times to pandas Timestamp series
            time_idx = pd.to_datetime(times)

            # Identify the primary scientific variable in the NetCDF
            # (e.g. 'tasmax', 'pr', 'mrso')
            possible_vars = ["tasmax", "pr", "mrso", "soil_moisture", "precipitation", "temperature"]
            var_names = [v for v in possible_vars if v in ds.variables]
            if not var_names:
                # If standard names not found, take the first non-coordinate 3D variable
                var_names = [v for v in ds.variables if len(ds[v].dims) >= 3 and v not in [lat_name, lon_name, time_name]]
            
            if not var_names:
                print("   ❌ Error: No climate variable found in dataset.")
                ds.close()
                continue
                
            var_name = var_names[0]
            print(f"   🏷️  Variable detected: '{var_name}'")

            # Extract variable values
            data_arr = ds[var_name].values
            
            # Check unit scales and apply conversions if necessary
            # tasmax (Kelvin -> Celsius): subtract 273.15
            unit_offset = 0.0
            unit_scale = 1.0
            if var_name == "tasmax" and np.nanmax(data_arr) > 100.0:
                print("     🌡️ Converting Temperature from Kelvin to Celsius")
                unit_offset = -273.15
            # CORDEX pr is in kg/m2/s (liquid water equivalent). We convert to mm/day (multiply by 86400)
            elif var_name == "pr" and np.nanmax(data_arr) < 0.01:
                print("     🌧️ Converting Precipitation from kg/m²/s to mm/day")
                unit_scale = 86400.0
            # mrso/soil_moisture (Total soil moisture content kg/m2 -> volumetric m3/m3 approx by dividing by depth ~1000mm)
            elif var_name == "mrso" and np.nanmax(data_arr) > 1.0:
                print("     🌱 Normalizing Soil Moisture to volumetric fractions")
                unit_scale = 1.0 / 1000.0

            # Map the grids to each district centroid
            for district, (lat_c, lon_c) in district_centroids.items():
                lat_idx, lon_idx = get_nearest_grid_index(lats, lons, lat_c, lon_c)
                
                # Slice the time dimension at this specific grid point coordinate
                # Handles grid dimensions (time, lat, lon) or (time, y, x)
                grid_series = data_arr[:, lat_idx, lon_idx] * unit_scale + unit_offset
                
                for t_dt, val in zip(time_idx, grid_series):
                    date_str = t_dt.strftime("%Y-%m-%d")
                    if date_str not in district_timeseries[district]:
                        district_timeseries[district][date_str] = {}
                    
                    # Normalize keys to match CMIP6 CSV requirements
                    if var_name == "tasmax":
                        district_timeseries[district][date_str]["tmax"] = float(val)
                    elif var_name == "pr":
                        district_timeseries[district][date_str]["precipitation"] = float(val)
                    elif var_name == "mrso":
                        # Bound volumetric soil moisture between typical agricultural ranges
                        district_timeseries[district][date_str]["soil_moisture"] = float(np.clip(val, 0.02, 0.45))

            ds.close()

        # Save the aggregated CSVs for each district
        print(f"💾 Saving processed CSV forecasts for {len(MP_DISTRICTS)} districts...")
        for district, date_dict in district_timeseries.items():
            if not date_dict:
                continue
            
            # Format to DataFrame
            rows = []
            for d_str, vals in date_dict.items():
                rows.append({
                    "date": d_str,
                    "tmax": vals.get("tmax", 30.0),
                    "soil_moisture": vals.get("soil_moisture", 0.18),
                    "precipitation": vals.get("precipitation", 0.0)
                })
                
            df = pd.DataFrame(rows).sort_values("date")
            district_slug = district.replace(" ", "_").replace("-", "_").lower()
            out_file = os.path.join(output_dir, f"cmip6_{district_slug}_{scenario_app}.csv")
            
            df.to_csv(out_file, index=False)
            
        print(f"✅ Aggregated climate forecast CSVs saved to {output_dir}")
        
    return True

if __name__ == "__main__":
    parse_and_process_netcdf()
