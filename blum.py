import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from scipy.interpolate import griddata
import xarray as xr

# Optional: only import netCDF4 when needed
try:
    from netCDF4 import Dataset, num2date
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

# --------------------------------------------------
# 1. CONFIG & THEME
# --------------------------------------------------
SITE_NAME = "Blumm Blummmmm Blummmmmmmmmmmmm🐠🫧"
st.set_page_config(page_title=SITE_NAME, layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(to bottom, #0D1B2A, #1B263B, #415A77); }
h1, h2, h3, p, span, label { color: #E0E1DD !important; }
.stButton>button { background-color: #0077B6; color: white; border-radius: 12px; font-weight: bold; }
.footer { position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 12px; color: #778DA9; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------
if 'final_excels' not in st.session_state:
    st.session_state.final_excels = []

# --------------------------------------------------
# 3. UI HEADER
# --------------------------------------------------
st.title(SITE_NAME)
st.subheader("Hey Mate! Blum blum… decoding Copernicus depths 🌊🫧")

# --------------------------------------------------
# 4. FILE UPLOAD
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Drop your Copernicus (.nc) files here", 
    type=["nc"], 
    accept_multiple_files=True
)

# --------------------------------------------------
# 5. VARIABLE & EXTENT SELECTION
# --------------------------------------------------
copernicus_selections = {}
if uploaded_files:
    st.markdown("### 🗂 Select Variables and Clipping Extent for each file")
    for file in uploaded_files:
        try:
            # Try reading via netCDF4 first
            try:
                if NETCDF4_AVAILABLE:
                    ds_test = Dataset(io.BytesIO(file.getvalue()))
                    ds_test.close()
                    ds = xr.open_dataset(io.BytesIO(file.getvalue()), engine="netcdf4")
                else:
                    raise Exception("netCDF4 not available, using h5netcdf fallback")
            except Exception:
                # fallback to h5netcdf
                ds = xr.open_dataset(io.BytesIO(file.getvalue()), engine="h5netcdf")

            # Detect spatial variables
            spatial_vars = [var for var in ds.data_vars if {'lat','lon'}.issubset(ds[var].dims)]
            if not spatial_vars:
                st.error(f"❌ {file.name}: No spatial variables detected.")
                continue

            # Multi-select
            vars_selected = st.multiselect(f"{file.name} - Select variables to decode", spatial_vars, key=f"vars_{file.name}")
            if not vars_selected:
                st.warning(f"⚠️ {file.name}: No variables selected, will skip.")
                continue

            # Determine min/max
            lat_vals = ds.lat.values
            lon_vals = ds.lon.values
            lat_ascending = lat_vals[0] < lat_vals[-1]
            lat_min_val, lat_max_val = (float(lat_vals[0]), float(lat_vals[-1])) if lat_ascending else (float(lat_vals[-1]), float(lat_vals[0]))
            lon_min_val, lon_max_val = float(lon_vals.min()), float(lon_vals.max())

            # Streamlit 3-decimal inputs
            col1, col2 = st.columns(2)
            with col1:
                lat_min = st.number_input(f"{file.name} - Min Latitude", min_value=lat_min_val, max_value=lat_max_val, value=lat_min_val, step=0.001, format="%.3f", key=f"latmin_{file.name}")
                lat_max = st.number_input(f"{file.name} - Max Latitude", min_value=lat_min_val, max_value=lat_max_val, value=lat_max_val, step=0.001, format="%.3f", key=f"latmax_{file.name}")
            with col2:
                lon_min = st.number_input(f"{file.name} - Min Longitude", min_value=lon_min_val, max_value=lon_max_val, value=lon_min_val, step=0.001, format="%.3f", key=f"lonmin_{file.name}")
                lon_max = st.number_input(f"{file.name} - Max Longitude", min_value=lon_min_val, max_value=lon_max_val, value=lon_max_val, step=0.001, format="%.3f", key=f"lonmax_{file.name}")

            copernicus_selections[file.name] = {
                "variables": vars_selected,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max
            }
            ds.close()
        except Exception as e:
            st.error(f"⚠️ Could not read {file.name}: {e}")

# --------------------------------------------------
# 6. COPERNICUS HANDLER FUNCTION (with fallback)
# --------------------------------------------------
def handle_copernicus_file(file, selection):
    results = []
    try:
        # Try netCDF4 engine first
        try:
            if NETCDF4_AVAILABLE:
                ds = xr.open_dataset(io.BytesIO(file.getvalue()), engine="netcdf4")
            else:
                raise Exception("netCDF4 not available, fallback to h5netcdf")
        except Exception:
            ds = xr.open_dataset(io.BytesIO(file.getvalue()), engine="h5netcdf")

        lat_min, lat_max = selection["lat_min"], selection["lat_max"]
        lon_min, lon_max = selection["lon_min"], selection["lon_max"]

        # Swap lat if descending
        if ds.lat.values[0] > ds.lat.values[-1]:
            lat_min, lat_max = lat_max, lat_min

        for var in selection["variables"]:
            subset = ds[var].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            df = subset.to_dataframe(name="value").reset_index()
            df = df.dropna(subset=["value"])  # remove missing

            # Convert longitudes >180 to 0-360
            if df['lon'].max() > 180:
                df['lon'] = df['lon'] % 360
                lon_min = lon_min % 360
                lon_max = lon_max % 360

            # Create QGIS-like grid
            grid_spacing = 0.125
            grid_lons = np.arange(lon_min, lon_max + 0.0001, grid_spacing)
            grid_lats = np.arange(lat_min, lat_max + 0.0001, grid_spacing)
            grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons, indexing='ij')
            grid_points = np.c_[grid_lat.ravel(), grid_lon.ravel()]

            # Interpolate using original lat/lon
            points = df[['lat','lon']].values
            values = df['value'].values
            grid_values = griddata(points, values, grid_points, method='linear')

            # Fill remaining NaNs with nearest
            nan_mask = np.isnan(grid_values)
            if np.any(nan_mask):
                grid_values[nan_mask] = griddata(points, values, grid_points[nan_mask], method='nearest')

            # Build final DataFrame
            grid_df = pd.DataFrame({
                'lat': grid_lat.ravel().round(3),
                'lon': grid_lon.ravel().round(3),
                'value': grid_values
            })
            grid_df.sort_values(['lat','lon'], ascending=[False, True], inplace=True)

            out_name = file.name.replace(".nc", f"_{var}_qgis.xlsx")
            buffer = io.BytesIO()
            grid_df.to_excel(buffer, index=False, engine='openpyxl')
            results.append({"name": out_name, "content": buffer.getvalue()})

        ds.close()
        return results
    except Exception as e:
        st.error(f"⚠️ Failed to process {file.name}: {e}")
        return None

# --------------------------------------------------
# 7. PROCESSING LOGIC
# --------------------------------------------------
if uploaded_files and st.button("🚀 Process & Generate All Reports"):
    st.session_state.final_excels = []
    progress_bar = st.progress(0.0)

    total_files = len(uploaded_files)
    for idx, file in enumerate(uploaded_files, start=1):
        st.info(f"Processing {file.name}...")
        if file.name in copernicus_selections:
            results = handle_copernicus_file(file, copernicus_selections[file.name])
            if results:
                st.session_state.final_excels.extend(results)
        else:
            st.warning(f"⚠️ {file.name} skipped: No selections made.")
        progress_bar.progress(idx / total_files)

    st.success(f"✅ Successfully processed {len(st.session_state.final_excels)} file(s)!")
    st.snow()

# --------------------------------------------------
# 8. ZIP & DOWNLOAD SECTION
# --------------------------------------------------
if st.session_state.final_excels:
    st.divider()
    st.markdown("### 📥 Download Results")
    
    zip_out = io.BytesIO()
    with zipfile.ZipFile(zip_out, "w") as zf:
        for file_data in st.session_state.final_excels:
            zf.writestr(file_data["name"], file_data["content"])
    zip_out.seek(0)

    st.download_button(
        label="📦 DOWNLOAD ALL AS ZIP",
        data=zip_out,
        file_name="copernicus_ocean_data.zip",
        mime="application/zip",
        use_container_width=True,
        key="zip_download_btn"
    )

    with st.expander("Or download files individually"):
        for idx, f in enumerate(st.session_state.final_excels):
            st.download_button(
                label=f"⬇️ {f['name']}", 
                data=f['content'], 
                file_name=f['name'],
                key=f"dl_{idx}"
            )

st.markdown('<div class="footer">💀🌊Copernicus Blum Blumm Version 1.4.2@ak💦 Not your Ordinary Blub Obviously !!!!🌊💀</div>', unsafe_allow_html=True)
