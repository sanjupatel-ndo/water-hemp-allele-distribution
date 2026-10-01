"""
Kreiner Lab - September 14th 2026
Resampling pipeline
Primary contributor(s): Sanju Patel, Julian Huang

Objective: Pull environmental data for samples with varying degrees of location 
uncertainty

Methodology: For each sample, draw a circle where the sample may have been
collected. For each environmental variable, randomly sample some number of 
observations within this radius. For this first pass, report the mean and 
standard deviation of the sampling distribution for each environmental variable, 
for each sample.

Data requirements:
    Inputs: Waterhemp sample with a latitude and longtitude
"""
#Step 1: For each sample, create an area that represents where the sample was
#collected, taking into account differing levels of location uncertainty.

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

def create_buffers(obs_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Creates 1 km vector buffers around observation points. Assumes observations 
    from GBIF and has 'decimalLongitude' and 'decimalLatitude' keys

    Args:
        obs_df (pd.DataFrame): DataFrame containing GBIF observation coordinates 
                               with 'decimalLongitude' and 'decimalLatitude' 
                               columns.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with original points and their 
        corresponding circles of uncertainty.
    """
    geometry = [Point(lon, lat) for lon, lat in zip(obs_df['decimalLongitude'], obs_df['decimalLatitude'])]
    gdf = gpd.GeoDataFrame(obs_df, geometry=geometry, crs="EPSG:4326")

    gdf_utm = gdf.to_crs("EPSG:5070")
    gdf_utm["uncertainty_buffer"] = gdf_utm.geometry.buffer(gdf_utm["coordinateUncertaintyInMeters"])
    return gdf_utm


#Step 2: Start collecting the environmental data, and sampling if neccesary

import io
import re
import zipfile

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from googleapiclient.http import MediaIoBaseDownload

PRECIP_PRECIS_M = 800   # native PRISM precision
RES_TAG = "30s"         # tag for weather data with 800m precision

def build_zip_index(drive, folder_id: str) -> dict:
    """Map 'YYYYMM' -> list of Drive file dicts, using filenames only."""
    index, token = {}, None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and name contains '.zip' and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageSize=1000, pageToken=token,
        ).execute()
        for f in resp["files"]:
            m = re.search(r"_(\d{6})(?:_|\.)", f["name"])
            if m:
                index.setdefault(m.group(1), []).append(f)
        token = resp.get("nextPageToken")
        if not token:
            return index


def _download_bytes(drive, file_id: str) -> bytes:
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, drive.files().get_media(fileId=file_id),
                             chunksize=8 * 1024 * 1024)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def _pick_file(candidates: list, res_tag: str) -> dict:
    matches = [f for f in candidates if res_tag in f["name"]]
    return (matches or candidates)[0]


def collect_precip_data(
    a_df: gpd.GeoDataFrame,
    drive,
    index: dict,
    date_col: str = "eventDate",
    point_col: str = "geometry",
    rect_col: str = "uncertainty_buffer",
    uncertainty_col: str = "coordinateUncertaintyInMeters",
    n_samples: int = 30,
    seed: int = 0,
) -> gpd.GeoDataFrame:
    """
    For samples with an uncertainty more precise than the available precip.
    data, pulls appropriate data for the month of sample collection. For samples
    with greater uncertainty, sample some number of the precipitation values
    contained in the radius of uncertainty.
    The precision of the precip data is 800m.

    Args:
        a_df (gpd.GeoDataFrame): A GeoDataFrame with original points and their
                            corresponding rectangles of uncertainty.
        drive: authenticated Google Drive v3 service.
        index: output of build_zip_index() for the 'monthly' folder.
        date_col: collection date column (datetime-like).
        point_col: column holding the original point geometry.
        rect_col: column holding the uncertainty rectangle geometry.
        uncertainty_col: uncertainty in meters.
        n_samples: max number of cells to sample for high-uncertainty samples.
        seed: RNG seed so sampling is reproducible.

    Returns:
        gpd.GeoDataFrame: Same data frame but with monthly precipitation data
            (ppt_mm, ppt_std_mm, ppt_n_cells, ppt_method).
    """
    rng = np.random.default_rng(seed)
    df = a_df.copy()
    df["_yyyymm"] = pd.to_datetime(df[date_col]).dt.strftime("%Y%m")

    df["ppt_mm"] = np.nan
    df["ppt_std_mm"] = np.nan
    df["ppt_n_cells"] = 0
    df["ppt_method"] = None

    for yyyymm, group in df.groupby("_yyyymm"):
        if yyyymm not in index:
            print(f"No PRISM zip found for {yyyymm}; {len(group)} samples left as NaN")
            continue

        # one download + one open per month
        f = _pick_file(index[yyyymm], RES_TAG)
        with zipfile.ZipFile(io.BytesIO(_download_bytes(drive, f["id"]))) as zf:
            tif = next(n for n in zf.namelist() if n.lower().endswith(".tif"))
            tif_bytes = zf.read(tif)

        with MemoryFile(tif_bytes) as mem, mem.open() as src:
            # put geometries into the raster's CRS once per month
            pts = gpd.GeoSeries(group[point_col], crs=df.crs).to_crs(src.crs)
            rects = gpd.GeoSeries(group[rect_col], crs=df.crs).to_crs(src.crs)

            for idx, pt, rect in zip(group.index, pts, rects):
                precise = df.at[idx, uncertainty_col] <= PRECIP_PRECIS_M

                if precise:
                    val = next(src.sample([(pt.x, pt.y)], masked=True))[0]
                    if np.ma.is_masked(val):
                        continue
                    df.loc[idx, ["ppt_mm", "ppt_std_mm", "ppt_n_cells", "ppt_method"]] = \
                        [float(val), np.nan, 1, "point"]
                else:
                    arr, _ = rio_mask(src, [rect], crop=True, all_touched=True, filled=False)
                    vals = arr[0].compressed()   # drops nodata / outside-mask cells
                    if vals.size == 0:
                        continue
                    if vals.size > n_samples:
                        vals = rng.choice(vals, size=n_samples, replace=False)
                    df.loc[idx, ["ppt_mm", "ppt_std_mm", "ppt_n_cells", "ppt_method"]] = \
                        [float(vals.mean()), float(vals.std()), int(vals.size), "area"]

    return df.drop(columns="_yyyymm")
    

if __name__ == "__main__":
    df = pd.read_csv("1000m_amaranthus_with_proportions.csv")
    gdf = create_buffers(df)

    # setup drive auth (see above)
    PRISM_FOLDER_ID = "your_folder_id_here"
    index = build_zip_index(drive, PRISM_FOLDER_ID)

    result = collect_precip_data(
        gdf,
        drive,
        index,
        date_col="eventDate",
        uncertainty_col="coordinateUncertaintyInMeters",
    )
    result.drop(columns="geometry").to_csv("output_with_precip.csv", index=False)


