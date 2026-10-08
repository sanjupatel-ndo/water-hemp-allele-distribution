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
from typing import Optional

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

import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.mask import mask as rio_mask
from tqdm import tqdm
 
PRECIP_PRECIS_M = 800          # native PRISM 800m precision
PRISM_BASE = "https://services.nacse.org/prism/data/get"
RASTER_EXT = (".tif", ".bil")
ALL_VARIABLES = ["ppt", "tmin", "tmax", "tmean", "tdmean", "vpdmin", "vpdmax"]
 
 
# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def fetch_month_raster(
    yyyymm: str,
    element: str,
    cache_dir: str = "prism_cache",
    region: str = "us",
    res: str = "800m",
    pause: float = 2.0,
    retries: int = 4,
) -> Optional[Path]:
    """
    Download one monthly PRISM grid package and extract it. Returns the raster
    path, or None if PRISM has no grid for that month/variable. If the grid is
    already extracted in cache_dir, no request is made.
    """
    folder = Path(cache_dir) / f"{element}_{region}_{res}_{yyyymm}"
    if folder.exists():
        existing = [p for p in folder.glob("*") if p.suffix.lower() in RASTER_EXT]
        if existing:
            return existing[0]
 
    url = f"{PRISM_BASE}/{region}/{res}/{element}/{yyyymm}"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=180)
        except requests.RequestException as e:
            tqdm.write(f"  {element} {yyyymm}: request error ({e}); retry {attempt + 1}/{retries}")
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 200:
            folder.mkdir(parents=True, exist_ok=True)
            zpath = folder / f"{element}_{yyyymm}.zip"
            zpath.write_bytes(r.content)
            if not zipfile.is_zipfile(zpath):
                tqdm.write(f"  {element} {yyyymm}: response was not a zip (likely no grid available)")
                zpath.unlink()
                folder.rmdir()
                return None
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(folder)   # extract all so .bil/.hdr/.prj stay together
            zpath.unlink()
            time.sleep(pause)           # be polite to the server
            found = [p for p in folder.glob("*") if p.suffix.lower() in RASTER_EXT]
            return found[0] if found else None

        if r.status_code in (400, 404):
            tqdm.write(f"  {element} {yyyymm}: PRISM has no grid (HTTP {r.status_code})")
            return None

        tqdm.write(f"  {element} {yyyymm}: HTTP {r.status_code}; retry {attempt + 1}/{retries}")
        time.sleep(2 ** attempt)
 
    return None
 
 
# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def _cells_in_circle(src, circle) -> np.ndarray:
    """
    Valid raster values inside a circular uncertainty polygon. Uses cells whose
    center falls inside the circle first; falls back to every touched cell if
    no centers fall inside.
    """
    for all_touched in (False, True):
        try:
            arr, _ = rio_mask(src, [circle], crop=True, all_touched=all_touched, filled=False)
        except ValueError:   # circle doesn't overlap the grid
            return np.array([])
        vals = arr[0].compressed()
        if vals.size > 0:
            return vals
    return np.array([])
 
 
def _extract_value(src, pt, circle, precise, n_samples, rng):
    """Return (mean, std, n_cells, method) for one sample, or None."""
    if precise:
        val = next(src.sample([(pt.x, pt.y)], masked=True))[0]
        if np.ma.is_masked(val):
            return None
        return float(val), np.nan, 1, "point"
 
    if circle is None or circle.is_empty:
        return None
    vals = _cells_in_circle(src, circle)
    if vals.size == 0:
        return None
    if n_samples is not None and vals.size > n_samples:
        vals = rng.choice(vals, size=n_samples, replace=False)
    return float(vals.mean()), float(vals.std()), int(vals.size), "area"
 
 
# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def collect_prism_data(
    a_df: gpd.GeoDataFrame,
    variables: list[str] = ALL_VARIABLES,
    year_col: str = "year",
    month_col: str = "month",
    point_col: str = "geometry",
    circle_col: str = "uncertainty_buffer",
    uncertainty_col: str = "coordinateUncertaintyInMeters",
    n_samples: Optional[int] = None,
    seed: int = 0,
    res: str = "800m",
    cache_dir: str = "prism_cache",
    checkpoint_dir: str = "prism_checkpoints",
    keep_grids: bool = False,
) -> gpd.GeoDataFrame:
    """
    Match monthly PRISM variables to samples by location and collection month.
 
    Loops months on the outside and variables on the inside, so each grid
    (month x variable) is downloaded exactly once. After a month is finished,
    its results are written to a checkpoint file and (unless keep_grids=True)
    its grids are deleted. Rerunning skips any month that already has a
    checkpoint, so an interrupted run resumes where it left off.
 
    Samples with uncertainty <= 800 m get the value of the cell containing the
    point. Samples with greater uncertainty get the mean (and std) of the cells
    inside their uncertainty circle (all cells if n_samples is None, otherwise
    a random subset of n_samples).
 
    Output columns, per variable v: v, v_std. Shared: prism_n_cells,
    prism_method.
    """
    rng = np.random.default_rng(seed)
    df = a_df.copy()
 
    # YYYYMM key from separate year / month columns; NA if either is missing
    yy = pd.to_numeric(df[year_col], errors="coerce").astype("Int64").astype("string")
    mm = pd.to_numeric(df[month_col], errors="coerce").astype("Int64").astype("string").str.zfill(2)
    df["_yyyymm"] = yy + mm
    n_bad = df["_yyyymm"].isna().sum()
    if n_bad:
        print(f"{n_bad} samples have a missing year or month and will be left as NaN")
 
    ckpt = Path(checkpoint_dir) / (f"{res}_" + "_".join(variables))
    ckpt.mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
 
    months = sorted(df["_yyyymm"].dropna().unique())
    frames = []

    for yyyymm in tqdm(months, desc="months", unit="month"):
        ckpt_file = ckpt / f"{yyyymm}.csv"
        if ckpt_file.exists():
            frames.append(pd.read_csv(ckpt_file, index_col=0))
            continue

        group = df[df["_yyyymm"] == yyyymm]
        tqdm.write(f"{yyyymm}: {len(group)} samples")
        out = pd.DataFrame(index=group.index)
        for v in variables:
            out[v] = np.nan
            out[f"{v}_std"] = np.nan
        out["prism_n_cells"] = 0
        out["prism_method"] = None

        for v in tqdm(variables, desc=yyyymm, leave=False, unit="var"):
            raster_path = fetch_month_raster(yyyymm, v, cache_dir=cache_dir, res=res)
            if raster_path is None:
                continue
 
            with rasterio.open(raster_path) as src:
                # put geometries into the raster's CRS once per grid
                pts = gpd.GeoSeries(group[point_col], crs=df.crs).to_crs(src.crs)
                circles = gpd.GeoSeries(group[circle_col], crs=df.crs).to_crs(src.crs)
 
                for idx, pt, circle in zip(group.index, pts, circles):
                    unc = df.at[idx, uncertainty_col]
                    precise = pd.notna(unc) and unc <= PRECIP_PRECIS_M
                    res_tuple = _extract_value(src, pt, circle, precise, n_samples, rng)
                    if res_tuple is None:
                        continue
                    mean, std, n, method = res_tuple
                    out.at[idx, v] = mean
                    out.at[idx, f"{v}_std"] = std
                    if out.at[idx, "prism_n_cells"] == 0:
                        out.at[idx, "prism_n_cells"] = n
                        out.at[idx, "prism_method"] = method
 
            if not keep_grids:
                shutil.rmtree(raster_path.parent, ignore_errors=True)
 
        out.to_csv(ckpt_file)    # checkpoint: this month is done
        frames.append(out)
 
    if frames:
        results = pd.concat(frames)
        df = df.join(results)
    return df.drop(columns="_yyyymm")
 
 
if __name__ == "__main__":
    df = pd.read_csv("herbarium_metadata_2022_2026_datasets.csv")
    gdf = create_buffers(df)
 
    result = collect_prism_data(
        gdf,
        variables=ALL_VARIABLES,
        year_col="year",
        month_col="month",
        uncertainty_col="coordinateUncertaintyInMeters",
    )
    result.drop(columns=["geometry", "uncertainty_buffer"]).to_csv("herbarium_samples_2022_2026_withPRISMvars.csv", index=False)
