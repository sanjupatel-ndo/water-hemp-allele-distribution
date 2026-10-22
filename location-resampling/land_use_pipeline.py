"""
Kreiner Lab - October 19th 2026
Land use pipeline (HILDA+)
Primary contributor(s): Sanju Patel

Objective: Pull land use data for samples with varying degrees of location
uncertainty.

Methodology: For each sample, take the area where it may have been collected
and read every HILDA+ land use/cover cell whose centre falls inside that area.
Report the class composition of those cells, point composition, and a shannon
index for the diversity of land uses.

HILDA+ covers 1899-2019 annually and globally, so it reaches the pre-NLCD
records and the Ontario samples. The cost is resolution: cells are 0.01 degrees,
about 1.1 km north-south and 850 m east-west at 40 N.

Data requirements:
    Inputs: Waterhemp samples with a latitude, longitude, collection year, and
            a location uncertainty radius. Radius is taken from the coordinate
            precision columns written by coordinate_precision.py, or from an
            explicit column.
    Grids:  hildap_vGLOB-1.0-f_netcdf/hildaplus_GLOB-1-0-f_states.nc (2.7 GB).
"""

from __future__ import annotations   # anaconda env here is Python 3.9

import math
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------
# HILDA+ grid and class definitions
# --------------------------------------------------------------------------
HILDA_STATES = "hildap_vGLOB-1.0-f_netcdf/hildaplus_GLOB-1-0-f_states.nc"

# Clean global 0.01 degree raster, 18000 x 36000. The metadata .txt shipped with
# the download claims 18021 x 36041 and a Europe-only extent; both are stale.
# Indices are computed from these constants rather than from the stored lat/lon
# arrays, which are float32 and carry ~1e-5 of jitter.
CELL_DEG = 0.01
N_ROWS, N_COLS = 18000, 36000
LAT_TOP, LON_LEFT = 90.0, -180.0
M_PER_DEG = 111_320.0

HILDA_YEAR_MIN, HILDA_YEAR_MAX = 1899, 2019

# Forest subclasses 41-45 collapse into one Forest class: the HILDA+ docs state
# the subdivision is static (carried from ESA-CCI 1992) before 1992 and
# extrapolated before 1960, so it is not an observation in the historical period
# and must not inflate the diversity index.
HILDA_CLASSES = {
    11: "Urban",
    22: "Cropland",
    33: "Pasture",
    40: "Forest", 41: "Forest", 42: "Forest",
    43: "Forest", 44: "Forest", 45: "Forest",
    55: "Grass/shrubland",
    66: "Other land",
    77: "Water",
}
CLASS_ORDER = ["Urban", "Cropland", "Pasture", "Forest",
               "Grass/shrubland", "Other land", "Water"]
AGRICULTURAL_CLASSES = ["Cropland", "Pasture"]

# The docs name 99 as the fill value, but no 99 occurs in the grid; ocean and
# no-data are coded 0. Both are treated as missing.
NODATA_CODES = (0, 99)

# Below this cell count the Shannon index is returned as NaN. A meaningful index
# wants on the order of 25+ cells, i.e. a radius of about 3 km at this
# resolution, so treat low-cell results with suspicion even above the default.
MIN_CELLS_FOR_SHANNON = 2


# --------------------------------------------------------------------------
# Grid indexing
# --------------------------------------------------------------------------
def cell_index(lat: float, lon: float) -> tuple[int, int]:
    """Convert a coordinate to its HILDA+ (row, col) cell index, clamped."""
    row = int(math.floor((LAT_TOP - lat) / CELL_DEG))
    col = int(math.floor((lon - LON_LEFT) / CELL_DEG))
    return min(max(row, 0), N_ROWS - 1), min(max(col, 0), N_COLS - 1)


def resolve_radius(
    df: pd.DataFrame,
    radius_col: Optional[str] = None,
    min_radius_m: float = 0.0,
) -> pd.Series:
    """
    Decide the uncertainty radius, in metres, to use for each sample.

    Args:
        df (pd.DataFrame): Sample table.
        radius_col (Optional[str]): Explicit radius column, in metres.
        min_radius_m (float): Lower bound applied to every radius.

    Returns:
        pd.Series: Radius in metres, indexed like df.
    """
    if radius_col is not None:
        radius = pd.to_numeric(df[radius_col], errors="coerce")
    elif {"lat_precision_m", "lon_precision_m"} <= set(df.columns):
        # Those columns hold the full grid step; rounding uncertainty is half.
        step = pd.concat(
            [pd.to_numeric(df["lat_precision_m"], errors="coerce"),
             pd.to_numeric(df["lon_precision_m"], errors="coerce")],
            axis=1,
        ).max(axis=1)
        radius = step / 2.0
    elif "coordinateUncertaintyInMeters" in df.columns:
        radius = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
    else:
        raise KeyError(
            "no radius source: pass radius_col, or run coordinate_precision.py "
            "first to produce lat_precision_m / lon_precision_m"
        )

    return radius.fillna(0.0).clip(lower=min_radius_m)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def _cells_in_circle(
    window: np.ndarray,
    win_row0: int,
    win_col0: int,
    lat: float,
    lon: float,
    radius_m: float,
) -> np.ndarray:
    """
    Land use codes for cells whose centres lie within radius_m of a point.

    Distances are computed in index space with a cos(latitude) correction; at
    these radii the flat-earth approximation is well inside the 1 km cell size.
    A sub-cell radius leaves no centre inside, and falls back to the single cell
    containing the point.

    Args:
        window (np.ndarray): Land use codes for one year, as a 2D sub-grid.
        win_row0 (int): Grid row index of window row 0.
        win_col0 (int): Grid column index of window column 0.
        lat (float): Sample latitude.
        lon (float): Sample longitude.
        radius_m (float): Uncertainty radius in metres.

    Returns:
        np.ndarray: 1D array of class codes, nodata not yet removed.
    """
    row, col = cell_index(lat, lon)
    m_per_deg_lon = M_PER_DEG * math.cos(math.radians(lat))

    d_row = int(math.ceil(radius_m / (CELL_DEG * M_PER_DEG)))
    d_col = int(math.ceil(radius_m / (CELL_DEG * max(m_per_deg_lon, 1e-6))))

    r0 = max(row - d_row, win_row0)
    r1 = min(row + d_row + 1, win_row0 + window.shape[0])
    c0 = max(col - d_col, win_col0)
    c1 = min(col + d_col + 1, win_col0 + window.shape[1])
    if r0 >= r1 or c0 >= c1:
        return np.array([], dtype=np.uint8)

    block = window[r0 - win_row0:r1 - win_row0, c0 - win_col0:c1 - win_col0]

    if radius_m > 0:
        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        d_ns = (LAT_TOP - (rows + 0.5) * CELL_DEG - lat) * M_PER_DEG
        d_ew = (LON_LEFT + (cols + 0.5) * CELL_DEG - lon) * m_per_deg_lon
        inside = (d_ns[:, None] ** 2 + d_ew[None, :] ** 2) <= radius_m ** 2
        if inside.any():
            return block[inside]

    return np.array([window[row - win_row0, col - win_col0]], dtype=np.uint8)


def _composition(codes: np.ndarray, min_cells_for_shannon: int) -> Optional[dict]:
    """
    Turn a set of class codes into percentages, a majority class, and a Shannon
    index.

    Args:
        codes (np.ndarray): Raw HILDA+ class codes, including any nodata.
        min_cells_for_shannon (int): Below this cell count, shannon is NaN.

    Returns:
        Optional[dict]: Composition columns, or None if every cell was nodata.
    """
    valid = codes[~np.isin(codes, NODATA_CODES)]
    if valid.size == 0:
        return None

    names = pd.Series(valid).map(HILDA_CLASSES).dropna()
    if names.empty:
        return None

    counts = names.value_counts()
    total = float(counts.sum())
    shares = counts / total

    out = {f"pct_{c.lower().replace('/', '_').replace(' ', '_')}": 0.0
           for c in CLASS_ORDER}
    for cls, share in shares.items():
        out[f"pct_{cls.lower().replace('/', '_').replace(' ', '_')}"] = share * 100.0

    out["pct_agricultural"] = sum(
        shares.get(cls, 0.0) for cls in AGRICULTURAL_CLASSES) * 100.0
    out["hilda_majority_class"] = counts.idxmax()
    out["hilda_n_cells"] = int(total)
    out["hilda_shannon"] = (
        float(-(shares * np.log(shares)).sum())
        if total >= min_cells_for_shannon else np.nan
    )
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def collect_hilda_data(
    df: pd.DataFrame,
    states_path: str | Path = HILDA_STATES,
    year_col: str = "year",
    lat_col: str = "decimalLatitude",
    lon_col: str = "decimalLongitude",
    radius_col: Optional[str] = None,
    min_radius_m: float = 0.0,
    min_cells_for_shannon: int = MIN_CELLS_FOR_SHANNON,
    checkpoint_dir: str = "hilda_checkpoints",
) -> pd.DataFrame:
    """
    Match annual HILDA+ land use/cover to samples by location and collection
    year.

    Loops years on the outside, reading one window per year that covers all of
    that year's samples, so each year is decompressed once. Finished years are
    checkpointed and an interrupted run resumes.

    Only years present in the sample table are read. Samples collected outside
    1899-2019 are left as NaN rather than being filled from the nearest
    available year.

    Args:
        df (pd.DataFrame): Sample table with coordinates, year, and a radius
                           source (see resolve_radius).
        states_path (str | Path): Path to the HILDA+ LULC states netCDF.
        year_col (str): Collection year column.
        lat_col (str): Latitude column.
        lon_col (str): Longitude column.
        radius_col (Optional[str]): Explicit radius column, in metres.
        min_radius_m (float): Lower bound on the uncertainty radius.
        min_cells_for_shannon (int): Below this cell count, shannon is NaN.
        checkpoint_dir (str): Directory for per-year checkpoint files.

    Returns:
        pd.DataFrame: Copy of df with HILDA+ composition columns joined on.
    """
    states_path = Path(states_path)
    if not states_path.exists():
        raise FileNotFoundError(f"HILDA+ states grid not found: {states_path}")

    work = df.copy()
    lat = pd.to_numeric(work[lat_col], errors="coerce")
    lon = pd.to_numeric(work[lon_col], errors="coerce")
    year = pd.to_numeric(work[year_col], errors="coerce")
    radius = resolve_radius(work, radius_col, min_radius_m)

    has_input = lat.notna() & lon.notna() & year.notna()
    if (~has_input).any():
        print(f"{int((~has_input).sum())} samples lack a coordinate or year and "
              "will be left as NaN")

    in_range = year.between(HILDA_YEAR_MIN, HILDA_YEAR_MAX)
    n_outside = int((has_input & ~in_range).sum())
    if n_outside:
        print(f"{n_outside} samples were collected outside HILDA+ coverage "
              f"({HILDA_YEAR_MIN}-{HILDA_YEAR_MAX}) and will be left as NaN")

    usable = has_input & in_range

    ckpt = Path(checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    frames = []
    years = sorted(year[usable].unique())

    with h5py.File(states_path, "r") as f:
        grid = f["LULC_states"]

        for yr in tqdm(years, desc="years", unit="yr"):
            ckpt_file = ckpt / f"{int(yr)}.csv"
            if ckpt_file.exists():
                frames.append(pd.read_csv(ckpt_file, index_col=0))
                continue

            group = work.index[usable & (year == yr)]
            glat, glon, grad = lat[group], lon[group], radius[group]

            # Window covering every sample this year, padded by the largest
            # radius among them. The longitude pad uses whichever latitude in
            # the group is furthest from the equator, the worst case.
            max_r = float(grad.max())
            worst_lat = max(abs(float(glat.min())), abs(float(glat.max())))
            pad_lat = max_r / M_PER_DEG + CELL_DEG
            pad_lon = max_r / (M_PER_DEG * max(math.cos(math.radians(worst_lat)), 1e-6)) + CELL_DEG

            r_top, _ = cell_index(float(glat.max()) + pad_lat, 0.0)
            r_bot, _ = cell_index(float(glat.min()) - pad_lat, 0.0)
            _, c_left = cell_index(0.0, float(glon.min()) - pad_lon)
            _, c_right = cell_index(0.0, float(glon.max()) + pad_lon)
            r0, r1 = r_top, min(r_bot + 1, N_ROWS)
            c0, c1 = c_left, min(c_right + 1, N_COLS)

            window = grid[int(yr) - HILDA_YEAR_MIN, r0:r1, c0:c1]
            tqdm.write(f"{int(yr)}: {len(group)} samples, "
                       f"window {window.shape[0]}x{window.shape[1]} cells")

            rows = {}
            for idx in group:
                codes = _cells_in_circle(
                    window, r0, c0, float(lat[idx]), float(lon[idx]), float(radius[idx])
                )
                comp = _composition(codes, min_cells_for_shannon)
                if comp is None:
                    continue

                prow, pcol = cell_index(float(lat[idx]), float(lon[idx]))
                point_code = int(window[prow - r0, pcol - c0])
                comp["hilda_point_class"] = HILDA_CLASSES.get(point_code)
                comp["hilda_method"] = "point" if comp["hilda_n_cells"] == 1 else "area"
                rows[idx] = comp

            out = pd.DataFrame.from_dict(rows, orient="index")
            out.to_csv(ckpt_file)
            frames.append(out)

    if frames:
        work = work.join(pd.concat(frames))

    _report_coverage(work, has_input, in_range)
    return work


def _report_coverage(
    df: pd.DataFrame,
    has_input: pd.Series,
    in_range: pd.Series,
) -> None:
    """Print how many samples got land use data, and why the rest did not."""
    total = len(df)
    have = df["pct_agricultural"].notna() if "pct_agricultural" in df else pd.Series(
        False, index=df.index)
    no_input = ~has_input
    outside = has_input & ~in_range
    no_value = ~have & has_input & in_range

    print("\n--- HILDA+ coverage ---")
    print(f"total samples:                     {total}")
    print(f"land use assigned:                 {int(have.sum())}")
    print(f"  of which point lookups:          "
          f"{int((df.get('hilda_method') == 'point').sum())}")
    print(f"  of which area composites:        "
          f"{int((df.get('hilda_method') == 'area').sum())}")
    print(f"no data:                           "
          f"{int((no_input | outside | no_value).sum())}")
    print(f"  - missing coordinate or year:    {int(no_input.sum())}")
    print(f"  - collected outside "
          f"{HILDA_YEAR_MIN}-{HILDA_YEAR_MAX}:   {int(outside.sum())}")
    print(f"  - all cells ocean/no-data:       {int(no_value.sum())}")
    if "hilda_shannon" in df:
        print(f"shannon index computed:            {int(df['hilda_shannon'].notna().sum())}"
              f"  (NaN where < {MIN_CELLS_FOR_SHANNON} cells)")
    if "hilda_n_cells" in df:
        n = pd.to_numeric(df["hilda_n_cells"], errors="coerce")
        print(f"cells per sample:                  median {n.median():.0f}, "
              f"max {n.max():.0f}")


if __name__ == "__main__":
    df = pd.read_csv("herbarium_metadata_2022_2026_datasets.csv", low_memory=False)

    result = collect_hilda_data(
        df,
        year_col="year",
        lat_col="decimalLatitude",
        lon_col="decimalLongitude",
    )
    result.to_csv("herbarium_samples_withHILDA.csv", index=False)
    print("\nWrote herbarium_samples_withHILDA.csv")
    print("NOTE: check hilda_n_cells before using hilda_shannon, and do not "
          "compare it against a 30 m NLCD index.")
