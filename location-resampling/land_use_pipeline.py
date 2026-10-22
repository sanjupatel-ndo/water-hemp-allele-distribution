"""
Kreiner Lab - October 21st 2026
Land use pipeline (HILDA+)
Primary contributor(s): Sanju Patel

Objective: Pull land use data for samples with varying degrees of location
uncertainty.

Methodology: For each sample, take the area where it may have been collected
and read every HILDA+ land use/cover cell whose centre falls inside that area.
Report the class composition of those cells -- percent agricultural, percent of
each other class, the majority class, and a Shannon index -- plus the class at
the stated point itself.

HILDA+ is used because it is the only product covering the full span of this
collection: annual, 1899-2019, global, so it reaches the pre-NLCD herbarium
records and the Ontario samples that CONUS products exclude. The cost is
resolution: cells are 0.01 degrees, about 1.1 km north-south and 850 m
east-west at 40 N. Finer products (Annual NLCD at 30 m, 1985-2024 CONUS) should
be layered on top for the modern subset; HILDA+ is the backbone that makes the
whole series comparable.

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

# The grid is a clean global 0.01 degree raster: 18000 x 36000 cells with
# centres at 89.995 .. -89.995 and -179.995 .. 179.995. The stored latitude and
# longitude arrays are float32 and carry about 1e-5 of jitter, so cell indices
# are computed analytically from the grid definition rather than by searching
# those arrays. Note the metadata .txt shipped with the download claims
# 18021 x 36041 and a Europe-only extent; both are stale. Trust the file.
CELL_DEG = 0.01
N_ROWS, N_COLS = 18000, 36000
LAT_TOP, LON_LEFT = 90.0, -180.0
M_PER_DEG = 111_320.0

HILDA_YEAR_MIN, HILDA_YEAR_MAX = 1899, 2019

# Forest subclasses 41-45 are collapsed into a single Forest class. The HILDA+
# documentation states that the forest subdivision is static, carried from
# ESA-CCI 1992, for everything before 1992 and extrapolated before 1960 -- so
# subclass composition in the historical period is not an observation and must
# not be allowed to inflate a diversity index.
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

# The documentation names 99 as the fill value, but no 99 occurs anywhere in
# the grid; ocean and no-data are actually coded 0. Both are treated as missing.
NODATA_CODES = (0, 99)

# A Shannon index over a handful of cells is noise rather than a landscape
# measurement. Below this cell count the index is returned as NaN. Even at the
# default, treat the result with suspicion: a meaningful index wants on the
# order of 25+ cells, which at 1 km resolution means a radius of about 3 km.
MIN_CELLS_FOR_SHANNON = 2


# --------------------------------------------------------------------------
# Grid indexing
# --------------------------------------------------------------------------
def cell_index(lat: float, lon: float) -> tuple[int, int]:
    """
    Convert a coordinate to its HILDA+ (row, col) cell index.

    Args:
        lat (float): Latitude in degrees north.
        lon (float): Longitude in degrees east.

    Returns:
        tuple[int, int]: Row and column index, clamped to the grid.
    """
    row = int(math.floor((LAT_TOP - lat) / CELL_DEG))
    col = int(math.floor((lon - LON_LEFT) / CELL_DEG))
    return min(max(row, 0), N_ROWS - 1), min(max(col, 0), N_COLS - 1)


def cell_center(row: int, col: int) -> tuple[float, float]:
    """
    Return the centre coordinate of a HILDA+ cell.

    Args:
        row (int): Row index.
        col (int): Column index.

    Returns:
        tuple[float, float]: Latitude and longitude of the cell centre.
    """
    return (LAT_TOP - (row + 0.5) * CELL_DEG, LON_LEFT + (col + 0.5) * CELL_DEG)


def resolve_radius(
    df: pd.DataFrame,
    radius_col: Optional[str] = None,
    min_radius_m: float = 0.0,
) -> pd.Series:
    """
    Decide the uncertainty radius, in metres, to use for each sample.

    With no radius_col, the radius is the rounding uncertainty implied by the
    coordinate precision columns from coordinate_precision.py: half the coarser
    of the two grid steps, since a value rounded to a grid lies within half a
    step of what was written. Those columns hold the full step, hence the
    halving here.

    This is a floor on total location uncertainty, not the whole of it.
    Georeferencing error -- how vague the original locality text was, and
    whether the coordinate is a gazetteer centroid -- is usually the larger
    term and is not yet available. Pass radius_col once it is.

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

    Distances are computed in index space with a cos(latitude) correction
    rather than by reprojecting and masking geometry. Over a radius of at most
    a few hundred kilometres the local flat-earth approximation is well inside
    the 1 km cell size, and it avoids building a polygon per sample.

    If the radius is smaller than a cell -- true for most of this collection,
    since HILDA+ cells are about 1 km and most coordinates are more precise
    than that -- no centre falls inside, and the single cell containing the
    point is returned instead.

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

    # Half-extent of the search box, in cells, on each axis.
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

    # Sub-cell radius: fall back to the one cell containing the point.
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

    Loops over collection years on the outside, reading one window per year
    that covers every sample collected in that year, then extracting each
    sample from the in-memory window. This keeps one decompression pass per
    year instead of one per sample. Each finished year is checkpointed, so an
    interrupted run resumes where it stopped.

    Samples outside the 1899-2019 range of the product are clamped to the
    nearest available year and flagged in hilda_year_offset; the land use at a
    sample collected in 1870 is read from the 1899 grid, which is the closest
    HILDA+ can get, and that gap must be reported in any analysis.

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

    usable = lat.notna() & lon.notna() & year.notna()
    if (~usable).any():
        print(f"{int((~usable).sum())} samples lack a coordinate or year and "
              "will be left as NaN")

    year_used = year.clip(HILDA_YEAR_MIN, HILDA_YEAR_MAX)
    work["hilda_year_used"] = year_used
    work["hilda_year_offset"] = (year_used - year).abs()
    n_clamped = int(((work["hilda_year_offset"] > 0) & usable).sum())
    if n_clamped:
        print(f"{n_clamped} samples fall outside HILDA+ coverage "
              f"({HILDA_YEAR_MIN}-{HILDA_YEAR_MAX}) and were clamped to the "
              "nearest year; see hilda_year_offset")

    ckpt = Path(checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    frames = []
    years = sorted(year_used[usable].unique())

    with h5py.File(states_path, "r") as f:
        grid = f["LULC_states"]

        for yr in tqdm(years, desc="years", unit="yr"):
            ckpt_file = ckpt / f"{int(yr)}.csv"
            if ckpt_file.exists():
                frames.append(pd.read_csv(ckpt_file, index_col=0))
                continue

            group = work.index[usable & (year_used == yr)]
            glat, glon, grad = lat[group], lon[group], radius[group]

            # One window covering every sample in this year, padded by the
            # largest radius among them. A degree of longitude shortens by
            # cos(latitude), so the same radius spans more degrees of longitude
            # than of latitude; the pad is computed at whichever latitude in
            # this group is furthest from the equator, which is the worst case.
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

    _report_coverage(work, usable)
    return work


def _report_coverage(df: pd.DataFrame, usable: pd.Series) -> None:
    """
    Print how many samples ended up with land use data, and why the rest did
    not.

    Args:
        df (pd.DataFrame): Result table.
        usable (pd.Series): Mask of samples that had a coordinate and a year.
    """
    total = len(df)
    have = df["pct_agricultural"].notna() if "pct_agricultural" in df else pd.Series(
        False, index=df.index)
    no_input = ~usable
    no_value = ~have & usable

    print("\n--- HILDA+ coverage ---")
    print(f"total samples:                     {total}")
    print(f"land use assigned:                 {int(have.sum())}")
    print(f"  of which point lookups:          "
          f"{int((df.get('hilda_method') == 'point').sum())}")
    print(f"  of which area composites:        "
          f"{int((df.get('hilda_method') == 'area').sum())}")
    print(f"no data:                           {int((no_input | no_value).sum())}")
    print(f"  - missing coordinate or year:    {int(no_input.sum())}")
    print(f"  - all cells ocean/no-data:       {int(no_value.sum())}")
    if "hilda_year_offset" in df:
        off = pd.to_numeric(df["hilda_year_offset"], errors="coerce")
        print(f"year clamped to HILDA+ range:      {int((off > 0).sum())}"
              f"  (max gap {int(off.max()) if off.notna().any() else 0} yr)")
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
    print("NOTE: HILDA+ cells are ~1 km. Shannon over a handful of cells is not "
          "a landscape measurement -- check hilda_n_cells before using it, and "
          "do not compare it against a 30 m NLCD index.")
