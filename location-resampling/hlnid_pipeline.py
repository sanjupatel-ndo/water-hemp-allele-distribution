"""
Kreiner Lab - November 4th 2026
Land use pipeline (HLNID)
Primary contributor(s): Sanju Patel

Objective: Pull land use / land cover and crop type for samples at the finest
location precision each sample supports.

Methodology: For each sample, read every HLNID LULC cell whose centre falls
within the sample's own location uncertainty, and report the class composition
of those cells plus the class at the point itself. No radius floor is applied:
where the uncertainty is smaller than a cell, the result is the single cell
containing the coordinate. HLNID_n_cells says which case a row is.

HLNID against HILDA+: HLNID separates the crops waterhemp grows in (Corn,
Soybean, Small grains, Alfalfa, Non-alfalfa hay, Other crop) where HILDA+
collapses them into one Cropland class, and splits developed land into four
intensities. The cost is coverage: HLNID is conterminous US only and starts in
1938, so Ontario and pre-1938 samples are NaN here. The two are complementary.

Classes are reported under their exact HLNID names, with no grouping into
agricultural and non-agricultural.

Data requirements:
    Inputs: Samples with latitude, longitude, collection year, and a location
            uncertainty radius. Radius comes from the coordinate precision
            columns written by coordinate_precision.py, or from an explicit
            column.
    Grids:  HLNID_48sec_20260406/lulc/CONUS_vegtype_YYYY_48sec.tif, 1938-2020,
            from https://doi.org/10.5061/dryad.r4xgxd2rh (CC0). Download
            HLNID_48sec_20260406.zip (682 MB) and unzip it beside this script.
            The 150 arc-sec release also works -- point lulc_dir at it. Grids
            are matched by glob, since the Dryad readme names them
            vegtype_YYYY.tif but the archive ships CONUS_vegtype_YYYY_48sec.tif.
"""

from __future__ import annotations   # anaconda env here is Python 3.9

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

HLNID_LULC_DIR = "HLNID_48sec_20260406/lulc"
VEGTYPE_GLOB = "*vegtype_*.tif"
PREFIX = "HLNID_"

M_PER_DEG = 111_320.0
HLNID_YEAR_MIN, HLNID_YEAR_MAX = 1938, 2020

# Codes and names exactly as the HLNID readme gives them. Forest is split by
# leaf habit rather than collapsed as in the HILDA+ pipeline: here the division
# comes from NLCD/LCMAP/FORE-SCE for the year in question, not carried forward
# from a single reference year.
HLNID_CLASSES = {
    4: "Evergreen forest",
    5: "Deciduous forest",
    8: "Mixed forest",
    10: "Grassland",
    11: "Shrubland",
    15: "Perennial ice/snow",
    16: "Corn",
    17: "Soybean",
    18: "Small grains",
    19: "Alfalfa",
    20: "Non-alfalfa hay",
    21: "Pasture / range",
    22: "Developed / high intensity",
    23: "Developed / medium intensity",
    24: "Developed / low intensity",
    25: "Developed / open space",
    26: "Herbaceous wetland",
    27: "Woody wetland",
    30: "Barren",
    98: "Open water",
    99: "Other crop",
}
CLASS_ORDER = list(HLNID_CLASSES.values())

# 0 is the off-CONUS background; any raster-declared nodata is added at read time.
NODATA_CODES = (0,)


def _slug(name: str) -> str:
    """'Pasture / range' -> 'HLNID_pct_pasture_range'."""
    return PREFIX + "pct_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


PCT_COLUMNS = [_slug(c) for c in CLASS_ORDER]


def find_lulc_dir(lulc_dir: str | Path = HLNID_LULC_DIR) -> Path:
    """
    Locate the directory holding the annual vegtype rasters. Accepts that
    directory or any parent of it, so the wrapper folder the archive unzips to
    is fine.
    """
    root = Path(lulc_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"HLNID directory not found: {root}. Download "
            "HLNID_48sec_20260406.zip from https://doi.org/10.5061/dryad.r4xgxd2rh "
            "and unzip it beside this script."
        )
    if any(root.glob(VEGTYPE_GLOB)):
        return root

    hits = sorted({p.parent for p in root.rglob(VEGTYPE_GLOB)})
    if not hits:
        raise FileNotFoundError(f"no {VEGTYPE_GLOB} found under {root}")
    if len(hits) > 1:
        raise ValueError(
            f"several vegtype directories under {root} ({', '.join(map(str, hits))}); "
            "pass the one you want as lulc_dir"
        )
    return hits[0]


def resolve_radius(
    df: pd.DataFrame,
    radius_col: Optional[str] = None,
) -> pd.Series:
    """
    Location uncertainty in metres per sample. A missing uncertainty becomes 0,
    which reads the single cell containing the coordinate.
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

    return radius.fillna(0.0).clip(lower=0.0)


class _Grid:
    """
    Geometry of one HLNID raster, read from the file rather than hardcoded so
    either resolution works. The circle maths below is in degrees, so a
    projected raster is rejected instead of being silently mis-sampled.
    """

    def __init__(self, src: rasterio.DatasetReader):
        t = src.transform
        if t.b or t.d:
            raise ValueError(f"{src.name}: rotated raster, not supported")
        if src.crs is None or not src.crs.is_geographic:
            raise ValueError(
                f"{src.name}: expected a geographic (lat/lon) CRS, got {src.crs}"
            )
        self.lon_left = t.c
        self.lat_top = t.f
        self.lon_step = t.a          # positive, degrees per column
        self.lat_step = -t.e         # positive, degrees per row (north-up)
        self.n_rows = src.height
        self.n_cols = src.width

    def cell_index(self, lat: float, lon: float) -> tuple[int, int]:
        """(row, col) of the cell containing a coordinate; may be off-grid."""
        row = int(math.floor((self.lat_top - lat) / self.lat_step))
        col = int(math.floor((lon - self.lon_left) / self.lon_step))
        return row, col

    def contains(self, row: int, col: int) -> bool:
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols


def _read_circle(
    src: rasterio.DatasetReader,
    grid: _Grid,
    lat: float,
    lon: float,
    radius_m: float,
) -> np.ndarray:
    """
    Class codes (nodata not yet removed) for cells whose centres lie within
    radius_m of a point, or the single cell containing the point when the
    radius is sub-cell. Empty if the point is off the raster.

    Distances are computed in index space with a cos(latitude) correction; at
    these radii the flat-earth approximation is well inside the cell size.
    """
    row, col = grid.cell_index(lat, lon)
    m_per_deg_lon = M_PER_DEG * math.cos(math.radians(lat))

    d_row = int(math.ceil(radius_m / (grid.lat_step * M_PER_DEG)))
    d_col = int(math.ceil(radius_m / (grid.lon_step * max(m_per_deg_lon, 1e-6))))

    r0 = max(row - d_row, 0)
    r1 = min(row + d_row + 1, grid.n_rows)
    c0 = max(col - d_col, 0)
    c1 = min(col + d_col + 1, grid.n_cols)
    if r0 >= r1 or c0 >= c1:
        return np.array([], dtype=np.uint8)

    block = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))

    if radius_m > 0:
        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        d_ns = (grid.lat_top - (rows + 0.5) * grid.lat_step - lat) * M_PER_DEG
        d_ew = (grid.lon_left + (cols + 0.5) * grid.lon_step - lon) * m_per_deg_lon
        inside = (d_ns[:, None] ** 2 + d_ew[None, :] ** 2) <= radius_m ** 2
        if inside.any():
            return block[inside]

    if not grid.contains(row, col):
        return np.array([], dtype=block.dtype)
    return np.array([block[row - r0, col - c0]], dtype=block.dtype)


def _composition(codes: np.ndarray, nodata_codes: tuple) -> Optional[dict]:
    """
    Class codes to percentages, a majority class and a cell count. None if
    every cell was nodata.
    """
    if codes.size == 0:
        return None
    valid = codes[~np.isin(codes, nodata_codes)]
    if valid.size == 0:
        return None

    names = pd.Series(valid).map(HLNID_CLASSES).dropna()
    if names.empty:
        return None

    counts = names.value_counts()
    total = float(counts.sum())

    out = {col: 0.0 for col in PCT_COLUMNS}
    for cls, share in (counts / total).items():
        out[_slug(cls)] = share * 100.0

    out[PREFIX + "majority_class"] = counts.idxmax()
    out[PREFIX + "n_cells"] = int(total)
    return out


def collect_hlnid_data(
    df: pd.DataFrame,
    lulc_dir: str | Path = HLNID_LULC_DIR,
    year_col: str = "year",
    lat_col: str = "decimalLatitude",
    lon_col: str = "decimalLongitude",
    radius_col: Optional[str] = None,
    checkpoint_dir: str = "hlnid_checkpoints",
) -> pd.DataFrame:
    """
    Match annual HLNID LULC to samples by location and collection year, reading
    each over its own location uncertainty. Returns a copy of df with the
    HLNID_ columns joined on.

    Loops years on the outside so each annual GeoTIFF is opened once. Finished
    years are checkpointed and an interrupted run resumes. Samples collected
    outside 1938-2020 are left as NaN rather than filled from the nearest
    available year.
    """
    lulc = find_lulc_dir(lulc_dir)

    work = df.copy()
    lat = pd.to_numeric(work[lat_col], errors="coerce")
    lon = pd.to_numeric(work[lon_col], errors="coerce")
    year = pd.to_numeric(work[year_col], errors="coerce")
    radius = resolve_radius(work, radius_col)

    has_input = lat.notna() & lon.notna() & year.notna()
    in_range = year.between(HLNID_YEAR_MIN, HLNID_YEAR_MAX)
    usable = has_input & in_range

    # Checkpoints are only valid for the grid they were written under, so they
    # are kept per resolution rather than silently reused.
    ckpt = Path(checkpoint_dir) / lulc.parent.name
    ckpt.mkdir(parents=True, exist_ok=True)

    frames = []
    for yr in tqdm(sorted(year[usable].unique()), desc="years", unit="yr"):
        ckpt_file = ckpt / f"{int(yr)}.csv"
        if ckpt_file.exists():
            frames.append(pd.read_csv(ckpt_file, index_col=0))
            continue

        matches = sorted(lulc.glob(f"*vegtype_{int(yr)}_*.tif")) or \
            sorted(lulc.glob(f"*vegtype_{int(yr)}.tif"))
        if not matches:
            raise FileNotFoundError(
                f"no vegtype raster for {int(yr)} in {lulc}: the year is inside "
                f"{HLNID_YEAR_MIN}-{HLNID_YEAR_MAX} but the file is not there"
            )
        if len(matches) > 1:
            raise ValueError(
                f"several vegtype rasters for {int(yr)} in {lulc}: "
                f"{', '.join(p.name for p in matches)}"
            )
        tif = matches[0]

        rows = {}
        with rasterio.open(tif) as src:
            grid = _Grid(src)
            nodata = NODATA_CODES if src.nodata is None else (
                tuple(NODATA_CODES) + (int(src.nodata),))

            for idx in work.index[usable & (year == yr)]:
                comp = _composition(
                    _read_circle(src, grid, float(lat[idx]), float(lon[idx]),
                                 float(radius[idx])),
                    nodata,
                )
                if comp is None:
                    continue

                prow, pcol = grid.cell_index(float(lat[idx]), float(lon[idx]))
                point_code = (int(src.read(1, window=Window(pcol, prow, 1, 1))[0, 0])
                              if grid.contains(prow, pcol) else None)
                comp[PREFIX + "point_class"] = HLNID_CLASSES.get(point_code)
                comp[PREFIX + "radius_m"] = float(radius[idx])
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
    n_cells_col = PREFIX + "n_cells"
    have = (df[n_cells_col].notna() if n_cells_col in df
            else pd.Series(False, index=df.index))
    no_input = ~has_input
    outside = has_input & ~in_range
    # HLNID is CONUS only, so off-CONUS samples land here: outside the mapped
    # area reads as background.
    no_value = ~have & has_input & in_range

    print("\n--- HLNID coverage ---")
    print(f"total samples:                   {len(df)}")
    print(f"land use assigned:               {int(have.sum())}")
    print(f"no data:                         "
          f"{int((no_input | outside | no_value).sum())}")
    print(f"  - missing coordinate or year:  {int(no_input.sum())}")
    print(f"  - collected outside "
          f"{HLNID_YEAR_MIN}-{HLNID_YEAR_MAX}: {int(outside.sum())}")
    print(f"  - outside CONUS / background:  {int(no_value.sum())}")

    if n_cells_col in df:
        n = pd.to_numeric(df[n_cells_col], errors="coerce")
        print(f"cells per sample:                median {n.median():.0f}, "
              f"max {n.max():.0f}   ({int((n == 1).sum())} read as a single cell)")
    if PREFIX + "radius_m" in df:
        r = pd.to_numeric(df[PREFIX + "radius_m"], errors="coerce")
        print(f"uncertainty radius (m):          median {r.median():.0f}, "
              f"max {r.max():.0f}")
    if PREFIX + "majority_class" in df:
        print("majority class:")
        for cls, n in df[PREFIX + "majority_class"].value_counts().items():
            print(f"  {cls:<30} {n}")


if __name__ == "__main__":
    df = pd.read_csv("herbarium_metadata_2022_2026_datasets.csv", low_memory=False)

    result = collect_hlnid_data(
        df,
        year_col="year",
        lat_col="decimalLatitude",
        lon_col="decimalLongitude",
    )
    result.to_csv("herbarium_samples_withHLNID.csv", index=False)
