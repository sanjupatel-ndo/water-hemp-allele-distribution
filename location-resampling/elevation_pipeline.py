"""
Kreiner Lab - November 11th 2026
Elevation pipeline (SRTM 30 m)
Primary contributor(s): Sanju Patel

Objective: Attach a ground elevation to every sample. Herbarium labels do not
record one and ClimateNA needs it. Each uncertainty radius is flattened to the 
point elevation.

Data requirements:
    Inputs:  Samples with a latitude and longitude.
    Network: api.opentopodata.org, 100 points per call, one call a second.
             Lookups are cached in the file climateNA_pipeline.r also reads.
"""

from __future__ import annotations   # anaconda env here is Python 3.9

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

ELEV_API = "https://api.opentopodata.org/v1/srtm30m"
CACHE_FILE = "climatena_cache/elevations.csv"
BATCH = 100
PAUSE = 1.0


def fetch_elevations(
    lat: pd.Series,
    lon: pd.Series,
    cache_file: str | Path = CACHE_FILE,
    pause: float = PAUSE,
    retries: int = 4,
) -> pd.Series:
    """Elevation in metres per coordinate, NaN where the DEM has no value."""
    cache_file = Path(cache_file)
    key = lat.round(6).astype(str) + "," + lon.round(6).astype(str)
    key[lat.isna() | lon.isna()] = None

    cache = (pd.read_csv(cache_file).set_index("key")["elev"]
             if cache_file.exists() else pd.Series(dtype=float, name="elev"))
    cache.index = cache.index.astype(str)

    missing = sorted(set(key.dropna()) - set(cache.index))
    for start in tqdm(range(0, len(missing), BATCH), desc="elevation", unit="batch"):
        chunk = missing[start:start + BATCH]
        found = {}

        for attempt in range(retries):
            try:
                r = requests.get(ELEV_API, params={"locations": "|".join(chunk)},
                                 timeout=180)
            except requests.RequestException as e:
                tqdm.write(f"  request error ({e}); retry {attempt + 1}/{retries}")
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                found = {k: v["elevation"]
                         for k, v in zip(chunk, r.json()["results"])}
                break
            tqdm.write(f"  HTTP {r.status_code}; retry {attempt + 1}/{retries}")
            time.sleep(2 ** attempt)

        cache = pd.concat([cache, pd.Series(found or {k: None for k in chunk},
                                            dtype=float, name="elev")])
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache.rename_axis("key").to_csv(cache_file)
        time.sleep(pause)

    return key.map(cache).astype(float)


def add_elevation(
    df: pd.DataFrame,
    lat_col: str = "decimalLatitude",
    lon_col: str = "decimalLongitude",
    cache_file: str | Path = CACHE_FILE,
) -> pd.DataFrame:
    """Return a copy of df with an elevation column."""
    work = df.copy()
    work["elevation"] = fetch_elevations(
        pd.to_numeric(work[lat_col], errors="coerce"),
        pd.to_numeric(work[lon_col], errors="coerce"),
        cache_file,
    )

    e = work["elevation"]
    print("\n--- elevation coverage ---")
    print(f"total samples:      {len(work)}")
    print(f"elevation assigned: {int(e.notna().sum())}")
    print(f"no data:            {int(e.isna().sum())}")
    if e.notna().any():
        print(f"metres:             median {e.median():.0f}, "
              f"range {e.min():.0f} to {e.max():.0f}")
    return work


if __name__ == "__main__":
    df = pd.read_csv("herbarium_metadata_2022_2026_datasets.csv", low_memory=False)

    result = add_elevation(df)
    result.to_csv("herbarium_samples_elevation.csv", index=False)
