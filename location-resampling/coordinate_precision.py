"""
Kreiner Lab - October 19th 2026
Coordinate rounding / precision detection
Primary contributor(s): Sanju Patel

Objective: Recover the true recorded precision of each sample's coordinates by
detecting the grid the written value was snapped to, instead of counting its
decimal places.

Rationale: Herbarium labels routinely record position in degrees-minutes or
degrees-minutes-seconds. Converting 30 deg 38' to decimal gives 30.633333,
which a decimal-place counter reads as six-digit (0.11 m) precision when the
real recorded precision is one arc-minute, about 1,855 m.

Output: adds lat_grid / lon_grid (the detected recording grid) and
lat_precision_m / lon_precision_m (that grid step in metres) to the herbarium
metadata table. The rounding uncertainty of a coordinate is half its grid step,
since the true value lies within half a step of what was written.

Scope: this is the ROUNDING component of location uncertainty only, and it is a
floor rather than a total. Georeferencing error -- how vague the original
locality text was, and whether the coordinate is a gazetteer centroid -- is
usually the larger term and is not handled here.

Data requirements:
    Inputs: coordinates read as STRINGS, not floats. Parsing to float first
            discards written precision, so read_csv is given dtype=str for the
            two coordinate columns.

Known caveat: a CSV that has already been read into pandas and written back out
has lost trailing zeros (34.950000 becomes 34.95), because to_csv emits the
shortest round-tripping representation. Grid detection tolerates this -- it
reads such a value as sitting on the 0.01 degree grid, which is the correct
conservative reading -- but written precision is understated for those records.
Where the original accession files are available, run this on those instead.
"""

import math

import pandas as pd

M_PER_DEG = 111_320.0
_ARCMIN = 1.0 / 60.0
_ARCSEC = 1.0 / 3600.0

# Candidate recording grids, coarsest first: the granularities a label is
# actually written at. Only whole degrees, whole arcminutes and whole arcseconds
# are included on the sexagesimal side, with the decimal decades interleaved so
# a plainly decimal coordinate is not forced onto a sexagesimal grid.
#
# Composite steps (2', 5', 10', 15', 30', and the arcsecond equivalents) are
# deliberately absent. Nobody records "to the nearest 2 arcminutes", but half of
# all true arcminute values land on that grid by chance and a fifth land on the
# 10 arcminute grid -- including them lets a chance match halve or quintuple the
# precision we claim. 38.4333333 is 26 arcmin, a 1 arcmin value, and must not be
# reported as 2 arcmin.
GRID_STEPS = sorted(
    [
        ("1 deg",         1.0),
        ("0.1 deg",       0.1),
        ("1 arcmin",       1 * _ARCMIN),
        ("0.01 deg",      0.01),
        ("0.001 deg",     0.001),
        ("1 arcsec",       1 * _ARCSEC),
        ("0.0001 deg",    0.0001),
        ("0.00001 deg",   0.00001),
        ("0.000001 deg",  0.000001),
    ],
    key=lambda g: -g[1],
)

# Cap on the step-relative part of the match tolerance. The binding term is
# normally the value's own written precision (see detect_step); this cap exists
# only to stop a coarse grid swallowing a value that sits near one of its lines
# by accident -- without it, 34.8 matches the 0.25 degree grid at 20% of a step.
REL_TOL = 0.02


def detect_step(text: str) -> tuple[str, float]:
    """
    Find the coarsest grid that a written coordinate sits on.

    Only grids at least as coarse as the written precision are tested: a value
    written to two decimals trivially sits on the 0.000001 degree grid, so
    finer candidates would always match and tell us nothing. Among the rest the
    coarsest match wins, because a value on the 1 arcmin grid also sits on every
    finer grid.

    The match tolerance is half a unit in the value's last written decimal
    place. That is the only thing that can legitimately displace a coordinate
    from its grid -- a true arcminute value written to seven decimals lands
    within 5e-8 of the line, one written to five decimals within 5e-6 -- so
    tying the tolerance to the step instead would be far too permissive on
    coarse grids, where 2% of a degree is 1.2 arcmin.

    Args:
        text (str): The coordinate exactly as written in the source file.

    Returns:
        tuple[str, float]: Grid label and grid step in degrees. Returns
        (None, nan) for a missing or unparseable coordinate. When no candidate
        matches, falls back to the written precision itself, labelled "<n> dp".
    """
    s = "" if text is None else str(text).strip()
    if not s or s.lower() in {"nan", "na", "none", "null"} or "e" in s.lower():
        return None, float("nan")
    try:
        value = float(s)
    except ValueError:
        return None, float("nan")

    ndp = len(s.split(".")[1]) if "." in s else 0
    finest = 10.0 ** (-ndp)
    written_tol = 0.5 * finest + 1e-12   # float slack for exact DMS conversions

    for label, step in GRID_STEPS:
        if step < finest * (1 - 1e-9):
            continue
        # math.remainder is the signed distance to the nearest multiple, and is
        # more accurate here than value/step - round(value/step).
        if abs(math.remainder(value, step)) <= min(written_tol, REL_TOL * step):
            return label, step

    return f"{ndp} dp", finest


if __name__ == "__main__":
    herbarium_file_path = "herbarium_metadata_2022_2026_datasets.csv"

    # Coordinates must stay as strings or their written precision is lost before
    # we can read it. Everything else is held as a string too, so that writing
    # the table back in place reproduces every existing column byte for byte
    # rather than nudging floats through repr.
    df = pd.read_csv(herbarium_file_path, dtype=str, low_memory=False)

    lat_grids, lon_grids, lat_prec, lon_prec = [], [], [], []
    for lat_text, lon_text in zip(df["decimalLatitude"], df["decimalLongitude"]):
        lat_label, lat_step = detect_step(lat_text)
        lon_label, lon_step = detect_step(lon_text)
        lat_grids.append(lat_label)
        lon_grids.append(lon_label)

        # A degree of longitude narrows by cos(latitude), so the same grid step
        # is a shorter distance east-west than north-south.
        lat_prec.append(lat_step * M_PER_DEG)
        try:
            cos_lat = math.cos(math.radians(float(lat_text)))
        except (TypeError, ValueError):
            cos_lat = float("nan")
        lon_prec.append(lon_step * M_PER_DEG * cos_lat)

    df["lat_grid"] = lat_grids
    df["lon_grid"] = lon_grids
    df["lat_precision_m"] = lat_prec
    df["lon_precision_m"] = lon_prec

    df.to_csv(herbarium_file_path, index=False)

    governing = df[["lat_precision_m", "lon_precision_m"]].max(axis=1)
    summary = (
        pd.DataFrame({"grid": df["lat_grid"], "precision_m": df["lat_precision_m"]})
        .dropna()
        .groupby("grid")
        .agg(n=("grid", "size"), precision_m=("precision_m", "median"))
        .sort_values("precision_m", ascending=False)
    )
    print(summary.to_string(float_format=lambda v: f"{v:,.3f}"))
    print(f"\nwrote lat/lon precision for {int(governing.notna().sum())} of {len(df)} samples")
    print(f"coarsest: {governing.max():,.0f} m   median: {governing.median():,.3f} m")
    print("NOTE: rounding uncertainty is half the grid step, and is a floor on "
          "total location uncertainty -- georeferencing error is not included.")
