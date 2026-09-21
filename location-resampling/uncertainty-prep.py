"""
Kreiner Lab - September 14th 2026
Resampling pipeline
Primary contributor(s): Sanju Patel

Objective: Report a location uncertainty for every sample. Additionally,
check if location data is properly reported and generally prep for resampling.
"""

import math
import pandas as pd
import geopandas as gpd

def _decimal_places(value: float) -> int:
    """Count significant decimal places by inspecting the string representation."""
    s = f"{value}".rstrip('0')
    if '.' not in s:
        return 0
    return len(s.split('.')[1])

def _coord_uncertainty_m(value: float, is_longitude: bool, lat: float) -> float:
    """
    Converts coordinate decimal-place precision to meters of uncertainty.
    One unit at the last decimal place represents 10^(-n) degrees.
    """
    n = _decimal_places(value)
    deg_precision = 10 ** (-n)
    if is_longitude:
        return deg_precision * 111_320 * math.cos(math.radians(lat))
    return deg_precision * 111_320

#provide a location uncertainty for every sample
def fill_uncertainty(loc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing coordinateUncertaintyInMeters values by inferring precision
    from the number of decimal places in decimalLatitude and decimalLongitude.
    Where lat and lon have different precisions, the larger uncertainty is used.

    Args:
        loc_df (pd.DataFrame): Data Frame containing the coordinate sets
    Returns:
        pd.DataFrame: Data Frame with an uncertainty value for every location
    """
    df = loc_df.copy()
    missing_mask = df["coordinateUncertaintyInMeters"].isna()

    for idx in df[missing_mask].index:
        lat = df.at[idx, "decimalLatitude"]
        lon = df.at[idx, "decimalLongitude"]
        lat_unc = _coord_uncertainty_m(lat, is_longitude=False, lat=lat)
        lon_unc = _coord_uncertainty_m(lon, is_longitude=True, lat=lat)
        df.at[idx, "coordinateUncertaintyInMeters"] = max(lat_unc, lon_unc)

    return df

