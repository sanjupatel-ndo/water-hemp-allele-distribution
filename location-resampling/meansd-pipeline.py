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
        corresponding 1 km buffers.
    """
    geometry = [Point(lon, lat) for lon, lat in zip(obs_df['decimalLongitude'], obs_df['decimalLatitude'])]
    gdf = gpd.GeoDataFrame(obs_df, geometry=geometry, crs="EPSG:4326")

    gdf_utm = gdf.to_crs("EPSG:32616")
    gdf_utm["uncertainty_buffer"] = gdf_utm.geometry.buffer(gdf_utm["coordinateUncertaintyInMeters"])
    return gdf_utm

gbif_observations_file_path = '1000m_amaranthus_with_proportions.csv'
df = pd.read_csv(gbif_observations_file_path)

gdf_utm = create_buffers(df)
print(gdf_utm.head())
