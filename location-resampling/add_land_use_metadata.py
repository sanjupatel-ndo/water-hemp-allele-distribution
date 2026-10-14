from __future__ import annotations

import argparse
from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterstats import zonal_stats
from shapely.geometry import Point
from shapely.ops import transform

from diversity_indices import calculate_shannon_index
from metadata_common import DEFAULT_NLCD_DIR, standardize_sample_table


NLCD_CLASSES = {
    11: "Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}

AGRICULTURAL_CLASSES = ["Cultivated Crops", "Pasture/Hay"]
NATURAL_CLASSES = [
    "Water",
    "Perennial Ice/Snow",
    "Barren Land",
    "Deciduous Forest",
    "Evergreen Forest",
    "Mixed Forest",
    "Shrub/Scrub",
    "Grassland/Herbaceous",
    "Woody Wetlands",
    "Emergent Herbaceous Wetlands",
]


def radius_suffix(radius_m: float) -> str:
    if float(radius_m).is_integer():
        return f"{int(radius_m)}m"
    return f"{str(radius_m).replace('.', '_')}m"


def build_geodesic_buffer(longitude: float, latitude: float, radius_m: float):
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} +datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True).transform

    point_local = transform(to_local, Point(longitude, latitude))
    return transform(to_wgs84, point_local.buffer(radius_m))


def build_sample_geodataframe(df: pd.DataFrame, radius_m: float) -> gpd.GeoDataFrame:
    geometry = [
        Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
        for lon, lat in zip(df["decimalLongitude"], df["decimalLatitude"])
    ]
    buffers = [
        build_geodesic_buffer(lon, lat, radius_m) if pd.notna(lon) and pd.notna(lat) else None
        for lon, lat in zip(df["decimalLongitude"], df["decimalLatitude"])
    ]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326").assign(buffers=buffers)


def _resolve_raster_path(nlcd_dir: Path, template: str, year: int) -> Path:
    raster_path = nlcd_dir / template.format(year=int(year))
    if not raster_path.exists():
        raise FileNotFoundError(f"NLCD raster for year {year} not found: {raster_path}")
    return raster_path


def _available_raster_years(nlcd_dir: Path, template: str) -> list[int]:
    pattern = re.escape(template).replace(r"\{year\}", r"(?P<year>\d{4})")
    years = []
    for path in nlcd_dir.iterdir():
        if not path.is_file():
            continue
        match = re.fullmatch(pattern, path.name)
        if match:
            years.append(int(match.group("year")))
    return sorted(set(years))


def _sample_point_classes(group: gpd.GeoDataFrame, src: rasterio.io.DatasetReader) -> pd.Series:
    transformer = Transformer.from_crs(group.crs, src.crs, always_xy=True)
    classes: dict[int, str | None] = {}

    for idx, point in group.geometry.items():
        if point is None or point.is_empty:
            classes[idx] = None
            continue
        x, y = transformer.transform(point.x, point.y)
        pixel_value = next(src.sample([(x, y)]))[0]
        classes[idx] = NLCD_CLASSES.get(int(pixel_value)) if pd.notna(pixel_value) else None

    return pd.Series(classes)


def _zonal_percentages(buffer_geometries, raster_path: Path) -> list[dict[str, float]]:
    stats = zonal_stats(
        list(buffer_geometries),
        raster_path,
        categorical=True,
        category_map=NLCD_CLASSES,
        all_touched=True,
        geojson_out=True,
    )

    results: list[dict[str, float]] = []
    for zone in stats:
        props = zone["properties"]
        counts = {k: v for k, v in props.items() if k != "id" and pd.notna(v)}
        total = float(sum(counts.values()))
        if total <= 0:
            results.append({})
            continue

        percentages = {k: (float(v) / total) * 100.0 for k, v in counts.items()}
        shannon = calculate_shannon_index(list(percentages.values()))
        percentages["shannon_index"] = shannon
        results.append(percentages)

    return results


def add_land_use_metadata(
    df: pd.DataFrame,
    radius_m: float = 1000.0,
    nlcd_dir: str | Path = DEFAULT_NLCD_DIR,
    nlcd_template: str = "Annual_NLCD_LndCov_{year}_CU_C1V0.tif",
    year_col: str = "year",
    lon_col: str | None = None,
    lat_col: str | None = None,
) -> pd.DataFrame:
    result = standardize_sample_table(df, lon_col=lon_col, lat_col=lat_col, year_col=year_col)
    gdf = build_sample_geodataframe(result, radius_m=radius_m)

    suffix = radius_suffix(radius_m)
    top_class_col = f"top_class_{suffix}"
    ag_col = f"proportion_agricultural_{suffix}"
    natural_col = f"proportion_natural_{suffix}"
    shannon_col = f"shannon_idx_{suffix}"

    for column in ["land_cover_class", top_class_col, ag_col, natural_col, shannon_col]:
        if column not in result.columns:
            result[column] = pd.NA

    valid_mask = (
        gdf.geometry.notna()
        & gdf["buffers"].notna()
        & gdf[year_col].notna()
    )
    nlcd_dir = Path(nlcd_dir)
    available_years = _available_raster_years(nlcd_dir, nlcd_template)
    if not available_years:
        raise FileNotFoundError(
            f"No NLCD rasters matching template '{nlcd_template}' were found in {nlcd_dir}."
        )
    earliest_year = available_years[0]
    valid_gdf = gdf.loc[valid_mask & (gdf[year_col] >= earliest_year)].copy()

    for year, group in valid_gdf.groupby(year_col):
        raster_path = _resolve_raster_path(nlcd_dir, nlcd_template, int(year))

        with rasterio.open(raster_path) as src:
            result.loc[group.index, "land_cover_class"] = _sample_point_classes(group, src)

            buffer_gdf = gpd.GeoDataFrame(
                geometry=group["buffers"],
                index=group.index,
                crs="EPSG:4326",
            ).to_crs(src.crs)
            proportions = _zonal_percentages(buffer_gdf.geometry, raster_path)

        for idx, prop_dict in zip(group.index, proportions):
            if not prop_dict:
                continue

            land_classes = {k: v for k, v in prop_dict.items() if k != "shannon_index"}
            if land_classes:
                result.at[idx, top_class_col] = max(land_classes.items(), key=lambda item: item[1])[0]
                result.at[idx, ag_col] = sum(prop_dict.get(cls, 0.0) for cls in AGRICULTURAL_CLASSES)
                result.at[idx, natural_col] = sum(prop_dict.get(cls, 0.0) for cls in NATURAL_CLASSES)
            result.at[idx, shannon_col] = prop_dict.get("shannon_index")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Add NLCD land-use metadata to a sample table.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV with coordinates and year.")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path.")
    parser.add_argument(
        "--radius",
        type=float,
        default=1000.0,
        help="Buffer radius in meters for zonal statistics (default: 1000).",
    )
    parser.add_argument(
        "--nlcd-dir",
        default=str(DEFAULT_NLCD_DIR),
        help="Directory containing Annual_NLCD_LndCov_<year>_CU_C1V0.tif files.",
    )
    parser.add_argument(
        "--nlcd-template",
        default="Annual_NLCD_LndCov_{year}_CU_C1V0.tif",
        help="Filename template relative to --nlcd-dir.",
    )
    parser.add_argument("--year-col", default="year", help="Name of the year column.")
    parser.add_argument("--lon-col", default=None, help="Override longitude column name.")
    parser.add_argument("--lat-col", default=None, help="Override latitude column name.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    enriched = add_land_use_metadata(
        df,
        radius_m=args.radius,
        nlcd_dir=args.nlcd_dir,
        nlcd_template=args.nlcd_template,
        year_col=args.year_col,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
    )
    enriched.to_csv(args.output, index=False)
    print(f"Wrote land-use metadata to: {args.output}")


if __name__ == "__main__":
    main()
