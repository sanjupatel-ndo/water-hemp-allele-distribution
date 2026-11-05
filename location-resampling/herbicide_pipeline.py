"""
Kreiner Lab - October 29th 2026
Herbicide use pipeline (USGS EPest)
Primary contributor(s): Sanju Patel

Objective: Attach county-level herbicide application estimates to herbarium
samples, by site of action group, for the year each sample was collected.

Methodology: Assign each sample to the US county containing its coordinate,
then join the USGS EPest county-year estimates for that county. Compound
estimates are summed into mode-of-action groups, which are the unit resistance
alleles map onto. Each group is reported both as raw kg applied and as kg per
hectare of county cropland; only the latter is comparable between samples,
since raw kg tracks how much cropland a county happens to have.

Groups are the full mode-of-action labels used in
herbarium_herbicide_groups_metadata.csv, not bare HRAC numbers, so the two
tables name the same things. That keeps chemistry the numbers would merge
apart: the three ACCase subgroups, the six PS II Group 5 subclasses, the five
VLCFA subclasses, and glyphosate separate from the other EPSPS compounds.

EPest estimates agricultural use only, allocated to counties from surveyed crop
acreage. A zero therefore means no estimated use on crops, not that nothing was
applied: roadside, railway and urban applications are outside its scope, which
matters for a weed that grows in exactly those places.

Samples outside EPest's 1992-2018 window are not simply missing. A compound
cannot have been applied before it existed, so a sample collected before a
group's first US registration is a true zero rather than a gap. Those rows are
written as 0.0 and flagged; samples between a group's registration and 1992,
and samples after 2018, are left NaN. The registration years below are
deliberately rounded down, so a zero is only ever claimed when it is safe. A
label inherits the first year of the HRAC number it names, which stays
conservative when a label is narrower than its group.

Sample status is summarized in hb_status, and the per-group zero-versus-NaN
distinction is what separates "known absent" from "unknown".

Data requirements:
    Inputs: Sample table with a latitude, longitude, and collection year.
    Grids:  County boundaries (auto-downloaded, cached) and, for the cropland
            denominator, the HILDA+ states netCDF used by land_use_pipeline.py.
    Network: EPest year files are downloaded once and cached under epest_cache/.
"""

from __future__ import annotations   # anaconda env here is Python 3.9

import math
import re
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from land_use_pipeline import (
    CELL_DEG,
    HILDA_STATES,
    HILDA_YEAR_MIN,
    LAT_TOP,
    LON_LEFT,
    M_PER_DEG,
    cell_index,
)

# Compound to mode of action, and the spelling variants EPest uses for the same
# active ingredient. Carried over from add_herbicide_metadata.py, which cannot
# be imported here because it depends on a metadata_common module that is not
# part of this repo. Herbicides only: anything absent is dropped, which is what
# removes the insecticides and fungicides EPest also publishes.
EPEST_HERBICIDE_MOA = {
    "BENSULFURON": "ALS inhibitor (Group 2)",
    "CHLORIMURON": "ALS inhibitor (Group 2)",
    "CHLORSULFURON": "ALS inhibitor (Group 2)",
    "CLORANSULAM-METHYL": "ALS inhibitor (Group 2)",
    "DICLOSULAM": "ALS inhibitor (Group 2)",
    "FLAZASULFURON": "ALS inhibitor (Group 2)",
    "FLORASULAM": "ALS inhibitor (Group 2)",
    "FLUCARBAZONE": "ALS inhibitor (Group 2)",
    "FORAMSULFURON": "ALS inhibitor (Group 2)",
    "HALOSULFURON": "ALS inhibitor (Group 2)",
    "IMAZAMOX": "ALS inhibitor (Group 2)",
    "IMAZAPIC": "ALS inhibitor (Group 2)",
    "IMAZAPYR": "ALS inhibitor (Group 2)",
    "IMAZAQUIN": "ALS inhibitor (Group 2)",
    "IMAZETHAPYR": "ALS inhibitor (Group 2)",
    "IMAZOSULFURON": "ALS inhibitor (Group 2)",
    "IODOSULFURON": "ALS inhibitor (Group 2)",
    "MESOSULFURON": "ALS inhibitor (Group 2)",
    "METSULFURON": "ALS inhibitor (Group 2)",
    "NICOSULFURON": "ALS inhibitor (Group 2)",
    "ORTHOSULFAMURON": "ALS inhibitor (Group 2)",
    "PENOXSULAM": "ALS inhibitor (Group 2)",
    "PRIMISULFURON": "ALS inhibitor (Group 2)",
    "PROPOXYCARBAZONE": "ALS inhibitor (Group 2)",
    "PROSULFURON": "ALS inhibitor (Group 2)",
    "PYRITHIOBAC-SODIUM": "ALS inhibitor (Group 2)",
    "PYROXSULAM": "ALS inhibitor (Group 2)",
    "RIMSULFURON": "ALS inhibitor (Group 2)",
    "SULFOSULFURON": "ALS inhibitor (Group 2)",
    "THIENCARBAZONE-METHYL": "ALS inhibitor (Group 2)",
    "THIFENSULFURON": "ALS inhibitor (Group 2)",
    "TRIASULFURON": "ALS inhibitor (Group 2)",
    "TRIBENURON": "ALS inhibitor (Group 2)",
    "TRIFLOXYSULFURON": "ALS inhibitor (Group 2)",
    "TRIFLUSULFURON": "ALS inhibitor (Group 2)",
    "GLYPHOSATE": "GLYPHOSATE",
    "SULFOSATE": "EPSPS Inhibitor (Group 9)",
    "ACIFLUORFEN": "PPO inhibitor (Group 14)",
    "CARFENTRAZONE-ETHYL": "PPO inhibitor (Group 14)",
    "FLUMICLORAC": "PPO inhibitor (Group 14)",
    "FLUMIOXAZIN": "PPO inhibitor (Group 14)",
    "FLUTHIACET-METHYL": "PPO inhibitor (Group 14)",
    "FOMESAFEN": "PPO inhibitor (Group 14)",
    "LACTOFEN": "PPO inhibitor (Group 14)",
    "OXADIAZON": "PPO inhibitor (Group 14)",
    "OXYFLUORFEN": "PPO inhibitor (Group 14)",
    "PYRAFLUFEN": "PPO inhibitor (Group 14)",
    "SAFLUFENACIL": "PPO inhibitor (Group 14)",
    "SULFENTRAZONE": "PPO inhibitor (Group 14)",
    "2,4-D": "Synthetic Auxin (Group 4)",
    "2,4-DB": "Synthetic Auxin (Group 4)",
    "AMINOPYRALID": "Synthetic Auxin (Group 4)",
    "CHLORAMBEN": "Synthetic Auxin (Group 4)",
    "CLOPYRALID": "Synthetic Auxin (Group 4)",
    "DICAMBA": "Synthetic Auxin (Group 4)",
    "DICHLORPROP": "Synthetic Auxin (Group 4)",
    "FLORPYRAUXIFEN": "Synthetic Auxin (Group 4)",
    "FLUROXYPYR": "Synthetic Auxin (Group 4)",
    "HALAUXIFEN": "Synthetic Auxin (Group 4)",
    "MCPA": "Synthetic Auxin (Group 4)",
    "MCPB": "Synthetic Auxin (Group 4)",
    "MECOPROP": "Synthetic Auxin (Group 4)",
    "PICLORAM": "Synthetic Auxin (Group 4)",
    "QUINCLORAC": "Synthetic Auxin (Group 4)",
    "TRICLOPYR": "Synthetic Auxin (Group 4)",
    "ATRAZINE": "PS II Inhibitor (Group 5) - Triazine",
    "PROMETRYN": "PS II Inhibitor (Group 5) - Triazine",
    "PROPAZINE": "PS II Inhibitor (Group 5) - Triazine",
    "SIMAZINE": "PS II Inhibitor (Group 5) - Triazine",
    "METRIBUZIN": "PS II Inhibitor (Group 5) - Triazinone",
    "HEXAZINONE": "PS II Inhibitor (Group 5) - Triazinone",
    "BROMACIL": "PS II Inhibitor (Group 5) - Uracil",
    "TERBACIL": "PS II Inhibitor (Group 5) - Uracil",
    "DIURON": "PS II Inhibitor (Group 5) - Urea",
    "FLUOMETURON": "PS II Inhibitor (Group 5) - Urea",
    "LINURON": "PS II Inhibitor (Group 5) - Urea",
    "TEBUTHIURON": "PS II Inhibitor (Group 5) - Urea",
    "DESMEDIPHAM": "PS II Inhibitor (Group 5) - Phenylcarbamate",
    "PHENMEDIPHAM": "PS II Inhibitor (Group 5) - Phenylcarbamate",
    "PROPANIL": "PS II Inhibitor (Group 5) - Anilide",
    "BENTAZONE": "PS II Inhibitor (Group 6) - Benzothiadiazinone",
    "BROMOXYNIL": "PS II Inhibitor (Group 6/24) - Nitrile (dual mode)",
    "DICHLOBENIL": "PS II Inhibitor (Group 6) - Nitrile",
    "CLETHODIM": "ACCase inhibitor (Group 1) - DIM",
    "CLODINAFOP": "ACCase inhibitor (Group 1) - FOP",
    "CYHALOFOP": "ACCase inhibitor (Group 1) - FOP",
    "FLUAZIFOP": "ACCase inhibitor (Group 1) - FOP",
    "PINOXADEN": "ACCase inhibitor (Group 1) - DEN",
    "QUIZALOFOP": "ACCase inhibitor (Group 1) - FOP",
    "SETHOXYDIM": "ACCase inhibitor (Group 1) - DIM",
    "TRALKOXYDIM": "ACCase inhibitor (Group 1) - DIM",
    "ACETOCHLOR": "VLCFA Inhibitor (Group 15) - Chloroacetamide",
    "ALACHLOR": "VLCFA Inhibitor (Group 15) - Chloroacetamide",
    "DIMETHENAMID": "VLCFA Inhibitor (Group 15) - Chloroacetamide",
    "DIMETHENAMID-P": "VLCFA Inhibitor (Group 15) - Chloroacetamide",
    "EPTC": "VLCFA Inhibitor (Group 15) - Thiocarbamate",
    "FLUFENACET": "VLCFA Inhibitor (Group 15) - Oxyacetamide",
    "METOLACHLOR": "VLCFA Inhibitor (Group 15) - Chloroacetamide",
    "METOLACHLOR-S": "VLCFA Inhibitor (Group 15) - Chloroacetamide",
    "NAPROPAMIDE": "VLCFA Inhibitor (Group 15) - Acetamide",
    "PYROXASULFONE": "VLCFA Inhibitor (Group 15) - Isoxazoline",
    "THIOBENCARB": "VLCFA Inhibitor (Group 15) - Thiocarbamate",
    "TRI-ALLATE": "VLCFA Inhibitor (Group 15) - Thiocarbamate",
    "BICYCLOPYRONE": "HPPD inhibitor (Group 27)",
    "ISOXAFLUTOLE": "HPPD inhibitor (Group 27)",
    "MESOTRIONE": "HPPD inhibitor (Group 27)",
    "PYRASULFOTOLE": "HPPD inhibitor (Group 27)",
    "TEMBOTRIONE": "HPPD inhibitor (Group 27)",
    "TOPRAMEZONE": "HPPD inhibitor (Group 27)",
    "BENFLURALIN": "Microtubule Assembly Inhibitor (Group 3) - Dinitroaniline",
    "BUTRALIN": "Microtubule Assembly Inhibitor (Group 3) - Dinitroaniline",
    "DITHIOPYR": "Microtubule Assembly Inhibitor (Group 3) - Pyridine",
    "ETHALFLURALIN": "Microtubule Assembly Inhibitor (Group 3) - Dinitroaniline",
    "ORYZALIN": "Microtubule Assembly Inhibitor (Group 3) - Dinitroaniline",
    "PENDIMETHALIN": "Microtubule Assembly Inhibitor (Group 3) - Dinitroaniline",
    "PRODIAMINE": "Microtubule Assembly Inhibitor (Group 3) - Dinitroaniline",
    "PROPYZAMIDE": "Microtubule Assembly Inhibitor (Group 3) - Benzamide",
    "TRIFLURALIN": "Microtubule Assembly Inhibitor (Group 3) - Dinitroaniline",
    "CHLORPROPHAM": "Mitosis Inhibitor (Group 18) - Carbamate",
    "CLOMAZONE": "Carotenoid Biosynthesis Inhibitor (Group 13)",
    "DCPA": "Mitosis Inhibitor (Group 3) - Benzoic acid",
    "DIFLUFENZOPYR": "Auxin Transport Inhibitor (Group 19)",
    "DINOSEB": "Uncoupler (Group 24) - Dinitrophenol",
    "DIQUAT": "PS I Electron Diverter (Group 22) - Bipyridilium",
    "ETHOFUMESATE": "Lipid synthesis inhibitor (Group 8)",
    "FLURIDONE": "Carotenoid Biosynthesis Inhibitor (Group 12)",
    "GLUFOSINATE": "Glutamine Synthetase Inhibitor (Group 10)",
    "INDAZIFLAM": "Cellulose Biosynthesis Inhibitor (Group 29)",
    "ISOXABEN": "Cellulose Biosynthesis Inhibitor (Group 21)",
    "MSMA": "Unknown MOA (Group 17) - Organoarsenic",
    "NORFLURAZON": "Carotenoid Biosynthesis Inhibitor (Group 12)",
    "PARAQUAT": "PS I Electron Diverter (Group 22) - Bipyridilium",
    "PELARGONIC": "Cell membrane disruptor (Group 26)",
}

ALIASES = {
    "S-METOLACHLOR": "METOLACHLOR-S",
    "TRIALLATE": "TRI-ALLATE",
    "GLUFOSINATE-AMMONIUM": "GLUFOSINATE",
    "PYRAFLUFEN-ETHYL": "PYRAFLUFEN",
    "HALAUXIFEN-METHYL": "HALAUXIFEN",
    "FLORPYRAUXIFEN-BENZYL": "FLORPYRAUXIFEN",
    "BROMOXYNIL OCTANOATE": "BROMOXYNIL",
    "DICHLORPROP-P": "DICHLORPROP",
}

EPEST_BASE = ("https://water.usgs.gov/nawqa/pnsp/usage/maps/county-level")

# EPest is published under three different layouts, not the one path the older
# add_herbicide_metadata.py assumed. 2013-2014 sit in a different directory,
# 2015-2018 are zipped, and 2017-2018 exclude California. There is no 2019
# release, so the window ends at 2018.
EPEST_MIN_YEAR, EPEST_MAX_YEAR = 1992, 2018
EPEST_LAYOUTS = {
    **{y: f"PesticideUseEstimates/EPest.county.estimates.{y}.txt"
       for y in range(1992, 2013)},
    **{y: f"PreliminaryEstimates/EPest.county.estimates.{y}.txt"
       for y in (2013, 2014)},
    **{y: f"PreliminaryEstimates/{y}PreliminaryEstimates.zip"
       for y in (2015, 2016)},
    **{y: f"PreliminaryEstimateNOCa/{y}PreliminaryEstimatesNoCA.zip"
       for y in (2017, 2018)},
}

COUNTY_URL = ("https://www2.census.gov/geo/tiger/GENZ2020/shp/"
              "cb_2020_us_county_500k.zip")

# First US registration per HRAC group, rounded DOWN to the earliest compound
# in the group so that a "pre-introduction" zero is never claimed too late.
# These are literature values and should be checked against the lab's own
# reference before the zeros are used in an analysis.
GROUP_FIRST_YEAR = {
    1: 1978,    # ACCase, diclofop
    2: 1982,    # ALS, chlorsulfuron
    3: 1958,    # microtubule assembly, DCPA
    4: 1945,    # synthetic auxin, 2,4-D
    5: 1952,    # PS II, the substituted ureas
    6: 1960,    # PS II nitriles / benzothiadiazinone
    8: 1968,    # lipid synthesis, ethofumesate
    9: 1974,    # EPSPS, glyphosate
    10: 1993,   # glutamine synthetase, glufosinate
    12: 1974,   # carotenoid, norflurazon
    13: 1986,   # carotenoid, clomazone
    14: 1980,   # PPO, acifluorfen
    15: 1957,   # VLCFA, EPTC
    17: 1960,   # organoarsenic, MSMA
    18: 1951,   # mitosis, chlorpropham
    19: 1999,   # auxin transport, diflufenzopyr
    21: 1988,   # cellulose, isoxaben
    22: 1958,   # PS I diverter, diquat
    24: 1945,   # uncoupler, dinoseb
    26: 1992,   # cell membrane, pelargonic acid
    27: 1996,   # HPPD, isoxaflutole
    29: 2010,   # cellulose, indaziflam
}

# Labels whose HRAC number cannot be read off the label text.
LABEL_GROUP_OVERRIDE = {"GLYPHOSATE": (9,)}

# EPest reports a low and a high estimate; they differ where survey coverage is
# thin. "high" matches the older script's default.
VALUE_KINDS = ("high", "low", "mean")

HILDA_CROPLAND_CODE = 22


def group_numbers(moa: str) -> tuple:
    """
    HRAC group number(s) named in a mode-of-action label.

    "GLYPHOSATE" carries no number in the MOA table inherited from
    add_herbicide_metadata.py, so its group is supplied here. A dual-mode label
    such as "(Group 6/24)" yields both numbers.
    """
    if moa in LABEL_GROUP_OVERRIDE:
        return LABEL_GROUP_OVERRIDE[moa]
    return tuple(int(n) for n in re.findall(r"\d+", str(moa)))


def moa_first_year(moa: str) -> Optional[int]:
    """
    Earliest US registration among the HRAC groups a label names, or None when
    the label names no known group.

    Taking the minimum keeps the pre-introduction zeros safe twice over: a
    dual-mode label gets the earlier of its two groups, and a label narrower
    than its group (say the DEN subgroup of ACCase) inherits the whole group's
    first year rather than its own later one. Both err toward claiming fewer
    zeros. Refining these per subclass would let more pre-1992 samples be
    called, but needs a registration date per subclass that this table does
    not have.
    """
    years = [GROUP_FIRST_YEAR[n] for n in group_numbers(moa) if n in GROUP_FIRST_YEAR]
    return min(years) if years else None


def epest_path(year: int, cache_dir: Path) -> Path:
    """Download one EPest year into the cache if absent, return the txt path."""
    if year not in EPEST_LAYOUTS:
        raise ValueError(f"no EPest release for {year}")

    remote = EPEST_LAYOUTS[year]
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / Path(remote).name

    if not local.exists():
        urlretrieve(f"{EPEST_BASE}/{remote}", local)

    if local.suffix != ".zip":
        return local

    # The 2016 archive's inner folder is misspelled "2106PreliminaryEstimates",
    # so the member is located by pattern rather than by an assumed path.
    extracted = cache_dir / f"EPest.county.estimates.{year}.txt"
    if not extracted.exists():
        with zipfile.ZipFile(local) as archive:
            member = next(
                name for name in archive.namelist()
                if name.endswith(f"{year}.txt")
            )
            extracted.write_bytes(archive.read(member))
    return extracted


def load_epest_year(
    year: int,
    moa_by_compound: dict,
    aliases: dict,
    value_kind: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """
    One EPest year as tidy rows of (fips, moa, kg), summed within MOA label.

    Compounds with no mode of action in moa_by_compound are dropped; that table
    covers herbicides only, so this is what excludes insecticides and fungicides.
    """
    table = pd.read_csv(
        epest_path(year, cache_dir),
        sep="\t",
        usecols=["COMPOUND", "STATE_FIPS_CODE", "COUNTY_FIPS_CODE",
                 "EPEST_LOW_KG", "EPEST_HIGH_KG"],
        dtype=str,
    )

    compound = table["COMPOUND"].str.upper().str.strip()
    compound = compound.map(lambda c: aliases.get(c, aliases.get(c.split(" ")[0], c)))
    moa = compound.map(moa_by_compound)

    low = pd.to_numeric(table["EPEST_LOW_KG"], errors="coerce")
    high = pd.to_numeric(table["EPEST_HIGH_KG"], errors="coerce")
    kg = {"low": low, "high": high, "mean": (low + high) / 2.0}[value_kind]

    out = pd.DataFrame({
        "fips": (table["STATE_FIPS_CODE"].str.zfill(2)
                 + table["COUNTY_FIPS_CODE"].str.zfill(3)),
        "moa": moa,
        "kg": kg,
    })
    out = out[out["moa"].notna()]

    return (out.groupby(["fips", "moa"], as_index=False)["kg"]
            .sum(min_count=1))


def county_boundaries(cache_dir: Path) -> gpd.GeoDataFrame:
    """County polygons in EPSG:4326, downloaded into the cache if absent."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / Path(COUNTY_URL).name
    if not local.exists():
        urlretrieve(COUNTY_URL, local)

    counties = gpd.read_file(f"zip://{local}")
    return (counties.rename(columns={"GEOID": "fips"})[["fips", "geometry"]]
            .to_crs(4326))


def assign_fips(
    df: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
) -> pd.Series:
    """
    FIPS of the county containing each sample, NA where there is none.

    Rows without a finite coordinate are dropped before the join rather than
    passed through as a point: a single POINT(x NaN) corrupts the spatial index
    and silently returns no match for unrelated, perfectly good rows.
    """
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    finite = np.isfinite(lat) & np.isfinite(lon)

    points = gpd.GeoDataFrame(
        df.loc[finite, []],
        geometry=gpd.points_from_xy(lon[finite], lat[finite]),
        crs=4326,
    )
    joined = gpd.sjoin(points, counties, how="left", predicate="within")
    joined = joined[~joined.index.duplicated()]

    fips = pd.Series(pd.NA, index=df.index, dtype="object")
    fips.loc[joined.index] = joined["fips"]
    return fips


def county_cropland_ha(
    wanted: set,
    counties: gpd.GeoDataFrame,
    states_path: Path,
    cache_file: Path,
) -> pd.Series:
    """
    Cropland hectares per (fips, year), counted from the HILDA+ grid.

    Uses the same grid the land use pipeline reads, so the denominator is
    year-matched to the sample rather than taken from a modern cropland census.
    Cell area shrinks with latitude, so each row's area is computed separately.
    Results are cached, since this is the slow step.
    """
    cached = pd.Series(dtype=float)
    if cache_file.exists():
        prior = pd.read_csv(cache_file, dtype={"fips": str})
        cached = prior.set_index(["fips", "year"])["cropland_ha"]

    missing = sorted(wanted - set(cached.index))
    if not missing:
        return cached

    geometry = counties.set_index("fips")["geometry"]
    rows = []
    with h5py.File(states_path, "r") as handle:
        grid = handle["LULC_states"]
        for fips, year in tqdm(missing, desc="county cropland", unit="cty-yr"):
            if fips not in geometry.index:
                continue
            poly = geometry.loc[fips]
            lon0, lat0, lon1, lat1 = poly.bounds

            r_top, _ = cell_index(lat1, 0.0)
            r_bot, _ = cell_index(lat0, 0.0)
            _, c_left = cell_index(0.0, lon0)
            _, c_right = cell_index(0.0, lon1)
            window = grid[int(year) - HILDA_YEAR_MIN,
                          r_top:r_bot + 1, c_left:c_right + 1]

            crop_rows, crop_cols = np.nonzero(window == HILDA_CROPLAND_CODE)
            if crop_rows.size == 0:
                rows.append({"fips": fips, "year": year, "cropland_ha": 0.0})
                continue

            cell_lat = LAT_TOP - (r_top + crop_rows + 0.5) * CELL_DEG
            cell_lon = LON_LEFT + (c_left + crop_cols + 0.5) * CELL_DEG
            inside = gpd.points_from_xy(cell_lon, cell_lat).within(poly)

            ns = CELL_DEG * M_PER_DEG
            ew = CELL_DEG * M_PER_DEG * np.cos(np.radians(cell_lat[inside]))
            rows.append({"fips": fips, "year": year,
                         "cropland_ha": float((ns * ew).sum() / 10_000.0)})

    fresh = pd.DataFrame(rows)
    if not fresh.empty:
        combined = pd.concat([cached.reset_index(), fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset=["fips", "year"], keep="last")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_file, index=False)
        cached = combined.set_index(["fips", "year"])["cropland_ha"]

    return cached


def add_herbicide_metadata(
    df: pd.DataFrame,
    moa_by_compound: dict = EPEST_HERBICIDE_MOA,
    aliases: dict = ALIASES,
    states_path: str | Path = HILDA_STATES,
    year_col: str = "year",
    lat_col: str = "decimalLatitude",
    lon_col: str = "decimalLongitude",
    value_kind: str = "high",
    cache_dir: str | Path = "epest_cache",
    with_cropland: bool = True,
) -> pd.DataFrame:
    """
    Join EPest county-year herbicide estimates to a sample table by mode of
    action.

    Returns a copy of df with, per group, "hb_kg_<label>" and (where the
    cropland denominator is available) "hb_kgha_<label>", plus hb_fips,
    hb_status and hb_n_groups_pre_introduction. <label> is the mode-of-action
    name exactly as herbarium_herbicide_groups_metadata.csv spells it. A 0.0
    means the group did not yet exist when the sample was collected; NaN means
    its use is unknown for that sample.
    """
    if value_kind not in VALUE_KINDS:
        raise ValueError(f"value_kind must be one of {VALUE_KINDS}")

    cache_dir = Path(cache_dir)
    work = df.copy()
    year = pd.to_numeric(work[year_col], errors="coerce")

    counties = county_boundaries(cache_dir)
    fips = assign_fips(work, counties, lat_col=lat_col, lon_col=lon_col)

    in_window = year.between(EPEST_MIN_YEAR, EPEST_MAX_YEAR)
    joinable = in_window & fips.notna()

    tall = pd.concat(
        [load_epest_year(int(yr), moa_by_compound, aliases, value_kind, cache_dir)
         .assign(year=int(yr))
         for yr in tqdm(sorted(year[joinable].unique()), desc="EPest years",
                        unit="yr")],
        ignore_index=True,
    ) if joinable.any() else pd.DataFrame(columns=["fips", "moa", "kg", "year"])

    # Every label the MOA table knows, whether or not it turned up in the years
    # actually read, so the column set does not shift with the sample table.
    labels = sorted(set(tall["moa"]) | set(moa_by_compound.values()))
    kg_cols = {m: f"hb_kg_{m}" for m in labels}

    wide = (tall.pivot_table(index=["year", "fips"], columns="moa",
                             values="kg", aggfunc="sum")
            if not tall.empty else pd.DataFrame())

    # Rows that joined get their county's value, with an absent group meaning
    # zero kg applied there that year rather than missing.
    for m in labels:
        work[kg_cols[m]] = np.nan
    if not wide.empty:
        keys = pd.MultiIndex.from_arrays([year[joinable], fips[joinable]])
        matched = wide.reindex(keys)
        for m in labels:
            if m in wide.columns:
                work.loc[joinable, kg_cols[m]] = matched[m].fillna(0.0).values
            else:
                work.loc[joinable, kg_cols[m]] = 0.0

    # Outside the window, a group that did not yet exist is a true zero. This
    # applies regardless of whether the sample is in the US.
    pre_intro = pd.DataFrame(False, index=work.index, columns=labels)
    for m in labels:
        first = moa_first_year(m)
        if first is None:
            continue
        before = year.notna() & (year < first) & ~joinable
        work.loc[before, kg_cols[m]] = 0.0
        pre_intro[m] = before

    cropland = pd.Series(dtype=float)
    if with_cropland and joinable.any():
        wanted = set(zip(fips[joinable], year[joinable].astype(int)))
        cropland = county_cropland_ha(
            wanted, counties, Path(states_path), cache_dir / "county_cropland.csv")

    work["hb_fips"] = fips
    work["hb_cropland_ha"] = [
        float(cropland.get((f, int(y)), np.nan))
        if pd.notna(f) and pd.notna(y) and not cropland.empty else np.nan
        for f, y in zip(fips, year)
    ]
    # A county with no HILDA cropland cells has no denominator, so intensity is
    # left NaN rather than becoming inf.
    for m in labels:
        with np.errstate(invalid="ignore", divide="ignore"):
            intensity = work[kg_cols[m]] / work["hb_cropland_ha"]
        work[f"hb_kgha_{m}"] = intensity.replace([np.inf, -np.inf], np.nan)

    work["hb_n_groups_pre_introduction"] = pre_intro.sum(axis=1)
    work["hb_status"] = _status(year, fips, joinable)

    _report_coverage(work, year, fips, labels)
    return work


def _status(year: pd.Series, fips: pd.Series, joinable: pd.Series) -> pd.Series:
    """One label per sample explaining where its herbicide values came from."""
    status = pd.Series("", index=year.index, dtype=object)
    status[year.isna()] = "no_year"
    status[fips.isna() & year.notna()] = "outside_us_counties"
    status[fips.notna() & year.notna() & (year < EPEST_MIN_YEAR)] = "pre_epest"
    status[fips.notna() & year.notna() & (year > EPEST_MAX_YEAR)] = "post_epest"
    status[joinable] = "epest"
    return status


def _report_coverage(
    df: pd.DataFrame,
    year: pd.Series,
    fips: pd.Series,
    labels: list,
) -> None:
    """Print where the herbicide values came from, and what is still unknown."""
    counts = df["hb_status"].value_counts()
    kg_cols = [f"hb_kg_{m}" for m in labels]
    measured = df["hb_status"] == "epest"
    zeros_only = ~measured & df[kg_cols].notna().any(axis=1)
    nothing = ~measured & ~zeros_only

    print("\n--- EPest coverage ---")
    print(f"total samples:                   {len(df)}")
    print(f"mode-of-action groups:           {len(labels)}")
    print(f"  measured use from EPest:       {int(measured.sum())}")
    print(f"  pre-introduction zeros only:   {int(zeros_only.sum())}")
    print(f"  no herbicide value:            {int(nothing.sum())}")
    print("sample status:")
    for label in ("epest", "pre_epest", "post_epest",
                  "outside_us_counties", "no_year"):
        print(f"  {label:30s} {int(counts.get(label, 0))}")
    if zeros_only.any():
        n = df.loc[zeros_only, "hb_n_groups_pre_introduction"]
        print(f"groups known absent, those rows:  median {n.median():.0f} "
              f"of {len(labels)}")
    print(f"counties matched:                {int(fips.nunique())}")
    if df["hb_cropland_ha"].notna().any():
        ha = df["hb_cropland_ha"]
        print(f"county cropland (ha):            median {ha.median():,.0f}, "
              f"zero in {int((ha == 0).sum())} samples")


if __name__ == "__main__":
    df = pd.read_csv("herbarium_samples_withHILDA.csv", low_memory=False)

    result = add_herbicide_metadata(df)
    result.to_csv("herbarium_samples_withHERBICIDE.csv", index=False)
    print("\nWrote herbarium_samples_withHERBICIDE.csv")

    ##Pull the new landuse, iron out herbicide, and pull climate NA.
    #also flag the samples with extrememe uncertainty errors
    ## disagreement between hlnid and hilda+ on cropland and other categories
