#######
#Kreiner Lab - November 11th 2026
#ClimateNA Pipeline
#Primary contributor(s): Sanju Patel
#
#Objective: Pull environmental data for samples with varying degrees of location
#uncertainty. ClimateNA doesn't use cells so pulling data at 800m points is 
#an arbitray decision.
#
#Methodology: For each sample, draw a circle based on the sample location
#uncertainty. Within the uncertainty circle, create a lattice with lines spaced
#800m apart. For each variable, collect data from every 800m x 800m square and 
#report the mean and standard deviation of the collected data (up to 100 squares
#per point). If the uncertainty is less than 800m, just report the value at the 
#point.
#
#
#Data requirements:
#   Inputs: Waterhemp samples with a latitude, longitude, elevation, collection
#           year, collection month and an uncertainty radius. Elevation comes 
#           from elevation_pipeline.py; points without one are skipped.
#   Package: ClimateNAr 3.2.0, from https://climatena.ca/downloads/ClimateNAr.zip
#           (187 MB), unzipped into an R library. It carries a native R
#           implementation of ClimateNA plus the 1901-2025 historical data, so
#           it runs locally with no executable, key or rate limit. It imports
#           terra, readr and data.table, and verifies its version against
#           climatena.ca on each call, so the first call of a session needs the
#           network.
#   Coverage: North America, 1901-2025. Samples outside that window, or without
#           a collection month, are left as NA rather than filled from the
#           nearest year.
#######

library(ClimateNAr)

M_PER_DEG <- 111320
CNA_YEAR_MIN <- 1901
CNA_YEAR_MAX <- 2025
CNA_NODATA <- -9999            # what ClimateNA returns outside its coverage

# Uncertainty at or below this is read as a single point, matching the PRISM
# pipeline's treatment of a sample that sits inside one 800 m cell.
POINT_RADIUS_M <- 800

# The circle is sampled on a lattice of this spacing, thinned to at most
# MAX_POINTS, which a 10 km circle would otherwise overshoot by two orders of
# magnitude for no gain in the mean.
POINT_SPACING_M <- 800
MAX_POINTS <- 100

# PRISM variable -> the ClimateNA monthly family holding the same quantity.
# ClimateNA has no dewpoint or vapour pressure deficit; it reports relative
# humidity, from which the last three are derived below.
VARIABLE_MAP <- c(ppt = "PPT", tmin = "Tmin", tmax = "Tmax",
                  tmean = "Tave", rh = "RH")
DERIVED_VARIABLES <- c("tdmean", "vpdmin", "vpdmax")
ALL_VARIABLES <- c(names(VARIABLE_MAP), DERIVED_VARIABLES)

# All twelve months of each family, which is what gets asked of ClimateNA: one
# call then serves every sample in the year whatever month it was collected in.
CNA_VARLIST <- as.vector(outer(VARIABLE_MAP, sprintf("%02d", 1:12), paste0))

PREFIX <- "cna_"


# --------------------------------------------------------------------------
# Sampling circle
# --------------------------------------------------------------------------
circle_points <- function(lat, lon, radius_m,
                          spacing_m = POINT_SPACING_M,
                          max_points = MAX_POINTS) {
  # Points to read for one sample: the coordinate itself when its uncertainty
  # is sub-cell, otherwise a lattice clipped to the uncertainty circle. The
  # lattice is centred on the coordinate, so the point itself is always in it.
  #
  # Thinning takes every nth lattice point rather than a random subset, so a
  # rerun reads the same points and the circle stays evenly covered.
  if (!is.finite(lat) || !is.finite(lon)) return(NULL)
  if (!is.finite(radius_m) || radius_m <= POINT_RADIUS_M) {
    return(data.frame(lat = lat, lon = lon))
  }

  offsets <- seq(-floor(radius_m / spacing_m), floor(radius_m / spacing_m)) * spacing_m
  grid <- expand.grid(dy = offsets, dx = offsets)
  grid <- grid[(grid$dx^2 + grid$dy^2) <= radius_m^2, , drop = FALSE]
  if (nrow(grid) == 0) return(data.frame(lat = lat, lon = lon))
  if (nrow(grid) > max_points) {
    grid <- grid[round(seq(1, nrow(grid), length.out = max_points)), , drop = FALSE]
  }

  # A degree of longitude narrows by cos(latitude); at these radii the
  # flat-earth approximation is well inside the 4 km ClimateNA baseline grid.
  m_per_deg_lon <- M_PER_DEG * cos(lat * pi / 180)
  data.frame(lat = lat + grid$dy / M_PER_DEG,
             lon = lon + grid$dx / max(m_per_deg_lon, 1e-6))
}


# --------------------------------------------------------------------------
# Climate
# --------------------------------------------------------------------------
fetch_climatena <- function(pts, year, varlist = CNA_VARLIST) {
  # Monthly ClimateNA variables for a data frame of lat/lon/elev, for one
  # historical year. Returns one row per input point, in input order.
  #
  # ClimateNAr wants exactly five columns, in the order ID1, ID2, lat, long,
  # el, and reads the last three positionally. Handed a data frame it returns
  # the input with the requested variables bound on, and writes nothing, so
  # outDir only matters for its CSV and raster modes.
  input <- data.frame(ID1 = seq_len(nrow(pts)), ID2 = 1,
                      lat = pts$lat, long = pts$lon, el = pts$elev)

  # It rejects a single-row input (its validity check requires more than one
  # row), so a lone point is padded and the padding dropped again.
  padded <- nrow(input) == 1
  if (padded) input <- rbind(input, input)

  out <- try(ClimateNAr(inputFile = input,
                        periodList = sprintf("Year_%d.ann", year),
                        varList = varlist,
                        outDir = tempdir()), silent = TRUE)
  if (inherits(out, "try-error")) {
    message("  ", year, ": ClimateNAr failed -- ", conditionMessage(attr(out, "condition")))
    return(NULL)
  }

  out <- as.data.frame(out)
  if (padded) out <- out[1, , drop = FALSE]
  out
}


# --------------------------------------------------------------------------
# Derived humidity variables
# --------------------------------------------------------------------------
saturation_vp <- function(t_c) {
  # Saturation vapour pressure in hPa (Tetens), the PRISM vpd unit.
  6.108 * exp(17.27 * t_c / (t_c + 237.3))
}

derive_variables <- function(values) {
  # tdmean, vpdmin and vpdmax from ClimateNA's temperature and humidity, since
  # it reports neither dewpoint nor vpd.
  #
  # These are approximations and are not interchangeable with the PRISM columns
  # of the same name: PRISM averages daily values, while these come from
  # monthly mean relative humidity and monthly mean daily min/max temperature,
  # so both vpd extremes are damped.
  rh <- pmin(pmax(values$rh, 1), 100)
  gamma <- log(rh / 100) + 17.27 * values$tmean / (237.3 + values$tmean)

  values$tdmean <- 237.3 * gamma / (17.27 - gamma)
  values$vpdmin <- pmax(saturation_vp(values$tmin) * (1 - rh / 100), 0)
  values$vpdmax <- pmax(saturation_vp(values$tmax) * (1 - rh / 100), 0)
  values
}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
collect_climatena_data <- function(df,
                                   year_col = "year",
                                   month_col = "month",
                                   lat_col = "decimalLatitude",
                                   lon_col = "decimalLongitude",
                                   uncertainty_col = "coordinateUncertaintyInMeters",
                                   elevation_col = "elevation",
                                   max_points = MAX_POINTS,
                                   checkpoint_dir = "climatena_checkpoints") {
  # Match ClimateNA monthly variables to samples by location and collection
  # month, reading each over its uncertainty circle. Returns df with the cna_
  # columns joined on.
  #
  # Loops years on the outside, so each year's grid is opened once and every
  # point of that year goes in a single ClimateNAr call. Finished years are
  # checkpointed and an interrupted run resumes. Samples collected outside
  # 1901-2025, or with no collection month, are left as NA.
  if (!elevation_col %in% names(df)) {
    stop("no '", elevation_col, "' column -- run elevation_pipeline.py first")
  }
  lat <- suppressWarnings(as.numeric(df[[lat_col]]))
  lon <- suppressWarnings(as.numeric(df[[lon_col]]))
  year <- suppressWarnings(as.numeric(df[[year_col]]))
  month <- suppressWarnings(as.numeric(df[[month_col]]))
  radius <- suppressWarnings(as.numeric(df[[uncertainty_col]]))
  elevation <- suppressWarnings(as.numeric(df[[elevation_col]]))

  has_input <- !is.na(lat) & !is.na(lon) & !is.na(year) & !is.na(month)
  in_range <- !is.na(year) & year >= CNA_YEAR_MIN & year <= CNA_YEAR_MAX
  usable <- has_input & in_range

  value_cols <- paste0(PREFIX, c(rbind(ALL_VARIABLES, paste0(ALL_VARIABLES, "_sd"))))
  method_col <- paste0(PREFIX, "method")
  out_cols <- c(value_cols, paste0(PREFIX, c("n_points", "method", "radius_m")))

  dir.create(checkpoint_dir, showWarnings = FALSE, recursive = TRUE)
  frames <- list()

  for (yr in sort(unique(year[usable]))) {
    checkpoint <- file.path(checkpoint_dir, sprintf("%d.csv", yr))
    if (file.exists(checkpoint)) {
      frames[[length(frames) + 1]] <- read.csv(checkpoint, stringsAsFactors = FALSE)
      next
    }

    group <- which(usable & year == yr)
    message(sprintf("%d: %d samples", yr, length(group)))

    # Every point of every sample in this year, so the year is one pass.
    per_sample <- lapply(group, function(i) {
      circle_points(lat[i], lon[i], radius[i], max_points = max_points)
    })
    sizes <- vapply(per_sample, function(p) if (is.null(p)) 0L else nrow(p), integer(1))
    pts <- do.call(rbind, per_sample)
    pts$owner <- rep(group, sizes)
    pts$elev <- rep(elevation[group], sizes)

    # ClimateNA adjusts by elevation and has nothing to adjust by without one,
    # so a point the DEM could not place is dropped rather than sent at an
    # invented height.
    pts <- pts[!is.na(pts$elev), , drop = FALSE]
    if (nrow(pts) == 0) next

    climate <- fetch_climatena(pts, yr)
    if (is.null(climate)) {
      message("  ", yr, ": no data, leaving the year for the next run")
      next
    }

    result <- data.frame(row = group, stringsAsFactors = FALSE)
    for (v in out_cols) result[[v]] <- NA_real_
    result[[method_col]] <- NA_character_

    for (k in seq_along(group)) {
      i <- group[k]
      mine <- which(pts$owner == i)
      if (length(mine) == 0) next

      # The month's column out of each monthly family, nodata to NA.
      values <- as.data.frame(lapply(VARIABLE_MAP, function(family) {
        v <- suppressWarnings(as.numeric(climate[[sprintf("%s%02d", family, month[i])]][mine]))
        v[v <= CNA_NODATA] <- NA
        v
      }))
      names(values) <- names(VARIABLE_MAP)
      values <- derive_variables(values)

      keep <- rowSums(!is.na(values[, names(VARIABLE_MAP), drop = FALSE])) > 0
      if (!any(keep)) next
      values <- values[keep, , drop = FALSE]

      for (v in ALL_VARIABLES) {
        result[k, paste0(PREFIX, v)] <- mean(values[[v]], na.rm = TRUE)
        result[k, paste0(PREFIX, v, "_sd")] <-
          if (nrow(values) > 1) sd(values[[v]], na.rm = TRUE) else NA_real_
      }
      result[k, paste0(PREFIX, "n_points")] <- nrow(values)
      result[k, method_col] <- if (nrow(values) > 1) "area" else "point"
      result[k, paste0(PREFIX, "radius_m")] <- radius[i]
    }

    write.csv(result, checkpoint, row.names = FALSE)
    frames[[length(frames) + 1]] <- result
  }

  for (v in out_cols) {
    df[[v]] <- if (v == method_col) NA_character_ else NA_real_
  }
  if (length(frames) > 0) {
    joined <- do.call(rbind, frames)
    for (v in out_cols) df[joined$row, v] <- joined[[v]]
  }

  report_coverage(df, has_input, in_range)
  df
}


report_coverage <- function(df, has_input, in_range) {
  # Print how many samples ended up without climate data, split by cause, in
  # the same terms as the PRISM pipeline's report.
  present <- paste0(PREFIX, ALL_VARIABLES)
  no_values <- rowSums(!is.na(df[, present, drop = FALSE])) == 0
  partial <- rowSums(is.na(df[, present, drop = FALSE])) > 0 & !no_values

  no_date <- !has_input
  outside <- has_input & !in_range
  no_cna <- no_values & has_input & in_range

  cat("\n--- ClimateNA coverage ---\n")
  cat(sprintf("total samples:                        %d\n", nrow(df)))
  cat(sprintf("no data (any reason):                 %d\n", sum(no_values)))
  cat(sprintf("  - missing coordinate/year/month:    %d\n", sum(no_date)))
  cat(sprintf("  - collected outside %d-%d:      %d\n",
              CNA_YEAR_MIN, CNA_YEAR_MAX, sum(outside)))
  cat(sprintf("  - no ClimateNA value at location:   %d\n", sum(no_cna)))
  cat(sprintf("partial data (some variables only):   %d\n", sum(partial)))
  cat(sprintf("complete data:                        %d\n",
              nrow(df) - sum(no_values) - sum(partial)))

  n <- df[[paste0(PREFIX, "n_points")]]
  if (any(!is.na(n))) {
    cat(sprintf("points per sample:                    median %.0f, max %.0f   (%d read as a single point)\n",
                median(n, na.rm = TRUE), max(n, na.rm = TRUE),
                sum(n == 1, na.rm = TRUE)))
  }
}


# sys.nframe() is 0 only when this file is run, not when it is sourced, so the
# functions above can be loaded from another script the way the Python
# pipelines are imported.
if (sys.nframe() == 0) {
  # check.names = FALSE: the herbicide columns carry spaces and brackets in
  # their mode-of-action names, and R would otherwise rewrite every one of them.
  df <- read.csv("herbarium_samples_elevation.csv", check.names = FALSE,
                 stringsAsFactors = FALSE)
  df <- df[, !startsWith(names(df), PREFIX), drop = FALSE]

  result <- collect_climatena_data(df)
  write.csv(result, "herbarium_samples_climateNA.csv", row.names = FALSE)
}
