"""Tests for polar_reader.projection (pure geometry, no zarr required)."""

import mercantile
import numpy as np
import pytest

from polar_reader import ON_THE_FLY_ZOOM_THRESHOLD
from polar_reader.projection import tile_pixel_lonlat, pixel_to_polar

# KMOB — Mobile, AL  (NWS/ROC coordinates)
KMOB_LAT = 30.6794
KMOB_LON = -88.2397
KMOB_FIRST_GATE_M = 2125.0
KMOB_GATE_SPACING_M = 250.0
KMOB_MAX_RANGE_M = 230_000.0
KMOB_N_AZ = 720

# Representative z=15 tile near KMOB
_KMOB_TILE = mercantile.tile(KMOB_LON, KMOB_LAT, 15)


def test_tile_pixel_lonlat_shape():
    lon_arr, lat_arr = tile_pixel_lonlat(_KMOB_TILE)
    assert lon_arr.shape == (256, 256)
    assert lat_arr.shape == (256, 256)


def test_tile_pixel_lonlat_within_bounds():
    lon_arr, lat_arr = tile_pixel_lonlat(_KMOB_TILE)
    bounds = mercantile.bounds(_KMOB_TILE)
    tol = 1e-6
    assert float(lon_arr.min()) >= bounds.west - tol
    assert float(lon_arr.max()) <= bounds.east + tol
    assert float(lat_arr.min()) >= bounds.south - tol
    assert float(lat_arr.max()) <= bounds.north + tol


def test_tile_pixel_lonlat_dtype():
    lon_arr, lat_arr = tile_pixel_lonlat(_KMOB_TILE)
    assert lon_arr.dtype == np.float32
    assert lat_arr.dtype == np.float32


def test_pixel_to_polar_north_azimuth():
    """Due-north point at 50 km should have azimuth_idx == 0 and mask == 1."""
    north_lat = KMOB_LAT + 50_000.0 / 111_320.0
    pixel_lat = np.array([[north_lat]], dtype=np.float32)
    pixel_lon = np.array([[KMOB_LON]], dtype=np.float32)
    range_idx, azimuth_idx, mask = pixel_to_polar(
        KMOB_LAT, KMOB_LON,
        KMOB_FIRST_GATE_M, KMOB_GATE_SPACING_M, KMOB_MAX_RANGE_M, KMOB_N_AZ,
        pixel_lat, pixel_lon,
    )
    assert int(mask[0, 0]) == 1
    # Due north → bearing 0° → azimuth_idx 0 (or wraps to n_azimuths-1)
    assert int(azimuth_idx[0, 0]) in (0, KMOB_N_AZ - 1)


def test_pixel_to_polar_at_site_is_masked():
    """Pixel at the radar site itself (range=0) must be masked out."""
    pixel_lat = np.array([[KMOB_LAT]], dtype=np.float32)
    pixel_lon = np.array([[KMOB_LON]], dtype=np.float32)
    range_idx, azimuth_idx, mask = pixel_to_polar(
        KMOB_LAT, KMOB_LON,
        KMOB_FIRST_GATE_M, KMOB_GATE_SPACING_M, KMOB_MAX_RANGE_M, KMOB_N_AZ,
        pixel_lat, pixel_lon,
    )
    assert int(mask[0, 0]) == 0


def test_pixel_to_polar_beyond_range_is_masked():
    """Point 300 km away (> 230 km max range) must be masked."""
    far_lat = KMOB_LAT + 300_000.0 / 111_320.0
    pixel_lat = np.array([[far_lat]], dtype=np.float32)
    pixel_lon = np.array([[KMOB_LON]], dtype=np.float32)
    range_idx, azimuth_idx, mask = pixel_to_polar(
        KMOB_LAT, KMOB_LON,
        KMOB_FIRST_GATE_M, KMOB_GATE_SPACING_M, KMOB_MAX_RANGE_M, KMOB_N_AZ,
        pixel_lat, pixel_lon,
    )
    assert int(mask[0, 0]) == 0


def test_pixel_to_polar_range_idx_dtype():
    pixel_lat = np.array([[KMOB_LAT + 0.5]], dtype=np.float32)
    pixel_lon = np.array([[KMOB_LON]], dtype=np.float32)
    range_idx, azimuth_idx, mask = pixel_to_polar(
        KMOB_LAT, KMOB_LON,
        KMOB_FIRST_GATE_M, KMOB_GATE_SPACING_M, KMOB_MAX_RANGE_M, KMOB_N_AZ,
        pixel_lat, pixel_lon,
    )
    assert range_idx.dtype == np.uint16
    assert azimuth_idx.dtype == np.uint16
    assert mask.dtype == np.uint8


def test_pixel_to_polar_sentinels_are_zero_where_masked():
    """range_idx and azimuth_idx must be 0 wherever mask == 0."""
    # Mix of in-range and out-of-range pixels
    lats = np.array([[KMOB_LAT, KMOB_LAT + 300_000.0 / 111_320.0]], dtype=np.float32)
    lons = np.array([[KMOB_LON, KMOB_LON]], dtype=np.float32)
    range_idx, azimuth_idx, mask = pixel_to_polar(
        KMOB_LAT, KMOB_LON,
        KMOB_FIRST_GATE_M, KMOB_GATE_SPACING_M, KMOB_MAX_RANGE_M, KMOB_N_AZ,
        lats, lons,
    )
    masked = mask == 0
    assert (range_idx[masked] == 0).all()
    assert (azimuth_idx[masked] == 0).all()


def test_on_the_fly_threshold_exported():
    assert ON_THE_FLY_ZOOM_THRESHOLD == 15
