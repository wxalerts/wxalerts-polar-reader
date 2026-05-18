"""Self-contained polar projection for on-the-fly high-zoom tile rendering.

No pyproj dependency — pure numpy. Haversine + forward-bearing math is
identical to lutgen.geometry.pixel_to_polar (kept in sync manually).
"""

from __future__ import annotations

import math

import mercantile
import numpy as np

# WGS84 mean Earth radius — matches lutgen.geometry exactly.
EARTH_RADIUS_M: float = 6_371_008.8


def tile_pixel_lonlat(
    tile: mercantile.Tile, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) float32 arrays of shape (tile_size, tile_size).

    Pixel centres are computed via WGS84 linear interpolation within
    mercantile.bounds(tile). Accurate enough at z>=15 where the tile
    is <1.2 km wide.
    """
    bounds = mercantile.bounds(tile)
    cols = np.arange(tile_size, dtype=np.float32)
    rows = np.arange(tile_size, dtype=np.float32)

    lon = bounds.west + (cols + 0.5) / tile_size * (bounds.east - bounds.west)
    lat = bounds.north - (rows + 0.5) / tile_size * (bounds.north - bounds.south)

    lon_arr = np.tile(lon, (tile_size, 1))
    lat_arr = np.tile(lat.reshape(-1, 1), (1, tile_size))
    return lon_arr, lat_arr


def pixel_to_polar(
    site_lat: float,
    site_lon: float,
    first_range_gate_m: float,
    range_gate_spacing_m: float,
    max_range_m: float,
    n_azimuths: int,
    pixel_lat: np.ndarray,
    pixel_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map pixel lat/lon arrays to (range_idx, azimuth_idx, mask).

    Uses spherical Haversine + forward-bearing formulas identical to
    lutgen.geometry.pixel_to_polar. Operates in float32 throughout.
    Accepts arrays of any shape.

    Returns:
        range_idx:   uint16, index into the polar.zarr range axis
        azimuth_idx: uint16, index into the polar.zarr azimuth axis
        mask:        uint8,  1 where pixel is within radar coverage, else 0

    Sentinel: range_idx and azimuth_idx are 0 where mask == 0.
    """
    lat0 = math.radians(site_lat)
    lon0 = math.radians(site_lon)
    cos_lat0 = math.cos(lat0)
    sin_lat0 = math.sin(lat0)

    lat_r = np.radians(pixel_lat.astype(np.float32))
    lon_r = np.radians(pixel_lon.astype(np.float32))

    dt = lat_r.dtype
    _lat0 = dt.type(lat0)
    _cos_lat0 = dt.type(cos_lat0)
    _sin_lat0 = dt.type(sin_lat0)
    _2R = dt.type(2.0 * EARTH_RADIUS_M)

    dlat = lat_r - _lat0
    dlon = lon_r - dt.type(lon0)

    # Haversine great-circle distance
    a = np.sin(dlat * 0.5) ** 2 + _cos_lat0 * np.cos(lat_r) * np.sin(dlon * 0.5) ** 2
    range_m = _2R * np.arcsin(np.sqrt(a.clip(dt.type(0.0), dt.type(1.0))))

    # Forward bearing: 0° = North, clockwise (NEXRAD convention)
    x = np.cos(lat_r) * np.sin(dlon)
    y = _cos_lat0 * np.sin(lat_r) - _sin_lat0 * np.cos(lat_r) * np.cos(dlon)
    azimuth_deg = (np.degrees(np.arctan2(x, y)) + dt.type(360.0)) % dt.type(360.0)

    range_idx_f = (range_m - dt.type(first_range_gate_m)) / dt.type(range_gate_spacing_m)
    azimuth_idx_f = azimuth_deg / dt.type(360.0 / n_azimuths)
    max_range_idx = int(math.floor((max_range_m - first_range_gate_m) / range_gate_spacing_m))

    range_idx_i = np.floor(range_idx_f).astype(np.int32)
    azimuth_idx_i = np.floor(azimuth_idx_f).astype(np.int32)

    in_range = (
        (range_idx_i >= 0) & (range_idx_i < max_range_idx) & (range_m <= dt.type(max_range_m))
    )
    mask = in_range.astype(np.uint8)
    range_idx = np.where(in_range, range_idx_i, 0).astype(np.uint16)
    azimuth_idx = np.where(in_range, azimuth_idx_i % n_azimuths, 0).astype(np.uint16)

    return range_idx, azimuth_idx, mask
