# wxalerts-polar-reader

A [rio-tiler](https://cogeotiff.github.io/rio-tiler/) `BaseReader` implementation
that reads **nexrad-proc-v4 `schema_version=2` polar Zarr v3** volumes and serves
Web Mercator tiles to [TiTiler](https://developmentseed.org/titiler/).

## Purpose

`PolarRadarReader` is the library equivalent of `scripts/spot_check_tile.py` in
`wxalerts-nexrad-lut-gen`. It packages the proven per-tile data path — LUT lookup
→ fancy-index into polar.zarr → physical-unit decode → `ImageData` — into the
rio-tiler interface so it can be wired into a TiTiler service (Stage 3).

## Install

```bash
# From git (no PyPI release)
pip install git+https://github.com/wxalerts/wxalerts-polar-reader.git@main

# Pinned tag
pip install git+https://github.com/wxalerts/wxalerts-polar-reader.git@v0.1.0

# Dev install
pip install -e ".[dev]"
```

## Quick start

```python
from polar_reader import PolarRadarReader
from polar_reader.lut import LutCache

cache = LutCache("/opt/nexrad-luts")
reader = PolarRadarReader(
    input="s3://nexrad-data/KMOB/202605171728/polar.zarr",
    lut_cache=cache,
    s3_endpoint_url="http://minio.tailnet:9000",
    s3_key="...",
    s3_secret="...",
)

image = reader.tile(10, 20, 12, product="reflectivity", tilt="base")
# image.data  → (1, 256, 256) float32 dBZ
# image.mask  → (256, 256) uint8   (255 = valid, 0 = no data)
# image.bounds → Mercator bbox
```

### Available products

| Key | Units | Description |
|-----|-------|-------------|
| `reflectivity` | dBZ | Equivalent reflectivity factor |
| `velocity` | m s⁻¹ | Doppler radial velocity |
| `spectrum_width` | m s⁻¹ | Doppler spectrum width |
| `differential_reflectivity` | dB | ZDR |
| `cross_correlation_ratio` | — | RHO HV |
| `differential_phase` | degrees | PHI DP |

### Tilt selector

```python
reader.tile(..., tilt="base")   # lowest elevation (default)
reader.tile(..., tilt=1.5)      # nearest by elevation angle (degrees)
reader.tile(..., tilt=2)        # tilt_02 by index
```

### Info / metadata

```python
info = reader.info()
# {
#   "site_id": "KMOB",
#   "scan_time_utc": "2026-05-17T17:28:00+00:00",
#   "available_tilts": [{"name": "tilt_00", "elevation_deg": 0.5}, ...],
#   "available_products": ["reflectivity", "velocity", ...],
#   ...
# }
```

## Schema version requirement

This reader requires `schema_version=2` polar.zarr (nexrad-proc-v4's current
format with per-array CF-convention attrs: `scale_factor`, `add_offset`,
`_FillValue`). Older volumes raise `SchemaVersionMismatch`.

## LUT layout

LUTs are `.npz` files produced by `wxalerts-nexrad-lut-gen`:

```
{lut_dir}/
└── sites/
    └── KMOB/
        ├── z10.npz
        ├── z11.npz
        ├── z12.npz
        ├── z13.npz
        └── z14.npz
```

Each `.npz` contains:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `tile_index` | `(N, 2)` | int32 | `(x, y)` for each tile |
| `range_idx` | `(N, 256, 256)` | uint16 | Range gate index per pixel |
| `azimuth_idx` | `(N, 256, 256)` | uint16 | Azimuth ray index per pixel |
| `mask` | `(N, 256, 256)` | uint8 | 1 = in coverage, 0 = outside |

## Performance

- **Warm cache (LUT already in OS page cache):** target < 50 ms per tile.
- **Cold cache (first access):** < 500 ms (dominated by LUT load and zarr open).
- LUT files are opened with `mmap_mode='r'`; the OS page cache handles eviction.
- The full tilt array is loaded into RAM per `tile()` call (~1–5 MB). For high
  concurrency, consider pre-loading tilts at worker startup.

## Exceptions

| Exception | Raised when |
|-----------|-------------|
| `SchemaVersionMismatch` | polar.zarr is not schema_version 2 |
| `TileOutsideCoverage` | tile has no in-coverage pixels |
| `TiltNotFound` | tilt selector doesn't match any tilt group |
| `ProductNotAvailable` | product array missing from the tilt group |
| `LutNotFound` | no `.npz` for the requested site / zoom |
