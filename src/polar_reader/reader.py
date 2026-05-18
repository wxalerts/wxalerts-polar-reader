import math
import logging
from datetime import datetime
from typing import Union

import attrs
import mercantile
import numpy as np
import zarr
import zarr.storage
import s3fs
from rasterio.crs import CRS
from rio_tiler.io.base import BaseReader
from rio_tiler.models import ImageData
from rio_tiler.errors import TileOutsideBounds

from polar_reader._config import (
    SUPPORTED_SCHEMA_VERSIONS,
    COVERAGE_RADIUS_KM,
    DEFAULT_MINZOOM,
    DEFAULT_MAXZOOM,
)
from polar_reader import projection
from polar_reader.colormaps import COLORMAPS
from polar_reader.constants import ON_THE_FLY_ZOOM_THRESHOLD
from polar_reader.exceptions import (
    ProductNotAvailable,
    SchemaVersionMismatch,
    SiteNotFoundError,
    TileOutsideCoverage,
    TiltNotFound,
)
from polar_reader.lut import LutCache
from polar_reader.products import PRODUCTS, Product, get_product
from polar_reader.site_registry import SITE_GEOMETRY

log = logging.getLogger(__name__)


@attrs.define
class PolarRadarReader(BaseReader):
    """rio-tiler reader for nexrad-proc-v4 polar Zarr v3 volumes.

    Reads polar.zarr from MinIO/S3, looks up per-tile coordinates from
    pre-generated LUTs on local disk, fancy-indexes into the radar data,
    and returns physical-unit ImageData (no colormap applied — TiTiler does that).

    Parameters
    ----------
    input:
        s3:// URI (or local path) to the polar.zarr. Schema_version 2 required.
    lut_cache:
        LUT cache. Caller constructs this with the right lut_dir.
    s3_endpoint_url:
        S3/MinIO endpoint URL. None uses fsspec defaults (AWS).
    s3_key, s3_secret:
        S3 credentials.
    """

    input: str = attrs.field(kw_only=True)
    lut_cache: LutCache = attrs.field(kw_only=True)
    s3_endpoint_url: str | None = attrs.field(default=None, kw_only=True)
    s3_key: str | None = attrs.field(default=None, repr=False, kw_only=True)
    s3_secret: str | None = attrs.field(default=None, repr=False, kw_only=True)

    # Populated by __attrs_post_init__
    site_id: str = attrs.field(init=False)
    site_lat: float = attrs.field(init=False)
    site_lon: float = attrs.field(init=False)
    site_alt_m: float = attrs.field(init=False)
    scan_time_utc: datetime = attrs.field(init=False)
    vcp: int = attrs.field(init=False)
    total_tilts: int = attrs.field(init=False)
    schema_version: int = attrs.field(init=False)

    # rio-tiler interface fields
    bounds: tuple[float, float, float, float] = attrs.field(init=False)
    crs: CRS = attrs.field(init=False)
    minzoom: int = attrs.field(init=False, default=DEFAULT_MINZOOM)
    maxzoom: int = attrs.field(init=False, default=DEFAULT_MAXZOOM)

    _zarr_root: zarr.Group = attrs.field(init=False, repr=False)
    _tilt_elevations: dict[str, float] = attrs.field(
        init=False, factory=dict, repr=False
    )

    def __attrs_post_init__(self) -> None:
        """Open polar.zarr and read root attrs. Does NOT read array data."""
        self._zarr_root = self._open_zarr()
        root_attrs = dict(self._zarr_root.attrs)

        self.schema_version = int(root_attrs.get("schema_version", 0))
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionMismatch(
                f"polar.zarr schema_version {self.schema_version} not supported. "
                f"Supported: {SUPPORTED_SCHEMA_VERSIONS}"
            )

        self.site_id = root_attrs["site_id"]
        self.site_lat = float(root_attrs["site_lat"])
        self.site_lon = float(root_attrs["site_lon"])
        self.site_alt_m = float(root_attrs["site_alt_m"])
        self.scan_time_utc = datetime.fromisoformat(root_attrs["scan_time_utc"])
        self.vcp = int(root_attrs["vcp"])
        self.total_tilts = int(root_attrs["total_tilts"])

        dlat = COVERAGE_RADIUS_KM / 111.0
        dlon = COVERAGE_RADIUS_KM / (111.0 * math.cos(math.radians(self.site_lat)))
        self.bounds = (
            self.site_lon - dlon,
            self.site_lat - dlat,
            self.site_lon + dlon,
            self.site_lat + dlat,
        )
        self.crs = CRS.from_epsg(4326)

        for tilt_name in self._zarr_root.group_keys():
            if tilt_name.startswith("tilt_"):
                tilt_attrs = dict(self._zarr_root[tilt_name].attrs)
                self._tilt_elevations[tilt_name] = float(
                    tilt_attrs.get("elevation_deg", float("nan"))
                )

    def _open_zarr(self) -> zarr.Group:
        """Open the polar.zarr. Uses s3fs + FsspecStore for s3:// URIs."""
        if self.input.startswith("s3://"):
            fs = s3fs.S3FileSystem(
                anon=False,
                key=self.s3_key,
                secret=self.s3_secret,
                endpoint_url=self.s3_endpoint_url,
                use_ssl=(self.s3_endpoint_url or "").startswith("https"),
            )
            store = zarr.storage.FsspecStore(
                fs=fs, path=self.input.removeprefix("s3://")
            )
            return zarr.open_group(store, mode="r")
        else:
            return zarr.open_group(self.input, mode="r")

    def _resolve_tilt(self, tilt: Union[str, float, int]) -> str:
        """Map a tilt selector to a tilt_NN group name.

        - "base"  → tilt with minimum elevation_deg
        - float   → tilt with closest elevation_deg
        - int < total_tilts → tilt_<NN>; otherwise nearest by elevation
        """
        if not self._tilt_elevations:
            raise TiltNotFound("No tilt_NN groups found in zarr")

        if tilt == "base":
            return min(self._tilt_elevations, key=self._tilt_elevations.get)  # type: ignore[arg-type]

        if isinstance(tilt, int) and tilt < self.total_tilts:
            name = f"tilt_{tilt:02d}"
            if name in self._tilt_elevations:
                return name

        if isinstance(tilt, (int, float)):
            target = float(tilt)
            return min(
                self._tilt_elevations,
                key=lambda k: abs(self._tilt_elevations[k] - target),
            )

        raise TiltNotFound(f"Cannot parse tilt selector: {tilt!r}")

    @staticmethod
    def _read_zarr_product_attrs(
        tilt_group: zarr.Group, product_obj: Product, tilt_name: str
    ) -> tuple[zarr.Array, float, float, int]:
        """Validate product availability and read scale/offset attrs."""
        if product_obj.zarr_key not in tilt_group:
            available = list(tilt_group.array_keys())
            raise ProductNotAvailable(
                f"Product '{product_obj.name}' not in {tilt_name}. Available: {available}"
            )
        product_arr: zarr.Array = tilt_group[product_obj.zarr_key]  # type: ignore[assignment]
        prod_attrs = dict(product_arr.attrs)
        scale_factor = float(prod_attrs.get("scale_factor", product_obj.scale))  # type: ignore[arg-type]
        add_offset = float(prod_attrs.get("add_offset", product_obj.offset))  # type: ignore[arg-type]
        fill_value = int(prod_attrs.get("_FillValue", product_obj.fill_value))  # type: ignore[arg-type]
        return product_arr, scale_factor, add_offset, fill_value

    @staticmethod
    def _decode_values(
        raw: np.ndarray,
        coverage_mask: np.ndarray,
        product_obj: Product,
        scale_factor: float,
        add_offset: float,
        fill_value: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply NEXRAD ICD sentinel logic and scale/offset to raw array.

        Returns (physical float32, output_mask uint8) where output_mask
        follows the rio-tiler convention: 255 = valid, 0 = no data.
        """
        if product_obj.dtype == "uint8":
            sentinel_mask = (raw == fill_value) | (raw == 255)
        else:
            sentinel_mask = raw == fill_value
        valid_data = (~sentinel_mask) & (coverage_mask > 0)
        physical = raw.astype(np.float32) * scale_factor + add_offset
        output_mask = (valid_data * 255).astype(np.uint8)
        return physical, output_mask

    def _render_tile_lut(
        self,
        tile: mercantile.Tile,
        product: str,
        tilt_group: zarr.Group,
        tilt_name: str,
    ) -> ImageData:
        """Render a tile using the pre-generated LUT (z <= 14 path)."""
        lut_entry = self.lut_cache.lookup_tile(self.site_id, tile.z, tile.x, tile.y)
        if lut_entry is None or lut_entry.mask.sum() == 0:
            raise TileOutsideCoverage(
                f"Tile z{tile.z}/{tile.x}/{tile.y} not in coverage for {self.site_id}"
            )

        product_obj = get_product(product)
        product_arr, scale_factor, add_offset, fill_value = self._read_zarr_product_attrs(
            tilt_group, product_obj, tilt_name
        )

        # Load the full array and fancy-index — cheaper than per-chunk zarr reads
        full_array = np.array(product_arr)
        raw = full_array[lut_entry.azimuth_idx, lut_entry.range_idx]

        physical, output_mask = self._decode_values(
            raw, lut_entry.mask, product_obj, scale_factor, add_offset, fill_value
        )

        bbox = mercantile.xy_bounds(tile.x, tile.y, tile.z)
        return ImageData(
            np.expand_dims(physical, 0),
            output_mask,
            assets=[self.input],
            crs=CRS.from_epsg(3857),
            bounds=(bbox.left, bbox.bottom, bbox.right, bbox.top),
            band_names=[product],
            metadata={
                "site_id": self.site_id,
                "scan_time_utc": self.scan_time_utc.isoformat(),
                "vcp": self.vcp,
                "tilt": tilt_name,
                "elevation_deg": self._tilt_elevations[tilt_name],
                "product": product,
                "units": product_obj.units,
            },
        )

    def _render_tile_on_the_fly(
        self,
        tile: mercantile.Tile,
        product: str,
        tilt_group: zarr.Group,
        tilt_name: str,
    ) -> ImageData:
        """Render a tile via on-the-fly polar projection (z >= 15 path)."""
        site_geom = SITE_GEOMETRY.get(self.site_id)
        if site_geom is None:
            raise SiteNotFoundError(
                f"Site '{self.site_id}' not in site registry; cannot render on-the-fly"
            )

        product_obj = get_product(product)
        product_arr, scale_factor, add_offset, fill_value = self._read_zarr_product_attrs(
            tilt_group, product_obj, tilt_name
        )

        lon_arr, lat_arr = projection.tile_pixel_lonlat(tile)
        range_idx, azimuth_idx, mask = projection.pixel_to_polar(
            site_geom.lat,
            site_geom.lon,
            site_geom.first_range_gate_m,
            site_geom.range_gate_spacing_m,
            site_geom.max_range_m,
            site_geom.n_azimuths,
            lat_arr,
            lon_arr,
        )

        if mask.sum() == 0:
            raise TileOutsideCoverage(
                f"Tile z{tile.z}/{tile.x}/{tile.y} not in coverage for {self.site_id}"
            )

        full_array = np.array(product_arr)
        raw = full_array[azimuth_idx, range_idx]

        physical, output_mask = self._decode_values(
            raw, mask, product_obj, scale_factor, add_offset, fill_value
        )

        bbox = mercantile.xy_bounds(tile.x, tile.y, tile.z)
        return ImageData(
            np.expand_dims(physical, 0),
            output_mask,
            assets=[self.input],
            crs=CRS.from_epsg(3857),
            bounds=(bbox.left, bbox.bottom, bbox.right, bbox.top),
            band_names=[product],
            metadata={
                "site_id": self.site_id,
                "scan_time_utc": self.scan_time_utc.isoformat(),
                "vcp": self.vcp,
                "tilt": tilt_name,
                "elevation_deg": self._tilt_elevations[tilt_name],
                "product": product,
                "units": product_obj.units,
            },
        )

    def tile(
        self,
        tile_x: int,
        tile_y: int,
        tile_z: int,
        *,
        product: str = "reflectivity",
        tilt: Union[str, float, int] = "base",
        colormap: str | None = None,
        **kwargs,
    ) -> ImageData:
        """Render a single 256×256 Web Mercator tile.

        Returns ImageData with:
        - data: (1, 256, 256) float32 in physical units (dBZ, m/s, etc.)
        - mask: (256, 256) uint8; 255 = valid, 0 = no data
        - bounds: Mercator bbox
        - crs: EPSG:3857

        Zoom dispatch:
        - z < ON_THE_FLY_ZOOM_THRESHOLD  → LUT-backed path
        - z >= ON_THE_FLY_ZOOM_THRESHOLD → on-the-fly polar projection

        The colormap parameter is accepted for API compatibility but is NOT
        applied here — rio-tiler / TiTiler apply it at the endpoint layer.
        """
        if tile_z < self.minzoom or tile_z > self.maxzoom:
            raise TileOutsideBounds(
                f"Zoom {tile_z} outside reader range [{self.minzoom}, {self.maxzoom}]"
            )

        tilt_name = self._resolve_tilt(tilt)
        tilt_group: zarr.Group = self._zarr_root[tilt_name]  # type: ignore[assignment]
        t = mercantile.Tile(tile_x, tile_y, tile_z)

        if tile_z >= ON_THE_FLY_ZOOM_THRESHOLD:
            return self._render_tile_on_the_fly(t, product, tilt_group, tilt_name)
        else:
            return self._render_tile_lut(t, product, tilt_group, tilt_name)

    def info(self) -> dict:  # type: ignore[override]
        """Metadata for the /info endpoint."""
        return {
            "site_id": self.site_id,
            "site_lat": self.site_lat,
            "site_lon": self.site_lon,
            "site_alt_m": self.site_alt_m,
            "scan_time_utc": self.scan_time_utc.isoformat(),
            "vcp": self.vcp,
            "total_tilts": self.total_tilts,
            "schema_version": self.schema_version,
            "bounds": self.bounds,
            "available_tilts": [
                {"name": k, "elevation_deg": v}
                for k, v in sorted(
                    self._tilt_elevations.items(), key=lambda kv: kv[1]
                )
            ],
            "available_products": list(PRODUCTS.keys()),
            "available_colormaps": list(COLORMAPS.keys()),
            "minzoom": self.minzoom,
            "maxzoom": self.maxzoom,
        }

    def statistics(self, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "Statistics not supported for radar volumes. Use info() instead."
        )

    def preview(self, **kwargs) -> ImageData:  # type: ignore[override]
        """Lowest-zoom tile containing the radar site."""
        radar_tile = mercantile.tile(self.site_lon, self.site_lat, self.minzoom)
        return self.tile(radar_tile.x, radar_tile.y, self.minzoom)

    def point(self, lon: float, lat: float, **kwargs):  # type: ignore[override]
        raise NotImplementedError("Point sampling not implemented yet.")

    def part(self, bbox, **kwargs) -> ImageData:  # type: ignore[override]
        raise NotImplementedError(
            "Arbitrary bbox rendering not supported. Use tile() for Mercator tiles."
        )

    def feature(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "Feature-based rendering not supported. Use tile() for Mercator tiles."
        )
