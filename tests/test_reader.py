import pytest
import numpy as np

from polar_reader.reader import PolarRadarReader
from polar_reader.lut import LutCache
from polar_reader.exceptions import (
    SchemaVersionMismatch,
    TileOutsideCoverage,
    ProductNotAvailable,
    TiltNotFound,
)


def test_reader_opens_and_reads_attrs(sample_zarr_path, sample_lut_dir):
    reader = PolarRadarReader(input=sample_zarr_path, lut_cache=LutCache(sample_lut_dir))
    assert reader.site_id == "KTEST"
    assert reader.site_lat == pytest.approx(30.6794)
    assert reader.site_lon == pytest.approx(-88.2397)
    assert reader.vcp == 215
    assert reader.total_tilts == 2
    assert reader.schema_version == 2


def test_reader_renders_in_coverage_tile(sample_zarr_path, sample_lut_dir):
    reader = PolarRadarReader(input=sample_zarr_path, lut_cache=LutCache(sample_lut_dir))
    image = reader.tile(10, 20, 10, product="reflectivity", tilt="base")

    assert image.data.shape == (1, 256, 256)
    assert image.mask.shape == (256, 256)

    # Gate 150 with raw=130: 130 * 0.5 + (-33) = 32.0 dBZ
    valid_pixels = image.data[0][image.mask > 0]
    assert valid_pixels.size > 0
    np.testing.assert_allclose(valid_pixels.mean(), 32.0, atol=0.1)


def test_reader_raises_on_out_of_coverage_tile(sample_zarr_path, sample_lut_dir):
    reader = PolarRadarReader(input=sample_zarr_path, lut_cache=LutCache(sample_lut_dir))
    with pytest.raises(TileOutsideCoverage):
        reader.tile(9999, 9999, 10)


def test_reader_resolves_tilt_base(sample_zarr_path, sample_lut_dir):
    reader = PolarRadarReader(input=sample_zarr_path, lut_cache=LutCache(sample_lut_dir))
    assert reader._resolve_tilt("base") == "tilt_00"


def test_reader_resolves_tilt_by_elevation(sample_zarr_path, sample_lut_dir):
    reader = PolarRadarReader(input=sample_zarr_path, lut_cache=LutCache(sample_lut_dir))
    assert reader._resolve_tilt(1.5) == "tilt_01"
    assert reader._resolve_tilt(0.7) == "tilt_00"  # closer to 0.5 than 1.5


def test_reader_resolves_tilt_by_index(sample_zarr_path, sample_lut_dir):
    reader = PolarRadarReader(input=sample_zarr_path, lut_cache=LutCache(sample_lut_dir))
    assert reader._resolve_tilt(0) == "tilt_00"
    assert reader._resolve_tilt(1) == "tilt_01"


def test_reader_raises_on_unknown_product(sample_zarr_path, sample_lut_dir):
    reader = PolarRadarReader(input=sample_zarr_path, lut_cache=LutCache(sample_lut_dir))
    with pytest.raises(ProductNotAvailable):
        reader.tile(10, 20, 10, product="differential_reflectivity")


def test_reader_info_returns_complete_dict(sample_zarr_path, sample_lut_dir):
    reader = PolarRadarReader(input=sample_zarr_path, lut_cache=LutCache(sample_lut_dir))
    info = reader.info()
    assert info["site_id"] == "KTEST"
    assert info["schema_version"] == 2
    assert len(info["available_tilts"]) == 2
    assert "reflectivity" in info["available_products"]
    assert "nws_reflectivity" in info["available_colormaps"]


def test_reader_schema_version_mismatch(tmp_path):
    import zarr as _zarr

    bad_zarr = str(tmp_path / "bad.zarr")
    root = _zarr.open_group(bad_zarr, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "schema_version": 1,
            "site_id": "KBAD",
            "site_lat": 30.0,
            "site_lon": -88.0,
            "site_alt_m": 0.0,
            "scan_time_utc": "2026-01-01T00:00:00+00:00",
            "vcp": 11,
            "total_tilts": 1,
        }
    )
    from polar_reader.lut import LutCache as LC

    with pytest.raises(SchemaVersionMismatch):
        PolarRadarReader(
            input=bad_zarr,
            lut_cache=LC(str(tmp_path)),
        )


def test_reader_bounds_are_wgs84(sample_zarr_path, sample_lut_dir):
    reader = PolarRadarReader(input=sample_zarr_path, lut_cache=LutCache(sample_lut_dir))
    w, s, e, n = reader.bounds
    assert w < reader.site_lon < e
    assert s < reader.site_lat < n


def test_reader_velocity_product(sample_zarr_path, sample_lut_dir):
    reader = PolarRadarReader(input=sample_zarr_path, lut_cache=LutCache(sample_lut_dir))
    image = reader.tile(10, 20, 10, product="velocity", tilt="base")
    assert image.data.shape == (1, 256, 256)
    assert image.band_names == ["velocity"]
