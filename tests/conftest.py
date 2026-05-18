from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import zarr

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_zarr_path(tmp_path_factory):
    """A small but valid schema_version=2 polar.zarr."""
    path = tmp_path_factory.mktemp("polar") / "sample.zarr"

    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "schema_version": 2,
            "site_id": "KTEST",
            "site_lat": 30.6794,
            "site_lon": -88.2397,
            "site_alt_m": 63.0,
            "scan_time_utc": datetime(
                2026, 5, 17, 17, 0, 0, tzinfo=timezone.utc
            ).isoformat(),
            "vcp": 215,
            "total_tilts": 2,
        }
    )

    n_rays, n_gates = 360, 500
    for tilt_idx, elev in enumerate([0.5, 1.5]):
        tilt = root.create_group(f"tilt_{tilt_idx:02d}")
        tilt.attrs.update(
            {
                "elevation_deg": elev,
                "n_rays": n_rays,
                "n_gates": n_gates,
                "azimuth_first_deg": 0.0,
                "range_step_m": 250.0,
                "range_first_m": 2125.0,
                "scan_time_utc": datetime(
                    2026, 5, 17, 17, 0, 0, tzinfo=timezone.utc
                ).isoformat(),
                "vcp": 215,
                "sweep_index": tilt_idx,
            }
        )

        # Synthetic reflectivity: ring of echoes at gates 100-200
        # raw byte 130 → 130 * 0.5 + (-33) = 32 dBZ
        ref = np.zeros((n_rays, n_gates), dtype=np.uint8)
        ref[:, 100:200] = 130
        ref_arr = tilt.create_array(
            "reflectivity",
            shape=ref.shape,
            dtype=ref.dtype,
            chunks=(90, 460),
        )
        ref_arr[:] = ref
        ref_arr.attrs.update(
            {
                "scale_factor": 0.5,
                "add_offset": -33.0,
                "_FillValue": 0,
                "units": "dBZ",
                "long_name": "Equivalent reflectivity factor",
            }
        )

        vel = np.zeros((n_rays, n_gates), dtype=np.int16)
        vel_arr = tilt.create_array(
            "velocity",
            shape=vel.shape,
            dtype=vel.dtype,
            chunks=(90, 460),
        )
        vel_arr[:] = vel
        vel_arr.attrs.update(
            {
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "_FillValue": -32768,
                "units": "m s-1",
            }
        )

    return str(path)


@pytest.fixture
def sample_lut_dir(tmp_path):
    """A minimal LUT directory: site KTEST, zoom 10, two tiles."""
    site_dir = tmp_path / "sites" / "KTEST"
    site_dir.mkdir(parents=True)

    n_tiles = 2
    tile_index = np.array([[10, 20], [11, 21]], dtype=np.int32)

    # Both tiles index into the echo ring at gates 100-200; azimuth 0
    range_idx = np.full((n_tiles, 256, 256), 150, dtype=np.uint16)
    azimuth_idx = np.full((n_tiles, 256, 256), 0, dtype=np.uint16)
    mask = np.ones((n_tiles, 256, 256), dtype=np.uint8)

    np.savez(
        site_dir / "z10.npz",
        tile_index=tile_index,
        range_idx=range_idx,
        azimuth_idx=azimuth_idx,
        mask=mask,
    )

    return str(tmp_path)
