#!/usr/bin/env python3
"""Regenerate the kmob_sample.zarr test fixture from scratch.

Run from the repo root:
    python tests/fixtures/make_fixtures.py

The fixture is a synthetic but structurally valid schema_version=2 polar.zarr
with 2 tilts, 360 rays, 500 gates. No real radar data is used.
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

FIXTURE_PATH = Path(__file__).parent / "kmob_sample.zarr"
N_RAYS, N_GATES = 360, 500


def main() -> None:
    if FIXTURE_PATH.exists():
        import shutil
        shutil.rmtree(FIXTURE_PATH)

    root = zarr.open_group(str(FIXTURE_PATH), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "schema_version": 2,
            "site_id": "KMOB",
            "site_lat": 30.6794,
            "site_lon": -88.2397,
            "site_alt_m": 63.0,
            "scan_time_utc": datetime(2026, 5, 17, 17, 28, 0, tzinfo=timezone.utc).isoformat(),
            "vcp": 215,
            "total_tilts": 2,
        }
    )

    for tilt_idx, elev in enumerate([0.5, 1.5]):
        tilt = root.create_group(f"tilt_{tilt_idx:02d}")
        tilt.attrs.update(
            {
                "elevation_deg": elev,
                "n_rays": N_RAYS,
                "n_gates": N_GATES,
                "azimuth_first_deg": 0.0,
                "range_step_m": 250.0,
                "range_first_m": 2125.0,
                "scan_time_utc": datetime(2026, 5, 17, 17, 28, 0, tzinfo=timezone.utc).isoformat(),
                "vcp": 215,
                "sweep_index": tilt_idx,
            }
        )

        # Reflectivity: echo ring at gates 100-200 (32 dBZ), rest no-echo
        ref = np.zeros((N_RAYS, N_GATES), dtype=np.uint8)
        ref[:, 100:200] = 130  # 130 * 0.5 + (-33) = 32.0 dBZ
        ref_arr = tilt.create_array("reflectivity", shape=ref.shape, dtype=ref.dtype, chunks=(90, 500))
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

        # Velocity: zero everywhere
        vel = np.zeros((N_RAYS, N_GATES), dtype=np.int16)
        vel_arr = tilt.create_array("velocity", shape=vel.shape, dtype=vel.dtype, chunks=(90, 500))
        vel_arr[:] = vel
        vel_arr.attrs.update(
            {
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "_FillValue": -32768,
                "units": "m s-1",
                "long_name": "Doppler radial velocity",
            }
        )

    print(f"Wrote fixture to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
