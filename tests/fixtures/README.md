# Test Fixtures

## kmob_sample.zarr

A small synthetic but structurally valid `schema_version=2` polar.zarr (KMOB, Mobile AL).

- 2 tilts (0.5° and 1.5°)
- 360 rays × 500 gates
- Reflectivity: synthetic echo ring at gates 100–200 (raw byte 130 → 32 dBZ)
- Velocity: zeroed out

**This directory is not committed to git** (zarr stores contain many small files).
Run `make_fixtures.py` to regenerate it:

```bash
python tests/fixtures/make_fixtures.py
```
