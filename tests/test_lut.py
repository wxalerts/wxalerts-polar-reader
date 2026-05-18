import numpy as np
import pytest

from polar_reader.lut import LutCache
from polar_reader.exceptions import LutNotFound


def test_cache_loads_lut(sample_lut_dir):
    cache = LutCache(sample_lut_dir)
    entry = cache.lookup_tile("KTEST", 10, 10, 20)
    assert entry is not None
    assert entry.range_idx.shape == (256, 256)
    assert entry.azimuth_idx.shape == (256, 256)
    assert entry.mask.shape == (256, 256)


def test_cache_returns_none_for_missing_tile(sample_lut_dir):
    cache = LutCache(sample_lut_dir)
    entry = cache.lookup_tile("KTEST", 10, 9999, 9999)
    assert entry is None


def test_cache_raises_for_missing_site(sample_lut_dir):
    cache = LutCache(sample_lut_dir)
    with pytest.raises(LutNotFound):
        cache.lookup_tile("KMISSING", 10, 10, 20)


def test_cache_raises_for_missing_zoom(sample_lut_dir):
    cache = LutCache(sample_lut_dir)
    with pytest.raises(LutNotFound):
        cache.lookup_tile("KTEST", 13, 10, 20)  # only z10 in fixture


def test_cache_second_tile_found(sample_lut_dir):
    cache = LutCache(sample_lut_dir)
    entry = cache.lookup_tile("KTEST", 10, 11, 21)
    assert entry is not None


def test_cache_lru_eviction(tmp_path):
    """Oldest (site, zoom) entry is evicted when max_sites is exceeded."""
    import numpy as np

    def _make_site(lut_root, site_id, x, y):
        site_dir = lut_root / "sites" / site_id
        site_dir.mkdir(parents=True)
        np.savez(
            site_dir / "z10.npz",
            tile_index=np.array([[x, y]], dtype=np.int32),
            range_idx=np.zeros((1, 256, 256), dtype=np.uint16),
            azimuth_idx=np.zeros((1, 256, 256), dtype=np.uint16),
            mask=np.ones((1, 256, 256), dtype=np.uint8),
        )

    _make_site(tmp_path, "KSITE1", 10, 20)
    _make_site(tmp_path, "KSITE2", 5, 6)

    cache = LutCache(str(tmp_path), max_sites=1)
    cache.lookup_tile("KSITE1", 10, 10, 20)
    assert ("KSITE1", 10) in cache._cache

    # Loading KSITE2 should evict KSITE1
    cache.lookup_tile("KSITE2", 10, 5, 6)
    assert ("KSITE1", 10) not in cache._cache
    assert ("KSITE2", 10) in cache._cache


def test_cache_stats(sample_lut_dir):
    cache = LutCache(sample_lut_dir, max_sites=50)
    cache.lookup_tile("KTEST", 10, 10, 20)
    stats = cache.stats()
    assert stats["cached_sites"] == 1
    assert stats["max"] == 50
    assert ("KTEST", 10) in stats["keys"]


def test_lut_entry_values_are_copies(sample_lut_dir):
    """Entries must be copies so mutations don't corrupt the mmap."""
    cache = LutCache(sample_lut_dir)
    entry1 = cache.lookup_tile("KTEST", 10, 10, 20)
    entry2 = cache.lookup_tile("KTEST", 10, 10, 20)
    assert entry1 is not entry2
    entry1.range_idx[0, 0] = 0xFFFF
    assert entry2.range_idx[0, 0] != 0xFFFF
