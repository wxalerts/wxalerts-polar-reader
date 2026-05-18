import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polar_reader.exceptions import LutNotFound


@dataclass(frozen=True, slots=True)
class LutEntry:
    """Per-tile slice into a radar's polar coordinates."""

    range_idx: np.ndarray    # (256, 256) uint16
    azimuth_idx: np.ndarray  # (256, 256) uint16
    mask: np.ndarray         # (256, 256) uint8


class LutCache:
    """Per-process LRU cache of LUT files loaded via mmap.

    NPZ files are loaded with mmap_mode='r' so the OS page cache handles
    memory pressure. Hot tiles stay in RAM; cold ones evict transparently.
    The cache holds the wrapper objects; the OS holds the bytes.
    """

    def __init__(self, lut_dir: str | Path, max_sites: int = 200) -> None:
        self._lut_dir = Path(lut_dir)
        self._max = max_sites
        self._cache: OrderedDict[tuple[str, int], dict] = OrderedDict()
        self._lock = threading.RLock()

    def _load(self, site_id: str, zoom: int) -> dict:
        path = self._lut_dir / "sites" / site_id / f"z{zoom}.npz"
        if not path.exists():
            raise LutNotFound(f"No LUT at {path}")
        data = np.load(str(path), mmap_mode="r")
        return {
            "tile_index": data["tile_index"],
            "range_idx": data["range_idx"],
            "azimuth_idx": data["azimuth_idx"],
            "mask": data["mask"],
        }

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def lookup_tile(self, site_id: str, zoom: int, x: int, y: int) -> LutEntry | None:
        """Return the LutEntry for this tile, or None if not in the LUT.

        Thread-safe: the lock covers only OrderedDict mutations.
        Raises LutNotFound if no NPZ exists for (site_id, zoom).
        """
        key = (site_id, zoom)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                lut = self._cache[key]
            else:
                lut = self._load(site_id, zoom)
                self._cache[key] = lut
                self._evict_if_needed()

        tile_index = lut["tile_index"]
        matches = np.where((tile_index[:, 0] == x) & (tile_index[:, 1] == y))[0]
        if len(matches) == 0:
            return None
        row = int(matches[0])
        return LutEntry(
            range_idx=lut["range_idx"][row].copy(),
            azimuth_idx=lut["azimuth_idx"][row].copy(),
            mask=lut["mask"][row].copy(),
        )

    def stats(self) -> dict:
        with self._lock:
            return {
                "cached_sites": len(self._cache),
                "max": self._max,
                "keys": list(self._cache.keys()),
            }
