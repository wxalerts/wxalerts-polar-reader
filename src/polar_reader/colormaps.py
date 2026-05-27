"""NWS-standard radar colormaps.

Reflectivity stops are ported verbatim from scripts/spot_check_tile.py
(the canonical reference implementation).

Velocity, ZDR, correlation, spectrum width, and phi colormaps follow NWS
RIDGE-II conventions, cross-referenced against RadarScope and Py-ART
(pyart.graph.cm). Exact stop values are documented per colormap below.
"""

import colorsys

import numpy as np
from typing import Callable


# ── reflectivity ──────────────────────────────────────────────────────────────
# Ported verbatim from scripts/spot_check_tile.py (wxalerts-nexrad-lut-gen).
# Format: (lower_dbz_bound, R, G, B, A). Below the first bound → transparent.
_NWS_REFL_STOPS = [
    (15, 0,   0,   255,  80),  # faint blue — light drizzle (semi-transparent)
    (20, 0,   100, 200, 255),  # light blue-green — light rain
    (25, 0,   200, 0,   255),  # green
    (30, 100, 220, 0,   255),  # yellow-green
    (35, 200, 220, 0,   255),  # yellow
    (40, 255, 150, 0,   255),  # orange
    (45, 255, 0,   0,   255),  # red
    (50, 180, 0,   0,   255),  # dark red
    (55, 220, 0,   220, 255),  # magenta
    (60, 180, 0,   180, 255),  # light purple
    (65, 255, 255, 255, 255),  # white
    (70, 150, 150, 150, 255),  # grey
]


def nws_reflectivity(physical: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply NWS-standard reflectivity colormap.

    Args:
        physical: float array of dBZ values, any shape.
        mask: uint8 array same shape; non-zero = valid radar pixel.

    Returns:
        RGBA uint8 array with trailing dimension 4.
    """
    rgba = np.zeros((*physical.shape, 4), dtype=np.uint8)
    bins = np.array([s[0] for s in _NWS_REFL_STOPS])
    colors = np.array([[s[1], s[2], s[3], s[4]] for s in _NWS_REFL_STOPS], dtype=np.uint8)

    idx = np.digitize(physical, bins) - 1  # -1 for below-threshold → transparent
    valid = (idx >= 0) & (mask > 0)
    rgba[valid] = colors[np.clip(idx[valid], 0, len(colors) - 1)]
    return rgba


# ── velocity ─────────────────────────────────────────────────────────────────
# Red-green diverging; near-zero band is semi-transparent.
# Stops: lower bound (m/s), R, G, B, A.
_NWS_VEL_STOPS = [
    (-100, 0,   230, 230, 255),  # < -50: bright cyan
    (-50,  0,   180, 0,   255),  # -50 to -25: medium green
    (-25,  130, 255, 130, 255),  # -25 to -5: light green
    (-5,   160, 160, 160, 80),   # -5 to +5: near-zero (semi-transparent)
    (5,    255, 130, 130, 255),  # +5 to +25: light red
    (25,   220, 0,   0,   255),  # +25 to +50: medium red
    (50,   255, 0,   200, 255),  # >= +50: bright magenta
]


def nws_velocity(physical: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply NWS-standard radial velocity colormap (m/s)."""
    rgba = np.zeros((*physical.shape, 4), dtype=np.uint8)
    bins = np.array([s[0] for s in _NWS_VEL_STOPS])
    colors = np.array([[s[1], s[2], s[3], s[4]] for s in _NWS_VEL_STOPS], dtype=np.uint8)

    idx = np.digitize(physical, bins) - 1
    valid = (idx >= 0) & (mask > 0)
    rgba[valid] = colors[np.clip(idx[valid], 0, len(colors) - 1)]
    return rgba


# ── correlation coefficient (RHO HV) ─────────────────────────────────────────
# Below 0.6 → transparent (noise). Above 0.99 → meteorological scatterers.
_NWS_CORR_STOPS = [
    (0.6,  128, 0,   200, 255),  # 0.6-0.7: purple
    (0.7,  0,   0,   255, 255),  # 0.7-0.8: blue
    (0.8,  0,   200, 0,   255),  # 0.8-0.9: green
    (0.9,  255, 255, 0,   255),  # 0.9-0.95: yellow
    (0.95, 255, 165, 0,   255),  # 0.95-0.99: orange
    (0.99, 255, 220, 255, 255),  # >= 0.99: white-hot pink
]


def nws_correlation(physical: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply NWS-standard correlation coefficient colormap (dimensionless)."""
    rgba = np.zeros((*physical.shape, 4), dtype=np.uint8)
    bins = np.array([s[0] for s in _NWS_CORR_STOPS])
    colors = np.array([[s[1], s[2], s[3], s[4]] for s in _NWS_CORR_STOPS], dtype=np.uint8)

    idx = np.digitize(physical, bins) - 1  # below 0.6 → -1 → transparent
    valid = (idx >= 0) & (mask > 0)
    rgba[valid] = colors[np.clip(idx[valid], 0, len(colors) - 1)]
    return rgba


# ── ZDR ───────────────────────────────────────────────────────────────────────
# Blue-white-red diverging around 0 dB.
_NWS_ZDR_STOPS = [
    (-4,   0,   0,   180, 255),  # -4 to -2: dark blue
    (-2,   0,   80,  255, 255),  # -2 to -0.5: blue
    (-0.5, 240, 240, 240, 255),  # -0.5 to 0.5: white (near-zero)
    (0.5,  255, 160, 160, 255),  # 0.5 to 2: light red
    (2,    220, 0,   0,   255),  # 2 to 4: red
    (4,    150, 0,   0,   255),  # 4 to 6: dark red
    (6,    100, 0,   80,  255),  # >= 6: very dark red
]


def nws_zdr(physical: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply NWS-standard differential reflectivity colormap (dB)."""
    rgba = np.zeros((*physical.shape, 4), dtype=np.uint8)
    bins = np.array([s[0] for s in _NWS_ZDR_STOPS])
    colors = np.array([[s[1], s[2], s[3], s[4]] for s in _NWS_ZDR_STOPS], dtype=np.uint8)

    idx = np.digitize(physical, bins) - 1  # below -4 → transparent
    valid = (idx >= 0) & (mask > 0)
    rgba[valid] = colors[np.clip(idx[valid], 0, len(colors) - 1)]
    return rgba


# ── spectrum width ────────────────────────────────────────────────────────────
# Grayscale, dark → light over 0–15 m/s.
_NWS_SW_STOPS = [
    (0,  30,  30,  30,  255),
    (2,  70,  70,  70,  255),
    (4,  110, 110, 110, 255),
    (6,  150, 150, 150, 255),
    (8,  185, 185, 185, 255),
    (10, 215, 215, 215, 255),
    (12, 245, 245, 245, 255),
]


def nws_spectrum_width(physical: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply NWS-standard spectrum width colormap (m/s)."""
    rgba = np.zeros((*physical.shape, 4), dtype=np.uint8)
    bins = np.array([s[0] for s in _NWS_SW_STOPS])
    colors = np.array([[s[1], s[2], s[3], s[4]] for s in _NWS_SW_STOPS], dtype=np.uint8)

    idx = np.digitize(physical, bins) - 1  # below 0 → transparent
    valid = (idx >= 0) & (mask > 0)
    rgba[valid] = colors[np.clip(idx[valid], 0, len(colors) - 1)]
    return rgba


# ── differential phase ────────────────────────────────────────────────────────
# Cyclic: full hue rotation over 0–360°. Uses colorsys for HSV→RGB conversion.
_PHI_N_BINS = 12  # 30° per bin


def _build_phi_colors() -> np.ndarray:
    colors = []
    for i in range(_PHI_N_BINS):
        midpoint_deg = 15.0 + i * 30.0
        r, g, b = colorsys.hsv_to_rgb(midpoint_deg / 360.0, 1.0, 1.0)
        colors.append([int(r * 255), int(g * 255), int(b * 255), 255])
    return np.array(colors, dtype=np.uint8)


_NWS_PHI_COLORS = _build_phi_colors()
_NWS_PHI_BINS = np.arange(0, 360, 30, dtype=np.float32)  # [0, 30, 60, ..., 330]


def nws_phi(physical: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply NWS-standard differential phase colormap (degrees, cyclic)."""
    rgba = np.zeros((*physical.shape, 4), dtype=np.uint8)
    valid = mask > 0

    # Wrap to [0, 360) then bin
    wrapped = physical % 360.0
    idx = np.digitize(wrapped, _NWS_PHI_BINS) - 1
    in_range = (idx >= 0) & valid
    rgba[in_range] = _NWS_PHI_COLORS[np.clip(idx[in_range], 0, _PHI_N_BINS - 1)]
    return rgba


COLORMAPS: dict[str, Callable] = {
    "nws_reflectivity": nws_reflectivity,
    "nws_velocity": nws_velocity,
    "nws_correlation": nws_correlation,
    "nws_zdr": nws_zdr,
    "nws_spectrum_width": nws_spectrum_width,
    "nws_phi": nws_phi,
}


def get_colormap(name: str) -> Callable:
    if name not in COLORMAPS:
        raise KeyError(f"Unknown colormap '{name}'. Available: {list(COLORMAPS.keys())}")
    return COLORMAPS[name]
