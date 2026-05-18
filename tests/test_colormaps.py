import numpy as np
import pytest

from polar_reader.colormaps import (
    get_colormap,
    nws_reflectivity,
    nws_velocity,
    nws_correlation,
    nws_zdr,
    nws_spectrum_width,
    nws_phi,
    COLORMAPS,
)


def _ones_mask(shape):
    return np.ones(shape, dtype=np.uint8)


def _zeros_mask(shape):
    return np.zeros(shape, dtype=np.uint8)


# ── output shape and dtype ────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(COLORMAPS.keys()))
def test_colormap_output_shape(name):
    fn = COLORMAPS[name]
    physical = np.zeros((10, 10), dtype=np.float32)
    mask = _ones_mask((10, 10))
    out = fn(physical, mask)
    assert out.shape == (10, 10, 4)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", list(COLORMAPS.keys()))
def test_colormap_masked_pixels_are_transparent(name):
    fn = COLORMAPS[name]
    physical = np.full((5, 5), 50.0, dtype=np.float32)
    mask = _zeros_mask((5, 5))
    out = fn(physical, mask)
    # With mask=0, alpha must be 0 everywhere
    assert (out[..., 3] == 0).all()


# ── reflectivity specific ─────────────────────────────────────────────────────

def test_nws_reflectivity_below_threshold_transparent():
    physical = np.array([[0.0, 4.9]], dtype=np.float32)
    out = nws_reflectivity(physical, _ones_mask(physical.shape))
    assert (out[..., 3] == 0).all()


def test_nws_reflectivity_45dbz_is_red():
    physical = np.array([[47.0]], dtype=np.float32)  # 45-50 → red
    out = nws_reflectivity(physical, _ones_mask(physical.shape))
    r, g, b, a = out[0, 0]
    assert a == 255
    assert r == 255
    assert g == 0
    assert b == 0


def test_nws_reflectivity_70dbz_is_grey():
    physical = np.array([[72.0]], dtype=np.float32)
    out = nws_reflectivity(physical, _ones_mask(physical.shape))
    r, g, b, a = out[0, 0]
    assert a == 255
    assert r == g == b == 150


def test_nws_reflectivity_5dbz_lower_bound():
    """Exactly 5 dBZ should be colored (first stop), not transparent."""
    physical = np.array([[5.0]], dtype=np.float32)
    out = nws_reflectivity(physical, _ones_mask(physical.shape))
    assert out[0, 0, 3] == 255


def test_nws_reflectivity_batch_shape():
    physical = np.random.uniform(-5, 75, size=(4, 256, 256)).astype(np.float32)
    mask = _ones_mask((4, 256, 256))
    out = nws_reflectivity(physical, mask)
    assert out.shape == (4, 256, 256, 4)


# ── velocity ─────────────────────────────────────────────────────────────────

def test_nws_velocity_positive_values_have_red_tint():
    physical = np.array([[30.0]], dtype=np.float32)  # +25 to +50: medium red
    out = nws_velocity(physical, _ones_mask(physical.shape))
    assert out[0, 0, 3] == 255
    assert out[0, 0, 0] > out[0, 0, 2]  # R > B


def test_nws_velocity_negative_values_have_green_tint():
    physical = np.array([[-30.0]], dtype=np.float32)  # -50 to -25: medium green
    out = nws_velocity(physical, _ones_mask(physical.shape))
    assert out[0, 0, 3] == 255
    assert out[0, 0, 1] > out[0, 0, 0]  # G > R


# ── correlation ───────────────────────────────────────────────────────────────

def test_nws_correlation_below_threshold_transparent():
    physical = np.array([[0.5]], dtype=np.float32)  # below 0.6
    out = nws_correlation(physical, _ones_mask(physical.shape))
    assert out[0, 0, 3] == 0


def test_nws_correlation_high_value_colored():
    physical = np.array([[1.0]], dtype=np.float32)  # >= 0.99
    out = nws_correlation(physical, _ones_mask(physical.shape))
    assert out[0, 0, 3] == 255


# ── ZDR ───────────────────────────────────────────────────────────────────────

def test_nws_zdr_near_zero_is_white():
    physical = np.array([[0.0]], dtype=np.float32)  # -0.5 to 0.5 → white
    out = nws_zdr(physical, _ones_mask(physical.shape))
    r, g, b, a = out[0, 0]
    assert a == 255
    # Near white: all channels should be high
    assert r > 200 and g > 200 and b > 200


def test_nws_zdr_negative_is_blue():
    physical = np.array([[-3.0]], dtype=np.float32)  # -4 to -2 or -2 to -0.5
    out = nws_zdr(physical, _ones_mask(physical.shape))
    assert out[0, 0, 3] == 255
    assert out[0, 0, 2] > out[0, 0, 0]  # B > R (blue tint)


# ── spectrum width ────────────────────────────────────────────────────────────

def test_nws_spectrum_width_below_zero_transparent():
    physical = np.array([[-1.0]], dtype=np.float32)
    out = nws_spectrum_width(physical, _ones_mask(physical.shape))
    assert out[0, 0, 3] == 0


def test_nws_spectrum_width_grayscale():
    """All three color channels should be equal (greyscale)."""
    physical = np.array([[5.0]], dtype=np.float32)
    out = nws_spectrum_width(physical, _ones_mask(physical.shape))
    r, g, b, a = out[0, 0]
    assert a == 255
    assert r == g == b


# ── phi ───────────────────────────────────────────────────────────────────────

def test_nws_phi_is_cyclic():
    """0° and 360° should map to the same color."""
    p0 = np.array([[0.0]], dtype=np.float32)
    p360 = np.array([[360.0]], dtype=np.float32)
    mask = _ones_mask((1, 1))
    np.testing.assert_array_equal(nws_phi(p0, mask), nws_phi(p360, mask))


def test_nws_phi_full_range_colored():
    angles = np.arange(0, 360, 10, dtype=np.float32).reshape(1, -1)
    out = nws_phi(angles, _ones_mask(angles.shape))
    assert (out[..., 3] == 255).all()


# ── registry ─────────────────────────────────────────────────────────────────

def test_get_colormap_returns_callable():
    fn = get_colormap("nws_reflectivity")
    assert callable(fn)


def test_get_colormap_raises_on_unknown():
    with pytest.raises(KeyError):
        get_colormap("not_a_colormap")
