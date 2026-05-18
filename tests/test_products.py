import pytest

from polar_reader.products import PRODUCTS, Product, get_product


def test_all_expected_products_present():
    expected = {
        "reflectivity",
        "velocity",
        "spectrum_width",
        "differential_reflectivity",
        "cross_correlation_ratio",
        "differential_phase",
    }
    assert set(PRODUCTS.keys()) == expected


def test_each_product_is_product_instance():
    for name, prod in PRODUCTS.items():
        assert isinstance(prod, Product), f"{name} is not a Product"


@pytest.mark.parametrize(
    "name,expected_dtype",
    [
        ("reflectivity", "uint8"),
        ("velocity", "int16"),
        ("spectrum_width", "uint8"),
        ("differential_reflectivity", "uint8"),
        ("cross_correlation_ratio", "uint8"),
        ("differential_phase", "int16"),
    ],
)
def test_product_dtype(name, expected_dtype):
    assert PRODUCTS[name].dtype == expected_dtype


def test_reflectivity_scale_matches_icd():
    p = PRODUCTS["reflectivity"]
    assert p.scale == 0.5
    assert p.offset == -33.0
    assert p.fill_value == 0
    assert p.units == "dBZ"


def test_velocity_scale_matches_icd():
    p = PRODUCTS["velocity"]
    assert p.scale == 0.01
    assert p.offset == 0.0
    assert p.fill_value == -32768


def test_product_zarr_key_matches_name():
    for name, prod in PRODUCTS.items():
        assert prod.zarr_key == name, f"{name}: zarr_key mismatch"


def test_product_physical_range_is_ordered():
    for name, prod in PRODUCTS.items():
        lo, hi = prod.physical_range
        assert lo < hi, f"{name}: physical_range must be (min, max)"


def test_get_product_returns_correct():
    p = get_product("reflectivity")
    assert p.name == "reflectivity"
    assert p is PRODUCTS["reflectivity"]


def test_get_product_raises_on_unknown():
    with pytest.raises(KeyError, match="Unknown product"):
        get_product("nonexistent_product")


def test_product_has_default_colormap():
    for name, prod in PRODUCTS.items():
        assert prod.default_colormap, f"{name} missing default_colormap"


def test_product_is_frozen():
    """Products must be immutable (frozen dataclass)."""
    p = get_product("reflectivity")
    with pytest.raises((AttributeError, TypeError)):
        p.scale = 99.0  # type: ignore[misc]
