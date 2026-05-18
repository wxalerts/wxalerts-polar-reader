from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Product:
    """Encoding metadata for one radar moment.

    Matches nexrad-proc-v4 schema_version 2 NEXRAD ICD constants. These
    mirror the zarr array attrs but serve as a fallback / sanity-check.
    """

    name: str
    zarr_key: str
    dtype: str
    scale: float
    offset: float
    fill_value: int
    units: str
    long_name: str
    physical_range: tuple[float, float]
    default_colormap: str


PRODUCTS: dict[str, Product] = {
    "reflectivity": Product(
        name="reflectivity",
        zarr_key="reflectivity",
        dtype="uint8",
        scale=0.5,
        offset=-33.0,
        fill_value=0,
        units="dBZ",
        long_name="Equivalent reflectivity factor",
        physical_range=(-30.0, 80.0),
        default_colormap="nws_reflectivity",
    ),
    "velocity": Product(
        name="velocity",
        zarr_key="velocity",
        dtype="int16",
        scale=0.01,
        offset=0.0,
        fill_value=-32768,
        units="m s-1",
        long_name="Doppler radial velocity",
        physical_range=(-100.0, 100.0),
        default_colormap="nws_velocity",
    ),
    "spectrum_width": Product(
        name="spectrum_width",
        zarr_key="spectrum_width",
        dtype="uint8",
        scale=0.5,
        offset=-63.5,
        fill_value=0,
        units="m s-1",
        long_name="Doppler spectrum width",
        physical_range=(0.0, 15.0),
        default_colormap="nws_spectrum_width",
    ),
    "differential_reflectivity": Product(
        name="differential_reflectivity",
        zarr_key="differential_reflectivity",
        dtype="uint8",
        scale=0.0625,
        offset=-7.875,
        fill_value=0,
        units="dB",
        long_name="Differential reflectivity",
        physical_range=(-4.0, 8.0),
        default_colormap="nws_zdr",
    ),
    "cross_correlation_ratio": Product(
        name="cross_correlation_ratio",
        zarr_key="cross_correlation_ratio",
        dtype="uint8",
        scale=0.00333,
        offset=0.20833,
        fill_value=0,
        units="1",
        long_name="Copolar cross-correlation coefficient",
        physical_range=(0.0, 1.05),
        default_colormap="nws_correlation",
    ),
    "differential_phase": Product(
        name="differential_phase",
        zarr_key="differential_phase",
        dtype="int16",
        scale=0.043945,
        offset=-360.0,
        fill_value=-32768,
        units="degrees",
        long_name="Differential phase",
        physical_range=(0.0, 360.0),
        default_colormap="nws_phi",
    ),
}


def get_product(name: str) -> Product:
    if name not in PRODUCTS:
        raise KeyError(f"Unknown product '{name}'. Available: {list(PRODUCTS.keys())}")
    return PRODUCTS[name]
