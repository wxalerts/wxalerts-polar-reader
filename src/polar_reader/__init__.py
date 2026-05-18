"""polar_reader — rio-tiler BaseReader for nexrad-proc-v4 polar Zarr v3 volumes."""

from polar_reader.colormaps import COLORMAPS, get_colormap
from polar_reader.constants import ON_THE_FLY_ZOOM_THRESHOLD
from polar_reader.products import PRODUCTS, get_product
from polar_reader.reader import PolarRadarReader

__all__ = [
    "PolarRadarReader",
    "PRODUCTS",
    "get_product",
    "COLORMAPS",
    "get_colormap",
    "ON_THE_FLY_ZOOM_THRESHOLD",
]
