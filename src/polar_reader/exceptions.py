class PolarReaderError(Exception):
    """Base exception for polar_reader package."""


class TileOutsideCoverage(PolarReaderError):
    """The requested tile has no in-coverage pixels for this radar."""


class TiltNotFound(PolarReaderError):
    """The requested tilt (by index, elevation, or 'base') doesn't exist."""


class ProductNotAvailable(PolarReaderError):
    """The requested product isn't present in the selected tilt."""


class LutNotFound(PolarReaderError):
    """No LUT exists for this site/zoom combination."""


class SchemaVersionMismatch(PolarReaderError):
    """The polar.zarr's schema_version is not supported by this reader."""


class SiteNotFoundError(PolarReaderError):
    """The requested site ID is not in the site registry."""
