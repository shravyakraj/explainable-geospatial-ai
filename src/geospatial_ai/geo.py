"""Geospatial grid definitions and coordinate reference systems."""

from dataclasses import dataclass
import affine
import pyproj

@dataclass(frozen=True)
class StudyGrid:
    """Locked study region grid definition for Rondônia (Tile 20LMR)."""
    # EPSG:4326 Bounds
    west: float = -63.80
    south: float = -9.05
    east: float = -63.15
    north: float = -8.50

    # Target Projected CRS
    crs: str = "EPSG:32720"

    # Pixel resolution (meters)
    pixel_size: float = 10.0

    # Origin in EPSG:32720 (Easting, Northing)
    origin_x: float = 412080.0
    origin_y: float = 9060420.0

    # Patch grid specifications
    patch_size: int = 256
    cols_patches: int = 27
    rows_patches: int = 23

    @property
    def width_pixels(self) -> int:
        return self.cols_patches * self.patch_size  # 27 * 256 = 6912

    @property
    def height_pixels(self) -> int:
        return self.rows_patches * self.patch_size  # 23 * 256 = 5888

    @property
    def total_patches(self) -> int:
        return self.cols_patches * self.rows_patches  # 621

    @property
    def transform(self) -> affine.Affine:
        return affine.Affine(
            self.pixel_size, 0.0, self.origin_x,
            0.0, -self.pixel_size, self.origin_y
        )

    @property
    def bounds_projected(self) -> tuple[float, float, float, float]:
        minx = self.origin_x
        maxy = self.origin_y
        maxx = minx + self.width_pixels * self.pixel_size
        miny = maxy - self.height_pixels * self.pixel_size
        return (minx, miny, maxx, maxy)

LOCKED_GRID = StudyGrid()
