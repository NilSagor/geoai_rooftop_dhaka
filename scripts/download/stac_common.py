"""Shared STAC plumbing: catalog, bbox, target grid, chip enumeration, and warping."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math

import numpy as np
import planetary_computer
import pystac
import pystac_client
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DHAKA_BBOX_4326: tuple[float, float, float, float] = (90.00, 23.53, 90.52, 24.02)
TARGET_CRS = "EPSG:32646"


@dataclass(frozen=True)
class Grid:
    """Regular grid snapped to multiples of `res_m` in `crs`."""

    crs: str
    res_m: int
    xmin: float
    ymax: float
    width_px: int
    height_px: int

    @property
    def xmax(self) -> float:
        return self.xmin + self.width_px * self.res_m

    @property
    def ymin(self) -> float:
        return self.ymax - self.height_px * self.res_m

    @property
    def transform(self) -> Affine:
        """Top-left pixel-edge affine transform for rasterio."""
        return Affine.translation(self.xmin, self.ymax) * Affine.scale(
            self.res_m, -self.res_m
        )

    def chip_transform(self, row_px: int, col_px: int) -> Affine:
        """Affine transform for a chip anchored at `(row_px, col_px)` on this grid."""
        return self.transform * Affine.translation(col_px, row_px)

    def chip_bounds(self, row_px: int, col_px: int, chip_px: int) -> tuple[float, float, float, float]:
        """Return `(left, bottom, right, top)` in `self.crs` for a chip."""
        left = self.xmin + col_px * self.res_m
        top = self.ymax - row_px * self.res_m
        right = left + chip_px * self.res_m
        bottom = top - chip_px * self.res_m
        return (left, bottom, right, top)

    def stride_px(self, chip_px: int, overlap: float = 0.5) -> int:
        if chip_px <= 0:
            raise ValueError(f"chip_px must be positive, got {chip_px}")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")
        return max(1, int(round(chip_px * (1.0 - overlap))))

    def chip_origins(
        self, chip_px: int, overlap: float = 0.5
    ) -> Iterator[tuple[int, int]]:
        """Yield `(row_px, col_px)` top-left corners; final row/col stay flush."""
        stride = self.stride_px(chip_px, overlap)
        yield from chip_origins_from_shape(
            height_px=self.height_px,
            width_px=self.width_px,
            chip_px=chip_px,
            stride_px=stride,
        )


def _axis_starts(total_px: int, chip_px: int, stride_px: int) -> list[int]:
    """Compute 1D chip start indices with the last chip anchored flush to `total_px`."""
    if total_px <= chip_px:
        return [0]
    last_start = total_px - chip_px
    starts = list(range(0, last_start + 1, stride_px))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def chip_origins_from_shape(
    height_px: int,
    width_px: int,
    chip_px: int,
    stride_px: int,
) -> Iterator[tuple[int, int]]:
    """Yield `(row_px, col_px)` top-left chip origins for a raster of given dimensions."""
    if stride_px <= 0:
        raise ValueError(f"stride_px must be positive, got {stride_px}")
    rows = _axis_starts(height_px, chip_px, stride_px)
    cols = _axis_starts(width_px, chip_px, stride_px)
    for row in rows:
        for col in cols:
            yield row, col


def dhaka_grid(
    res_m: int,
    bbox_4326: tuple[float, float, float, float] = DHAKA_BBOX_4326,
    target_crs: str = TARGET_CRS,
) -> Grid:
    """Build a snapped grid covering `bbox_4326` in `target_crs`."""
    if res_m <= 0:
        raise ValueError(f"res_m must be positive, got {res_m}")

    min_lon, min_lat, max_lon, max_lat = bbox_4326
    corners_lon = [min_lon, max_lon, min_lon, max_lon]
    corners_lat = [min_lat, max_lat, max_lat, min_lat]

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    xs, ys = transformer.transform(corners_lon, corners_lat)

    xmin = math.floor(min(xs) / res_m) * res_m
    xmax = math.ceil(max(xs) / res_m) * res_m
    ymin = math.floor(min(ys) / res_m) * res_m
    ymax = math.ceil(max(ys) / res_m) * res_m

    width_px = int((xmax - xmin) / res_m)
    height_px = int((ymax - ymin) / res_m)

    return Grid(
        crs=target_crs,
        res_m=res_m,
        xmin=float(xmin),
        ymax=float(ymax),
        width_px=width_px,
        height_px=height_px,
    )


def open_catalog(stac_url: str = STAC_URL) -> pystac_client.Client:
    """Open and sign assets for a Microsoft Planetary Computer STAC catalog."""
    return pystac_client.Client.open(
        stac_url,
        modifier=planetary_computer.sign_inplace,
    )


def search_stac(
    catalog: pystac_client.Client,
    collections: Sequence[str] = ("sentinel-2-l2a",),
    bbox: tuple[float, float, float, float] = DHAKA_BBOX_4326,
    datetime: str | None = None,
    max_cloud_cover: float | None = 20.0,
    limit: int = 100,
) -> list[pystac.Item]:
    """Search STAC items and return them sorted by cloud cover ascending, then date."""
    query: dict[str, dict[str, float]] | None = None
    if max_cloud_cover is not None:
        query = {"eo:cloud_cover": {"lt": max_cloud_cover}}

    search = catalog.search(
        collections=list(collections),
        bbox=list(bbox),
        datetime=datetime,
        query=query,
        limit=limit,
    )
    items = list(search.items())
    items.sort(
        key=lambda item: (
            item.properties.get("eo:cloud_cover", float("inf")),
            item.datetime.isoformat() if item.datetime else "",
        )
    )
    return items


def reproject_chip_to_grid(
    href: str,
    grid: Grid,
    row_px: int,
    col_px: int,
    chip_px: int,
    resampling: Resampling = Resampling.bilinear,
    dst_nodata: float = np.nan,
    pad_px: int = 4,
) -> tuple[np.ndarray, Affine]:
    """Read a windowed region from `href` and reproject onto a single `Grid` chip.

    Returns `(chip_array, chip_transform)` with shape `(chip_px, chip_px)` and dtype `float32`.
    """
    dst_left, dst_bottom, dst_right, dst_top = grid.chip_bounds(row_px, col_px, chip_px)
    dst_transform = grid.chip_transform(row_px, col_px)
    dst_array = np.full((chip_px, chip_px), dst_nodata, dtype=np.float32)

    with rasterio.open(href) as src:
        if src.crs is None:
            raise ValueError(f"Source raster has no CRS: {href}")

        src_left, src_bottom, src_right, src_top = transform_bounds(
            grid.crs,
            src.crs,
            dst_left,
            dst_bottom,
            dst_right,
            dst_top,
            densify_pts=21,
        )
        src_window = (
            from_bounds(src_left, src_bottom, src_right, src_top, transform=src.transform)
            .round_offsets()
            .round_lengths()
            .pad(pad_px)
            .intersection(Window(0, 0, src.width, src.height))
        )
        if src_window.width <= 0 or src_window.height <= 0:
            return dst_array, dst_transform

        src_data = src.read(1, window=src_window, masked=True).filled(np.nan).astype(np.float32)
        src_transform = src.window_transform(src_window)
        src_nodata = src.nodata if src.nodata is not None else np.nan

        reproject(
            source=src_data,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=grid.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    return dst_array, dst_transform




# """Shared STAC plumbing: catalog, bbox, target grid, and chip enumeration."""

# from collections.abc import Iterator
# from dataclasses import dataclass
# import math

# import planetary_computer
# import pystac_client
# from pyproj import Transformer
# from rasterio.transform import Affine

# STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# DHAKA_BBOX_4326: tuple[float, float, float, float] = (90.00, 23.53, 90.52, 24.02)
# TARGET_CRS = "EPSG:32646"


# @dataclass(frozen=True)
# class Grid:
#     """Regular grid snapped to multiples of `res_m` in `crs`."""

#     crs: str
#     res_m: int
#     xmin: float
#     ymax: float
#     width_px: int
#     height_px: int

#     @property
#     def xmax(self) -> float:
#         return self.xmin + self.width_px * self.res_m

#     @property
#     def ymin(self) -> float:
#         return self.ymax - self.height_px * self.res_m

#     @property
#     def transform(self) -> Affine:
#         """Top-left pixel-edge affine transform for rasterio."""
#         return Affine.translation(self.xmin, self.ymax) * Affine.scale(
#             self.res_m, -self.res_m
#         )

#     def stride_px(self, chip_px: int, overlap: float = 0.5) -> int:
#         if chip_px <= 0:
#             raise ValueError(f"chip_px must be positive, got {chip_px}")
#         if not (0.0 <= overlap < 1.0):
#             raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")
#         return max(1, int(round(chip_px * (1.0 - overlap))))

#     def chip_origins(
#         self, chip_px: int, overlap: float = 0.5
#     ) -> Iterator[tuple[int, int]]:
#         """Yield `(row_px, col_px)` top-left corners; final row/col stay flush."""
#         stride = self.stride_px(chip_px, overlap)
#         yield from chip_origins_from_shape(
#             height_px=self.height_px,
#             width_px=self.width_px,
#             chip_px=chip_px,
#             stride_px=stride,
#         )


# def _axis_starts(total_px: int, chip_px: int, stride_px: int) -> list[int]:
#     """Compute 1D chip start indices with the last chip anchored flush to `total_px`."""
#     if total_px <= chip_px:
#         return [0]
#     last_start = total_px - chip_px
#     starts = list(range(0, last_start + 1, stride_px))
#     if starts[-1] != last_start:
#         starts.append(last_start)
#     return starts


# def chip_origins_from_shape(
#     height_px: int,
#     width_px: int,
#     chip_px: int,
#     stride_px: int,
# ) -> Iterator[tuple[int, int]]:
#     """Yield `(row_px, col_px)` top-left chip origins for a raster of given dimensions."""
#     if stride_px <= 0:
#         raise ValueError(f"stride_px must be positive, got {stride_px}")
#     rows = _axis_starts(height_px, chip_px, stride_px)
#     cols = _axis_starts(width_px, chip_px, stride_px)
#     for row in rows:
#         for col in cols:
#             yield row, col


# def dhaka_grid(
#     res_m: int,
#     bbox_4326: tuple[float, float, float, float] = DHAKA_BBOX_4326,
#     target_crs: str = TARGET_CRS,
# ) -> Grid:
#     """Build a snapped grid covering `bbox_4326` in `target_crs`."""
#     if res_m <= 0:
#         raise ValueError(f"res_m must be positive, got {res_m}")

#     min_lon, min_lat, max_lon, max_lat = bbox_4326
#     corners_lon = [min_lon, max_lon, min_lon, max_lon]
#     corners_lat = [min_lat, max_lat, max_lat, min_lat]

#     transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
#     xs, ys = transformer.transform(corners_lon, corners_lat)

#     xmin = math.floor(min(xs) / res_m) * res_m
#     xmax = math.ceil(max(xs) / res_m) * res_m
#     ymin = math.floor(min(ys) / res_m) * res_m
#     ymax = math.ceil(max(ys) / res_m) * res_m

#     width_px = int((xmax - xmin) / res_m)
#     height_px = int((ymax - ymin) / res_m)

#     return Grid(
#         crs=target_crs,
#         res_m=res_m,
#         xmin=float(xmin),
#         ymax=float(ymax),
#         width_px=width_px,
#         height_px=height_px,
#     )


# def open_catalog(stac_url: str = STAC_URL) -> pystac_client.Client:
#     """Open and sign assets for a Microsoft Planetary Computer STAC catalog."""
#     return pystac_client.Client.open(
#         stac_url,
#         modifier=planetary_computer.sign_inplace,
#     )