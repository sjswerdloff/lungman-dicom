"""Shrink an RT Structure Set's contours without changing what they enclose beyond a stated tolerance.

rt-utils already drops collinear points (OpenCV CHAIN_APPROX_SIMPLE), but a pixel-traced outline of any
sloped or curved edge is a staircase in which every one-pixel step is a corner, so most points survive.
Ramer-Douglas-Peucker (scikit-image `approximate_polygon`) removes them while keeping every original
point within `tolerance_mm` of the simplified outline. Coordinates are then written to `decimals` places:
0.001 mm is a micron, far below anything dosimetric.

Each contour's optional (Type 3) ContourImageSequence is also dropped: it names the CT slice the contour
lies on, which its z already says, and it was about a quarter of the file. The Structure Set's own
series-level reference to every CT image is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pydicom
from skimage.measure import approximate_polygon

DEFAULT_TOLERANCE_MM = 0.35  # about half a 0.625 mm voxel; below ~0.22 mm a pixel staircase is not reduced at all
DEFAULT_DECIMALS = 3
PLANAR_Z_TOLERANCE_MM = 1e-6  # rt-utils writes one z per contour exactly; anything more is a real tilt


@dataclass(frozen=True)
class ReductionStats:
    """Point counts before and after reduction."""

    contours: int
    points_before: int
    points_after: int


def simplify_contour(points: np.ndarray, tolerance_mm: float) -> np.ndarray:
    """Simplify one closed planar contour, (N, 3) in mm, keeping its z.

    Contours of fewer than 4 points are returned unchanged: there is nothing to simplify, and a single-
    pixel structure must not collapse.

    Raises:
        ValueError: if `tolerance_mm` is negative, or the contour's points do not share one z (the
            simplified outline is written at the first point's z, so a non-planar contour would be
            flattened silently).
    """
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm must be non-negative")
    if len(points) and np.ptp(points[:, 2]) > PLANAR_Z_TOLERANCE_MM:
        raise ValueError(f"contour is not axial-planar: z spans {np.ptp(points[:, 2]):.6g} mm")
    if len(points) < 4 or tolerance_mm == 0:
        return points
    closed = np.vstack([points[:, :2], points[:1, :2]])  # approximate_polygon treats first == last as closed
    reduced = approximate_polygon(closed, tolerance=tolerance_mm)
    if np.array_equal(reduced[0], reduced[-1]):
        reduced = reduced[:-1]
    if len(reduced) < 3:  # never reduce a real outline below a triangle
        return points
    return np.column_stack([reduced, np.full(len(reduced), points[0, 2])])


def _format(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return "0" if float(text) == 0 else text.rstrip("0").rstrip(".")


def reduce_rtstruct(
    ds: pydicom.Dataset, tolerance_mm: float = DEFAULT_TOLERANCE_MM, decimals: int = DEFAULT_DECIMALS
) -> ReductionStats:
    """Simplify every contour in place, write its coordinates to `decimals` places, and drop its
    ContourImageSequence.

    Returns:
        Contour and point counts before and after.
    """
    if not 0 <= decimals <= 6:
        raise ValueError("decimals must be between 0 and 6")
    contours = before = after = 0
    for roi in ds.ROIContourSequence:
        for contour in getattr(roi, "ContourSequence", []):
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            reduced = simplify_contour(points, tolerance_mm)
            contour.ContourData = [_format(v, decimals) for v in reduced.ravel()]
            contour.NumberOfContourPoints = len(reduced)
            if "ContourImageSequence" in contour:
                del contour.ContourImageSequence
            contours += 1
            before += len(points)
            after += len(reduced)
    return ReductionStats(contours, before, after)
