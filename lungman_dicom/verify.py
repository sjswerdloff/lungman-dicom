"""Verify an RTSTRUCT against the label volume WITHOUT using rt-utils.

convert.py writes through rt-utils; reading back through rt-utils would share its geometry
assumptions. This reads ContourData with pydicom, maps patient millimetres to voxel indices
from the CT headers, and fills the polygons with scikit-image (even-odd, so nested contours
become holes), including the boundary pixels the contours run through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from skimage.draw import line, polygon

from lungman_dicom.convert import LungmanConversionError


@dataclass(frozen=True)
class RoiComparison:
    """Agreement between one ROI's contours and its reference mask."""

    name: str
    reference_voxels: int
    contour_voxels: int
    dice: float


def rasterize_roi(rtstruct: pydicom.Dataset, roi_number: int, ct: list[pydicom.Dataset]) -> np.ndarray:
    """Fill every contour of one ROI onto the CT grid, (slices, rows, cols) in instance order."""
    rows, cols = int(ct[0].Rows), int(ct[0].Columns)
    mask = np.zeros((len(ct), rows, cols), dtype=bool)
    z_to_index = {round(float(d.ImagePositionPatient[2]), 3): i for i, d in enumerate(ct)}
    x0, y0 = float(ct[0].ImagePositionPatient[0]), float(ct[0].ImagePositionPatient[1])
    dy, dx = (float(v) for v in ct[0].PixelSpacing)  # PixelSpacing is (row spacing, column spacing)
    boundary = np.zeros_like(mask)
    roi_contours = next(r for r in rtstruct.ROIContourSequence if int(r.ReferencedROINumber) == roi_number)
    for contour in getattr(roi_contours, "ContourSequence", []):
        points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
        zs = {round(z, 3) for z in points[:, 2]}
        if len(zs) != 1:
            raise LungmanConversionError(f"ROI {roi_number}: a contour is not planar")
        z = zs.pop()
        if z not in z_to_index:
            raise LungmanConversionError(f"ROI {roi_number}: contour at z={z} matches no CT slice")
        cc = (points[:, 0] - x0) / dx
        rr = (points[:, 1] - y0) / dy
        r_idx, c_idx = polygon(rr, cc, shape=(rows, cols))
        filled = np.zeros((rows, cols), dtype=bool)
        filled[r_idx, c_idx] = True
        mask[z_to_index[z]] ^= filled
        # The contours run through the CENTRES of boundary pixels, and those pixels belong to
        # the structure (rt-utils' own read-back is exact under that convention). A strict
        # polygon fill excludes them, which costs a 1-2 pixel thick structure most of itself.
        ri, ci = np.round(rr).astype(int), np.round(cc).astype(int)
        for k in range(len(ri)):
            lr, lc = line(ri[k], ci[k], ri[(k + 1) % len(ri)], ci[(k + 1) % len(ri)])
            boundary[z_to_index[z], lr.clip(0, rows - 1), lc.clip(0, cols - 1)] = True
    return mask | boundary


def compare(rtstruct_path: Path, ct: list[pydicom.Dataset], references: dict[str, np.ndarray]) -> list[RoiComparison]:
    """Compare each named ROI in the file with its reference mask."""
    rtstruct = pydicom.dcmread(rtstruct_path)
    numbers = {r.ROIName: int(r.ROINumber) for r in rtstruct.StructureSetROISequence}
    results: list[RoiComparison] = []
    for name, reference in references.items():
        if name not in numbers:
            raise LungmanConversionError(f"ROI {name!r} missing from {rtstruct_path}")
        drawn = rasterize_roi(rtstruct, numbers[name], ct)
        overlap = np.logical_and(drawn, reference).sum()
        total = drawn.sum() + reference.sum()
        results.append(
            RoiComparison(name, int(reference.sum()), int(drawn.sum()), float(2 * overlap / total) if total else 1.0)
        )
    return results


def stl_bounds(stl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) of an STL mesh's vertices, in the mesh's own units and frame."""
    from stl import mesh as stl_mesh

    vertices = stl_mesh.Mesh.from_file(str(stl_path)).vectors.reshape(-1, 3)
    return vertices.min(axis=0), vertices.max(axis=0)


def mask_bounds_mm(mask: np.ndarray, ct: list[pydicom.Dataset]) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) in patient mm of the voxel centres in `mask`."""
    s, r, c = np.nonzero(mask)
    x0, y0 = float(ct[0].ImagePositionPatient[0]), float(ct[0].ImagePositionPatient[1])
    dy, dx = (float(v) for v in ct[0].PixelSpacing)
    z = np.array([float(d.ImagePositionPatient[2]) for d in ct])
    xyz = np.column_stack([x0 + c * dx, y0 + r * dy, z[s]])
    return xyz.min(axis=0), xyz.max(axis=0)
