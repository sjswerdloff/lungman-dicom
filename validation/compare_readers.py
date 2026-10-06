"""Rasterise Lungman RTSTRUCT variants the way OpenTPS and dicompyler-core (OnkoDICOM's DVH) do; compare to labels.

OpenTPS: its own readDicomCT / readDicomStruct / ROIContour.getBinaryMask on the CT grid (run, not copied).
dicompyler-core: get_contour_mask's rule (matplotlib Path.contains_points on pixel centres, XOR per plane),
evaluated only over each contour's bounding box, which gives the same pixels as the full-grid call.
The rule is reproduced here, not called: get_contour_mask tests every dose-grid point per contour,
which is too slow for 240,000 contours.

Usage (in an environment with opentps_core and matplotlib):
    python validation/compare_readers.py <work dir from dump_reference.py> <RTSTRUCT> [<RTSTRUCT> ...]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pydicom
from matplotlib.path import Path as MplPath
from skimage.measure import label as connected_components

from opentps.core.io.dicomIO import readDicomCT, readDicomStruct

work = Path(sys.argv[1])
variants = sys.argv[2:]
ref = np.load(work / "ref.npz")
meta = json.loads((work / "ref.json").read_text())
labels, zs = ref["labels"], ref["z"]

ct = readDicomCT(meta["ct_files"])
ox, oy, oz = (float(v) for v in ct.origin)
sx, sy, sz = (float(v) for v in ct.spacing)
nx, ny, nz = (int(v) for v in ct.gridSize)
# reference slice for each OpenTPS z index
k_of = {int(round((z - oz) / sz)): s for s, z in enumerate(zs)}
assert sorted(k_of) == list(range(nz)), "CT z grid does not match the label slices"
order = np.array([k_of[k] for k in range(nz)])


def ref_xyz(mask_srs: np.ndarray) -> np.ndarray:
    """(slice, row, col) in instance order -> OpenTPS (x, y, z) ascending z."""
    return np.transpose(mask_srs[order], (2, 1, 0))


refs = {}
for roi in meta["rois"]:
    m = labels == roi["value"]
    refs[roi["name"]] = m
    if roi["name"] in meta["tumours"]:
        comps = connected_components(m, connectivity=3)
        sizes = np.bincount(comps.ravel())[1:]
        for rank, comp in enumerate(np.argsort(-sizes) + 1, 1):
            refs[f"{roi['name']}_{rank}"] = comps == comp


def strict_mask(ds: pydicom.Dataset, roi_number: int) -> np.ndarray:
    mask = np.zeros((nx, ny, nz), dtype=bool)
    roi = next(
        r for r in ds.ROIContourSequence if int(r.ReferencedROINumber) == roi_number
    )
    for c in getattr(roi, "ContourSequence", []):
        p = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
        k = int(round((p[0, 2] - oz) / sz))
        i0 = max(int(np.floor((p[:, 0].min() - ox) / sx)) - 1, 0)
        i1 = min(int(np.ceil((p[:, 0].max() - ox) / sx)) + 2, nx)
        j0 = max(int(np.floor((p[:, 1].min() - oy) / sy)) - 1, 0)
        j1 = min(int(np.ceil((p[:, 1].max() - oy) / sy)) + 2, ny)
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        pts = np.column_stack([ox + ii.ravel() * sx, oy + jj.ravel() * sy])
        inside = MplPath(p[:, :2]).contains_points(pts).reshape(ii.shape)
        mask[i0:i1, j0:j1, k] ^= inside
    return mask


def dice(a: np.ndarray, b: np.ndarray) -> float:
    t = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / t) if t else 1.0


for path in variants:
    ds = pydicom.dcmread(path)
    struct = readDicomStruct(path)
    numbers = {r.ROIName: int(r.ROINumber) for r in ds.StructureSetROISequence}
    print(f"=== {Path(path).name}")
    print(
        f"{'ROI':22s} {'ref':>9s} | {'OpenTPS dice':>12s} {'vol':>6s} | {'strict dice':>11s} {'vol':>6s}"
    )
    t = time.time()
    for name, m in refs.items():
        r = ref_xyz(m)
        otps = (
            struct.getContourByName(name)
            .getBinaryMask(ct.origin, ct.gridSize, ct.spacing)
            .imageArray.astype(bool)
        )
        st = strict_mask(ds, numbers[name])
        print(
            f"{name:22s} {int(r.sum()):9d} | {dice(otps, r):12.4f} {otps.sum() / r.sum():6.3f} |"
            f" {dice(st, r):11.4f} {st.sum() / r.sum():6.3f}",
            flush=True,
        )
    print(f"({time.time() - t:.0f}s)")
