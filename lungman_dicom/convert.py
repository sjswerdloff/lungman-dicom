"""Convert the Lungman phantom label volume into a DICOM RT Structure Set.

The Lungman archive (Vidal & Tugwell-Allsup, 2024) ships CD1's CT series and a label volume:
one int16 TIFF per CT slice (SEGMENTATION/SG*.tiff) plus labels.dat
(label, mean HU, name). The STL meshes in MESHES/ were derived from these labels, so the
label volume is the source of truth and is already on the CT grid.

Slice correspondence: SG{i:06d}.tiff is the slice with InstanceNumber i+1. The CT FILENAMES
are not in slice order (CT000000 is instance 158), so filename order must never be used.
This was established by sampling the CT inside each label: in instance order the structures
whose labels.dat HU is a measurement (trachea, heart, bronchus, diaphragm, skin, sheets_low)
agree within ~3 HU, and every other ordering or in-plane flip is off by hundreds.
`check_alignment` repeats that test at run time and refuses to write on a mismatch.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import tifffile
from rt_utils import RTStructBuilder
from skimage.measure import label as connected_components

from lungman_dicom.reduce import DEFAULT_DECIMALS, DEFAULT_TOLERANCE_MM, ReductionStats, reduce_rtstruct

# Labels whose labels.dat HU was measured on the CT (the bone entries are assigned values,
# all exactly 600.357 or 300.357, and cannot be used to check alignment).
ALIGNMENT_LABELS: dict[str, float] = {"trachea": 10.0, "heart": 10.0, "bronchus": 10.0}

TUMOUR_LABELS = ("tumours_100HU", "tumours_630HU")


class LungmanConversionError(Exception):
    """Raised when the inputs are inconsistent and no structure set must be written."""


@dataclass(frozen=True)
class LabelInfo:
    """One row of labels.dat."""

    value: int
    mean_hu: float
    name: str


def read_labels(labels_path: Path) -> list[LabelInfo]:
    """Parse labels.dat: whitespace-separated label value, mean HU, file name (`.mha` stripped)."""
    rows: list[LabelInfo] = []
    for line_no, line in enumerate(labels_path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3:
            raise LungmanConversionError(f"{labels_path}:{line_no}: expected 3 fields, got {len(parts)}")
        rows.append(LabelInfo(int(parts[0]), float(parts[1]), parts[2].removesuffix(".mha")))
    if len({r.value for r in rows}) != len(rows):
        raise LungmanConversionError(f"{labels_path}: duplicate label values")
    return rows


def read_ct_series(ct_dir: Path) -> list[pydicom.Dataset]:
    """Read the CT series and return it in InstanceNumber order, validated.

    Raises:
        LungmanConversionError: if instances are not exactly 1..N, the series is mixed, the
            orientation is not axial identity, or slice spacing along the instance order is
            not uniform.
    """
    files = sorted(glob.glob(str(ct_dir / "*")))
    datasets = [pydicom.dcmread(f) for f in files if Path(f).is_file()]
    datasets = [d for d in datasets if getattr(d, "Modality", None) == "CT"]
    if not datasets:
        raise LungmanConversionError(f"no CT images in {ct_dir}")
    if len({d.SeriesInstanceUID for d in datasets}) != 1:
        raise LungmanConversionError(f"{ct_dir} holds more than one series")
    datasets.sort(key=lambda d: int(d.InstanceNumber))
    instances = [int(d.InstanceNumber) for d in datasets]
    if instances != list(range(1, len(datasets) + 1)):
        raise LungmanConversionError("InstanceNumbers are not exactly 1..N")
    for d in datasets:
        if [round(float(v), 6) for v in d.ImageOrientationPatient] != [1, 0, 0, 0, 1, 0]:
            raise LungmanConversionError(f"instance {d.InstanceNumber}: non-identity ImageOrientationPatient")
    z = np.array([float(d.ImagePositionPatient[2]) for d in datasets])
    steps = np.round(np.diff(z), 4)
    if len(set(steps.tolist())) != 1 or steps[0] == 0:
        raise LungmanConversionError(f"slice spacing along instance order is not uniform: {sorted(set(steps.tolist()))}")
    return datasets


def read_label_volume(seg_dir: Path, ct: list[pydicom.Dataset]) -> np.ndarray:
    """Stack SG*.tiff into a (slices, rows, cols) array indexed like `ct` (instance order)."""
    files = sorted(glob.glob(str(seg_dir / "SG*.tiff")))
    if len(files) != len(ct):
        raise LungmanConversionError(f"{len(files)} label slices for {len(ct)} CT slices")
    volume = np.stack([tifffile.imread(f) for f in files])
    expected = (len(ct), int(ct[0].Rows), int(ct[0].Columns))
    if volume.shape != expected:
        raise LungmanConversionError(f"label volume {volume.shape} does not match CT {expected}")
    return volume


def hu_volume(ct: list[pydicom.Dataset]) -> np.ndarray:
    """Return the CT in HU, (slices, rows, cols), instance order."""
    return np.stack(
        [d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept) for d in ct]
    )


def check_alignment(hu: np.ndarray, labels: np.ndarray, infos: list[LabelInfo]) -> dict[str, float]:
    """Verify the label volume sits on the CT by comparing measured and tabulated mean HU.

    Returns:
        The absolute HU difference for each alignment label.

    Raises:
        LungmanConversionError: if any alignment label is missing or exceeds its tolerance.
    """
    by_name = {i.name: i for i in infos}
    diffs: dict[str, float] = {}
    for name, tolerance in ALIGNMENT_LABELS.items():
        info = by_name.get(name)
        if info is None:
            raise LungmanConversionError(f"alignment label {name!r} missing from labels.dat")
        mask = labels == info.value
        if not mask.any():
            raise LungmanConversionError(f"alignment label {name!r} has no voxels")
        diffs[name] = abs(float(hu[mask].mean()) - info.mean_hu)
        if diffs[name] > tolerance:
            raise LungmanConversionError(
                f"{name}: CT mean differs from labels.dat by {diffs[name]:.1f} HU (> {tolerance}); "
                "label volume is not aligned with this CT"
            )
    return diffs


def build_rtstruct(
    ct_dir: Path,
    ct: list[pydicom.Dataset],
    labels: np.ndarray,
    infos: list[LabelInfo],
    split_tumours: bool = True,
):
    """Build an RTSTRUCT with one ROI per label and, optionally, one per tumour component.

    rt-utils takes a (rows, cols, slices) mask ordered like ITS OWN sorted series, so each
    slice is mapped by SOPInstanceUID -> InstanceNumber -> label slice rather than by
    assuming either order.
    """
    rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))
    index_by_uid = {d.SOPInstanceUID: int(d.InstanceNumber) - 1 for d in ct}
    order = [index_by_uid[s.SOPInstanceUID] for s in rtstruct.series_data]

    def to_rtutils(mask: np.ndarray) -> np.ndarray:
        return np.transpose(mask[order], (1, 2, 0))

    for info in infos:
        mask = labels == info.value
        if not mask.any():
            continue
        rtstruct.add_roi(mask=to_rtutils(mask), name=info.name)
        if split_tumours and info.name in TUMOUR_LABELS:
            components = connected_components(mask, connectivity=3)
            sizes = np.bincount(components.ravel())[1:]
            for rank, comp in enumerate(np.argsort(-sizes) + 1, 1):
                rtstruct.add_roi(mask=to_rtutils(components == comp), name=f"{info.name}_{rank}")
    return rtstruct


def convert(
    lungman_dir: Path,
    output: Path,
    split_tumours: bool = True,
    full: bool = False,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
) -> tuple[dict[str, float], ReductionStats | None]:
    """Convert CD1 + SEGMENTATION into an RTSTRUCT at `output`.

    By default every contour is simplified to within `tolerance_mm` and written to 0.001 mm (see
    reduce.py); `full=True` keeps every traced point at full precision.

    Returns:
        The alignment diffs, and the reduction stats (None when `full`).
    """
    ct_dir = lungman_dir / "CD1" / "DICOM" / "ST000000" / "SE000000"
    ct = read_ct_series(ct_dir)
    infos = read_labels(lungman_dir / "SEGMENTATION" / "labels.dat")
    labels = read_label_volume(lungman_dir / "SEGMENTATION", ct)
    diffs = check_alignment(hu_volume(ct), labels, infos)
    rtstruct = build_rtstruct(ct_dir, ct, labels, infos, split_tumours)
    stats = None if full else reduce_rtstruct(rtstruct.ds, tolerance_mm, DEFAULT_DECIMALS)
    output.parent.mkdir(parents=True, exist_ok=True)
    rtstruct.save(str(output))
    return diffs, stats
