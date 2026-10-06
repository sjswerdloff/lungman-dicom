"""Regions the Lungman label volume does not carry, derived from it by stated rules.

The archive labels the inserts and the soft-tissue filler ("skin"), but not the lungs and not a body
outline. Planning needs both. These are DERIVED: they follow the rules below, not a label in the data.

- body: on each slice, the union of every label except the couch, with its holes filled. The lung
  air and any other unlabelled space enclosed by phantom material is inside it. The `sheets_*`
  labels are the CT couch (flat layers across the whole image width, behind the phantom), not the
  phantom, and are left out.
- lungs: the 3D-connected regions of body that are unlabelled or `bronchioles` (the vessel tree
  inside the lung air), keeping each region at least a tenth the size of the largest. Trachea,
  bronchus, tumours and every other insert are excluded. Smaller unlabelled regions (air gaps between
  phantom parts) are not lung and are dropped.
  The two lungs may or may not be connected to each other; the rule keeps both either way.
- left and right: the lungs split at the x of the spine's centroid; the midline column itself goes
  to the right lung, so the two together are the whole lung region. Patient x increases towards the
  patient's left in DICOM's patient coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from lungman_dicom.convert import LabelInfo, LungmanConversionError

VESSEL_LABEL = "bronchioles"
MIN_FRACTION_OF_LARGEST = 0.1
SPINE_LABELS = ("spine-hard-650", "spine-soft-650")
COUCH_PREFIX = "sheets"


@dataclass(frozen=True)
class DerivedRegions:
    """Masks on the label grid, (slices, rows, cols) in the label volume's own slice order."""

    body: np.ndarray
    lung_left: np.ndarray
    lung_right: np.ndarray
    midline_column: int

    @property
    def lungs(self) -> np.ndarray:
        """Both lungs."""
        both: np.ndarray = self.lung_left | self.lung_right
        return both


def body_mask(labels: np.ndarray, couch_values: list[int]) -> np.ndarray:
    """Fill the holes of the labelled region, less the couch, on each slice.

    A couch-labelled voxel is never body, even where phantom material encloses it (Lungman has one).
    """
    couch = np.isin(labels, couch_values)
    phantom = (labels > 0) & ~couch
    return np.stack([ndimage.binary_fill_holes(s) for s in phantom]) & ~couch


def derive_regions(labels: np.ndarray, infos: list[LabelInfo], x_increases_with_column: bool = True) -> DerivedRegions:
    """Derive the body outline and the two lungs from the label volume.

    Args:
        labels: (slices, rows, cols) label values, 0 where unlabelled.
        infos: the rows of labels.dat.
        x_increases_with_column: True for the identity row direction (1, 0, 0), which
            `read_ct_series` enforces.

    Raises:
        LungmanConversionError: if the vessel or spine labels are missing, or no lung region is found.
    """
    value = {i.name: i.value for i in infos}
    missing = [n for n in (VESSEL_LABEL, *SPINE_LABELS) if n not in value]
    if missing:
        raise LungmanConversionError(f"labels needed to derive the lungs are missing: {', '.join(missing)}")
    body = body_mask(labels, [i.value for i in infos if i.name.startswith(COUCH_PREFIX)])
    candidate = body & np.isin(labels, [0, value[VESSEL_LABEL]])
    components, count = ndimage.label(candidate)
    if count == 0:
        raise LungmanConversionError("no unlabelled region inside the body: cannot derive the lungs")
    sizes = np.bincount(components.ravel())[1:]
    kept = np.nonzero(sizes >= MIN_FRACTION_OF_LARGEST * sizes.max())[0] + 1
    lungs = np.isin(components, kept)

    spine_columns = np.nonzero(np.isin(labels, [value[n] for n in SPINE_LABELS]))[2]
    if spine_columns.size == 0:
        raise LungmanConversionError("the spine labels are empty: cannot place the midline")
    midline = round(float(spine_columns.mean()))
    left_side = np.zeros(labels.shape, dtype=bool)
    if x_increases_with_column:
        left_side[:, :, midline + 1 :] = True
    else:
        left_side[:, :, :midline] = True
    return DerivedRegions(body, lungs & left_side, lungs & ~left_side, midline)
