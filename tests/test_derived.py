"""Tests for lungman_dicom.derived: body outline and lungs from a label volume."""

from __future__ import annotations

import numpy as np
import pytest
from lungman_fixture import LUNGMAN, needs_lungman

from lungman_dicom import convert as cv
from lungman_dicom.derived import body_mask, derive_regions

SKIN, VESSEL, TUMOUR, SPINE_HARD, SPINE_SOFT = 5, 1, 15, 16, 17
INFOS = [
    cv.LabelInfo(VESSEL, -463.0, "bronchioles"),
    cv.LabelInfo(SKIN, -12.0, "skin"),
    cv.LabelInfo(TUMOUR, 90.0, "tumours_100HU"),
    cv.LabelInfo(SPINE_HARD, 600.0, "spine-hard-650"),
    cv.LabelInfo(SPINE_SOFT, 300.0, "spine-soft-650"),
]


def _phantom() -> np.ndarray:
    """4 slices of 20 x 30: a soft-tissue block with two lung cavities either side of a spine column.

    The right cavity (low columns) holds a vessel voxel and a tumour voxel. A one-voxel unlabelled
    gap sits apart from both cavities.
    """
    lab = np.zeros((4, 20, 30), dtype=np.uint8)
    lab[:, 2:18, 2:28] = SKIN
    lab[:, 5:15, 5:12] = 0  # right lung cavity
    lab[:, 5:15, 18:25] = 0  # left lung cavity
    lab[:, 8:12, 14:16] = SPINE_HARD
    lab[:, 12, 14:16] = SPINE_SOFT
    lab[1, 7, 7] = VESSEL
    lab[1, 9, 9] = TUMOUR
    lab[2, 16, 14] = 0  # isolated gap, not lung
    return lab


class TestBodyMask:
    def test_fills_enclosed_space_and_nothing_outside(self) -> None:
        body = body_mask(_phantom(), [])
        assert body[:, 2:18, 2:28].all()
        assert int(body.sum()) == 4 * 16 * 26

    def test_the_couch_and_the_gap_it_encloses_are_outside_the_body(self) -> None:
        lab = _phantom()
        couch = 12
        lab[:, 18, :] = couch  # a full-width layer touching the block
        lab[:, 19, 0] = couch
        lab[:, 19, 29] = couch  # with row 18, closes an air gap along row 19 against the image edge
        assert int(body_mask(lab, [couch]).sum()) == 4 * 16 * 26
        assert int(body_mask(lab, []).sum()) > 4 * 16 * 26


class TestDeriveRegions:
    def test_lungs_are_the_cavities_with_vessels_and_without_the_tumour(self) -> None:
        regions = derive_regions(_phantom(), INFOS)
        assert regions.lung_right[1, 7, 7]  # the vessel voxel is lung
        assert not regions.lungs[1, 9, 9]  # the tumour voxel is not
        assert int(regions.lung_left.sum()) == 4 * 10 * 7

    def test_two_unconnected_lungs_are_both_kept_and_a_small_gap_is_dropped(
        self,
    ) -> None:
        regions = derive_regions(_phantom(), INFOS)
        assert not regions.lungs[2, 16, 14]
        assert int(regions.lung_right.sum()) == 4 * 10 * 7 - 1  # the right cavity less its tumour voxel

    @pytest.mark.parametrize(("extra", "kept"), [(3, False), (4, True)])
    def test_a_region_is_kept_from_a_tenth_of_the_largest(self, extra: int, kept: bool) -> None:
        lab = _phantom()
        lab[:, 5:15, 5:12] = SKIN  # one cavity of 280 voxels remains, so a tenth is 28
        lab[0, 16, 3:27] = 0  # 24 voxels, enclosed by soft tissue
        lab[1, 16, 3 : 3 + extra] = 0  # 27 or 28 in all, joined through the slice below
        regions = derive_regions(lab, INFOS)
        assert bool(regions.lungs[0, 16, 10]) is kept
        assert int(regions.lungs.sum()) == 280 + (24 + extra if kept else 0)

    def test_left_is_the_high_x_side_and_the_two_do_not_overlap(self) -> None:
        regions = derive_regions(_phantom(), INFOS)
        assert regions.midline_column in (14, 15)
        assert np.nonzero(regions.lung_left)[2].min() > regions.midline_column
        assert np.nonzero(regions.lung_right)[2].max() <= regions.midline_column
        assert not (regions.lung_left & regions.lung_right).any()

    def test_couch_labels_are_left_out_of_the_body(self) -> None:
        lab = _phantom()
        lab[:, 18, :] = 12
        infos = [*INFOS, cv.LabelInfo(12, 204.0, "sheets_med")]
        assert not derive_regions(lab, infos).body[:, 18, :].any()

    def test_left_follows_the_x_direction(self) -> None:
        flipped = derive_regions(_phantom(), INFOS, x_increases_with_column=False)
        assert np.nonzero(flipped.lung_left)[2].max() < flipped.midline_column

    @pytest.mark.parametrize("dropped", ["bronchioles", "spine-hard-650"])
    def test_refuses_when_a_needed_label_is_missing(self, dropped: str) -> None:
        with pytest.raises(cv.LungmanConversionError, match=dropped):
            derive_regions(_phantom(), [i for i in INFOS if i.name != dropped])


@needs_lungman
class TestRealData:
    def test_lungs_are_low_density_and_hold_no_insert_but_the_vessels(self) -> None:
        ct = cv.read_ct_series(LUNGMAN / "CD1" / "DICOM" / "ST000000" / "SE000000")
        infos = cv.read_labels(LUNGMAN / "SEGMENTATION" / "labels.dat")
        labels = cv.read_label_volume(LUNGMAN / "SEGMENTATION", ct)
        regions = derive_regions(labels, infos)
        hu = cv.hu_volume(ct)
        value = {i.name: i.value for i in infos}
        assert set(np.unique(labels[regions.lungs]).tolist()) <= {
            0,
            value["bronchioles"],
        }
        assert float(np.median(hu[regions.lungs])) < -900
        voxel_cm3 = 0.625 * 0.625 * 0.7 / 1000
        left, right = (
            regions.lung_left.sum() * voxel_cm3,
            regions.lung_right.sum() * voxel_cm3,
        )
        assert 1500 < left < 3000 and 1500 < right < 3000
        heart = labels == value["heart"]
        assert regions.body[heart].all()
        couch = np.isin(labels, [i.value for i in infos if i.name.startswith("sheets")])
        assert couch.any() and not regions.body[couch].any()
        assert np.nonzero(heart)[2].mean() > regions.midline_column  # the heart lies to the patient's left
