"""Tests for lungman_dicom.reduce: fewer points, bounded deviation, nothing collapses."""

from __future__ import annotations

import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from lungman_dicom.reduce import reduce_rtstruct, simplify_contour

Z = -243.4


def _staircase(steps: int = 20, pitch: float = 0.625) -> np.ndarray:
    """A pixel-traced diagonal edge closed into a triangle: every step is a corner."""
    pts = []
    for k in range(steps):
        pts += [(k * pitch, k * pitch), ((k + 1) * pitch, k * pitch)]
    pts += [(steps * pitch, steps * pitch), (0.0, steps * pitch)]
    return np.column_stack([np.array(pts), np.full(len(pts), Z)])


def _max_deviation(original: np.ndarray, reduced: np.ndarray) -> float:
    """Largest distance from an original point to the closed reduced polyline."""
    segs = np.vstack([reduced[:, :2], reduced[:1, :2]])
    worst = 0.0
    for p in original[:, :2]:
        best = np.inf
        for a, b in zip(segs[:-1], segs[1:]):
            ab = b - a
            t = 0.0 if not ab.any() else np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
            best = min(best, float(np.linalg.norm(p - (a + t * ab))))
        worst = max(worst, best)
    return worst


class TestSimplifyContour:
    def test_staircase_shrinks_within_tolerance(self) -> None:
        stair = _staircase()
        reduced = simplify_contour(stair, 0.5)
        assert len(reduced) < len(stair) / 4
        assert _max_deviation(stair, reduced) <= 0.5 + 1e-9

    def test_tolerance_below_half_pitch_keeps_the_staircase(self) -> None:
        stair = _staircase()
        # A 0.625 mm staircase deviates ~0.22 mm from its diagonal, so 0.1 mm cannot flatten it.
        assert len(simplify_contour(stair, 0.1)) > len(stair) / 2

    def test_square_keeps_its_four_corners(self) -> None:
        side = np.linspace(0, 10, 17)
        sq = np.array([(x, 0) for x in side[:-1]] + [(10, y) for y in side[:-1]]
                      + [(x, 10) for x in side[::-1][:-1]] + [(0, y) for y in side[::-1][:-1]])
        sq = np.column_stack([sq, np.full(len(sq), Z)])
        reduced = simplify_contour(sq, 0.1)
        assert len(reduced) == 4
        assert {tuple(p) for p in reduced[:, :2]} == {(0, 0), (10, 0), (10, 10), (0, 10)}

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_tiny_contours_are_untouched(self, n: int) -> None:
        pts = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.full(n, Z)])
        assert np.array_equal(simplify_contour(pts, 5.0), pts)

    def test_never_collapses_below_a_triangle(self) -> None:
        sliver = np.array([[0, 0, Z], [10, 0.01, Z], [20, 0, Z], [10, -0.01, Z]], dtype=float)
        assert len(simplify_contour(sliver, 1.0)) >= 3

    def test_z_is_preserved(self) -> None:
        assert np.all(simplify_contour(_staircase(), 0.5)[:, 2] == Z)

    def test_zero_tolerance_changes_nothing(self) -> None:
        stair = _staircase()
        assert np.array_equal(simplify_contour(stair, 0.0), stair)

    def test_negative_tolerance_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            simplify_contour(_staircase(), -0.1)


def _rtstruct(contours: list[np.ndarray]) -> Dataset:
    roi = Dataset()
    roi.ReferencedROINumber = 1
    roi.ContourSequence = Sequence()
    for pts in contours:
        c = Dataset()
        c.ContourGeometricType = "CLOSED_PLANAR"
        c.NumberOfContourPoints = len(pts)
        c.ContourData = [float(v) for v in pts.ravel()]
        roi.ContourSequence.append(c)
    ds = Dataset()
    ds.ROIContourSequence = Sequence([roi])
    return ds


class TestReduceRtstruct:
    def test_counts_and_point_numbers_stay_consistent(self) -> None:
        ds = _rtstruct([_staircase(), _staircase(5)])
        stats = reduce_rtstruct(ds, 0.5, 3)
        assert stats.contours == 2
        assert stats.points_before == len(_staircase()) + len(_staircase(5))
        assert stats.points_after < stats.points_before
        for c in ds.ROIContourSequence[0].ContourSequence:
            assert int(c.NumberOfContourPoints) * 3 == len(c.ContourData)

    def test_coordinates_written_to_three_decimals(self) -> None:
        pts = np.array([[1.23456789, -2.000049, Z], [3.3333333, 4.0, Z], [0.0001, 7.1, Z]])
        ds = _rtstruct([pts])
        reduce_rtstruct(ds, 0.1, 3)
        values = [str(v) for v in ds.ROIContourSequence[0].ContourSequence[0].ContourData]
        assert values[:6] == ["1.235", "-2", "-243.4", "3.333", "4", "-243.4"]
        assert values[6] == "0"  # 0.0001 rounds to zero, written without a sign
        assert all(len(v.split(".")[-1]) <= 3 for v in values if "." in v)

    @pytest.mark.parametrize("decimals", [-1, 7])
    def test_out_of_range_decimals_refused(self, decimals: int) -> None:
        with pytest.raises(ValueError, match="decimals"):
            reduce_rtstruct(_rtstruct([_staircase()]), 0.1, decimals)
