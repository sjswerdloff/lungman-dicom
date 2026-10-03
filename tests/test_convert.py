"""Tests for lungman_dicom.convert: every refusal is a way a wrong structure set could be written."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from lungman_dicom import convert as cv

LUNGMAN = Path.home() / "Downloads" / "lungman_data"


def _write_ct(
    directory: Path,
    filename: str,
    instance: int,
    z: float,
    series_uid: str,
    orientation: list[float] | None = None,
    pixels: np.ndarray | None = None,
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(directory / filename), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.SeriesInstanceUID = series_uid
    ds.InstanceNumber = instance
    ds.ImagePositionPatient = [0.0, 0.0, z]
    ds.ImageOrientationPatient = orientation or [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    data = pixels if pixels is not None else np.zeros((4, 4), dtype=np.uint16)
    ds.Rows, ds.Columns = data.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.PixelData = data.tobytes()
    ds.save_as(directory / filename, enforce_file_format=True)


def _series(tmp_path: Path, instances_and_z: list[tuple[int, float]], **kwargs) -> Path:
    uid = generate_uid()
    # Filenames deliberately NOT in instance order, as in the real data.
    for n, (instance, z) in enumerate(reversed(instances_and_z)):
        _write_ct(tmp_path, f"CT{n:06d}", instance, z, uid, **kwargs)
    return tmp_path


class TestReadCtSeries:
    def test_returns_instance_order_not_filename_order(self, tmp_path: Path) -> None:
        ct = cv.read_ct_series(_series(tmp_path, [(1, -10.0), (2, -10.7), (3, -11.4)]))
        assert [int(d.InstanceNumber) for d in ct] == [1, 2, 3]

    def test_refuses_a_gap_in_instance_numbers(self, tmp_path: Path) -> None:
        with pytest.raises(cv.LungmanConversionError, match="1..N"):
            cv.read_ct_series(_series(tmp_path, [(1, -10.0), (3, -11.4)]))

    def test_refuses_non_uniform_spacing(self, tmp_path: Path) -> None:
        with pytest.raises(cv.LungmanConversionError, match="not uniform"):
            cv.read_ct_series(_series(tmp_path, [(1, -10.0), (2, -10.7), (3, -12.0)]))

    def test_refuses_duplicate_positions(self, tmp_path: Path) -> None:
        with pytest.raises(cv.LungmanConversionError, match="not uniform"):
            cv.read_ct_series(_series(tmp_path, [(1, -10.0), (2, -10.0)]))

    def test_refuses_oblique_orientation(self, tmp_path: Path) -> None:
        with pytest.raises(cv.LungmanConversionError, match="ImageOrientationPatient"):
            cv.read_ct_series(_series(tmp_path, [(1, 0.0), (2, 1.0)], orientation=[1, 0, 0, 0, 0.9, 0.1]))

    def test_refuses_two_series(self, tmp_path: Path) -> None:
        _write_ct(tmp_path, "a", 1, 0.0, generate_uid())
        _write_ct(tmp_path, "b", 2, 1.0, generate_uid())
        with pytest.raises(cv.LungmanConversionError, match="more than one series"):
            cv.read_ct_series(tmp_path)

    def test_refuses_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(cv.LungmanConversionError, match="no CT"):
            cv.read_ct_series(tmp_path)


class TestReadLabels:
    def test_parses_tab_and_space_separated_rows_and_strips_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "labels.dat"
        f.write_text("14 \t-662.0720\ttumours_630HU.mha\n15\t90.357100\ttumours_100HU.mha\n\n")
        assert cv.read_labels(f) == [cv.LabelInfo(14, -662.072, "tumours_630HU"), cv.LabelInfo(15, 90.3571, "tumours_100HU")]

    def test_refuses_a_malformed_row(self, tmp_path: Path) -> None:
        f = tmp_path / "labels.dat"
        f.write_text("14 -662.0\n")
        with pytest.raises(cv.LungmanConversionError, match="expected 3 fields"):
            cv.read_labels(f)

    def test_refuses_duplicate_values(self, tmp_path: Path) -> None:
        f = tmp_path / "labels.dat"
        f.write_text("3 -915.5 trachea.mha\n3 28.3 heart.mha\n")
        with pytest.raises(cv.LungmanConversionError, match="duplicate"):
            cv.read_labels(f)


class TestCheckAlignment:
    INFOS = [cv.LabelInfo(3, -915.5, "trachea"), cv.LabelInfo(6, 28.3, "heart"), cv.LabelInfo(2, -25.9, "bronchus")]

    def _volumes(self) -> tuple[np.ndarray, np.ndarray]:
        labels = np.zeros((2, 4, 4), dtype=np.int16)
        labels[0, 0, :] = 3
        labels[0, 1, :] = 6
        labels[1, 0, :] = 2
        hu = np.zeros((2, 4, 4), dtype=np.float32)
        hu[labels == 3], hu[labels == 6], hu[labels == 2] = -915.5, 28.3, -25.9
        return hu, labels

    def test_accepts_aligned_volumes(self) -> None:
        hu, labels = self._volumes()
        diffs = cv.check_alignment(hu, labels, self.INFOS)
        assert max(diffs.values()) < 1e-3

    def test_refuses_reversed_slice_order(self) -> None:
        hu, labels = self._volumes()
        with pytest.raises(cv.LungmanConversionError, match="not aligned"):
            cv.check_alignment(hu[::-1], labels, self.INFOS)

    def test_tolerance_boundary(self) -> None:
        hu, labels = self._volumes()
        hu[labels == 3] += 10.0  # exactly at tolerance: accepted
        cv.check_alignment(hu, labels, self.INFOS)
        hu[labels == 3] += 0.01
        with pytest.raises(cv.LungmanConversionError, match="trachea"):
            cv.check_alignment(hu, labels, self.INFOS)

    def test_refuses_missing_alignment_label(self) -> None:
        hu, labels = self._volumes()
        with pytest.raises(cv.LungmanConversionError, match="missing"):
            cv.check_alignment(hu, labels, self.INFOS[:2])


@pytest.mark.skipif(not (LUNGMAN / "SEGMENTATION" / "labels.dat").exists(), reason="Lungman data not on this host")
class TestRealData:
    """The real archive: filename order is wrong, instance order is right."""

    def test_instance_order_passes_and_filename_order_fails(self) -> None:
        ct = cv.read_ct_series(LUNGMAN / "CD1" / "DICOM" / "ST000000" / "SE000000")
        infos = cv.read_labels(LUNGMAN / "SEGMENTATION" / "labels.dat")
        labels = cv.read_label_volume(LUNGMAN / "SEGMENTATION", ct)
        hu = cv.hu_volume(ct)
        assert max(cv.check_alignment(hu, labels, infos).values()) < 5.0
        by_filename = sorted(range(len(ct)), key=lambda i: Path(ct[i].filename).name)
        with pytest.raises(cv.LungmanConversionError, match="not aligned"):
            cv.check_alignment(hu[by_filename], labels, infos)
