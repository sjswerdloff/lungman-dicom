"""Dump the Lungman reference (label volume, slice z, ROI definitions) for compare_readers.py.

Usage: python validation/dump_reference.py <lungman_data dir> <work dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

from lungman_dicom import convert as cv

LUNGMAN = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
ct_dir = LUNGMAN / "CD1" / "DICOM" / "ST000000" / "SE000000"
ct = cv.read_ct_series(ct_dir)
infos = cv.read_labels(LUNGMAN / "SEGMENTATION" / "labels.dat")
labels = cv.read_label_volume(LUNGMAN / "SEGMENTATION", ct)
z = np.array([float(d.ImagePositionPatient[2]) for d in ct])
np.savez_compressed(out / "ref.npz", labels=labels.astype(np.uint8), z=z)
meta = {
    "rois": [
        {"name": i.name, "value": i.value} for i in infos if (labels == i.value).any()
    ],
    "tumours": list(cv.TUMOUR_LABELS),
    "ct_files": [str(p) for p in sorted(ct_dir.iterdir()) if p.is_file()],
}
(out / "ref.json").write_text(json.dumps(meta))
print("ok", labels.shape, labels.max())
