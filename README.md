# lungman-dicom

This converts the Lungman chest phantom's label volume into a DICOM RT Structure Set on its CT, and derives the regions the labels lack (body outline, lungs). The structure set and masks are used to test proton dose engines and treatment planning tools.

**The phantom data is not in this repository.** It is *CT scans, 3D segmentations, digital radiograph and 3D surfaces of the Lungman phantom*, Vidal & Tugwell-Allsup, 2024, and is distributed under its own terms. Only CD1's CT series is segmented.

This is research software. It is not a medical device.

## Usage

    uv sync
    uv run python -m lungman_dicom ~/Downloads/lungman_data output/lungman_CD1_RTSTRUCT.dcm

By default the contours are simplified (every original point stays within 0.35 mm of the outline) and written to 0.001 mm, and the per-contour image references are left out; this takes the file from 167 MB to 55 MB. `--full` keeps every traced point, at full precision, with the references. **Use `--full` for OpenTPS**: it resamples simplified outlines and they shift (see `validation/README.md`).

The output has one ROI per label in `labels.dat`, plus one ROI per tumour component, largest first:
`tumours_100HU_1..3` (inserts around +90 HU) and `tumours_630HU_1..3` (around −660 HU).

## What the data does not tell you

- **The CT filenames are not in slice order.** `CT000000` is instance 158. `SG{i}.tiff` belongs to the slice with InstanceNumber `i+1`, which is descending z. This was established by sampling the CT inside each label. In instance order, the structures whose `labels.dat` HU is a measurement agree within about 3 HU (trachea, heart, bronchus, diaphragm, skin, sheets_low). Every other ordering or in-plane flip is off by hundreds. `check_alignment` repeats the test on every run, and nothing is written if it fails.
- **The bone HU values in `labels.dat` are assigned, not measured.** They are exactly 600.357 or 300.357, so they cannot be used for the alignment check.
- **The STL meshes are not in patient coordinates.** They are in mm, centred on the volume, with z reversed into slice order. From the bounding boxes, STL ≈ (x − 8.0, y + 134.2, −(z + 189.25)). The label volume is the source of truth and is already on the CT grid.
- **Contours run through the centres of boundary pixels, and those pixels belong to the structure.** A reader that counts only pixel centres strictly inside an outline loses the boundary ring of every structure: measured on the `--full` file, the tumour inserts come back at 0.65 to 0.78 of their volume and thin hard bone at 0.23 to 0.45. A reader that counts the pixels the outline passes through recovers the tumours exactly. `validation/README.md` has the method and the per-structure table.
- **`sheets_*` is the CT couch, and `spine-*` is the spine and the rib cage together.** There is no rib, lung, body or spinal cord label. `lungman_dicom.derived` derives the body outline and the lungs by stated rules.

## Verification

`lungman_dicom.verify` reads the RTSTRUCT with pydicom, without rt-utils. It rasterises every contour onto the CT grid, counting the pixels each outline passes through, and compares the result with the label masks. On 2026-10-03, with every traced point kept (today's `--full`), all 19 labels and all 6 tumour components matched voxel for voxel (Dice 1.0000). With the default simplification the lowest Dice is 0.963 (hard spine and ribs) and the tumours are at 0.991 or better; the table for all 25 is in [validation/README.md](validation/README.md).

## Development

    uv sync
    uv run pre-commit install
    uv run ruff check lungman_dicom tests validation && uv run mypy && uv run pytest

CI runs the same checks. The tests that need the phantom data read its location from the environment variable `LUNGMAN_DATA`. When the variable is not set they look in `~/Downloads/lungman_data` and skip if the data is not there. When it is set and the data is missing, the test run stops with an error. `uv sync --group autoseg` adds TotalSegmentator and SimpleITK, which nothing in the package needs.

## Licence

Apache 2.0; see [`LICENSE`](LICENSE).
