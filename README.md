# lungman-autoseg

This converts the Lungman chest phantom's label volume into a DICOM RT Structure Set on its CT. The structure set is used to test proton dose engines (MCsquare-portable against Open-MCsquare). The project is also the place to test OnkoDICOM's TotalSegmentator auto-contouring without CUDA.

Data: *CT scans, 3D segmentations, digital radiograph and 3D surfaces of the Lungman phantom*, Vidal & Tugwell-Allsup, 2024. Only CD1's CT series is segmented.

## Usage

    uv sync
    uv run python -m lungman_dicom ~/Downloads/lungman_data output/lungman_CD1_RTSTRUCT.dcm

The output has one ROI per label in `labels.dat`, plus one ROI per tumour component, largest first:
`tumours_100HU_1..3` (inserts around +90 HU) and `tumours_630HU_1..3` (around −660 HU).

## What the data does not tell you

- **The CT filenames are not in slice order.** `CT000000` is instance 158. `SG{i}.tiff` belongs to the slice with InstanceNumber `i+1`, which is descending z. This was established by sampling the CT inside each label. In instance order, the structures whose `labels.dat` HU is a measurement agree within about 3 HU (trachea, heart, bronchus, diaphragm, skin, sheets_low). Every other ordering or in-plane flip is off by hundreds. `check_alignment` repeats the test on every run, and nothing is written if it fails.
- **The bone HU values in `labels.dat` are assigned, not measured.** They are exactly 600.357 or 300.357, so they cannot be used for the alignment check.
- **The STL meshes are not in patient coordinates.** They are in mm, centred on the volume, with z reversed into slice order. From the bounding boxes, STL ≈ (x − 8.0, y + 134.2, −(z + 189.25)). The label volume is the source of truth and is already on the CT grid.
- **Contours run through the centres of boundary pixels, and those pixels belong to the structure.** A consumer that fills polygons strictly will under-fill structures one or two pixels thick: sternum-soft drops to Dice 0.88 that way. The tumour ROIs are exact either way.

## Verification

`lungman_dicom.verify` reads the RTSTRUCT with pydicom, without rt-utils. It rasterises every contour onto the CT grid and compares the result with the label masks. On 2026-10-03, all 19 labels and all 6 tumour components matched voxel for voxel (Dice 1.0000).
