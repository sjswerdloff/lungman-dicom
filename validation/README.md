# Reader comparison: which voxels does a receiver recover from the RTSTRUCT?

rt-utils traces each outline through the CENTRES of a structure's edge pixels. Whether a receiver
gets the labelled voxels back depends on whether it counts the pixels an outline passes through.
DICOM PS3.3 C.8.8.6.3 says: "Points in space lying along the path defined by the contour are
considered to be part of the ROI."

## Method

Two readers rasterise the same RTSTRUCT onto the CT grid, and each mask is compared with the
Lungman label volume, voxel for voxel. "Volume" is reader voxels divided by reference voxels.

- **OpenTPS** (commit 78850535): its own `readDicomCT`, `readDicomStruct` and
  `ROIContour.getBinaryMask(ct.origin, ct.gridSize, ct.spacing)` are run. It fills with PIL
  `polygon(outline=1, fill=1)`, so pixels the outline passes through count, and combines the
  contours on a plane by XOR.
- **Strict** is the rule in dicompyler-core 0.5.7 `dvhcalc.get_contour_mask`, which OnkoDICOM's DVH
  uses: `matplotlib.path.Path(contour).contains_points(pixel centres)`, XOR per plane. The rule is
  reproduced in `compare_readers.py` over each contour's bounding box, not called, because the
  original tests the whole grid for every contour. matplotlib 3.10.8.

Run (2026-10-04, macOS arm64; Pillow 12.1.0, numpy 2.4.1, scikit-image 0.26.0):

    python validation/dump_reference.py <lungman_data> <work>          # lungman-dicom environment
    python validation/compare_readers.py <work> <RTSTRUCT> [...]       # opentps_core + matplotlib

The files were written by `python -m lungman_dicom` at commit 9712e60: `--full` (every traced
point, full precision) and the default (0.35 mm Douglas-Peucker, 0.001 mm coordinates).

## Results: `--full`

| ROI | Reference voxels | OpenTPS Dice | OpenTPS volume | Strict Dice | Strict volume |
|---|---|---|---|---|---|
| bronchioles | 1272807 | 0.9573 | 0.918 | 0.5150 | 0.347 |
| bronchus | 285769 | 0.9977 | 0.995 | 0.7345 | 0.580 |
| trachea | 165315 | 1.0000 | 1.000 | 0.9429 | 0.892 |
| diaphram | 4272090 | 1.0000 | 1.000 | 0.9928 | 0.986 |
| skin | 29718041 | 0.9825 | 0.966 | 0.9768 | 0.955 |
| heart | 2703185 | 1.0000 | 1.000 | 0.9824 | 0.966 |
| sheets_low | 535605 | 0.9952 | 0.990 | 0.6630 | 0.496 |
| sheets_med | 1287855 | 1.0000 | 1.000 | 0.8829 | 0.790 |
| sheets_high | 293335 | 1.0000 | 1.000 | 0.7615 | 0.615 |
| tumours_630HU | 3783 | 1.0000 | 1.000 | 0.8619 | 0.757 |
| tumours_630HU_1 | 1737 | 1.0000 | 1.000 | 0.8750 | 0.778 |
| tumours_630HU_2 | 1440 | 1.0000 | 1.000 | 0.8719 | 0.773 |
| tumours_630HU_3 | 606 | 1.0000 | 1.000 | 0.7964 | 0.662 |
| tumours_100HU | 2932 | 1.0000 | 1.000 | 0.8571 | 0.750 |
| tumours_100HU_1 | 1792 | 1.0000 | 1.000 | 0.8765 | 0.780 |
| tumours_100HU_2 | 815 | 1.0000 | 1.000 | 0.8399 | 0.724 |
| tumours_100HU_3 | 325 | 1.0000 | 1.000 | 0.7873 | 0.649 |
| spine-hard-650 | 296051 | 0.9843 | 0.969 | 0.3890 | 0.241 |
| spine-soft-650 | 3006429 | 0.9239 | 0.858 | 0.8529 | 0.744 |
| scaps-hard-550 | 69844 | 0.9708 | 0.943 | 0.4205 | 0.266 |
| scaps-soft-550 | 1014279 | 0.9481 | 0.901 | 0.8824 | 0.790 |
| sternum-hard-550 | 32241 | 0.9961 | 0.992 | 0.3805 | 0.235 |
| sternum-soft-550 | 189886 | 0.8772 | 0.781 | 0.8588 | 0.753 |
| clavicle-hard-700 | 27861 | 0.9581 | 0.920 | 0.6171 | 0.446 |
| clavicle-soft-700 | 107163 | 0.8824 | 0.790 | 0.7752 | 0.633 |

OpenTPS recovers every tumour exactly. The strict reader loses the edge ring: tumours come back at
0.65 to 0.78 of their volume and hard bone at 0.23 to 0.45.

## Results: default (0.35 mm)

| ROI | Reference voxels | OpenTPS Dice | OpenTPS volume | Strict Dice | Strict volume |
|---|---|---|---|---|---|
| bronchioles | 1272807 | 0.7892 | 0.880 | 0.6832 | 0.519 |
| bronchus | 285769 | 0.8858 | 0.957 | 0.8252 | 0.702 |
| trachea | 165315 | 0.9752 | 0.994 | 0.9664 | 0.935 |
| diaphram | 4272090 | 0.9985 | 0.997 | 0.9959 | 0.992 |
| skin | 29718041 | 0.9799 | 0.965 | 0.9845 | 0.970 |
| heart | 2703185 | 0.9911 | 0.995 | 0.9898 | 0.980 |
| sheets_low | 535605 | 0.7530 | 0.901 | 0.7430 | 0.591 |
| sheets_med | 1287855 | 0.8853 | 0.957 | 0.9049 | 0.826 |
| sheets_high | 293335 | 0.8080 | 0.903 | 0.8248 | 0.702 |
| tumours_630HU | 3783 | 0.9581 | 0.974 | 0.9123 | 0.839 |
| tumours_630HU_1 | 1737 | 0.9640 | 0.969 | 0.9151 | 0.843 |
| tumours_630HU_2 | 1440 | 0.9630 | 0.970 | 0.9234 | 0.858 |
| tumours_630HU_3 | 606 | 0.9975 | 0.998 | 0.8767 | 0.781 |
| tumours_100HU | 2932 | 0.9787 | 0.983 | 0.9080 | 0.832 |
| tumours_100HU_1 | 1792 | 0.9969 | 0.994 | 0.9211 | 0.854 |
| tumours_100HU_2 | 815 | 0.8953 | 0.968 | 0.8927 | 0.806 |
| tumours_100HU_3 | 325 | 0.9354 | 0.954 | 0.8715 | 0.772 |
| spine-hard-650 | 296051 | 0.8159 | 0.923 | 0.5279 | 0.359 |
| spine-soft-650 | 3006429 | 0.9076 | 0.867 | 0.9228 | 0.857 |
| scaps-hard-550 | 69844 | 0.6711 | 0.934 | 0.5432 | 0.373 |
| scaps-soft-550 | 1014279 | 0.9012 | 0.895 | 0.9385 | 0.884 |
| sternum-hard-550 | 32241 | 0.6563 | 0.977 | 0.5847 | 0.413 |
| sternum-soft-550 | 189886 | 0.8820 | 0.793 | 0.9071 | 0.830 |
| clavicle-hard-700 | 27861 | 0.8139 | 0.882 | 0.7765 | 0.635 |
| clavicle-soft-700 | 107163 | 0.8429 | 0.798 | 0.8788 | 0.784 |

Simplified outlines no longer sit on pixel centres. OpenTPS fits its fill grid to the outline's own
extent and then resamples onto the CT, so its masks shift: Dice drops although volume stays close.

## Limits

- One phantom, one CT grid (0.625 mm pixels), one machine.
- The strict column reproduces dicompyler-core's rule; it is not a run of dicompyler-core or OnkoDICOM.
- Pixel centres lying exactly on the path are the deciding case, so the strict result may differ
  between matplotlib versions (3.10.8 here).
