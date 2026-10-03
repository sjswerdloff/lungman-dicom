"""Command line: python -m lungman_dicom <lungman_data dir> <output RTSTRUCT path> [--full] [--tolerance-mm MM]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lungman_dicom.convert import LungmanConversionError, convert
from lungman_dicom.reduce import DEFAULT_TOLERANCE_MM


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert the Lungman label volume to a DICOM RTSTRUCT on CD1's CT.")
    parser.add_argument("lungman_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-split-tumours", action="store_true", help="do not add one ROI per tumour component")
    parser.add_argument("--full", action="store_true", help="keep every traced contour point at full precision (large file)")
    parser.add_argument("--tolerance-mm", type=float, default=DEFAULT_TOLERANCE_MM,
                        help=f"maximum deviation when simplifying contours (default {DEFAULT_TOLERANCE_MM} mm)")
    args = parser.parse_args()
    try:
        diffs, stats = convert(args.lungman_dir, args.output, split_tumours=not args.no_split_tumours,
                               full=args.full, tolerance_mm=args.tolerance_mm)
    except LungmanConversionError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}; alignment |dHU| " + ", ".join(f"{k} {v:.1f}" for k, v in diffs.items()))
    if stats is not None:
        print(f"contour points {stats.points_before} -> {stats.points_after} across {stats.contours} contours")
    return 0


if __name__ == "__main__":
    sys.exit(main())
