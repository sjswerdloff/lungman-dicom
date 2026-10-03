"""Command line: python -m lungman_dicom <lungman_data dir> <output RTSTRUCT path> [--no-split-tumours]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lungman_dicom.convert import LungmanConversionError, convert


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert the Lungman label volume to a DICOM RTSTRUCT on CD1's CT.")
    parser.add_argument("lungman_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-split-tumours", action="store_true", help="do not add one ROI per tumour component")
    args = parser.parse_args()
    try:
        diffs = convert(args.lungman_dir, args.output, split_tumours=not args.no_split_tumours)
    except LungmanConversionError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}; alignment |dHU| " + ", ".join(f"{k} {v:.1f}" for k, v in diffs.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
