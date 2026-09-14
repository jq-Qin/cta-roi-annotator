from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CTA lower-limb ROI binary-mask annotator")
    parser.add_argument("--image-dir", type=Path, help="Folder containing CTA slices")
    parser.add_argument("--mask-dir", type=Path, help="Separate output folder for masks")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.image_dir is None) != (args.mask_dir is None):
        raise SystemExit("--image-dir and --mask-dir must be specified together")
    from .ui import run_app

    return run_app(args.image_dir, args.mask_dir)


if __name__ == "__main__":
    raise SystemExit(main())

