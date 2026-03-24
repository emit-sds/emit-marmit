#!/usr/bin/env python
"""
Apply a binary soil mask to an ENVI reflectance file.

Non-soil pixels (mask == 0) are set to zero in the output.

Usage:
    python apply_mask.py <input_reflectance.hdr> <mask.hdr> <output.hdr>

Author: Nayma Binte Nur
"""

import sys
import os
import numpy as np
import spectral.io.envi as envi


def apply_mask(input_hdr, mask_hdr, output_hdr):
    refl = envi.open(input_hdr).load()
    mask = envi.open(mask_hdr).load().squeeze().astype(bool)

    if refl.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Shape mismatch: reflectance {refl.shape[:2]} vs mask {mask.shape[:2]}"
        )

    masked = refl.copy()
    masked[~mask] = 0

    header = envi.read_envi_header(input_hdr)
    os.makedirs(os.path.dirname(os.path.abspath(output_hdr)), exist_ok=True)
    envi.save_image(output_hdr, masked, metadata=header, interleave="bsq", force=True)
    print(f"Masked reflectance saved to: {output_hdr.replace('.hdr', '')}")
    print(f"  Valid pixels: {mask.sum():,} / {mask.size:,}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    apply_mask(sys.argv[1], sys.argv[2], sys.argv[3])