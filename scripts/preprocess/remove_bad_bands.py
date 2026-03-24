"""
Remove atmospheric absorption bad bands from an ENVI reflectance file.

Reads bad band center wavelengths from a text file and removes the nearest
matching bands from the ENVI image, writing a cleaned output file.

Usage:
    python remove_bad_bands.py <input.hdr> <output.hdr> <bad_bands_list.txt>

Author: Nayma Binte Nur
"""

import numpy as np
import spectral
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=spectral.io.spyfile.NaNValueWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="spectral.io.envi")

spectral.settings.envi_support_nonlowercase_params = True


def get_band_indices_from_wavelengths(header_file, bad_wavelengths):
    metadata = spectral.envi.read_envi_header(header_file)
    wavelengths = np.array(metadata.get("wavelength", []), dtype=float)
    if wavelengths.size == 0:
        raise ValueError("No wavelength information found in header file.")
    return [np.abs(wavelengths - bw).argmin() for bw in bad_wavelengths], wavelengths


def remove_bad_bands(input_file, output_file, bad_bands_file):
    bad_wavelengths = []
    with open(bad_bands_file, 'r') as f:
        for line in f:
            try:
                bad_wavelengths.append(float(line.strip()))
            except ValueError:
                print(f"Skipping invalid entry: {line.strip()}")

    img = spectral.envi.open(input_file)
    data = img.load()
    if data.ndim != 3:
        raise ValueError(f"Unexpected image shape: {data.shape}")

    header_file = input_file.replace(".hdr", "") + ".hdr"
    bad_bands, wavelengths = get_band_indices_from_wavelengths(header_file, bad_wavelengths)
    max_bands = data.shape[2]
    bad_bands = sorted(set(b for b in bad_bands if 0 <= b < max_bands))
    good_bands = [i for i in range(max_bands) if i not in bad_bands]
    cleaned_data = data[:, :, good_bands].copy()
    refined_wavelengths = wavelengths[good_bands].tolist()

    original_dtype = data.dtype
    metadata = spectral.envi.read_envi_header(header_file)
    metadata["wavelength"] = [str(w) for w in refined_wavelengths]
    metadata["bands"] = len(good_bands)

    if "byte order" not in metadata:
        metadata["byte order"] = 0
    if "interleave" not in metadata:
        metadata["interleave"] = img.metadata.get("interleave", "bsq")
    if "data type" not in metadata:
        metadata["data type"] = str(spectral.io.envi._dtype_to_envi[original_dtype])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    spectral.envi.save_image(output_file, cleaned_data, dtype=original_dtype, interleave=metadata["interleave"], force=True)
    spectral.envi.write_envi_header(output_file.replace(".hdr", "") + ".hdr", metadata)


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    bad_bands_txt = sys.argv[3]
    remove_bad_bands(input_file, output_file, bad_bands_txt)