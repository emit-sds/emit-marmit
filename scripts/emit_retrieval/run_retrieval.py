"""
EMIT per-pixel MARMIT soil moisture retrieval.

Loads EMIT Level-2A reflectance and solar zenith angle, applies a wavelength
subset mask, and runs MARMIT spectral inversion in parallel across all valid
pixels. Outputs L (equivalent water layer thickness) and MEWT (L x epsilon)
maps in mm as ENVI and PNG files.

Usage: called from run_emit_retrieval.sh or directly via argparse.

Author: Nayma Binte Nur
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))

from marmit_inversion import *
from marmit_model import *

import numpy as np
import pandas as pd
import spectral.io.envi as envi
from scipy.interpolate import interp1d
from multiprocessing import Pool
import argparse
import matplotlib.pyplot as plt
import time
import os
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def find_envi_data_file(hdr_file):
    base_path = os.path.splitext(hdr_file)[0]
    possible_extensions = ['', '.dat', '.img', '.bsq', '.bil', '.bip']
    for ext in possible_extensions:
        data_file = base_path + ext
        if os.path.exists(data_file):
            return data_file
    raise FileNotFoundError(f"No data file found matching header: {hdr_file}")


def process_pixel(args):
    i, j, refl_meas_wet, R_soil_endmembers, alpha, n, solar_zenith_value = args

    if np.all((refl_meas_wet <= 0) | np.isnan(refl_meas_wet)):
        return i, j, np.nan, np.nan, None, None

    try:
        L, epsilon_opt, r_dry_optimized, predicted_reflectance = perform_inversion(
            refl_meas_wet, R_soil_endmembers, alpha, n, solar_zenith_value, method="least_squares"
        )
        mean_water_thickness = L * epsilon_opt
        return i, j, L, mean_water_thickness, predicted_reflectance, r_dry_optimized

    except Exception as e:
        print(f"Error at Pixel ({i}, {j}): {e}")
        return i, j, np.nan, np.nan, None, None


def save_as_png(data, output_file, label, cmap='viridis'):
    valid = np.isfinite(data)
    if valid.any():
        vmin, vmax = np.nanpercentile(data[valid], [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(data[valid])), float(np.nanmax(data[valid]) + 1e-6)
    else:
        vmin, vmax = 0, 1

    arr = np.ma.masked_invalid(data)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad((1, 1, 1, 1))  # NaN → white

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    ax.axis('off')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_optimal_processes(mem_per_proc_gb=2.0):
    if _PSUTIL:
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        return max(1, min(int(available_gb // mem_per_proc_gb), os.cpu_count()))
    return max(1, os.cpu_count() // 2)


# ------------------- Main -------------------
if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_reflectance_file', type=str, required=True)
    parser.add_argument('--soil_spectra_file', type=str, required=True)
    parser.add_argument('--solar_zenith_file', type=str, required=True)
    parser.add_argument('--optical_properties_file', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--n_processes', type=int, default=None)
    parser.add_argument('--wavelength_ranges', type=str, required=False,
                        help='Comma-separated wavelength ranges in nm, e.g., "900-1350,1400-1800"')

    args = parser.parse_args()

    input_reflectance_file = args.input_reflectance_file
    soil_spectra_file = args.soil_spectra_file
    data_file = args.optical_properties_file
    output_folder = args.output_folder
    output_file = os.path.join(output_folder, 'equivalent_water_layer_thickness_map')
    output_file_mwlt = os.path.join(output_folder, 'mean_equivalent_water_layer_thickness_map')
    output_file_pr = os.path.join(output_folder, 'predicted_reflectance_map')
    output_file_do = os.path.join(output_folder, 'r_dry_optimized_map')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ---- Load data ----
    wavelengths_soil, R_soil_endmembers = load_spectra(soil_spectra_file)
    data = pd.read_csv(data_file)
    wavelength_data = data['Wavelength (nm)'].values
    alpha_data = data['absorption coefficient'].values
    n_data = data['relative refractive index'].values

    hdr_reflectance = input_reflectance_file + '.hdr'
    img = envi.open(hdr_reflectance, find_envi_data_file(hdr_reflectance))
    reflectance_data = img.load()
    hdr_metadata = envi.read_envi_header(hdr_reflectance)
    wavelength_emit = np.array(hdr_metadata['wavelength'], dtype=float)

    hdr_solar_zenith = args.solar_zenith_file + '.hdr'
    img_sza = envi.open(hdr_solar_zenith, find_envi_data_file(hdr_solar_zenith))
    solar_zenith_data = img_sza.load()

    # ---- Interpolate optical properties and soil reflectance ----
    alpha_interp = interp1d(wavelength_data, alpha_data, kind='linear', fill_value="extrapolate")
    n_interp = interp1d(wavelength_data, n_data, kind='linear', fill_value="extrapolate")
    alpha_full = alpha_interp(wavelength_emit)
    n_full = n_interp(wavelength_emit)
    R_soil_interp = interp1d(wavelengths_soil, R_soil_endmembers, axis=1, kind='linear', fill_value='extrapolate')
    R_soil_full = R_soil_interp(wavelength_emit)

    # ---- Apply wavelength subset mask if specified ----
    if args.wavelength_ranges:
        band_ranges = []
        for r in args.wavelength_ranges.split(","):
            try:
                low, high = map(float, r.strip().split("-"))
                band_ranges.append((low, high))
            except Exception:
                raise ValueError(f"Invalid wavelength range format: {r}")

        band_mask = np.zeros_like(wavelength_emit, dtype=bool)
        for (low, high) in band_ranges:
            band_mask |= (wavelength_emit >= low) & (wavelength_emit <= high)

        wavelength_emit = wavelength_emit[band_mask]
        alpha = alpha_full[band_mask]
        n = n_full[band_mask]
        R_soil_endmembers = R_soil_full[:, band_mask]
        reflectance_data = reflectance_data[:, :, band_mask]

        print(f"Applied wavelength mask. Using {np.sum(band_mask)} bands.")
    else:
        alpha = alpha_full
        n = n_full
        R_soil_endmembers = R_soil_full

    # ---- Pixel processing ----
    pixel_indices = [
        (
            i,
            j,
            reflectance_data[i, j, :].flatten(),
            R_soil_endmembers,
            alpha,
            n,
            solar_zenith_data[i, j]
        )
        for i in range(reflectance_data.shape[0])
        for j in range(reflectance_data.shape[1])
    ]

    if args.n_processes is None or args.n_processes == 0:
        num_processes = get_optimal_processes()
    else:
        num_processes = args.n_processes

    print(f"Using {num_processes} parallel processes.")

    from multiprocessing import get_context
    ctx = get_context("spawn")
    with ctx.Pool(processes=num_processes) as pool:
        results = pool.map(process_pixel, pixel_indices)

    # ---- Store results ----
    L_map = np.zeros((reflectance_data.shape[0], reflectance_data.shape[1]))
    mean_water_thickness_map = np.zeros((reflectance_data.shape[0], reflectance_data.shape[1]))
    predicted_reflectance_map = np.zeros((reflectance_data.shape[0], reflectance_data.shape[1], len(wavelength_emit)))
    r_dry_map = np.zeros((reflectance_data.shape[0], reflectance_data.shape[1], len(wavelength_emit)))

    for i, j, L, mean_water_thickness, predicted_reflectance, r_dry_optimized in results:
        L_map[i, j] = L
        mean_water_thickness_map[i, j] = mean_water_thickness
        if predicted_reflectance is not None:
            predicted_reflectance_map[i, j, :] = predicted_reflectance
        if r_dry_optimized is not None:
            r_dry_map[i, j, :] = r_dry_optimized

    # ---- Convert to mm ----
    L_map_mm = L_map * 10.0
    mewt_map_mm = mean_water_thickness_map * 10.0

    # ---- Save outputs ----
    save_as_png(L_map_mm, output_file + '.png',
                label='Equivalent Water Layer Thickness, L (mm)')
    save_as_png(mewt_map_mm, output_file_mwlt + '.png',
                label='Mean Equivalent Water Layer Thickness, \u03b5\u00d7L (mm)')

    header = envi.read_envi_header(hdr_reflectance)
    header['data type'] = 4

    header['description'] = 'Equivalent Water Layer Thickness (L) map (mm)'
    header['band names'] = ['Equivalent Water Layer Thickness L (mm)']
    envi.save_image(output_file + '.hdr', L_map_mm, metadata=header, interleave='bsq', force=True)

    header['description'] = 'Mean Equivalent Water Layer Thickness (epsilon x L) map (mm)'
    header['band names'] = ['Mean Equivalent Water Layer Thickness epsilon x L (mm)']
    envi.save_image(output_file_mwlt + '.hdr', mewt_map_mm, metadata=header, interleave='bsq', force=True)
    header['wavelength'] = list(wavelength_emit.astype(str))
    envi.save_image(output_file_pr + '.hdr', predicted_reflectance_map, metadata=header, interleave='bsq', force=True)
    envi.save_image(output_file_do + '.hdr', r_dry_map, metadata=header, interleave='bsq', force=True)
    np.savetxt(os.path.join(output_folder, "wavelengths_used.txt"), wavelength_emit, fmt="%.2f")

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")