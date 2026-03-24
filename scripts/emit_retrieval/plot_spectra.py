"""
Plot measured vs. modeled spectral fits for selected EMIT pixels.

Selects pixels spanning the MEWT range, plots combined and individual
measured/predicted reflectance spectra, and saves pixel data to CSV.

Usage: called from run_emit_retrieval.sh or directly via argparse.

Author: Nayma Binte Nur
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import spectral.io.envi as envi
import pandas as pd
import argparse
import os

# ------------- Argument Parser -------------
parser = argparse.ArgumentParser(description='Plot pixels with diverse L values.')
parser.add_argument('--l_map', type=str, required=True, help='Path to equivalent water thickness map (.img)')
parser.add_argument('--predicted', type=str, required=True, help='Path to predicted reflectance image (.img)')
parser.add_argument('--measured', type=str, required=True, help='Path to real measured reflectance image (.img)')
parser.add_argument('--l_epsilon', type=str, required=True)

args = parser.parse_args()


def get_envi_files(base_path):
    """
    Given a base path, return the correct (hdr_file, data_file) tuple.
    Handles .img, .dat, or extension-less base paths.
    """
    if base_path.endswith('.hdr'):
        hdr_file = base_path
        base = base_path[:-4]
    else:
        base = base_path.replace('.img', '').replace('.dat', '')
        hdr_file = base + '.hdr'

    if os.path.isfile(base + '.img'):
        data_file = base + '.img'
    elif os.path.isfile(base + '.dat'):
        data_file = base + '.dat'
    else:
        raise FileNotFoundError(f"Cannot find .img or .dat file for base: {base}")

    return hdr_file, data_file


# Get valid pairs
measured_hdr, measured_data_file = get_envi_files(args.measured)
predicted_hdr, predicted_data_file = get_envi_files(args.predicted)
l_map_hdr, l_map_data_file = get_envi_files(args.l_map)
l_epsilon_hdr, l_epsilon_data_file = get_envi_files(args.l_epsilon)

# Load the ENVI data
measured_data = envi.open(measured_hdr, measured_data_file).load()
predicted_data = envi.open(predicted_hdr, predicted_data_file).load()
l_map_data = envi.open(l_map_hdr, l_map_data_file).load()
l_epsilon_map_data = envi.open(l_epsilon_hdr, l_epsilon_data_file).load()

# ------------- Load Wavelengths -------------
pred_meta = envi.read_envi_header(predicted_hdr)
meas_meta = envi.read_envi_header(measured_hdr)
wavelengths_pred = np.array(pred_meta['wavelength'], dtype=float)[:predicted_data.shape[2]]
wavelengths_meas = np.array(meas_meta['wavelength'], dtype=float)[:measured_data.shape[2]]

# ------------- Select Spectral Regions -------------
selected_ranges = [(500, 1320), (1506, 1732), (2063, 2497)]

# ------------- Find Diverse Pixels by L -------------
rows, cols = l_map_data.shape[:2]
l_flat = l_map_data.ravel()
valid_mask = ~np.isnan(l_flat) & (l_flat > 0)
valid_l = l_flat[valid_mask]
valid_indices = np.where(valid_mask)[0]

quantiles = np.quantile(valid_l, np.linspace(0.01, 0.6, 150))
unique_coords = []
for q_val in quantiles:
    diff = np.abs(valid_l - q_val)
    closest_idx = valid_indices[np.argmin(diff)]
    y, x = np.unravel_index(closest_idx, (rows, cols))
    unique_coords.append((x, y))

# ------------- Create Output Folder -------------
plot_dir = os.path.join(os.path.dirname(args.l_map), 'pixel_spectra_plots/low_midhigh')
os.makedirs(plot_dir, exist_ok=True)


def select_and_plot_epsL_pixels(epsL_map, measured_data, predicted_data, l_map_data,
                                wavelengths_meas, wavelengths_pred, selected_ranges,
                                min_epsL, max_epsL, num_pixels=10, round_decimals=2,
                                base_output_dir="./pixel_spectra_plots", save_csv=True):

    if epsL_map.ndim == 3 and epsL_map.shape[2] == 1:
        epsL_map = np.squeeze(epsL_map)
    if epsL_map.ndim != 2:
        raise ValueError(f"Expected 2D ε×L map, but got shape: {epsL_map.shape}")

    rows, cols = epsL_map.shape
    product = epsL_map.ravel()

    valid_mask = ~np.isnan(product) & (product > 0)
    valid_product = product[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    sorted_order = np.argsort(valid_product)
    sorted_product = valid_product[sorted_order]
    sorted_indices = valid_indices[sorted_order]

    seen = set()
    filtered_indices = []
    for val, idx in zip(sorted_product, sorted_indices):
        val_mm = val * 10
        rounded_val = round(val_mm, round_decimals)
        if min_epsL <= rounded_val <= max_epsL and rounded_val not in seen:
            seen.add(rounded_val)
            filtered_indices.append(idx)

    filtered_indices = np.array(filtered_indices)

    if len(filtered_indices) < num_pixels:
        raise ValueError(f"Only {len(filtered_indices)} unique εL values (in mm) in range {min_epsL}–{max_epsL}. Need at least {num_pixels}.")

    n_middle = num_pixels - 2
    lowest = filtered_indices[:1]
    highest = filtered_indices[-1:]
    middle_pool = filtered_indices[1:-1]
    middle_indices = np.linspace(0, len(middle_pool) - 1, n_middle, dtype=int)
    middle = middle_pool[middle_indices]

    selected_indices = np.concatenate([lowest, middle, highest])
    selected_coords = [np.unravel_index(idx, (rows, cols)) for idx in selected_indices]
    selected_coords = [(x, y) for y, x in selected_coords]

    # === Font sizes ===
    FS_LABEL  = 18
    FS_TICK   = 15
    FS_LEGEND = 13

    # === Color palette — visually distinct, colorblind-friendly ===
    COLORS = [
        '#1f77b4', '#d62728', '#2ca02c', '#9467bd',
        '#e377c2', '#8c564b', '#17becf', '#bcbd22',
        '#ff7f0e', '#7f7f7f',
    ]

    # === Plot all spectra (measured & predicted) ===
    range_folder = f"{min_epsL:.2f}_{max_epsL:.2f}".replace('.', 'p')
    combined_plot_dir = os.path.join(base_output_dir, range_folder)
    os.makedirs(combined_plot_dir, exist_ok=True)

    individual_plot_dir = os.path.join(combined_plot_dir, "individual")
    os.makedirs(individual_plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    legend_handles = []
    for i, (x, y) in enumerate(selected_coords):
        real_spectrum = measured_data[y, x, :].flatten()
        pred_spectrum = predicted_data[y, x, :].flatten()
        product_val = float(epsL_map[y, x]) * 10
        color = COLORS[i % len(COLORS)]

        for j, (start, end) in enumerate(selected_ranges):
            meas_mask = (wavelengths_meas >= start) & (wavelengths_meas <= end)
            pred_mask = (wavelengths_pred >= start) & (wavelengths_pred <= end)
            ax.plot(wavelengths_meas[meas_mask], real_spectrum[meas_mask],
                    linestyle='-', color=color, linewidth=1.8)
            ax.plot(wavelengths_pred[pred_mask], pred_spectrum[pred_mask],
                    linestyle='--', color=color, linewidth=1.4, alpha=0.80)

        legend_handles.append(mlines.Line2D([0], [0], color=color, linestyle='-',
                                            linewidth=2.0,
                                            label=f"Measured   ε×L = {product_val:.3f} mm"))
        legend_handles.append(mlines.Line2D([0], [0], color=color, linestyle='--',
                                            linewidth=1.6,
                                            label=f"Predicted    ε×L = {product_val:.3f} mm"))
        if i < len(selected_coords) - 1:
            legend_handles.append(mlines.Line2D([0], [0], linestyle='none', label=''))

    ax.set_xlabel("Wavelength (nm)", fontsize=FS_LABEL)
    ax.set_ylabel("Reflectance", fontsize=FS_LABEL)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(handles=legend_handles, fontsize=FS_LEGEND,
              loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=True, framealpha=0.9, edgecolor='#cccccc')
    fig.tight_layout()
    fig.subplots_adjust(right=0.68)
    fig.savefig(os.path.join(combined_plot_dir, "combined_pixels.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # === Plot individual spectra ===
    for i, (x, y) in enumerate(selected_coords):
        real_spectrum = measured_data[y, x, :].flatten()
        pred_spectrum = predicted_data[y, x, :].flatten()
        product_val = float(epsL_map[y, x]) * 10

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        for j, (start, end) in enumerate(selected_ranges):
            meas_mask = (wavelengths_meas >= start) & (wavelengths_meas <= end)
            pred_mask = (wavelengths_pred >= start) & (wavelengths_pred <= end)

            ax.plot(wavelengths_meas[meas_mask], real_spectrum[meas_mask],
                    linestyle='-', color='#1f77b4', linewidth=1.8,
                    label='Measured' if j == 0 else "")
            ax.plot(wavelengths_pred[pred_mask], pred_spectrum[pred_mask],
                    linestyle='--', color='#d62728', linewidth=1.6,
                    label='Predicted' if j == 0 else "")

        ax.set_xlabel("Wavelength (nm)", fontsize=FS_LABEL)
        ax.set_ylabel("Reflectance", fontsize=FS_LABEL)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.text(0.50, 0.97, f"ε×L = {product_val:.3f} mm",
                transform=ax.transAxes, fontsize=FS_LEGEND,
                va='top', ha='center',
                bbox=dict(facecolor='lightyellow', alpha=0.85,
                          edgecolor='#aaaaaa', boxstyle='round,pad=0.3'))
        ax.legend(fontsize=FS_LEGEND, frameon=True, framealpha=0.9, edgecolor='#cccccc')
        fig.tight_layout()
        fig.savefig(os.path.join(individual_plot_dir,
                                 f"pixel_{i+1}_x{x}_y{y}_epsL_{product_val:.3f}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    # === Save to CSV ===
    if save_csv:
        csv_rows = []
        for i, (x, y) in enumerate(selected_coords):
            real_spectrum = measured_data[y, x, :].flatten()
            pred_spectrum = predicted_data[y, x, :].flatten()
            product_val = float(epsL_map[y, x]) * 10

            row = {"pixel_id": i + 1, "x": x, "y": y, "epsL_mm": product_val}
            for j, wl in enumerate(wavelengths_meas):
                row[f"meas_{int(wl)}nm"] = real_spectrum[j]
            for j, wl in enumerate(wavelengths_pred):
                row[f"pred_{int(wl)}nm"] = pred_spectrum[j]
            csv_rows.append(row)

        df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(base_output_dir, "selected_pixels.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved selected pixel data to {csv_path}")

    return selected_coords


min_epsL, max_epsL = (0, 0.2)
folder_name = f"epsL_{min_epsL:.2f}_to_{max_epsL:.2f}".replace('.', 'p')
full_plot_dir = os.path.join(plot_dir, folder_name)
os.makedirs(full_plot_dir, exist_ok=True)

select_and_plot_epsL_pixels(
    epsL_map=l_epsilon_map_data[:, :, 0],
    measured_data=measured_data,
    predicted_data=predicted_data,
    l_map_data=l_map_data[:, :, 0],
    wavelengths_meas=wavelengths_meas,
    wavelengths_pred=wavelengths_pred,
    selected_ranges=[(500, 1320), (1506, 1732), (2063, 2497)],
    min_epsL=min_epsL,
    max_epsL=max_epsL,
    num_pixels=6,
    round_decimals=2,
    base_output_dir=full_plot_dir
)