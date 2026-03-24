"""
Resample EMIT MEWT map to SMAP grid and export paired values for calibration.

For each SMAP pixel, computes the mean EMIT MEWT (mm) from valid sub-pixels.
SMAP pixels with less than min_coverage (default 0.4) valid EMIT sub-pixels
are excluded. Outputs preview PNGs and a paired CSV for logistic calibration.

Usage: called from run_emit_retrieval.sh or directly via argparse.

Author: Nayma Binte Nur
"""

import argparse
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main(emit_path, smap_path, output_dir, min_coverage=0.4):
    os.makedirs(output_dir, exist_ok=True)

    emit_ds = gdal.Open(emit_path)
    smap_ds = gdal.Open(smap_path)

    emit_data = emit_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    emit_gt = emit_ds.GetGeoTransform()

    smap_data = smap_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    smap_nodata = smap_ds.GetRasterBand(1).GetNoDataValue()
    if smap_nodata is not None:
        smap_data[smap_data == smap_nodata] = np.nan

    smap_gt = smap_ds.GetGeoTransform()
    smap_proj = smap_ds.GetProjection()
    smap_cols = smap_ds.RasterXSize
    smap_rows = smap_ds.RasterYSize

    # Resample EMIT to SMAP grid
    resampled_emit = np.full((smap_rows, smap_cols), np.nan, dtype=np.float32)
    for i in range(smap_rows):
        for j in range(smap_cols):
            x_min = smap_gt[0] + j * smap_gt[1]
            y_max = smap_gt[3] + i * smap_gt[5]
            x_max = x_min + smap_gt[1]
            y_min = y_max + smap_gt[5]

            col_start = int((x_min - emit_gt[0]) / emit_gt[1])
            col_end = int((x_max - emit_gt[0]) / emit_gt[1])
            row_start = int((y_max - emit_gt[3]) / emit_gt[5])
            row_end = int((y_min - emit_gt[3]) / emit_gt[5])

            col_start = max(0, col_start)
            col_end = min(emit_data.shape[1], col_end)
            row_start = max(0, row_start)
            row_end = min(emit_data.shape[0], row_end)

            if row_end > row_start and col_end > col_start:
                window = emit_data[row_start:row_end, col_start:col_end]
                if window.size == 0:
                    continue
                valid_mask_win = (~np.isnan(window)) & (window > 0)
                valid = window[valid_mask_win]
                coverage = valid.size / float(window.size)
                if coverage >= min_coverage and valid.size > 0:
                    resampled_emit[i, j] = valid.mean()

    # Crop both to valid region
    valid_mask = (~np.isnan(resampled_emit)) & (resampled_emit > 0)
    ys, xs = np.where(valid_mask)

    if ys.size == 0 or xs.size == 0:
        raise ValueError("No valid pixels found in resampled EMIT data.")

    row_min, row_max = int(ys.min()), int(ys.max()) + 1
    col_min, col_max = int(xs.min()), int(xs.max()) + 1

    emit_crop = resampled_emit[row_min:row_max, col_min:col_max]
    smap_crop = smap_data[row_min:row_max, col_min:col_max]

    crop_mask = valid_mask[row_min:row_max, col_min:col_max]
    smap_crop[~crop_mask] = np.nan

    # Flatten and apply combined valid mask
    flat_emit_all = emit_crop[crop_mask].flatten()
    flat_smap_all = smap_crop[crop_mask].flatten()
    final_valid_mask = (~np.isnan(flat_emit_all)) & (~np.isnan(flat_smap_all))

    flat_emit = flat_emit_all[final_valid_mask]
    flat_smap = flat_smap_all[final_valid_mask]

    # Reshape final valid mask back to 2D for image masking
    crop_mask_2d = crop_mask.copy()
    flat_mask = crop_mask_2d.flatten()
    flat_mask[flat_mask] = final_valid_mask
    final_mask_2d = flat_mask.reshape(crop_mask.shape)

    emit_crop_masked = np.full_like(emit_crop, np.nan)
    smap_crop_masked = np.full_like(smap_crop, np.nan)
    emit_crop_masked[final_mask_2d] = emit_crop[final_mask_2d]
    smap_crop_masked[final_mask_2d] = smap_crop[final_mask_2d]

    # Plot valid pixels
    for arr, name in [(emit_crop_masked, "Resampled EMIT"), (smap_crop_masked, "SMAP")]:
        plt.figure(figsize=(8, 6))
        plt.imshow(arr, cmap='viridis')
        label = "Mean Equivalent Water Layer Thickness, \u03b5\u00d7L (mm)" if "EMIT" in name else "Soil Moisture Content (cm\u00b3/cm\u00b3)"
        plt.colorbar(label=label)
        plt.title(f"{name} - Pixels Used in Scatter Plot")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_common_valid.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print("Number of valid resampled EMIT pixels:", np.sum(valid_mask))
    print("Final number of pixels used:", flat_emit.size)

    # Save paired values to CSV
    scatter_csv_path = os.path.join(output_dir, "emit_vs_smap_paired_values.csv")
    scatter_df = pd.DataFrame({
        "EMIT_Equivalent_Water_Thickness_mm": flat_emit,
        "SMAP_Soil_Moisture_cm3_per_cm3": flat_smap
    })
    scatter_df.to_csv(scatter_csv_path, index=False)
    print(f"Paired values saved to: {scatter_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emit_path', required=True, help='Path to EMIT image')
    parser.add_argument('--smap_path', required=True, help='Path to SMAP image')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--min_coverage', type=float, default=0.4,
                        help='Minimum EMIT coverage ratio inside each SMAP pixel (0–1) to accept (default: 0.4)')
    args = parser.parse_args()
    main(args.emit_path, args.smap_path, args.output_dir, args.min_coverage)