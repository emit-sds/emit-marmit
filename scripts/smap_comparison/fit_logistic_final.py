"""
Logistic calibration of EMIT MEWT to SMAP soil moisture content.

Fits a 3-parameter logistic function (Huber loss, robust to outliers) to paired
EMIT MEWT (mm) and SMAP SMC (cm3/cm3) values. Outputs calibration curve,
validation scatter plot, and optionally full-resolution EMIT-derived SMC maps.

Usage: called from run_emit_retrieval.sh or directly via argparse.

Author: Nayma Binte Nur
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from sklearn.metrics import mean_squared_error
from scipy.optimize import least_squares
from osgeo import gdal
import os

parser = argparse.ArgumentParser()
parser.add_argument('--paired_csv', required=True,
                    help='CSV from compare_smap_emit.py '
                         '(columns: EMIT_Equivalent_Water_Thickness_mm, SMAP_Soil_Moisture_cm3_per_cm3)')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--scene_dirs', nargs='+', default=[],
                    help='Per-scene TIF directories as "SCENE_ID:PATH" entries '
                         '(must contain smap_cropped_common.tif and '
                         'emit_resampled_cropped_common.tif)')
parser.add_argument('--emit_full_dirs', nargs='+', default=[],
                    help='Full-resolution EMIT MEWT paths as "SCENE_ID:PATH" entries '
                         '(mean_equivalent_water_layer_thickness_map.img per scene)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
output_dir = args.output_dir


# ── Model & metrics ───────────────────────────────────────────────────────────

def logistic_function(phi, K, psi, alpha):
    x = psi * phi
    return np.where(
        x >= 0,
        K / (1 + alpha * np.exp(-x)),
        (K * np.exp(x)) / (alpha + np.exp(x))
    )


def coefficient_of_determination(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(args.paired_csv)
phi = df["EMIT_Equivalent_Water_Thickness_mm"].to_numpy()
smc = df["SMAP_Soil_Moisture_cm3_per_cm3"].to_numpy()

valid_mask = np.isfinite(phi) & np.isfinite(smc) & (phi > 0) & (smc > 0)
phi, smc = phi[valid_mask], smc[valid_mask]

# ── Min-max scale ─────────────────────────────────────────────────────────────

phi_min, phi_max = phi.min(), phi.max()
smc_min, smc_max = smc.min(), smc.max()
phi_scale = max(1e-12, phi_max - phi_min)
smc_scale = max(1e-12, smc_max - smc_min)
phi_scaled = (phi - phi_min) / phi_scale
smc_scaled = (smc - smc_min) / smc_scale

# ── Robust logistic fit (Huber loss) ─────────────────────────────────────────

def _residuals(params, phi, y):
    return logistic_function(phi, *params) - y

result = least_squares(
    _residuals,
    x0=[0.5, 5.0, 5.0],
    args=(phi_scaled, smc_scaled),
    bounds=([0, 0, 0], [2, 50, 100]),
    loss='huber',
    f_scale=0.1,
    max_nfev=5000,
)
best_K, best_psi, best_alpha = result.x
smc_est_scaled = logistic_function(phi_scaled, best_K, best_psi, best_alpha)
smc_est = smc_est_scaled * smc_scale + smc_min

r2_val = coefficient_of_determination(smc, smc_est)
nrmse  = np.sqrt(mean_squared_error(smc, smc_est)) / np.mean(smc)

print("\nRobust Logistic Fit (Huber loss)")
print(f"   R²    = {r2_val:.4f}")
print(f"   NRMSE = {nrmse:.4f}")
print(f"   K = {best_K:.4f},  psi = {best_psi:.4f},  alpha = {best_alpha:.4f}")

# ── Plot style ────────────────────────────────────────────────────────────────

mpl.rcParams.update({
    'font.family':       'sans-serif',
    'axes.labelsize':    15,
    'axes.titlesize':    15,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   10,
    'legend.framealpha': 0.85,
    'legend.edgecolor':  '0.8',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '0.88',
    'grid.linewidth':    0.6,
    'figure.dpi':        150,
})


def _clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(direction='out', length=4, width=0.8)


cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'viridis_dark', plt.cm.viridis(np.linspace(0.15, 0.60, 256))
)
norm = mpl.colors.Normalize(vmin=smc.min(), vmax=smc.max())

phi_smooth_scaled   = np.linspace(phi_scaled.min(), phi_scaled.max(), 500)
phi_smooth_unscaled = phi_smooth_scaled * phi_scale + phi_min
smc_curve           = logistic_function(phi_smooth_scaled, best_K, best_psi, best_alpha) * smc_scale + smc_min

eq_str = (r"$\mathrm{{SM}} = \dfrac{{{:.2f}}}{{1 + {:.2f}\,e^{{-{:.2f}\phi}}}}$"
          .format(best_K * smc_scale + smc_min, best_alpha, best_psi))

mn = min(smc.min(), smc_est.min())
mx = max(smc.max(), smc_est.max())
_pad = (mx - mn) * 0.06

# ── Plot A: Calibration curve ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))
_clean_ax(ax)
sc = ax.scatter(phi, smc, c=smc, cmap=cmap, norm=norm,
                s=65, alpha=0.6, linewidths=0, zorder=3, label='Data')
ax.plot(phi_smooth_unscaled, smc_curve, color='#222222', lw=2.0, zorder=4, label='Fitted logistic')
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label('SMAP Soil Moisture (cm³/cm³)', fontsize=13)
cb.ax.tick_params(labelsize=9)
ax.text(0.04, 0.92, eq_str, transform=ax.transAxes, fontsize=12, va='top', ha='left')
ax.set_xlabel('EMIT MEWT (mm)')
ax.set_ylabel('SMAP Soil Moisture (cm³/cm³)')
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'calibration.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: calibration.png")

# ── Plot B: Scatter observed vs estimated ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(5.5, 5.5))
_clean_ax(ax)
sc = ax.scatter(smc, smc_est, c=smc, cmap=cmap, norm=norm,
                s=65, alpha=0.6, linewidths=0, zorder=3, label='Data')
ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='1:1 line')
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label('SMAP Soil Moisture (cm³/cm³)', fontsize=13)
cb.ax.tick_params(labelsize=9)
ax.text(0.05, 0.93,
        f'$R^2 = {r2_val:.3f}$\nNRMSE $= {nrmse:.3f}$',
        transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='0.8', linewidth=0.8))
ax.set_xlabel('SMAP Soil Moisture (cm³/cm³)')
ax.set_ylabel('EMIT-derived Soil Moisture (cm³/cm³)')
ax.legend(loc='lower right', frameon=True)
ax.set_xlim([mn - _pad, mx + _pad])
ax.set_ylim([mn - _pad, mx + _pad])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scatter.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: scatter.png")

# ── Combined: calibration (A) | scatter (B) ───────────────────────────────────

fig = plt.figure(figsize=(14, 5))
gs  = mgridspec.GridSpec(1, 2, figure=fig, wspace=0.52)

ax1 = fig.add_subplot(gs[0, 0])
_clean_ax(ax1)
sc1 = ax1.scatter(phi, smc, c=smc, cmap=cmap, norm=norm,
                  s=65, alpha=0.6, linewidths=0, zorder=3, label='Data')
ax1.plot(phi_smooth_unscaled, smc_curve, color='#222222', lw=2.0, zorder=4, label='Fitted logistic')
ax1.text(0.04, 0.92, eq_str, transform=ax1.transAxes, fontsize=11, va='top', ha='left')
ax1.set_xlabel('EMIT MEWT (mm)')
ax1.set_ylabel('SMAP Soil Moisture (cm³/cm³)')
ax1.legend(loc='lower right', frameon=True)
ax1.text(-0.12, 1.02, '(A)', transform=ax1.transAxes, fontsize=13, fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
_clean_ax(ax2)
sc2 = ax2.scatter(smc, smc_est, c=smc, cmap=cmap, norm=norm,
                  s=65, alpha=0.6, linewidths=0, zorder=3, label='Data')
ax2.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='1:1 line')
ax2.text(0.05, 0.93,
         f'$R^2 = {r2_val:.3f}$\nNRMSE $= {nrmse:.3f}$',
         transform=ax2.transAxes, fontsize=11, va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='0.8', linewidth=0.8))
ax2.set_xlabel('SMAP Soil Moisture (cm³/cm³)')
ax2.set_ylabel('EMIT-derived Soil Moisture (cm³/cm³)')
ax2.legend(loc='lower right', frameon=True)
ax2.set_xlim([mn - _pad, mx + _pad])
ax2.set_ylim([mn - _pad, mx + _pad])
ax2.text(-0.12, 1.02, '(B)', transform=ax2.transAxes, fontsize=13, fontweight='bold')

cb = fig.colorbar(sc2, ax=[ax1, ax2], pad=0.05, shrink=0.85)
cb.set_label('SMAP Soil Moisture (cm³/cm³)', fontsize=13)
cb.ax.tick_params(labelsize=9)

plt.savefig(os.path.join(output_dir, 'combined.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: combined.png")


# ── Spatial maps (per scene, if provided) ─────────────────────────────────────

if args.scene_dirs:
    spatial_root = os.path.join(output_dir, 'spatial_maps')
    os.makedirs(spatial_root, exist_ok=True)

    scenes_info = [entry.split(':', 1) for entry in args.scene_dirs]
    scene_data  = []

    for sid, sdir in scenes_info:
        smap_tif = os.path.join(sdir, 'smap_cropped_common.tif')
        emit_tif = os.path.join(sdir, 'emit_resampled_cropped_common.tif')
        if not os.path.exists(smap_tif) or not os.path.exists(emit_tif):
            print(f"  WARNING: Missing TIFs for scene {sid} in {sdir} — skipping")
            continue

        smap_ds  = gdal.Open(smap_tif)
        smap_arr = smap_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        smap_nd  = smap_ds.GetRasterBand(1).GetNoDataValue()
        if smap_nd is not None and not np.isnan(float(smap_nd)):
            smap_arr[smap_arr == smap_nd] = np.nan

        emit_ds  = gdal.Open(emit_tif)
        emit_arr = emit_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        emit_nd  = emit_ds.GetRasterBand(1).GetNoDataValue()
        if emit_nd is not None and not np.isnan(float(emit_nd)):
            emit_arr[emit_arr == emit_nd] = np.nan

        # emit_resampled_cropped_common.tif stores MEWT in mm (×10 applied in compare_smap_emit.py)
        valid    = np.isfinite(emit_arr) & (emit_arr > 0)
        smc_emit = np.full_like(emit_arr, np.nan)
        if valid.any():
            phi_px        = emit_arr[valid]
            phi_scaled_px = np.clip((phi_px - phi_min) / phi_scale, 0, None)
            smc_emit[valid] = logistic_function(phi_scaled_px, best_K, best_psi, best_alpha) * smc_scale + smc_min

        scene_data.append({'id': sid, 'smap': smap_arr, 'emit_ewt': emit_arr, 'emit_smc': smc_emit})

    if scene_data:
        cmap_sp = 'viridis'

        def _pct(vals, lo=2, hi=98):
            v = vals[np.isfinite(vals) & (vals > 0)]
            return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v.size else (0.0, 1.0)

        smap_all = np.concatenate([d['smap'].ravel()     for d in scene_data])
        ewt_all  = np.concatenate([d['emit_ewt'].ravel() for d in scene_data])
        smc_all  = np.concatenate([d['emit_smc'].ravel() for d in scene_data])
        vmin_smap, vmax_smap = _pct(smap_all)
        vmin_ewt,  vmax_ewt  = _pct(ewt_all)
        vmin_smc,  vmax_smc  = _pct(smc_all)

        for d in scene_data:
            sid       = d['id']
            scene_out = os.path.join(spatial_root, f'scene_{sid}')
            os.makedirs(scene_out, exist_ok=True)

            for arr, fname, label in [
                (d['smap'],     f'smap_{sid}.png',     'SMAP Soil Moisture (cm³/cm³)'),
                (d['emit_ewt'], f'emit_ewt_{sid}.png', 'EMIT Water Layer Thickness (mm)'),
                (d['emit_smc'], f'emit_smc_{sid}.png', 'EMIT-derived SM (cm³/cm³)'),
            ]:
                vmin = vmin_smap if 'smap' in fname else (vmin_ewt if 'ewt' in fname else vmin_smc)
                vmax = vmax_smap if 'smap' in fname else (vmax_ewt if 'ewt' in fname else vmax_smc)
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(arr, cmap=cmap_sp, vmin=vmin, vmax=vmax)
                cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
                cb.set_label(label, fontsize=13)
                cb.ax.tick_params(labelsize=9)
                ax.axis('off')
                plt.savefig(os.path.join(scene_out, fname), dpi=300, bbox_inches='tight')
                plt.close()

            # Side-by-side SMAP vs EMIT MEWT
            fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(14, 5))
            fig.subplots_adjust(wspace=0.30)
            im_s = ax_s.imshow(d['smap'],     cmap=cmap_sp, vmin=vmin_smap, vmax=vmax_smap)
            im_e = ax_e.imshow(d['emit_ewt'], cmap=cmap_sp, vmin=vmin_ewt,  vmax=vmax_ewt)
            ax_s.axis('off')
            ax_e.axis('off')
            ax_s.text(-0.04, 1.04, '(A)', transform=ax_s.transAxes, fontsize=13, fontweight='bold', va='bottom')
            ax_e.text(-0.04, 1.04, '(B)', transform=ax_e.transAxes, fontsize=13, fontweight='bold', va='bottom')
            cb_s = fig.colorbar(im_s, ax=ax_s, shrink=0.7, pad=0.02)
            cb_s.set_label('SMAP Soil Moisture (cm³/cm³)', fontsize=11)
            cb_s.ax.tick_params(labelsize=9)
            cb_e = fig.colorbar(im_e, ax=ax_e, shrink=0.7, pad=0.02)
            cb_e.set_label('EMIT Water Layer Thickness (mm)', fontsize=11)
            cb_e.ax.tick_params(labelsize=9)
            plt.savefig(os.path.join(scene_out, f'comparison_{sid}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: spatial_maps/scene_{sid}/")

        # Combined multi-scene figure
        n   = len(scene_data)
        fig = plt.figure(figsize=(4.8 * n + 2.5, 8.5))
        gs_m = mgridspec.GridSpec(2, n, figure=fig, wspace=0.04, hspace=0.12,
                                  left=0.10, right=0.84, top=0.93, bottom=0.04)
        ims_smap_list, ims_ewt_list = [], []
        for col, d in enumerate(scene_data):
            ax_top = fig.add_subplot(gs_m[0, col])
            im_s = ax_top.imshow(d['smap'], cmap=cmap_sp, vmin=vmin_smap, vmax=vmax_smap, aspect='auto')
            ims_smap_list.append(im_s)
            ax_top.axis('off')
            ax_top.text(0.5, 1.02, f'Scene {d["id"]}', transform=ax_top.transAxes,
                        fontsize=12, fontweight='bold', ha='center', va='bottom')
            ax_bot = fig.add_subplot(gs_m[1, col])
            im_e = ax_bot.imshow(d['emit_ewt'], cmap=cmap_sp, vmin=vmin_ewt, vmax=vmax_ewt, aspect='auto')
            ims_ewt_list.append(im_e)
            ax_bot.axis('off')

        fig.text(0.03, 0.73, 'SMAP\nSM',   fontsize=12, fontweight='bold', rotation=90, va='center', ha='center')
        fig.text(0.03, 0.27, 'EMIT\nMEWT', fontsize=12, fontweight='bold', rotation=90, va='center', ha='center')
        cax_top = fig.add_axes([0.86, 0.53, 0.018, 0.38])
        cb_top  = fig.colorbar(ims_smap_list[-1], cax=cax_top)
        cb_top.set_label('SMAP Soil Moisture (cm³/cm³)', fontsize=13)
        cb_top.ax.tick_params(labelsize=9)
        cax_bot = fig.add_axes([0.86, 0.07, 0.018, 0.38])
        cb_bot  = fig.colorbar(ims_ewt_list[-1], cax=cax_bot)
        cb_bot.set_label('EMIT Water Layer Thickness (mm)', fontsize=13)
        cb_bot.ax.tick_params(labelsize=9)
        plt.savefig(os.path.join(spatial_root, 'all_scenes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: spatial_maps/all_scenes.png")


# ── Full-resolution EMIT-derived SMC maps ─────────────────────────────────────

if args.emit_full_dirs:
    spatial_root_full = os.path.join(output_dir, 'spatial_maps')
    os.makedirs(spatial_root_full, exist_ok=True)

    full_scene_data = []
    for entry in args.emit_full_dirs:
        sid, ewt_path = entry.split(':', 1)
        if not os.path.exists(ewt_path):
            print(f"  WARNING: EWT file not found for scene {sid}: {ewt_path}")
            continue

        ds = gdal.Open(ewt_path)
        if ds is None:
            print(f"  WARNING: Could not open {ewt_path} — skipping scene {sid}")
            continue

        ewt_arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        nd = ds.GetRasterBand(1).GetNoDataValue()
        if nd is not None and not np.isnan(float(nd)):
            ewt_arr[ewt_arr == nd] = np.nan

        # mean_equivalent_water_layer_thickness_map is already in mm

        valid    = np.isfinite(ewt_arr) & (ewt_arr > 0)
        smc_full = np.full_like(ewt_arr, np.nan)
        if valid.any():
            phi_px        = ewt_arr[valid]
            phi_scaled_px = np.clip((phi_px - phi_min) / phi_scale, 0, None)
            smc_full[valid] = logistic_function(phi_scaled_px, best_K, best_psi, best_alpha) * smc_scale + smc_min

        full_scene_data.append({'id': sid, 'smc': smc_full})

    if full_scene_data:
        all_smc_vals = np.concatenate([
            d['smc'][np.isfinite(d['smc']) & (d['smc'] > 0)] for d in full_scene_data
        ])
        vmin_f = float(np.percentile(all_smc_vals, 2))  if all_smc_vals.size else 0.0
        vmax_f = float(np.percentile(all_smc_vals, 98)) if all_smc_vals.size else 1.0

        for d in full_scene_data:
            sid       = d['id']
            scene_out = os.path.join(spatial_root_full, f'scene_{sid}')
            os.makedirs(scene_out, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(d['smc'], cmap='viridis', vmin=vmin_f, vmax=vmax_f)
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label('EMIT-derived SM (cm³/cm³)', fontsize=13)
            cb.ax.tick_params(labelsize=9)
            ax.axis('off')
            plt.savefig(os.path.join(scene_out, f'emit_smc_full_{sid}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: spatial_maps/scene_{sid}/emit_smc_full_{sid}.png")

        n   = len(full_scene_data)
        fig = plt.figure(figsize=(7 * n + 1.5, 6))
        gs_f = mgridspec.GridSpec(1, n, figure=fig, wspace=0.04,
                                  left=0.04, right=0.86, top=0.93, bottom=0.04)
        ims_f = []
        for col, d in enumerate(full_scene_data):
            ax = fig.add_subplot(gs_f[0, col])
            im = ax.imshow(d['smc'], cmap='viridis', vmin=vmin_f, vmax=vmax_f)
            ims_f.append(im)
            ax.axis('off')
            ax.text(0.5, 1.02, f'Scene {d["id"]}', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', ha='center', va='bottom')
        cax = fig.add_axes([0.88, 0.12, 0.022, 0.74])
        cb  = fig.colorbar(ims_f[-1], cax=cax)
        cb.set_label('EMIT-derived SM (cm³/cm³)', fontsize=12)
        cb.ax.tick_params(labelsize=9)
        plt.savefig(os.path.join(spatial_root_full, 'emit_smc_full_all_scenes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: spatial_maps/emit_smc_full_all_scenes.png")