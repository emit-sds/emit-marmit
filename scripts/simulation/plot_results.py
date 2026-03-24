"""
Plotting functions for MARMIT noise sensitivity analysis results.

All plots target MEWT (Mean Equivalent Water Thickness = L x epsilon).
Spectral fits are shown over valid wavelength ranges only (500-1320,
1507-1766, 2064-2300 nm). Each SNR level has a consistent color across
all figure types.

Author: Nayma Binte Nur
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import os

# ── Global style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':        'sans-serif',
    'axes.labelsize':     13,
    'axes.titlesize':     13,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'legend.fontsize':    10,
    'legend.framealpha':  0.8,
    'legend.edgecolor':   '0.8',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.color':         '0.88',
    'grid.linewidth':     0.6,
    'figure.dpi':         150,
})

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_RANGES = [(500, 1320), (1507, 1766), (2064, 2300)]

# Consistent color per SNR level (high → low SNR = blue → red)
SNR_COLORS = {
    np.inf: '#111111',
    600:    '#1f77b4',   # deep blue
    400:    '#2ca02c',   # deep green
    200:    '#e07b00',   # deep orange
    100:    '#d62728',   # deep red
}


def _snr_label(snr):
    return '∞ (clean)' if np.isinf(snr) else str(int(snr))


def _snr_color(snr):
    return SNR_COLORS.get(snr, 'gray')


def _clean_ax(ax):
    """Apply clean scientific style to an axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(direction='out', length=4, width=0.8)


def _mewt_stats(d):
    """
    Compute mean ± std of MAE, NRMSE, R² across realizations.
    d : DataFrame for one SNR level, with a 'realization' column.
    Returns (mae_mean, mae_std, nrmse_mean, nrmse_std, r2_mean, r2_std).
    """
    per = []
    for _, dr in d.groupby('realization'):
        true = dr['MEWT_true'].values * 10
        ret  = dr['MEWT_retrieved'].values * 10
        err  = true - ret
        mae   = np.mean(np.abs(err))
        rng   = true.max() - true.min()
        nrmse = np.sqrt(np.mean(err**2)) / rng if rng > 0 else np.inf
        ss    = np.sum((true - true.mean())**2)
        r2    = 1 - np.sum(err**2) / ss if ss > 0 else -np.inf
        per.append((mae, nrmse, r2))
    per = np.array(per)
    ddof = 1 if len(per) > 1 else 0
    return (per[:, 0].mean(), per[:, 0].std(ddof=ddof),
            per[:, 1].mean(), per[:, 1].std(ddof=ddof),
            per[:, 2].mean(), per[:, 2].std(ddof=ddof))


# ── Noise sensitivity curves ──────────────────────────────────────────────────

def plot_noise_sensitivity_curves(stats_df, output_dir):
    """
    MEWT MAE, NRMSE, and R² vs SNR. Each SNR point is colored consistently.
    Saved to: output_dir/noise_sensitivity_curves.png
    """
    finite = stats_df[~np.isinf(stats_df['snr'])].copy()
    clean  = stats_df[np.isinf(stats_df['snr'])]

    snr_vals     = finite['snr'].values
    mae_vals     = finite['MEWT_MAE_mean'].values * 10
    mae_errs     = finite['MEWT_MAE_std'].values * 10
    nrmse_vals   = finite['MEWT_NRMSE_mean'].values
    nrmse_errs   = finite['MEWT_NRMSE_std'].values
    r2_vals      = finite['MEWT_R2_mean'].values
    r2_errs      = finite['MEWT_R2_std'].values
    point_colors = [_snr_color(s) for s in snr_vals]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    def _plot_curve(ax, yvals, yerrs, baseline_val, ylabel):
        _clean_ax(ax)
        ax.set_xscale('log')
        ax.set_xlim([snr_vals.max() * 1.5, snr_vals.min() * 0.7])
        ax.set_xticks(snr_vals)
        ax.set_xticklabels([str(int(s)) for s in snr_vals])
        ax.set_xlabel('SNR')

        ax.plot(snr_vals, yvals, '-', lw=1.0, color='0.65', zorder=1)
        handles = []
        for s, y, ye, c in zip(snr_vals, yvals, yerrs, point_colors):
            ax.errorbar(s, y, yerr=ye, fmt='none', color=c,
                        capsize=3, capthick=0.8, elinewidth=0.8, zorder=2)
            h = ax.scatter(s, y, color=c, s=40, zorder=3,
                           linewidths=0.5, edgecolors='white')
            handles.append(h)

        if baseline_val is not None:
            ax.axhline(baseline_val, color=_snr_color(np.inf),
                       lw=1.0, ls='--', alpha=0.7, label='No-noise baseline')

        ymin = (yvals - yerrs).min()
        ymax = (yvals + yerrs).max()
        pad = (ymax - ymin) * 0.25 if ymax > ymin else 0.05
        ax.set_ylim(ymin - pad, ymax + pad)

        ax.set_ylabel(ylabel)
        return handles

    handles = _plot_curve(axes[0], mae_vals, mae_errs,
                          clean['MEWT_MAE_mean'].values[0] * 10 if len(clean) > 0 else None,
                          'Mean MEWT MAE (mm)')

    _plot_curve(axes[1], nrmse_vals, nrmse_errs,
                clean['MEWT_NRMSE_mean'].values[0] if len(clean) > 0 else None,
                'Mean MEWT NRMSE')

    _plot_curve(axes[2], r2_vals, r2_errs,
                clean['MEWT_R2_mean'].values[0] if len(clean) > 0 else None,
                'Mean MEWT R²')
    axes[2].axhline(0, color='0.5', lw=0.8, ls=':')

    # Panel labels (A), (B), (C)
    for ax, letter in zip(axes, ['(A)', '(B)', '(C)']):
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='bottom', ha='left')

    # Shared legend below panels — use subplots_adjust, not tight_layout
    labels = [f'SNR = {int(s)}' for s in snr_vals]
    fig.legend(handles, labels, loc='lower center', ncol=len(snr_vals),
               fontsize=9, framealpha=0.9, edgecolor='0.8',
               bbox_to_anchor=(0.5, 0.0))
    plt.subplots_adjust(top=0.96, bottom=0.22, wspace=0.35)

    plt.savefig(os.path.join(output_dir, 'noise_sensitivity_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: noise_sensitivity_curves.png")


# ── Scatter plots ─────────────────────────────────────────────────────────────

def plot_scatter_by_snr(results_df, output_dir, snr_levels_to_plot=None):
    """
    Scatter: retrieved vs true MEWT for each SNR level.
    Saved to: output_dir/scatter_by_snr/scatter_snr_{value}.png
    """
    if snr_levels_to_plot is None:
        snr_levels_to_plot = sorted(results_df['snr'].unique(), reverse=True)

    scatter_dir = os.path.join(output_dir, 'scatter_by_snr')
    os.makedirs(scatter_dir, exist_ok=True)

    for snr in snr_levels_to_plot:
        d = results_df[results_df['snr'] == snr]
        if len(d) == 0:
            continue

        color = _snr_color(snr)
        label = _snr_label(snr)

        dmean = d.groupby(['L_true', 'epsilon_true', 'MEWT_true'],
                          as_index=False)['MEWT_retrieved'].mean()
        true_vals = dmean['MEWT_true'].values * 10
        ret_vals  = dmean['MEWT_retrieved'].values * 10
        mae_m, mae_s, nrmse_m, nrmse_s, r2_m, r2_s = _mewt_stats(d)

        fig, ax = plt.subplots(figsize=(5, 5))
        _clean_ax(ax)

        ax.scatter(true_vals, ret_vals, color=color, alpha=0.7, s=20, linewidths=0)
        lims = [min(true_vals.min(), ret_vals.min()) - 0.02,
                max(true_vals.max(), ret_vals.max()) + 0.02]
        ax.plot(lims, lims, 'k--', lw=1.2, label='1:1 line')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('True MEWT (mm)')
        ax.set_ylabel('Retrieved MEWT (mm)')
        ax.set_title(f'SNR = {label}', fontweight='bold')
        ax.legend(loc='upper left')
        ax.text(0.97, 0.05,
                f'MAE   = {mae_m:.3f} ± {mae_s:.3f} mm'
                f'\nNRMSE = {nrmse_m:.3f} ± {nrmse_s:.3f}'
                f'\nR²    = {r2_m:.3f} ± {r2_s:.3f}',
                transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='0.8', linewidth=0.8))

        plt.tight_layout()
        fname = f'scatter_snr_{"inf" if np.isinf(snr) else int(snr)}.png'
        plt.savefig(os.path.join(scatter_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: scatter_by_snr/{fname}")


# ── Spectral fit plots ────────────────────────────────────────────────────────


def plot_combined_2rows_5cols(results_df, samples, wl_valid, endmembers,
                               alpha, n, theta, output_dir,
                               perform_inversion_fn, add_noise_fn,
                               snr_levels=None):
    """
    Combined figure: scatter (top row) | spectral fits (bottom row).
    One column per SNR level. Saved to: output_dir/combined_2rows_5cols.png
    """
    if snr_levels is None:
        snr_levels = [np.inf, 600, 400, 200, 100]

    n_cols = len(snr_levels)
    mewt_colors = plt.cm.plasma(np.linspace(0.05, 0.92, len(samples)))

    fig = plt.figure(figsize=(14, 6.5))
    gs = GridSpec(2, n_cols, figure=fig,
                  height_ratios=[1, 1.3],
                  wspace=0.18, hspace=0.42,
                  left=0.07, right=0.98)

    fit_axes = []

    for col, snr in enumerate(snr_levels):
        color = _snr_color(snr)
        label = _snr_label(snr)

        # ── Top row: scatter ──────────────────────────────────────────────
        ax_s = fig.add_subplot(gs[0, col])
        _clean_ax(ax_s)

        d = results_df[results_df['snr'] == snr]
        dmean = d.groupby(['L_true', 'epsilon_true', 'MEWT_true'],
                          as_index=False)['MEWT_retrieved'].mean()
        true_v = dmean['MEWT_true'].values * 10
        ret_v  = dmean['MEWT_retrieved'].values * 10
        mae_m, mae_s, nrmse_m, nrmse_s, r2_m, r2_s = _mewt_stats(d)

        ax_s.scatter(true_v, ret_v, color=color, alpha=0.7, s=14, linewidths=0)
        lims = [min(true_v.min(), ret_v.min()) - 0.02,
                max(true_v.max(), ret_v.max()) + 0.02]
        ax_s.plot(lims, lims, 'k--', lw=1.0)
        ax_s.set_xlim(lims)
        ax_s.set_ylim(lims)
        ax_s.set_xlabel('True MEWT (mm)', fontsize=8)
        ax_s.set_title(f'SNR = {label}', fontweight='bold',
                       color=color, fontsize=10)
        ax_s.text(0.97, 0.05,
                  f'MAE={mae_m:.3f}±{mae_s:.3f}'
                  f'\nNRMSE={nrmse_m:.3f}±{nrmse_s:.3f}'
                  f'\nR²={r2_m:.3f}±{r2_s:.3f}',
                  transform=ax_s.transAxes, fontsize=6.5, va='bottom', ha='right',
                  bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                            edgecolor='0.8', linewidth=0.6))
        if col == 0:
            ax_s.set_ylabel('Retrieved MEWT (mm)', fontsize=8)
        else:
            ax_s.tick_params(labelleft=False)

        # Panel labels: (A)-(E) top row, (F)-(J) bottom row
        top_labels    = ['(A)', '(B)', '(C)', '(D)', '(E)']
        bottom_labels = ['(F)', '(G)', '(H)', '(I)', '(J)']
        ax_s.text(-0.10, 1.06, top_labels[col], transform=ax_s.transAxes,
                  fontsize=11, fontweight='bold', va='bottom', ha='left')

        # ── Bottom row: spectral fits ─────────────────────────────────────
        ax_f = fig.add_subplot(gs[1, col])
        _clean_ax(ax_f)
        fit_axes.append(ax_f)

        for mc, sample in zip(mewt_colors, samples):
            refl_clean   = sample['refl_clean']
            true_L       = sample['true_L']
            true_epsilon = sample['true_epsilon']
            true_MEWT    = true_L * true_epsilon

            refl_noisy = add_noise_fn(refl_clean, snr)
            try:
                L_est, eps_est, _, refl_pred = perform_inversion_fn(
                    refl_noisy, endmembers, alpha, n, theta
                )
            except Exception as e:
                print(f"  Fit failed SNR={label}, MEWT={true_MEWT*10:.2f} mm: {e}")
                continue

            for start, end in VALID_RANGES:
                mask = (wl_valid >= start) & (wl_valid <= end)
                ax_f.plot(wl_valid[mask], refl_noisy[mask],
                          color=mc, lw=0.6, alpha=0.45)
                ax_f.plot(wl_valid[mask], refl_pred[mask],
                          color=mc, lw=1.4, ls='--')

        ax_f.set_xlabel('Wavelength (nm)', fontsize=8)
        if col == 0:
            ax_f.set_ylabel('Reflectance', fontsize=8)
        else:
            ax_f.tick_params(labelleft=False)
        ax_f.text(-0.10, 1.04, bottom_labels[col], transform=ax_f.transAxes,
                  fontsize=11, fontweight='bold', va='bottom', ha='left')

    # Row labels on the left margin
    fig.text(0.015, 0.74, 'Retrieval Scatter', va='center',
             rotation='vertical', fontsize=10, fontweight='bold')
    fig.text(0.015, 0.30, 'Spectral Fits', va='center',
             rotation='vertical', fontsize=10, fontweight='bold')

    # Shared legend for spectral fits below the figure
    legend_lines, legend_labels = [], []
    for mc, sample in zip(mewt_colors, samples):
        true_MEWT = sample['true_L'] * sample['true_epsilon']
        line, = fit_axes[-1].plot([], [], color=mc, lw=1.2)
        legend_lines.append(line)
        legend_labels.append(f'MEWT = {true_MEWT*10:.2f} mm')
    meas_line, = fit_axes[-1].plot([], [], color='0.3', lw=1.2)
    model_line, = fit_axes[-1].plot([], [], color='0.3', lw=1.4, ls='--')
    legend_lines += [meas_line, model_line]
    legend_labels += ['measured', 'modeled']

    fig.legend(legend_lines, legend_labels,
               loc='lower center', ncol=len(legend_lines),
               fontsize=7.5, framealpha=0.9, edgecolor='0.8',
               bbox_to_anchor=(0.5, -0.04))

    plt.savefig(os.path.join(output_dir, 'combined_2rows_5cols.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: combined_2rows_5cols.png")


def plot_spectral_fits(samples, wl_valid, endmembers, alpha, n, theta,
                       output_dir, perform_inversion_fn, add_noise_fn,
                       snr_levels_to_plot=None):
    """
    One figure per SNR level. All 5 MEWT levels overlaid on a single axes,
    color-coded by MEWT. Legend placed outside to the right to avoid overlap.

    Saved to: output_dir/spectral_fits/spectral_fits_snr_{value}.png
    """
    if snr_levels_to_plot is None:
        snr_levels_to_plot = [np.inf, 600, 400, 200, 100]

    spectral_dir = os.path.join(output_dir, 'spectral_fits')
    os.makedirs(spectral_dir, exist_ok=True)

    mewt_colors = plt.cm.plasma(np.linspace(0.05, 0.92, len(samples)))

    for snr in snr_levels_to_plot:
        snr_str   = 'inf' if np.isinf(snr) else str(int(snr))
        snr_color = _snr_color(snr)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        _clean_ax(ax)
        ax.set_title(f'SNR = {_snr_label(snr)}',
                     fontsize=13, fontweight='bold', color=snr_color)

        for mewt_color, sample in zip(mewt_colors, samples):
            refl_clean    = sample['refl_clean']
            true_L        = sample['true_L']
            true_epsilon  = sample['true_epsilon']
            true_MEWT     = true_L * true_epsilon

            refl_noisy = add_noise_fn(refl_clean, snr)

            try:
                L_est, eps_est, _, refl_predicted = perform_inversion_fn(
                    refl_noisy, endmembers, alpha, n, theta
                )
            except Exception as e:
                print(f"  Spectral fit failed SNR={_snr_label(snr)}, "
                      f"MEWT={true_MEWT * 10:.2f} mm: {e}")
                continue

            MEWT_est = L_est * eps_est
            lbl = (f'MEWT = {true_MEWT * 10:.2f} mm  '
                   f'(retrieved: {MEWT_est * 10:.2f} mm)')

            for j, (start, end) in enumerate(VALID_RANGES):
                mask = (wl_valid >= start) & (wl_valid <= end)
                ax.plot(wl_valid[mask], refl_noisy[mask],
                        color=mewt_color, lw=0.8, alpha=0.45,
                        label=lbl if j == 0 else '_nolegend_')
                ax.plot(wl_valid[mask], refl_predicted[mask],
                        color=mewt_color, lw=2.0, ls='--',
                        label='_nolegend_')

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')

        # Legend outside the plot on the right
        leg = ax.legend(title='— measured   - - modeled',
                        loc='upper left',
                        bbox_to_anchor=(1.01, 1),
                        borderaxespad=0,
                        fontsize=9,
                        title_fontsize=9,
                        framealpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(spectral_dir, f'spectral_fits_snr_{snr_str}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: spectral_fits/spectral_fits_snr_{snr_str}.png")
