#!/usr/bin/env python
"""
Noise sensitivity analysis for MARMIT retrieval.

Inverts simulated reflectance spectra across a grid of L and epsilon values
at multiple SNR levels (inf, 600, 400, 200, 100) to assess retrieval robustness.
For each noisy SNR level, 20 independent noise realizations are used to compute
mean +/- std statistics.

Noise model: signal-dependent Gaussian noise per band
    noise_std[band] = (1 / SNR) * reflectance[band]

Outputs saved to output/simulation/noise_sensitivity/:
  - noise_sensitivity_results.csv
  - statistics_by_snr.csv
  - noise_sensitivity_curves.png
  - scatter_by_snr/
  - spectral_fits/

Author: Nayma Binte Nur
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os

from data_utils import load_water_properties, load_spectra, load_bad_bands, build_good_mask
from marmit_inversion import perform_inversion
from plot_results import (plot_noise_sensitivity_curves, plot_scatter_by_snr,
                          plot_spectral_fits, plot_combined_2rows_5cols)


# ============================================
# Noise
# ============================================

def add_noise(spectrum, snr):
    """
    Add signal-dependent Gaussian noise at a given SNR.

    noise_std[band] = (1/SNR) * reflectance[band]

    Parameters
    ----------
    spectrum : array, clean reflectance spectrum
    snr      : float, SNR value; use np.inf for no noise

    Returns
    -------
    noisy_spectrum : array
    """
    if np.isinf(snr):
        return spectrum.copy()
    noise_std = spectrum / snr
    noise = np.random.normal(0, noise_std, size=spectrum.shape)
    return np.maximum(spectrum + noise, 0.0)


# ============================================
# Noise Sensitivity Loop
# ============================================

def run_noise_sensitivity(sim_df, wl_cols_valid, endmembers, alpha, n, theta,
                          snr_levels, output_dir, n_realizations=20):
    """
    Run retrieval across ALL simulated spectra at each SNR level.

    For noisy SNR levels, the analysis is repeated n_realizations times with
    independent noise draws so that statistics (mean ± std) can be computed
    across realizations. SNR = inf is run once (no noise, deterministic).

    Parameters
    ----------
    sim_df          : DataFrame, simulated reflectance dataset
    wl_cols_valid   : list, valid wavelength column names
    endmembers      : array, [n_endmembers, n_valid_bands]
    alpha           : array, water absorption coefficient at valid bands
    n               : array, water refractive index at valid bands
    theta           : float, solar zenith angle (degrees)
    snr_levels      : list, SNR values to test (np.inf = no noise)
    output_dir      : str, output directory
    n_realizations  : int, number of independent noise realizations per SNR

    Returns
    -------
    DataFrame with retrieval results for all samples, SNR levels, realizations
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    n_total = len(sim_df)

    for snr in snr_levels:
        snr_label = 'inf' if np.isinf(snr) else str(int(snr))
        n_real = 1 if np.isinf(snr) else n_realizations
        print(f"  SNR {snr_label:>6}  ({n_total} spectra × {n_real} realizations)")

        for realization in range(n_real):
            for idx, row in sim_df.iterrows():
                true_L       = row['L']
                true_epsilon = row['epsilon']
                true_MEWT    = row['mean_equivalent_water_thickness']
                refl_clean   = row[wl_cols_valid].values.astype(float)

                refl = add_noise(refl_clean, snr)

                try:
                    L_est, eps_est, _, _ = perform_inversion(refl, endmembers, alpha, n, theta)
                    results.append({
                        'snr':               snr,
                        'realization':       realization,
                        'L_true':            true_L,
                        'epsilon_true':      true_epsilon,
                        'MEWT_true':         true_MEWT,
                        'L_retrieved':       L_est,
                        'epsilon_retrieved': eps_est,
                        'MEWT_retrieved':    L_est * eps_est
                    })
                except Exception as e:
                    print(f"    Failed real={realization} idx={idx}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'noise_sensitivity_results.csv'), index=False)
    return df


# ============================================
# Statistics
# ============================================

def calculate_statistics(df, snr_levels):
    stats = []

    def _metrics(true, retrieved):
        err       = true - retrieved
        mae       = np.mean(np.abs(err))
        val_range = true.max() - true.min()
        nrmse     = np.sqrt(np.mean(err**2)) / val_range if val_range > 0 else np.inf
        ss_tot    = np.sum((true - np.mean(true))**2)
        r2        = 1 - np.sum(err**2) / ss_tot if ss_tot > 0 else -np.inf
        return mae, nrmse, r2

    for snr in snr_levels:
        d = df[df['snr'] == snr]
        if len(d) == 0:
            continue

        per_real = []
        for real in sorted(d['realization'].unique()):
            dr = d[d['realization'] == real]
            mae, nrmse, r2 = _metrics(dr['MEWT_true'].values, dr['MEWT_retrieved'].values)
            per_real.append({'mae': mae, 'nrmse': nrmse, 'r2': r2})

        pr = pd.DataFrame(per_real)
        stats.append({
            'snr':             snr,
            'n_realizations':  len(pr),
            'n_samples':       len(d) // len(pr),
            'MEWT_MAE_mean':   pr['mae'].mean(),
            'MEWT_MAE_std':    pr['mae'].std(ddof=1) if len(pr) > 1 else 0.0,
            'MEWT_NRMSE_mean': pr['nrmse'].mean(),
            'MEWT_NRMSE_std':  pr['nrmse'].std(ddof=1) if len(pr) > 1 else 0.0,
            'MEWT_R2_mean':    pr['r2'].mean(),
            'MEWT_R2_std':     pr['r2'].std(ddof=1) if len(pr) > 1 else 0.0,
        })

    return pd.DataFrame(stats)


# ============================================
# Main
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("NOISE SENSITIVITY ANALYSIS — SPECTRAL UNMIXING")
    print("=" * 60)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    repo_root   = os.path.join(script_dir, '..', '..')
    config_dir  = os.path.join(repo_root, 'config')
    data_common = os.path.join(repo_root, 'data', 'spectral_inputs')
    data_sim    = os.path.join(repo_root, 'data', 'simulation')
    output_dir  = os.path.join(repo_root, 'output', 'simulation', 'noise_sensitivity')
    os.makedirs(output_dir, exist_ok=True)

    # Load simulated reflectance
    print("\nLoading simulated reflectance...")
    sim_df      = pd.read_csv(os.path.join(data_sim, 'simulated_reflectance.csv'))
    meta_cols   = ['L', 'epsilon', 'mean_equivalent_water_thickness']
    wl_cols_all = [col for col in sim_df.columns if col not in meta_cols]
    wl_all      = np.array([float(col) for col in wl_cols_all])

    # Apply bad band mask
    bad_bands      = load_bad_bands(os.path.join(config_dir, 'bad_bands_list.txt'))
    good_mask      = build_good_mask(wl_all, bad_bands)
    wl_valid       = wl_all[good_mask]
    wl_cols_valid  = [wl_cols_all[i] for i in range(len(wl_cols_all)) if good_mask[i]]

    # Load water optical properties and interpolate to valid bands
    wl_water, alpha_water, n_water = load_water_properties(
        os.path.join(data_common, 'water_optical_properties.csv')
    )
    alpha_interp = interp1d(wl_water, alpha_water, kind='linear', fill_value='extrapolate')(wl_valid)
    n_interp     = interp1d(wl_water, n_water,     kind='linear', fill_value='extrapolate')(wl_valid)

    # Load spectral library and interpolate to valid bands
    wl_lib, lib_spectra   = load_spectra(os.path.join(data_common, 'spectral_library.csv'))
    endmembers_interp     = np.zeros((lib_spectra.shape[0], len(wl_valid)))
    for i in range(lib_spectra.shape[0]):
        endmembers_interp[i, :] = interp1d(
            wl_lib, lib_spectra[i, :], kind='linear', fill_value='extrapolate'
        )(wl_valid)

    theta      = 30.0  # solar zenith angle (degrees), same as simulation
    snr_levels = [np.inf, 600, 400, 200, 100]

    n_realizations = 20
    print(f"\nSNR levels: {['inf' if np.isinf(s) else int(s) for s in snr_levels]},  Noise realizations: {n_realizations}")

    # Run retrieval
    print("\nRunning noise sensitivity analysis...")
    results_df = run_noise_sensitivity(
        sim_df=sim_df,
        wl_cols_valid=wl_cols_valid,
        endmembers=endmembers_interp,
        alpha=alpha_interp,
        n=n_interp,
        theta=theta,
        snr_levels=snr_levels,
        output_dir=output_dir,
        n_realizations=n_realizations
    )

    # Statistics (mean ± std across realizations)
    print("\nCalculating statistics...")
    stats_df = calculate_statistics(results_df, snr_levels)
    stats_df.to_csv(os.path.join(output_dir, 'statistics_by_snr.csv'), index=False)
    print(stats_df[['snr', 'n_realizations', 'n_samples',
                     'MEWT_MAE_mean', 'MEWT_MAE_std',
                     'MEWT_NRMSE_mean', 'MEWT_R2_mean']].to_string(index=False))

    # Realization 0 used for illustrative scatter and spectral fit plots
    results_r0 = results_df[results_df['realization'] == 0].copy()

    # Plots
    print("\nGenerating plots...")
    plot_noise_sensitivity_curves(stats_df, output_dir)
    plot_scatter_by_snr(results_r0, output_dir)

    # Spectral fits: 5 representative samples spanning the MEWT range
    fit_params = [
        (0.025, 0.1),  # MEWT = 0.025 mm — very dry
        (0.050, 0.3),  # MEWT = 0.15  mm — slightly moist
        (0.050, 0.6),  # MEWT = 0.30  mm — moderate
        (0.100, 0.6),  # MEWT = 0.60  mm — wet
        (0.200, 0.8),  # MEWT = 1.60  mm — very wet
    ]
    fit_samples = []
    for L_val, eps_val in fit_params:
        match = sim_df[np.isclose(sim_df['L'], L_val) & np.isclose(sim_df['epsilon'], eps_val)]
        if len(match) == 0:
            print(f"  Warning: sample L={L_val}, epsilon={eps_val} not found in sim_df")
            continue
        fit_samples.append({
            'refl_clean':   match.iloc[0][wl_cols_valid].values.astype(float),
            'true_L':       L_val,
            'true_epsilon': eps_val
        })

    plot_spectral_fits(
        samples=fit_samples,
        wl_valid=wl_valid,
        endmembers=endmembers_interp,
        alpha=alpha_interp,
        n=n_interp,
        theta=theta,
        output_dir=output_dir,
        perform_inversion_fn=perform_inversion,
        add_noise_fn=add_noise
    )

    shared_fit_args = dict(
        samples=fit_samples,
        wl_valid=wl_valid,
        endmembers=endmembers_interp,
        alpha=alpha_interp,
        n=n_interp,
        theta=theta,
        output_dir=output_dir,
        perform_inversion_fn=perform_inversion,
        add_noise_fn=add_noise,
        snr_levels=snr_levels,
    )

    # 2 rows × 5 cols: scatter (top) | fits (bottom), one column per SNR
    plot_combined_2rows_5cols(results_df=results_df, **shared_fit_args)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - noise_sensitivity_results.csv")
    print("  - statistics_by_snr.csv")
    print("  - noise_sensitivity_curves.png")
    print("  - scatter_by_snr/")
    print("  - spectral_fits/")