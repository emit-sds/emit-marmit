"""
Data loading and preprocessing utilities for the simulation pipeline.

Provides loaders for water optical properties, spectral endmember libraries,
bad band lists, and a utility to build a valid-band boolean mask.

Author: Nayma Binte Nur
"""

import numpy as np
import pandas as pd


def load_water_properties(csv_path):
    """Load water absorption coefficient and refractive index from CSV."""
    df = pd.read_csv(csv_path)
    wl    = df['Wavelength (nm)'].values
    alpha = df['absorption coefficient'].values
    n     = df['relative refractive index'].values
    return wl, alpha, n


def load_spectra(csv_path):
    """Load spectral endmember library from CSV. Returns (wavelengths, spectra)."""
    df = pd.read_csv(csv_path, header=0)
    wavelengths = df.columns.astype(float).values
    spectra     = df.values
    return wavelengths, spectra


def load_bad_bands(txt_path):
    """Load bad band center wavelengths (nm) from text file."""
    with open(txt_path) as f:
        return np.array([float(line.strip()) for line in f if line.strip()])


def build_good_mask(wl_array, bad_bands, tol=1.0):
    """
    Build boolean mask: True = valid band, False = bad band.

    Parameters
    ----------
    wl_array  : array, wavelength values (nm)
    bad_bands : array, bad band center wavelengths (nm)
    tol       : float, matching tolerance in nm

    Returns
    -------
    good_mask : bool array, same length as wl_array
    """
    good_mask = np.ones(len(wl_array), dtype=bool)
    for bb in bad_bands:
        closest = np.argmin(np.abs(wl_array - bb))
        if np.abs(wl_array[closest] - bb) < tol:
            good_mask[closest] = False
    return good_mask