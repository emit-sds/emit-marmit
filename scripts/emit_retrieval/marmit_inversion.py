"""
MARMIT spectral inversion for EMIT per-pixel retrieval.

Retrieves water layer thickness (L, cm) and wet fraction (epsilon) by fitting
the MARMIT forward model to measured reflectance via LMFIT least-squares
optimization with simultaneous spectral library unmixing for the dry soil component.

Author: Nayma Binte Nur
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))

import lmfit
import numpy as np
import pandas as pd
from marmit_model import calc_refl


def load_spectra(csv_file):
    """Load spectral endmembers from CSV file."""
    df = pd.read_csv(csv_file)
    wavelengths = df.columns.astype(float)
    spectra = df.values
    return wavelengths, spectra


def linear_unmixing(params, endmembers):
    """Linearly mix spectral endmembers with parameters."""
    num = endmembers.shape[0]
    weights = np.array([params[f"f_{i}"].value for i in range(num)])
    return np.dot(weights, endmembers)


def optimize_soil_fractions(endmembers):
    """Initialize soil fraction parameters for LMFIT."""
    num = endmembers.shape[0]
    params = lmfit.Parameters()
    for i in range(num):
        params.add(f"f_{i}", value=1.0 / num)
    return params


def residual_vector(params, measured, endmembers, alpha, n, sza):
    """Residuals used for least-squares optimization."""
    L = params['L']
    epsilon = params['epsilon']
    r_dry = linear_unmixing(params, endmembers)
    estimated = calc_refl(alpha, L, n, sza, r_dry, epsilon)
    return (measured - estimated).flatten()


def perform_inversion(refl_meas, endmembers, alpha, n, sza, method="least_squares"):
    """Run optimization for one sample with the specified LMFIT method."""
    params = optimize_soil_fractions(endmembers)
    params.add('L', min=0.001, max=0.2)
    params.add('epsilon', min=0.001, max=1)

    mini = lmfit.Minimizer(
        residual_vector,
        params,
        fcn_args=(refl_meas, endmembers, alpha, n, sza),
        nan_policy='omit'
    )

    result = mini.minimize(method=method)

    L_opt = result.params['L'].value
    epsilon_opt = result.params['epsilon'].value
    r_dry = linear_unmixing(result.params, endmembers)
    predicted = calc_refl(alpha, L_opt, n, sza, r_dry, epsilon_opt)

    return L_opt, epsilon_opt, r_dry, predicted