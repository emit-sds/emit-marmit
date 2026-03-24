"""
MARMIT spectral unmixing retrieval (simulation version).

Retrieves soil moisture parameters (L, epsilon) by simultaneously unmixing
dry soil endmembers and fitting the MARMIT radiative transfer model via
LMFIT least-squares optimization. Shared by the simulation noise sensitivity
and EMIT retrieval pipelines.

Author: Nayma Binte Nur
"""

import numpy as np
import lmfit
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
from marmit_model import calc_refl


def linear_unmixing(params, endmembers):
    """Linearly mix spectral endmembers using lmfit parameters."""
    num     = endmembers.shape[0]
    weights = np.array([params[f'f_{i}'].value for i in range(num)])
    return np.dot(weights, endmembers)


def _init_params(endmembers):
    """Initialize lmfit Parameters with soil fraction and moisture variables."""
    num    = endmembers.shape[0]
    params = lmfit.Parameters()
    for i in range(num):
        params.add(f'f_{i}', value=1.0 / num)
    params.add('L',       min=0.001, max=0.2)
    params.add('epsilon', min=0.001, max=1.0)
    return params


def _residual(params, measured, endmembers, alpha, n, sza):
    """Residual vector for least-squares optimization."""
    L       = params['L']
    epsilon = params['epsilon']
    r_dry   = linear_unmixing(params, endmembers)
    modeled = calc_refl(alpha, L, n, sza, r_dry, epsilon)
    return (measured - modeled).flatten()


def perform_inversion(refl_meas, endmembers, alpha, n, sza, method='least_squares'):
    """
    Run MARMIT spectral unmixing inversion for one spectrum.

    Parameters
    ----------
    refl_meas  : array, measured reflectance (valid bands only)
    endmembers : array, [n_endmembers, n_bands]
    alpha      : array, water absorption coefficient at each band
    n          : array, water refractive index at each band
    sza        : float, solar zenith angle (degrees)
    method     : str, lmfit minimization method

    Returns
    -------
    L_opt       : float, retrieved water layer thickness (cm)
    epsilon_opt : float, retrieved wet fraction
    r_dry       : array, retrieved dry soil spectrum
    predicted   : array, modeled reflectance at retrieved parameters
    """
    params = _init_params(endmembers)

    result = lmfit.Minimizer(
        _residual,
        params,
        fcn_args=(refl_meas, endmembers, alpha, n, sza),
        nan_policy='omit'
    ).minimize(method=method)

    L_opt       = result.params['L'].value
    epsilon_opt = result.params['epsilon'].value
    r_dry       = linear_unmixing(result.params, endmembers)
    predicted   = calc_refl(alpha, L_opt, n, sza, r_dry, epsilon_opt)

    return L_opt, epsilon_opt, r_dry, predicted