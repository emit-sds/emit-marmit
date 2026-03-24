# EMIT-MARMIT: Soil Moisture Retrieval from Orbital Hyperspectral Imagery

A physics-based pipeline for retrieving surface soil moisture from NASA EMIT Level-2A reflectance data \[2, 3\] using an adapted version of the MARMIT radiative transfer model \[1\].

The pipeline modifies MARMIT to work without an in-scene dry soil reference by fitting the dry soil spectrum dynamically from a spectral library. This makes the approach sensor- and site-agnostic and applicable to large-scale orbital datasets. Retrieved mean equivalent water thickness (MEWT = L × ε, in mm) is calibrated against SMAP Level-4 soil moisture products \[4\] using a logistic function.

This repository covers two main components:

1. **Simulation-based evaluation** — retrieval is tested on synthetic spectra across a range of soil moisture conditions and SNR levels to assess model accuracy and noise robustness.
2. **EMIT retrieval and SMAP comparison** — the model is applied to real EMIT scenes to produce per-pixel MEWT maps, which are then resampled to the SMAP grid and calibrated to volumetric soil moisture content.

---

## Repository Structure

```
emit-marmit/
├── config/
│   └── bad_bands_list.txt              # EMIT bad band wavelengths to exclude
├── data/
│   ├── spectral_inputs/
│   │   ├── spectral_library.csv        # Dry soil and NPV endmember library
│   │   └── water_optical_properties.csv # Water absorption coefficient and refractive index
│   ├── simulation/
│   │   └── simulated_reflectance.csv   # Synthetic spectra for noise sensitivity analysis
│   ├── emit/                           # Example scene: Sevier Dry Lake, Utah, 2024-02-12
│   │   ├── reflectance/                # EMIT L2A reflectance (ENVI, download separately)
│   │   ├── sza/                        # EMIT L1B solar zenith angle (ENVI, 1 band, 0–90°)
│   │   └── mask/
│   │       ├── soil_mask.hdr/.img      # Binary soil mask (1 = bare soil, 0 = exclude)
│   │       └── soil_mask.png           # Preview
│   └── smap/
│       └── smap.tif                    # SMAP L4 soil moisture (cm³/cm³), same date
├── scripts/
│   ├── model/
│   │   └── marmit_model.py             # MARMIT radiative transfer forward model
│   ├── preprocess/
│   │   ├── remove_bad_bands.py         # Remove atmospheric absorption bands from ENVI file
│   │   └── apply_mask.py               # Apply binary soil mask to ENVI file
│   ├── emit_retrieval/
│   │   ├── marmit_inversion.py         # LMFIT-based per-pixel spectral inversion
│   │   ├── run_retrieval.py            # Pixel-level retrieval with multiprocessing
│   │   └── plot_spectra.py             # Measured vs. modeled spectral fit plots
│   ├── simulation/
│   │   ├── run_noise_sensitivity_analysis.py  # SNR robustness evaluation
│   │   ├── marmit_inversion.py         # Inversion (simulation version)
│   │   ├── data_utils.py               # Data loading utilities
│   │   └── plot_results.py             # Noise sensitivity curves, scatter, spectral fits
│   └── smap_comparison/
│       ├── compare_smap_emit.py        # Resample EMIT to SMAP grid, export paired CSV
│       └── fit_logistic_final.py       # Logistic calibration and final SMC map
├── run_emit_retrieval.sh               # Main pipeline entry point
├── run_emit_retrieval_cluster.sh       # SLURM cluster submission script
├── run_noise_sensitivity.sh            # Noise sensitivity analysis wrapper
└── requirements.txt
```

---

## Installation

```bash
conda create -n emit_marmit python=3.10
conda activate emit_marmit
conda install -c conda-forge gdal
pip install -r requirements.txt
```

---

## Data Requirements

| File | Source |
|---|---|
| EMIT L2A reflectance | [NASA LP DAAC](https://doi.org/10.5067/EMIT/EMITL2ARFL.001) |
| EMIT L1B OBS (solar zenith) | NASA LP DAAC (same granule) |
| Soil mask | Binary mask isolating bare soil pixels (non-soil materials excluded) |
| SMAP L4 surface soil moisture | [NASA NSIDC](https://nsidc.org/data/smap) — closest 3-hour product to EMIT overpass |

Place EMIT and SMAP files under `data/emit/` and `data/smap/` respectively, or pass paths as arguments. The Utah scene (granule `EMIT_L2A_RFL_001_20240212T214534_2404314_009`) is used as the example scene throughout this repository — the outputs shown here were generated from it. A bare-soil mask for this scene is included under `data/emit/mask/` to allow reproduction: it targets bare soil pixels by excluding non-soil surface materials to the extent possible given the available masking inputs.

---

## Usage

### Part 1 — Simulation-Based Noise Sensitivity Analysis

Tests retrieval accuracy on synthetic spectra across a grid of soil moisture conditions and SNR levels (∞, 600, 400, 200, 100). For each noisy SNR level, 20 independent noise realizations are used to compute mean ± std statistics.

```bash
bash run_noise_sensitivity.sh
```

Outputs are saved to `output/simulation/noise_sensitivity/`.

### Part 2 — EMIT Retrieval and SMAP Comparison

Applies the MARMIT inversion to a real EMIT scene, produces per-pixel L and MEWT maps, and calibrates against SMAP soil moisture.

```bash
bash run_emit_retrieval.sh \
    --reflectance  <path/to/emit_reflectance>  \
    --sza          <path/to/emit_obs>           \
    --mask         <path/to/soil_mask>          \
    --smap         <path/to/smap.tif>           \
    --output       <path/to/output_dir>         \
    --n_processes  8
```

All arguments are optional and fall back to the defaults for the example Utah scene. `--n_processes` controls the number of parallel workers for the pixel-level inversion (default: 8).

**Pipeline steps:**

| Step | Script | Description |
|---|---|---|
| 1 | `remove_bad_bands.py` | Remove atmospheric absorption bands |
| 2 | `apply_mask.py` | Mask non-soil pixels in reflectance and SZA |
| 3 | `run_retrieval.py` | Per-pixel MARMIT inversion → L and ε maps (mm) |
| 4 | `plot_spectra.py` | Measured vs. modeled spectral fit plots |
| 5 | `compare_smap_emit.py` | Resample EMIT MEWT to SMAP grid (≥ 40% coverage threshold) |
| 6 | `fit_logistic_final.py` | Fit logistic calibration curve, output SMC map (cm³/cm³) |

Step 6 runs automatically after Step 5 if the paired CSV exists.

For SLURM cluster submission, use `run_emit_retrieval_cluster.sh` (submit from within the repository directory).

---

## Key Parameters

| Parameter | Range | Units | Description |
|---|---|---|---|
| L (water layer thickness) | 0.001 – 2 | mm | Equivalent uniform water film thickness |
| ε (wet fraction) | 0.001 – 1 | — | Fraction of pixel surface that is wet |
| MEWT = L × ε | — | mm | Mean equivalent water thickness (input to calibration) |
| Min SMAP coverage | 0.4 | — | Minimum fraction of valid EMIT sub-pixels per SMAP cell |
| Wavelength ranges used | 500–1320, 1507–1766, 2064–2300 | nm | Excludes atmospheric water vapor absorption bands |

---

## Outputs

### Part 1 — Simulation

```
output/simulation/noise_sensitivity/
├── noise_sensitivity_results.csv       # Per-sample retrieval results across all SNR levels
├── statistics_by_snr.csv               # Mean ± std of MAE, NRMSE, R² per SNR level
├── noise_sensitivity_curves.png        # MAE, NRMSE, R² vs. SNR
├── combined_2rows_5cols.png            # Scatter + spectral fits, one column per SNR level
├── scatter_by_snr/
│   ├── scatter_snr_inf.png
│   ├── scatter_snr_600.png
│   ├── scatter_snr_400.png
│   ├── scatter_snr_200.png
│   └── scatter_snr_100.png
└── spectral_fits/
    ├── spectral_fits_snr_inf.png
    ├── spectral_fits_snr_600.png
    ├── spectral_fits_snr_400.png
    ├── spectral_fits_snr_200.png
    └── spectral_fits_snr_100.png
```

### Part 2 — EMIT Retrieval

All outputs are written under `--output` (default: `output/emit_retrieval/utah_20240212/`).

```
output/emit_retrieval/<scene>/
├── reflectance_cleaned/                # Bad-band-removed reflectance (ENVI)
├── masked/                             # Soil-masked reflectance and SZA (ENVI)
├── retrieval/
│   ├── equivalent_water_layer_thickness_map.{hdr,img,png}       # L map (mm)
│   ├── mean_equivalent_water_layer_thickness_map.{hdr,img,png}  # L×ε map (mm)
│   ├── predicted_reflectance_map.{hdr,img}
│   ├── r_dry_optimized_map.{hdr,img}
│   ├── wavelengths_used.txt
│   └── pixel_spectra_plots/            # Measured vs. modeled spectra (PNG + CSV)
└── smap_comparison/
    ├── emit_vs_smap_paired_values.csv  # Paired MEWT and SMAP SMC values
    ├── resampled_emit_common_valid.png # EMIT MEWT resampled to SMAP grid (preview)
    ├── smap_common_valid.png           # Corresponding SMAP pixels (preview)
    └── calibration/
        ├── calibration.png             # Logistic fit: MEWT vs. SMAP SMC
        ├── scatter.png                 # EMIT-derived vs. SMAP SMC validation
        └── combined.png
```

All retrieval maps (L and MEWT) are in mm.

---

## Model Overview

MARMIT \[1\] models wet soil reflectance as a linear mixture of dry and wet components:

```
R_model = ε · R_wet + (1 − ε) · R_dry
```

where `R_wet` is computed from Fresnel interface coefficients and Beer–Lambert water transmittance (`exp(−α·L)`). The dry soil spectrum `R_dry` is fit as a convex combination of spectral library endmembers, removing the need for an in-scene dry reference:

```
R_dry = Σ wᵢ · R_library,ᵢ    (wᵢ ≥ 0)
```

Inversion minimizes the squared spectral residual across all valid bands:

```
min_{L, ε, w}  Σ_λ [ R_obs(λ) − R_model(λ; L, ε, w) ]²
```

MEWT is then calibrated to volumetric soil moisture content via a logistic function fitted to paired SMAP observations.

---

## Citation

If you use this code, please cite:

> Nur, N. B., Thompson, D. R., Brodrick, P. G., Turmon, M., Carmon, N., Green, R. O., Bachmann, C. M., & Keebler, A. M. Soil Moisture from Orbital Imaging Spectroscopy. *Remote Sensing of Environment* (in review).

---

## References

\[1\] Bablet, A., Jacquemoud, S., Baret, F., & Olioso, A. (2018). The MARMIT model: A physical radiative transfer model for soil moisture and roughness from hyperspectral data. *Remote Sensing of Environment*, 217, 1–16. doi:10.1016/j.rse.2018.08.020

\[2\] Green, R. O., Mahowald, N., Ung, C., Thompson, D. R., et al. (2020). The Earth Surface Mineral Dust Source Investigation: An Earth science imaging spectroscopy mission. *2020 IEEE Aerospace Conference*, 1–15.

\[3\] Green, R. O. (2022). EMIT L2A estimated surface reflectance and uncertainty and masks 60 m v001. NASA LP DAAC. doi:10.5067/EMIT/EMITL2ARFL.001

\[4\] Entekhabi, D., Njoku, E. G., O'Neill, P. E., Kellogg, K. H., Crow, W. T., et al. (2010). The Soil Moisture Active Passive (SMAP) mission. *Proceedings of the IEEE*, 98, 704–716. doi:10.1109/JPROC.2010.2043918

---

## Acknowledgments

This work was carried out at the Jet Propulsion Laboratory, California Institute of Technology, under contract with NASA (80NM0018D0004). The EMIT mission is sponsored by the NASA Earth Science Division. Copyright 2025 California Institute of Technology. All Rights Reserved.