#!/bin/bash

# MARMIT EMIT Soil Moisture Retrieval Pipeline
#
# Usage:
#   bash run_emit_retrieval.sh \
#     --reflectance  <path/to/emit_reflectance>     \  # ENVI file (no extension)
#     --sza          <path/to/solar_zenith>          \  # ENVI file (no extension)
#     --mask         <path/to/soil_mask>             \  # ENVI mask file (no extension)
#     --smap         <path/to/smap.tif>              \  # SMAP GeoTIFF
#     --output       <path/to/output_dir>            \
#     --n_processes  <int>                              # parallel workers (default: auto)
#
# Prerequisites:
#   - EMIT reflectance scene downloaded from NASA Earthdata
#   - Soil mask applied (bare soil pixels only — see README)
#   - Corresponding SMAP soil moisture GeoTIFF for the same date

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Defaults (Utah scene, 2024-02-12) ----
REFL="$SCRIPT_DIR/data/emit/reflectance/EMIT_L2A_RFL_001_20240212T214534_2404314_009_ortho_subset_aid0000_reflectance"
SZA="$SCRIPT_DIR/data/emit/sza/EMIT_L1B_OBS_001_20240212T214534_2404314_009_ortho_subset_aid0000_obs"
MASK="$SCRIPT_DIR/data/emit/mask/soil_mask"
SMAP="$SCRIPT_DIR/data/smap/smap.tif"
OUT="$SCRIPT_DIR/output/emit_retrieval/utah_20240212"
N_PROCESSES=8

# ---- Parse arguments (override defaults) ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reflectance)  REFL="$2";        shift 2 ;;
        --sza)          SZA="$2";         shift 2 ;;
        --mask)         MASK="$2";        shift 2 ;;
        --smap)         SMAP="$2";        shift 2 ;;
        --output)       OUT="$2";         shift 2 ;;
        --n_processes)  N_PROCESSES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

BAD_BANDS="$SCRIPT_DIR/config/bad_bands_list.txt"
SOIL_SPECTRA="$SCRIPT_DIR/data/spectral_inputs/spectral_library.csv"
OPT_PROPS="$SCRIPT_DIR/data/spectral_inputs/water_optical_properties.csv"

CLEANED_DIR="$OUT/reflectance_cleaned"
MASKED_DIR="$OUT/masked"
RETRIEVAL_DIR="$OUT/retrieval"
SMAP_DIR="$OUT/smap_comparison"

mkdir -p "$CLEANED_DIR" "$MASKED_DIR" "$RETRIEVAL_DIR"

REFL_BASE=$(basename "$REFL")
SZA_BASE=$(basename "$SZA")

echo "============================================================"
echo "MARMIT EMIT Soil Moisture Retrieval"
echo "============================================================"

# === Step 1: Remove bad bands ===
echo ""
echo "== Step 1: Removing bad bands =="
CLEANED="$CLEANED_DIR/${REFL_BASE}_cleaned"
python "$SCRIPT_DIR/scripts/preprocess/remove_bad_bands.py" \
    "${REFL}.hdr" "${CLEANED}.hdr" "$BAD_BANDS"

# === Step 2: Apply soil mask to reflectance and SZA ===
if [[ -n "$MASK" ]]; then
    echo ""
    echo "== Step 2: Applying soil mask =="
    python "$SCRIPT_DIR/scripts/preprocess/apply_mask.py" \
        "${CLEANED}.hdr" "${MASK}.hdr" "${MASKED_DIR}/${REFL_BASE}_masked.hdr"
    python "$SCRIPT_DIR/scripts/preprocess/apply_mask.py" \
        "${SZA}.hdr" "${MASK}.hdr" "${MASKED_DIR}/${SZA_BASE}_masked.hdr"
    INPUT_REFL="${MASKED_DIR}/${REFL_BASE}_masked"
    INPUT_SZA="${MASKED_DIR}/${SZA_BASE}_masked"
else
    echo "== Step 2: No mask provided, using cleaned reflectance and original SZA =="
    INPUT_REFL="$CLEANED"
    INPUT_SZA="$SZA"
fi

# === Step 3: Run MARMIT retrieval ===
echo ""
echo "== Step 3: Running MARMIT retrieval =="
python "$SCRIPT_DIR/scripts/emit_retrieval/run_retrieval.py" \
    --input_reflectance_file  "$INPUT_REFL" \
    --soil_spectra_file       "$SOIL_SPECTRA" \
    --optical_properties_file "$OPT_PROPS" \
    --solar_zenith_file       "$INPUT_SZA" \
    --output_folder           "$RETRIEVAL_DIR" \
    --wavelength_ranges       "500-1320,1507-1766,2064-2300" \
    ${N_PROCESSES:+--n_processes $N_PROCESSES}

# === Step 4: Plot spectral fits ===
echo ""
echo "== Step 4: Plotting spectral fits =="
L_MAP="$RETRIEVAL_DIR/equivalent_water_layer_thickness_map"
MEWT_MAP="$RETRIEVAL_DIR/mean_equivalent_water_layer_thickness_map"
PRED_MAP="$RETRIEVAL_DIR/predicted_reflectance_map"

python "$SCRIPT_DIR/scripts/emit_retrieval/plot_spectra.py" \
    --l_map      "$L_MAP" \
    --predicted  "$PRED_MAP" \
    --measured   "$INPUT_REFL" \
    --l_epsilon  "$MEWT_MAP"

L_MAP="$RETRIEVAL_DIR/equivalent_water_layer_thickness_map"
MEWT_MAP="$RETRIEVAL_DIR/mean_equivalent_water_layer_thickness_map"
PRED_MAP="$RETRIEVAL_DIR/predicted_reflectance_map"

# === Step 5: Compare EMIT and SMAP (optional) ===
if [[ -n "$SMAP" ]]; then
    echo ""
    echo "== Step 5: Comparing EMIT and SMAP =="
    mkdir -p "$SMAP_DIR"
    python "$SCRIPT_DIR/scripts/smap_comparison/compare_smap_emit.py" \
        --emit_path    "${MEWT_MAP}.img" \
        --smap_path    "$SMAP" \
        --output_dir   "$SMAP_DIR" \
        --min_coverage 0.4

    # === Step 6: Fit logistic calibration curve ===
    PAIRED_CSV="$SMAP_DIR/emit_vs_smap_paired_values.csv"
    CALIB_DIR="$SMAP_DIR/calibration"

    if [[ -f "$PAIRED_CSV" ]]; then
        echo ""
        echo "== Step 6: Fitting logistic calibration curve =="
        mkdir -p "$CALIB_DIR"
        python "$SCRIPT_DIR/scripts/smap_comparison/fit_logistic_final.py" \
            --paired_csv "$PAIRED_CSV" \
            --output_dir "$CALIB_DIR"
    else
        echo "WARNING: Paired CSV not found, skipping calibration step."
    fi
fi

echo ""
echo "============================================================"
echo "DONE. Outputs saved to: $OUT"
echo "============================================================"


