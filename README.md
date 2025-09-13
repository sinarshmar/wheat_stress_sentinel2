## Weakly Supervised Wheat Stress Mapping (Ludhiana, Rabi 2023–24)

End-to-end pipeline using Sentinel-2, weak supervision and Random Forest. Two tracks: GEE for operational processing; Python for training and evaluation.

### Structure

```
├── Phase_1.js
├── Phase_2.js
├── Phase_3.js
├── Phase_4.js
├── Phase_5.js
├── Phase_5.5_&_6/
│   ├── GEE_exports/
│   │   ├── coordinates_labels_tiled_rabi_2023_24.csv
│   │   ├── predictors_image_tiled_rabi_2023_24-0000000000-0000000000.tif
│   │   ├── predictors_image_tiled_rabi_2023_24-0000000000-0000008704.tif
│   │   └── sampling_summary_rabi_2023_24.csv
│   ├── logs/
│   │   ├── phase_5_5_log.txt
│   │   └── phase_6_log.txt
│   ├── Phase_5.5_results/
│   │   ├── new_raster_statistics.csv
│   │   ├── phase_5_5_summary.txt
│   │   ├── predictors_image_MERGED.tif
│   │   ├── training_data_clean.csv
│   │   └── training_data_with_unmask.csv
│   ├── Phase_6_results/
│   │   ├── figures/
│   │   │   ├── confusion_roc.png
│   │   │   ├── feature_importance.png
│   │   │   └── model_comparison.png
│   │   ├── metrics/
│   │   │   ├── classification_report.json
│   │   │   ├── data_summary.json
│   │   │   ├── feature_importance.csv
│   │   │   ├── hyperparameter_search_results.csv
│   │   │   └── performance_metrics.json
│   │   ├── models/
│   │   │   ├── rf_model.pkl
│   │   │   └── rf_model_gee.json
│   │   └── phase_6_summary.txt
│   ├── Phase_5.5.py
│   ├── Phase_6.py
│   ├── tile_statistics_summary.csv
│   └── tree.py
│
├── Phase_7.js
│
├── Phase_7.5_Diagnose_phase_7_output/
│   ├── logs/
│   │   ├── diagnose_binary_log.txt
│   │   └── diagnose_probability_log.txt
│   ├── binary_classification_analysis.json
│   ├── daignose_prop.py
│   ├── diagnose_binary.py
│   ├── ludhiana_stress_map_rf_2023_24_binary.tif
│   ├── ludhiana_stress_map_rf_2023_24_probability_pct.tif
│   ├── probability_map_analysis.json
│   └── probability_map_analysis.png
├── Phase_8/
│   ├── logs/
│   │   └── phase_8_log.txt
│   ├── Phase_8_Validation/
│   │   ├── validation_guide.md
│   │   ├── validation_interface.html
│   │   ├── validation_points_complete.json
│   │   └── validation_summary.json
│   ├── Phase_8_validation_result/
│   │   └── validation_results.csv
│   ├── ludhiana_stress_map_rf_2023_24_probability_pct.tif
│   ├── Phase_8.py
│   └── tree.py
├── .gitignore
```

### Quickstart

**GEE (operational)**

1. Load `Phase_1.js` to `Phase_5.js` in the GEE Code Editor.
2. Set AOI, CRS and dates in `Phase_1.js`, then run sequentially to export tiles and labels.
3. After Python training, import `Phase_5.5_&_6/Phase_6_results/models/rf_model_gee.json` and run inference to export:

   * `*_probability_pct.tif` (0–100, uncalibrated)
   * `*_binary.tif` (thresholded)

**Python (research)**

```bash
# env
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib joblib rasterio geopandas shapely tqdm

# assemble training data
python Phase_5.5_&_6/Phase_5.5.py \
  --raster "Phase_5.5_&_6/GEE_exports/predictors_image_tiled_rabi_2023_24-*.tif" \
  --labels Phase_5.5_&_6/GEE_exports/coordinates_labels_tiled_rabi_2023_24.csv \
  --outdir Phase_5.5_&_6/Phase_5.5_results

# train and export model
python Phase_5.5_&_6/Phase_6.py \
  --train Phase_5.5_&_6/Phase_5.5_results/training_data_clean.csv \
  --outdir Phase_5.5_&_6/Phase_6_results --seed 42

# optional visual validation
python Phase_8/Phase_8.py --outdir Phase_8/Phase_8_validation_result
```

### Data sources

Sentinel-2 L2A SR, Sentinel-2 Cloud Probability, GAUL Level-2 (all via GEE).

### Wheat stress validtion webpage is hosted here:
https://wheatstressvalidation.tiiny.site/

### Licence

Code MIT. Text and figures CC BY 4.0. Third-party data under original licences.


### Detailed Technical Workflow
Check the fike Detailed_technical_workflow.png

