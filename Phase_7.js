/**
 * Phase 7 – GEE-native RF mapping (informed by Phase 6)
 * This phase TRAINS a new Random Forest inside GEE using the Phase-4/6 lenient rules.
 * It does NOT load sklearn trees. Differences vs Phase 6:
 *  - Own training sample (GEE sampled)
 *  - No sample weights (GEE limitation)
 *  - RF implementation (SMILE) ≠ sklearn (minor behavioural diffs)
 */


// Import previous phases
var p1 = require('users/lalutatu/wheatstress:Phase_1');
var p2 = require('users/lalutatu/wheatstress:Phase_2');
var p3 = require('users/lalutatu/wheatstress:Phase_3');
var p4 = require('users/lalutatu/wheatstress:Phase_4');

/* ========= 0) CONFIGURATION ========= */
var PARAMS7 = {
  // Model parameters from Phase 6 training (UPDATED from new run)
  modelSource: 'Phase6_Python_3049samples',  
  
  // RF Model Configuration (from NEW rf_model_gee.json)
  n_estimators: 50,
  max_depth: null,  // CHANGED: unlimited depth performed better
  min_samples_split: 20,  // CHANGED: increased from 5
  min_samples_leaf: 8,
  max_features: 0.5,  // CHANGED: 50% of features instead of log2
  
  // Feature names (must match training exactly)
  feature_names: [
    'NDVI_AUC', 'NDVI_drop', 'NDVI_peak', 'NDVI_slopeEarly',
    'NDWI_AUC', 'NDWI_drop', 'NDWI_peak',
    'GNDVI_AUC', 'GNDVI_drop',
    'SAVI_AUC', 'MSAVI2_AUC',
    'h_NDVI_drop', 'h_NDWI_drop', 'h_AUC_z', 'h_NDVI_AUC_cur'
  ],
  
  // UPDATED Feature importances from Phase 6 (3,049 samples)
  feature_importances: [
    0.0054, 0.1669, 0.0096, 0.0024,  // NDVI features
    0.0019, 0.3793, 0.0337,           // NDWI features (NDWI_drop now dominant!)
    0.0046, 0.0042,                   // GNDVI features
    0.0039, 0.0023,                   // SAVI, MSAVI2
    0.1092, 0.2674, 0.0032, 0.0061    // Helper features
  ],
  
  // Output configuration
  outputScale: 10,  // Keep 10m resolution
  outputCRS: 'EPSG:32643',
  
  // Classification thresholds
  stressThreshold: 0.5,  // Probability threshold for binary classification
  
  // Visualization
  showMaps: true,
  showStats: true,
  
  // Export configuration
  exportFolder: 'wheat_stress_phase7_final',  // Updated folder name
  exportPrefix: 'ludhiana_stress_map_rf_2023_24',
  
  // Training parameters for GEE RF
  trainingSampleSize: 10000,  // Increase from 5000 for better representation
  trainingSampleScale: 20,     // Match Phase 5 sampling scale
  trainingSeed: 42
};

/* ========= 1) GET BASE DATA ========= */
print('========== PHASE 7: FULL MAP INFERENCE (UPDATED) ==========');
print('Using RF parameters optimized from Phase 6 training');

var aoi = p1.aoi;
var cropMask = p1.cropMask;
var metrics = p2.metricsImage;
var helpers = p3.anomalyHelpers;

print('\n1. Loaded base data from previous phases');
print('   AOI:', aoi.area().divide(1e6), 'km²');

// Count cropland pixels at inference scale
var cropPixelCount = cropMask.reduceRegion({
  reducer: ee.Reducer.count(),
  geometry: aoi,
  scale: PARAMS7.outputScale,
  maxPixels: 1e10
});
print('   Cropland pixels at 10m:', cropPixelCount.get('cropMask'));

/* ========= 2) PREPARE PREDICTOR STACK ========= */
print('\n2. Preparing predictor stack...');

// Select and rename features to match training
var predictors = ee.Image.cat([
  // From Phase 2 metrics
  metrics.select(['NDVI_AUC', 'NDVI_drop', 'NDVI_peak', 'NDVI_slopeEarly']),
  metrics.select(['NDWI_AUC', 'NDWI_drop', 'NDWI_peak']),
  metrics.select(['GNDVI_AUC', 'GNDVI_drop']),
  metrics.select(['SAVI_AUC', 'MSAVI2_AUC']),
  
  // From Phase 3 helpers (with h_ prefix)
  helpers.select(['NDVI_drop', 'NDWI_drop', 'AUC_z', 'NDVI_AUC_cur'])
    .rename(['h_NDVI_drop', 'h_NDWI_drop', 'h_AUC_z', 'h_NDVI_AUC_cur'])
])
.updateMask(cropMask)
.clip(aoi);

print('   Predictor bands:', predictors.bandNames());

// Verify all features are present
var missingFeatures = ee.List(PARAMS7.feature_names)
  .removeAll(predictors.bandNames());
print('   Missing features check:', missingFeatures);

/* ========= 3) APPLY RULE-BASED BASELINE ========= */
print('\n3. Applying rule-based baseline for comparison...');

// Phase 4 thresholds (matching Phase 6 baseline)
var NDVI_DROP_THR = 0.02;
var NDWI_DROP_THR = 0.01;
var AUC_Z_THR = 1.5;

// Apply rules (OR logic from Phase 4)
var primary = predictors.select('NDVI_drop').gt(NDVI_DROP_THR);
var auxiliary = predictors.select('NDWI_drop').gt(NDWI_DROP_THR)
  .and(predictors.select('h_AUC_z').lt(AUC_Z_THR));

var baselineStress = primary.or(auxiliary)
  .rename('baseline_stress')
  .updateMask(cropMask);

// Calculate baseline statistics
var baselineStats = baselineStress.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: aoi,
  scale: 100,  // Coarser scale for speed
  maxPixels: 1e10
});

print('   Baseline stress fraction:', baselineStats.get('baseline_stress'));

/* ========= 4) RANDOM FOREST CLASSIFICATION ========= */
print('\n4. Setting up Random Forest classifier...');

// Prepare training data using Phase 4 labels with higher quality sampling
var labelImage = p4.label01;
var weightImage = p4.weight;

// Sample MORE points for better representation
var training = predictors.addBands(labelImage)
  .addBands(weightImage)
  .sample({
    region: aoi,
    scale: PARAMS7.trainingSampleScale,  // 20m to match Phase 5
    numPixels: PARAMS7.trainingSampleSize,  // 10000 samples
    seed: PARAMS7.trainingSeed,
    tileScale: 2  // Reduced for better memory management
  });

print('   Training samples collected:', training.size());

// Calculate actual feature count for variablesPerSplit
var numFeatures = ee.List(PARAMS7.feature_names).size();
var variablesPerSplit = numFeatures.multiply(PARAMS7.max_features).round();  // 50% of features

print('   Variables per split:', variablesPerSplit);

// Train classifier with UPDATED parameters
// Note: GEE doesn't support null max_depth, so we use a very large value
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: PARAMS7.n_estimators,
  variablesPerSplit: 7,  // 50% of 15 features ≈ 7
  minLeafPopulation: PARAMS7.min_samples_leaf,
  bagFraction: 0.8,
  maxNodes: 100000,  // Very large to simulate unlimited depth
  seed: PARAMS7.trainingSeed
})
.train({
  features: training,
  classProperty: 'label01',
  inputProperties: PARAMS7.feature_names
});

print('   Classifier trained with updated parameters');

// Get variable importance from trained model
var importance = classifier.explain();
print('   GEE RF variable importance:', importance);

/* ========= 5) APPLY CLASSIFIER ========= */
print('\n5. Applying classifier to full image...');

// Classify the entire image
var classified = predictors
  .select(PARAMS7.feature_names)
  .classify(classifier)
  .rename('rf_stress_class');

// Get probability for stressed class
var classifierProb = ee.Classifier.smileRandomForest({
  numberOfTrees: PARAMS7.n_estimators,
  variablesPerSplit: 7,  // 50% of features
  minLeafPopulation: PARAMS7.min_samples_leaf,
  bagFraction: 0.8,
  maxNodes: 100000,
  seed: PARAMS7.trainingSeed
})
.setOutputMode('PROBABILITY')
.train({
  features: training,
  classProperty: 'label01',
  inputProperties: PARAMS7.feature_names
});

var probability = predictors
  .select(PARAMS7.feature_names)
  .classify(classifierProb)
  .rename('stress_probability')
  .updateMask(cropMask);

print('   Classification complete');

/* ========= 6) CALCULATE STATISTICS ========= */
if (PARAMS7.showStats) {
  print('\n6. Calculating map statistics...');
  
  // RF classification stats at 100m for speed
  var rfStats = classified.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: aoi,
    scale: 100,
    maxPixels: 1e10
  });
  
  // Probability stats
  var probStats = probability.reduceRegion({
    reducer: ee.Reducer.percentile([5, 25, 50, 75, 95])
      .combine(ee.Reducer.mean(), '', true)
      .combine(ee.Reducer.stdDev(), '', true),
    geometry: aoi,
    scale: 100,
    maxPixels: 1e10
  });
  
  print('   RF stress fraction:', rfStats.get('rf_stress_class'));
  print('   Probability statistics:', probStats);
  
  // Area calculations at full resolution
  var pixelArea = ee.Image.pixelArea().divide(10000); // Convert to hectares
  
  var stressedArea = pixelArea
    .updateMask(classified.eq(1))
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: aoi,
      scale: PARAMS7.outputScale,
      maxPixels: 1e13,
      bestEffort: true
    });
  
  var totalArea = pixelArea
    .updateMask(cropMask)
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: aoi,
      scale: PARAMS7.outputScale,
      maxPixels: 1e13,
      bestEffort: true
    });
  
  print('   Stressed area (ha):', stressedArea.get('area'));
  print('   Total crop area (ha):', totalArea.get('area'));
  
  var stressPct = ee.Number(stressedArea.get('area'))
    .divide(totalArea.get('area')).multiply(100);
  print('   Stressed percentage:', stressPct);
}

/* ========= 7) VISUALIZATION ========= */
if (PARAMS7.showMaps) {
  print('\n7. Adding visualization layers...');
  
  Map.centerObject(aoi, 10);
  
  // Add crop mask
  Map.addLayer(cropMask.selfMask(), 
    {palette: ['lightgray'], opacity: 0.5}, 
    '1. Crop Mask', false);
  
  // Add baseline stress (binary)
  Map.addLayer(baselineStress.selfMask(), 
    {palette: ['orange'], min: 0, max: 1}, 
    '2. Baseline Stress (Rule-based)', false);
  
  // Add RF classification (binary)
  Map.addLayer(classified.selfMask(), 
    {palette: ['green', 'red'], min: 0, max: 1}, 
    '3. RF Classification (Binary)', true);
  
  // Add stress probability (continuous) - ENHANCED visualization
  Map.addLayer(probability, 
    {palette: ['#2E7D32', '#66BB6A', '#FDD835', '#FF8F00', '#D84315'], 
     min: 0, max: 1}, 
    '4. Stress Probability (0-1)', true);
  
  // Add high confidence stress (>80% probability) - UPDATED threshold
  var highConfidenceStress = probability.gt(0.8).selfMask();
  Map.addLayer(highConfidenceStress, 
    {palette: ['darkred']}, 
    '5. High Confidence Stress (>80%)', true);
  
  // Add moderate confidence stress (50-80% probability)
  var moderateConfidenceStress = probability.gt(0.5).and(probability.lte(0.8)).selfMask();
  Map.addLayer(moderateConfidenceStress, 
    {palette: ['orange']}, 
    '6. Moderate Confidence Stress (50-80%)', false);
  
  // Add district boundary
  Map.addLayer(aoi, {color: 'black'}, '7. District Boundary', true);
  
  // Add legend info
  print('   Legend:');
  print('   - Green: Healthy (0-20% stress probability)');
  print('   - Yellow: Low stress (20-50%)');
  print('   - Orange: Moderate stress (50-80%)');
  print('   - Red: High stress (>80%)');
}

/* ========= 8) CREATE EXPORT COMPOSITES ========= */
print('\n8. Preparing export composites...');

// Create multi-band export image with consistent data types
var exportImage = ee.Image.cat([
  // Classifications
  baselineStress.rename('baseline_stress').toFloat(),
  classified.rename('rf_stress_class').toFloat(),
  probability.rename('stress_probability').toFloat(),
  
  // Key predictors for validation
  predictors.select(['NDVI_drop', 'NDWI_drop', 'h_AUC_z']).toFloat(),
  
  // Add NDVI and NDWI peak values for context
  predictors.select(['NDVI_peak', 'NDWI_peak']).toFloat()
])
.updateMask(cropMask)
.clip(aoi);

print('   Export image bands:', exportImage.bandNames());

// Create enhanced visualization image
var visImage = probability
  .visualize({
    palette: ['#2E7D32', '#66BB6A', '#FDD835', '#FF8F00', '#D84315'], 
    min: 0, max: 1
  })
  .updateMask(cropMask)
  .clip(aoi);

/* ========= 9) EXPORT MAPS ========= */
print('\n9. Configuring exports...');

// Export 1: Multi-band analysis image
Export.image.toDrive({
  image: exportImage,
  description: PARAMS7.exportPrefix + '_analysis',
  folder: PARAMS7.exportFolder,
  fileNamePrefix: PARAMS7.exportPrefix + '_analysis',
  region: aoi,
  scale: PARAMS7.outputScale,
  crs: PARAMS7.outputCRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

// Export 2: Stress probability only (0-100 scale for easier interpretation)
Export.image.toDrive({
  image: probability.multiply(100).toUint8(),
  description: PARAMS7.exportPrefix + '_probability_pct',
  folder: PARAMS7.exportFolder,
  fileNamePrefix: PARAMS7.exportPrefix + '_probability_pct',
  region: aoi,
  scale: PARAMS7.outputScale,
  crs: PARAMS7.outputCRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

// Export 3: Enhanced RGB visualization
Export.image.toDrive({
  image: visImage,
  description: PARAMS7.exportPrefix + '_visualization',
  folder: PARAMS7.exportFolder,
  fileNamePrefix: PARAMS7.exportPrefix + '_visualization',
  region: aoi,
  scale: PARAMS7.outputScale,
  crs: PARAMS7.outputCRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export 4: Binary classification
Export.image.toDrive({
  image: classified.updateMask(cropMask).toUint8(),
  description: PARAMS7.exportPrefix + '_binary',
  folder: PARAMS7.exportFolder,
  fileNamePrefix: PARAMS7.exportPrefix + '_binary',
  region: aoi,
  scale: PARAMS7.outputScale,
  crs: PARAMS7.outputCRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export 5: Confidence zones (3 classes: low, moderate, high)
var confidenceZones = ee.Image(0)
  .where(probability.gt(0.3).and(probability.lte(0.5)), 1)  // Low stress
  .where(probability.gt(0.5).and(probability.lte(0.8)), 2)  // Moderate stress
  .where(probability.gt(0.8), 3)  // High stress
  .updateMask(cropMask)
  .rename('confidence_zones');

Export.image.toDrive({
  image: confidenceZones.toUint8(),
  description: PARAMS7.exportPrefix + '_confidence_zones',
  folder: PARAMS7.exportFolder,
  fileNamePrefix: PARAMS7.exportPrefix + '_confidence_zones',
  region: aoi,
  scale: PARAMS7.outputScale,
  crs: PARAMS7.outputCRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

/* ========= 10) SUMMARY STATISTICS EXPORT ========= */
// Create comprehensive summary statistics
var summaryDict = ee.Dictionary({
  'district': 'Ludhiana',
  'season': '2023-24 Rabi',
  'date_processed': ee.Date(Date.now()).format('YYYY-MM-dd'),
  'total_cropland_ha': totalArea.get('area'),
  'stressed_area_ha': stressedArea.get('area'),
  'stressed_percentage': stressPct,
  'baseline_stress_pct': ee.Number(baselineStats.get('baseline_stress')).multiply(100),
  'rf_stress_pct': ee.Number(rfStats.get('rf_stress_class')).multiply(100),
  'probability_mean': probStats.get('mean'),
  'probability_median': probStats.get('p50'),
  'probability_p75': probStats.get('p75'),
  'probability_p95': probStats.get('p95'),
  'training_samples_used': training.size()
});

var summaryStats = ee.Feature(null, summaryDict);

// Export statistics as CSV
Export.table.toDrive({
  collection: ee.FeatureCollection([summaryStats]),
  description: PARAMS7.exportPrefix + '_statistics',
  folder: PARAMS7.exportFolder,
  fileNamePrefix: PARAMS7.exportPrefix + '_statistics',
  fileFormat: 'CSV'
});

/* ========= 11) VALIDATION POINTS EXPORT ========= */
// Stratified sampling for validation: 50 stressed, 50 healthy
var stressedValidation = exportImage
  .updateMask(classified.eq(1))
  .sample({
    region: aoi,
    scale: 30,
    numPixels: 50,
    seed: 456,
    geometries: true
  })
  .map(function(f) { return f.set('expected_class', 1, 'class_name', 'stressed'); });

var healthyValidation = exportImage
  .updateMask(classified.eq(0))
  .sample({
    region: aoi,
    scale: 30,
    numPixels: 50,
    seed: 789,
    geometries: true
  })
  .map(function(f) { return f.set('expected_class', 0, 'class_name', 'healthy'); });

var verificationPoints = stressedValidation.merge(healthyValidation);

// Export verification points
Export.table.toDrive({
  collection: verificationPoints,
  description: PARAMS7.exportPrefix + '_verification_points',
  folder: PARAMS7.exportFolder,
  fileNamePrefix: PARAMS7.exportPrefix + '_verification_points',
  fileFormat: 'CSV'
});

/* ========= 12) FINAL SUMMARY ========= */
print('\n========== PHASE 7 COMPLETE ==========');
print('\nExports configured (7 files):');
print('1. ' + PARAMS7.exportPrefix + '_analysis.tif (multi-band)');
print('2. ' + PARAMS7.exportPrefix + '_probability_pct.tif (0-100%)');
print('3. ' + PARAMS7.exportPrefix + '_visualization.tif (RGB)');
print('4. ' + PARAMS7.exportPrefix + '_binary.tif (0/1)');
print('5. ' + PARAMS7.exportPrefix + '_confidence_zones.tif (3 zones)');
print('6. ' + PARAMS7.exportPrefix + '_statistics.csv');
print('7. ' + PARAMS7.exportPrefix + '_verification_points.csv');
print('\nCheck Tasks tab to run exports');
print('Files will be saved to Google Drive folder:', PARAMS7.exportFolder);
print('\nNext: Run exports, then proceed to Phase 8 (validation) or Phase 10 (write-up)');

// End of Phase 7 (Updated)