/**** PHASE 5 - TILED SAMPLING FOR MORE DATA ****/
// Modified version with tiled sampling to get 10,000+ samples

/* 0) IMPORTS */
var p1 = require('users/lalutatu/wheatstress:Phase_1');
var p2 = require('users/lalutatu/wheatstress:Phase_2');
var p3 = require('users/lalutatu/wheatstress:Phase_3');
var p4 = require('users/lalutatu/wheatstress:Phase_4');

print('========== PHASE 5 TILED SAMPLING ==========');

/* 1) BASIC SETUP */
var aoi = p1.aoi;
var PARAMS = p1.PARAMS;
var CRS = PARAMS.crs;
var SCALE = PARAMS.scale;
var RUN_TAG = (PARAMS.runTag || ('rabi_' + ee.Date(PARAMS.seasonStart).format('YYYY_YY')));

print('1. Basic params:', {
  'CRS': CRS,
  'SCALE': SCALE,
  'RUN_TAG': RUN_TAG,
  'AOI area (km²)': aoi.area().divide(1e6)
});

/* 2) SAMPLING PARAMETERS - ENHANCED FOR MORE SAMPLES */
var PH5 = {
  samplingScale: 20,  // Reduced from 30m to 20m for more pixels
  
  // Tiled sampling parameters
  useTiledSampling: true,
  tilesPerSide: 3,  // 3x3 = 9 tiles
  samplesPerTilePerClass: 1200,  // 1200 * 9 = 10,800 per class target
  
  // Fallback parameters
  numSamplesPerClass: 10000,  // If tiled sampling disabled
  
  seed: 42,
  exportFolder: 'gee_wheat_stress_exports',
  exportScale: 10  // Keep 10m for raster exports
};

// Calculate expected samples
var expectedSamplesPerClass = PH5.useTiledSampling ? 
  (PH5.tilesPerSide * PH5.tilesPerSide * PH5.samplesPerTilePerClass) : 
  PH5.numSamplesPerClass;

print('2. Sampling configuration:', {
  'Method': PH5.useTiledSampling ? 'Tiled Sampling' : 'Direct Sampling',
  'Tiles': PH5.useTiledSampling ? (PH5.tilesPerSide + 'x' + PH5.tilesPerSide) : 'N/A',
  'Target samples per class': expectedSamplesPerClass,
  'Target total samples': expectedSamplesPerClass * 2
});

/* 3) GET LABELS AND WEIGHTS */
var labelImg = ee.Image(p4.label01).rename('label').toInt16();
var weightImg = ee.Image(p4.weight).rename('weight').toFloat();
var cropMask = ee.Image(p1.cropMask);

// Verify we have both classes
var class0Count = labelImg.eq(0).selfMask().reduceRegion({
  reducer: ee.Reducer.count(),
  geometry: aoi,
  scale: 100,
  maxPixels: 1e9,
  bestEffort: true
}).get('label');

var class1Count = labelImg.eq(1).selfMask().reduceRegion({
  reducer: ee.Reducer.count(),
  geometry: aoi,
  scale: 100,
  maxPixels: 1e9,
  bestEffort: true
}).get('label');

print('3. Class distribution in AOI:', {
  'Healthy pixels (0)': class0Count,
  'Stressed pixels (1)': class1Count,
  'Ratio': ee.Number(class1Count).divide(class0Count)
});

/* 4) CREATE TILES FOR SAMPLING */
print('\n========== 4. CREATING TILES ==========');

// Function to create grid tiles
function createTileGrid(geometry, n) {
  var bounds = geometry.bounds();
  var coords = ee.List(bounds.coordinates().get(0));
  
  var xmin = ee.Number(ee.List(coords.get(0)).get(0));
  var ymin = ee.Number(ee.List(coords.get(0)).get(1));
  var xmax = ee.Number(ee.List(coords.get(2)).get(0));
  var ymax = ee.Number(ee.List(coords.get(2)).get(1));
  
  var width = xmax.subtract(xmin);
  var height = ymax.subtract(ymin);
  
  var tiles = ee.List([]);
  
  // Create grid
  for(var i = 0; i < n; i++) {
    for(var j = 0; j < n; j++) {
      var x1 = xmin.add(width.multiply(i).divide(n));
      var x2 = xmin.add(width.multiply(i + 1).divide(n));
      var y1 = ymin.add(height.multiply(j).divide(n));
      var y2 = ymin.add(height.multiply(j + 1).divide(n));
      
      var tile = ee.Feature(
        ee.Geometry.Rectangle([x1, y1, x2, y2]),
        {
          'tile_id': i * n + j,
          'tile_row': i,
          'tile_col': j
        }
      );
      tiles = tiles.add(tile);
    }
  }
  
  return ee.FeatureCollection(tiles);
}

// Create tiles
var tiles = createTileGrid(aoi, PH5.tilesPerSide);
print('4a. Created tiles:', tiles.size());

// Visualize tiles on map
Map.centerObject(aoi, 9);
Map.addLayer(tiles, {color: 'red'}, 'Sampling Tiles', true);
Map.addLayer(cropMask.selfMask(), {palette: ['gray'], opacity: 0.3}, 'Crop Mask', false);

/* 5) TILED STRATIFIED SAMPLING */
print('\n========== 5. TILED SAMPLING FOR COORDINATES + LABELS ==========');

// Create base image with essentials
var baseImg = ee.Image.cat([
  labelImg,
  weightImg,
  ee.Image.pixelLonLat().select(['longitude', 'latitude'], ['lon', 'lat'])
]).updateMask(cropMask);

// Function to sample from a single tile
function sampleFromTile(tileFeature) {
  var tileGeom = tileFeature.geometry();
  var tileId = tileFeature.get('tile_id');
  
  // Sample healthy pixels from this tile
  var tileHealthy = baseImg.updateMask(labelImg.eq(0))
    .sample({
      region: tileGeom,
      numPixels: PH5.samplesPerTilePerClass,
      scale: PH5.samplingScale,
      seed: ee.Number(PH5.seed).add(ee.Number(tileId)),
      geometries: false,
      tileScale: 2  // Reduced for memory efficiency
    })
    .map(function(f) {
      return f.set({
        'class_name': 'healthy',
        'tile_id': tileId,
        'season': PARAMS.seasonStart + '_to_' + PARAMS.seasonEnd,
        'run_tag': RUN_TAG
      });
    });
  
  // Sample stressed pixels from this tile
  var tileStressed = baseImg.updateMask(labelImg.eq(1))
    .sample({
      region: tileGeom,
      numPixels: PH5.samplesPerTilePerClass,
      scale: PH5.samplingScale,
      seed: ee.Number(PH5.seed).add(ee.Number(tileId)).add(100),
      geometries: false,
      tileScale: 2
    })
    .map(function(f) {
      return f.set({
        'class_name': 'stressed',
        'tile_id': tileId,
        'season': PARAMS.seasonStart + '_to_' + PARAMS.seasonEnd,
        'run_tag': RUN_TAG
      });
    });
  
  return tileHealthy.merge(tileStressed);
}

// Sample from all tiles
var allTileSamples = tiles.map(sampleFromTile);
var allSamples = ee.FeatureCollection(allTileSamples).flatten();

/* 6) COUNT SAMPLES BEFORE EXPORT */
print('\n========== 6. SAMPLE COUNT VERIFICATION ==========');

// Count samples by class
var healthyCount = allSamples.filter(ee.Filter.eq('class_name', 'healthy')).size();
var stressedCount = allSamples.filter(ee.Filter.eq('class_name', 'stressed')).size();
var totalCount = allSamples.size();

print('6a. Sample counts BEFORE export:');
print('   Healthy samples:', healthyCount);
print('   Stressed samples:', stressedCount);
print('   TOTAL samples:', totalCount);

// Get actual counts (forces evaluation)
healthyCount.evaluate(function(val) {
  print('   Healthy (evaluated):', val);
});
stressedCount.evaluate(function(val) {
  print('   Stressed (evaluated):', val);
});
totalCount.evaluate(function(val) {
  print('   TOTAL (evaluated):', val);
  if (val < 5000) {
    print('   ⚠️ WARNING: Fewer samples than expected! Check if tiles have enough pixels.');
  } else if (val < 10000) {
    print('   ⚠️ NOTE: Moderate sample size. Consider increasing samplesPerTilePerClass.');
  } else {
    print('   ✅ SUCCESS: Good sample size for training!');
  }
});

// Sample distribution by tile (first 3 tiles as preview)
print('\n6b. Sample distribution by tile (first 3):');
for (var t = 0; t < Math.min(3, PH5.tilesPerSide * PH5.tilesPerSide); t++) {
  var tileSamples = allSamples.filter(ee.Filter.eq('tile_id', t));
  print('   Tile', t, ':', tileSamples.size());
}

/* 7) ADD UNIQUE IDs */
print('\n========== 7. ADDING UNIQUE IDs ==========');

// Add sequential IDs to all samples - FIXED VERSION
var samplesWithId = allSamples.map(function(feature) {
  // Use the existing system:index as a unique identifier
  var sysIndex = feature.get('system:index');
  return feature.set({
    'sample_id': sysIndex,
    'unique_id': ee.String('sample_').cat(ee.String(sysIndex))
  });
});

// Alternative approach if above doesn't work
var samplesList = allSamples.toList(allSamples.size());
var samplesWithId2 = ee.FeatureCollection(
  samplesList.map(function(feature) {
    var feat = ee.Feature(feature);
    // Generate ID based on tile_id and system:index
    var tileId = feat.get('tile_id');
    var sysIdx = feat.get('system:index');
    return feat.set({
      'sample_id': ee.String(tileId).cat('_').cat(ee.String(sysIdx)),
      'row_number': feat.id()
    });
  })
);

// Use the simpler version
var finalSamples = samplesWithId;

print('7a. First sample with ID:', finalSamples.first());

/* 8) EXPORT COORDINATES + LABELS */
print('\n========== 8. EXPORT COORDINATES + LABELS ==========');

// Main export - using finalSamples with IDs
Export.table.toDrive({
  collection: finalSamples,
  description: 'coordinates_labels_tiled_' + RUN_TAG,
  fileNamePrefix: 'coordinates_labels_tiled_' + RUN_TAG,
  folder: PH5.exportFolder,
  fileFormat: 'CSV'
});

print('✅ Export 1: Coordinates + Labels (Tiled Sampling)');

/* 9) PREPARE PREDICTOR IMAGES */
print('\n========== 9. PREPARING PREDICTOR IMAGES ==========');

// Get metrics from Phase 2
var metrics = ee.Image(p2.metricsImage);

// Get helpers from Phase 3
var helpers = ee.Image(p3.anomalyHelpers);

// Select key predictors (same as original Phase 5)
var keyMetrics = metrics.select([
  'NDVI_AUC', 'NDVI_drop', 'NDVI_peak', 'NDVI_slopeEarly',
  'NDWI_AUC', 'NDWI_drop', 'NDWI_peak',
  'GNDVI_AUC', 'GNDVI_drop',
  'SAVI_AUC', 'MSAVI2_AUC'
]);

var keyHelpers = helpers.select([
  'NDVI_drop', 'NDWI_drop', 'AUC_z', 'NDVI_AUC_cur'
]).regexpRename('^(.*)$', 'h_$1');

// Combine predictors
var allPredictors = keyMetrics.addBands(keyHelpers)
  .updateMask(cropMask)
  .unmask(-9999)
  .float()
  .clip(aoi);

print('9a. Predictor bands:', allPredictors.bandNames());
print('9b. Number of predictor bands:', allPredictors.bandNames().size());

/* 10) EXPORT PREDICTORS AS GEOTIFF */
print('\n========== 10. EXPORT PREDICTOR RASTERS ==========');

// Export as multi-band GeoTIFF (same as original)
Export.image.toDrive({
  image: allPredictors.toFloat(),
  description: 'predictors_image_tiled_' + RUN_TAG,
  fileNamePrefix: 'predictors_image_tiled_' + RUN_TAG,
  folder: PH5.exportFolder,
  region: aoi,
  scale: PH5.exportScale,
  crs: CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

print('✅ Export 2: Predictor Image (GeoTIFF)');

/* 11) SMALL PREVIEW EXPORT (OPTIONAL) */
print('\n========== 11. PREVIEW EXPORT (OPTIONAL) ==========');

// Take first 100 samples of each class for preview - FIXED VERSION
var previewHealthy = finalSamples.filter(ee.Filter.eq('class_name', 'healthy')).limit(100);
var previewStressed = finalSamples.filter(ee.Filter.eq('class_name', 'stressed')).limit(100);
var previewSamples = previewHealthy.merge(previewStressed);

Export.table.toDrive({
  collection: previewSamples,
  description: 'PREVIEW_coordinates_labels_' + RUN_TAG,
  fileNamePrefix: 'PREVIEW_coordinates_labels_' + RUN_TAG,
  folder: PH5.exportFolder,
  fileFormat: 'CSV'
});

print('✅ Preview Export: 200 samples for quick testing');

/* 12) SUMMARY STATISTICS */
print('\n========== 12. SUMMARY ==========');

// Create summary feature
var summaryStats = ee.Feature(null, {
  'aoi_area_km2': aoi.area().divide(1e6),
  'sampling_method': 'tiled',
  'tiles_count': PH5.tilesPerSide * PH5.tilesPerSide,
  'sampling_scale_m': PH5.samplingScale,
  'target_per_class': expectedSamplesPerClass,
  'actual_healthy': healthyCount,
  'actual_stressed': stressedCount,
  'actual_total': totalCount,
  'export_date': ee.Date(Date.now()).format('YYYY-MM-dd'),
  'run_tag': RUN_TAG
});

Export.table.toDrive({
  collection: ee.FeatureCollection([summaryStats]),
  description: 'sampling_summary_' + RUN_TAG,
  fileNamePrefix: 'sampling_summary_' + RUN_TAG,
  folder: PH5.exportFolder,
  fileFormat: 'CSV'
});

print('✅ Export 3: Sampling summary statistics');

print('\n========== EXPECTED RESULTS ==========');
print('Target samples per class:', expectedSamplesPerClass);
print('Target total samples:', expectedSamplesPerClass * 2);
print('');
print('Files to be exported:');
print('1. coordinates_labels_tiled_' + RUN_TAG + '.csv (~10-20K rows)');
print('2. predictors_image_tiled_' + RUN_TAG + '.tif (15 bands)');
print('3. PREVIEW_coordinates_labels_' + RUN_TAG + '.csv (200 rows)');
print('4. sampling_summary_' + RUN_TAG + '.csv (statistics)');
print('');
print('Next steps:');
print('1. Run exports from Tasks tab');
print('2. Download CSV and GeoTIFF files');
print('3. Run Phase 5.5 spatial join (update input filenames)');
print('4. Proceed with Phase 6 training with more data!');

print('\n========== PHASE 5 TILED SAMPLING COMPLETE ==========');