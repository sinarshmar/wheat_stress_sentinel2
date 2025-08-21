/**** PHASE 5 — Stratified Sampling & Export (GEE JS) ****/

/* 0) IMPORTS */
var p1 = require('users/lalutatu/wheatstress:Phase_1');
var p2 = require('users/lalutatu/wheatstress:Phase_2');
var p3 = require('users/lalutatu/wheatstress:Phase_3');
var p4 = require('users/lalutatu/wheatstress:Phase_4');

/* 1) HANDLES */
var aoi      = p1.aoi;
var PARAMS   = p1.PARAMS;
var CRS      = PARAMS.crs;    // EPSG:32643
var SCALE    = PARAMS.scale;  // 10 m
var RUN_TAG  = (PARAMS.runTag || ('rabi_' + ee.Date(PARAMS.seasonStart).format('YYYY_YY')));

/* Phase-4: label + weight (required for sampling) */
var labelImg  = ee.Image(p4.label01).rename('label').toInt();
var weightImg = ee.Image(p4.weight).unmask(0).rename('weight');

/* Phase-1 crop mask → SAFE fallback to 1 if absent */
var cropMask = ee.Image(p1.cropMask);
var cropMaskSafe = ee.Image(ee.Algorithms.If(
  cropMask.bandNames().size().gt(0), cropMask, ee.Image.constant(1)
));

/* Phase-2 metrics (predictors) */
var HAS_METRICS = !!p2 && !!p2.metricsImage;
var metrics = HAS_METRICS ? ee.Image(p2.metricsImage) : null;

/* Phase-3 helpers (predictors), prefixed to avoid any name clash */
var HAS_HELPERS = !!p3 && !!p3.anomalyHelpers;
var helpersRen = (function(){
  if (!HAS_HELPERS) return null;
  var h = ee.Image(p3.anomalyHelpers);
  var names = h.bandNames();
  var pref  = names.map(function(b){ return ee.String('h_').cat(ee.String(b)); });
  return h.rename(pref);
})();

/* 2) CONFIG */
var PH5 = {
  gridKm: 5,
  perTileCapPerClass: 2000,    // cap per class per tile
  targetPerClass: 25000,       // global cap per class
  seed: 42,
  tileScale: 12,
  exportMode: 'single',        // 'byTile' | 'single'
  exportFolder: 'gee_phase5',
  includeGeometry: false
};

/* 3) GRID (5×5 km) — enumerate to create clean integer tile_id */
var proj = ee.Projection(CRS);
var rawGrid = aoi.coveringGrid(proj, PH5.gridKm * 1000);
var n = rawGrid.size();
var gridList = rawGrid.toList(n);
var grid = ee.FeatureCollection(ee.List.sequence(0, n.subtract(1)).map(function(i){
  var f = ee.Feature(gridList.get(i));
  return f.set('tile_id', ee.Number(i));   // clean numeric ID
}));

/* 4) tile_id raster */
var tileIdImg = ee.Image().paint(grid, 'tile_id')
  .rename('tile_id').toInt()
  .reproject({crs: CRS, scale: SCALE});

/* 5) lon/lat bands (avoid geometry in samples/export) */
var lonlat = ee.Image.pixelLonLat()
  .select(['latitude','longitude'], ['lat','lon'])
  .reproject({crs: CRS, scale: SCALE});

/* 6) BASE IMAGE FOR SAMPLING (tiny & safe) */
var baseImg = ee.Image.cat([labelImg, weightImg, tileIdImg, lonlat]).updateMask(cropMaskSafe);

/* 7) PER-TILE, PER-CLASS SAMPLING (explicit masks; no stratifiedSample) */
var perTileSamples = grid.map(function (tile) {
  var geom = ee.Feature(tile).geometry();
  var lbl  = baseImg.select('label');

  var img0 = baseImg.updateMask(lbl.eq(0));
  var img1 = baseImg.updateMask(lbl.eq(1));

  var s0 = img0.sample({
    region: geom,
    numPixels: PH5.perTileCapPerClass,
    scale: SCALE,
    seed: PH5.seed,
    tileScale: PH5.tileScale,
    geometries: false
  });

  var s1 = img1.sample({
    region: geom,
    numPixels: PH5.perTileCapPerClass,
    scale: SCALE,
    seed: PH5.seed + 1,
    tileScale: PH5.tileScale,
    geometries: false
  });

  // Add season/year/run_tag (tile_id/lat/lon already present)
  return s0.merge(s1).map(function (f) {
    return f.set({
      season: ee.String(PARAMS.seasonStart).cat('_').cat(PARAMS.seasonEnd),
      year: ee.Date(PARAMS.seasonStart).get('year'),
      run_tag: RUN_TAG
    });
  });
}).flatten();

/* 8) GLOBAL BALANCE (deterministic) */
var withRand  = perTileSamples.randomColumn('rand', PH5.seed);
var stressed  = withRand.filter(ee.Filter.eq('label', 1)).sort('rand').limit(PH5.targetPerClass);
var healthy   = withRand.filter(ee.Filter.eq('label', 0)).sort('rand').limit(PH5.targetPerClass);
var samplesBalanced = stressed.merge(healthy);

/* 9) ATTACH PREDICTORS SAFELY (sampleRegions onto points) */
// Slim predictors to keep CSV manageable
if (HAS_METRICS) {
  metrics = metrics.select([
    'NDVI_peak','NDVI_drop','NDVI_AUC','NDVI_slopeEarly','NDVI_datePeak_doy',
    'NDWI_drop','NDWI_AUC'
  ]);
}
if (HAS_HELPERS) {
  helpersRen = helpersRen.select(['h_AUC_z','h_NDVI_AUC_cur','h_sd_floor_applied']);
}

var predictorsImg;
if (HAS_METRICS && HAS_HELPERS) {
  predictorsImg = metrics.addBands(helpersRen);
} else if (HAS_METRICS) {
  predictorsImg = metrics;
} else if (HAS_HELPERS) {
  predictorsImg = helpersRen;
} else {
  // Fallback dummy band so sampleRegions has ≥1 band; drop later if desired.
  predictorsImg = ee.Image.constant(0).rename('pad');
}


predictorsImg = predictorsImg.toFloat();   // reduces payload size


var toExport = predictorsImg.sampleRegions({
  collection: samplesBalanced,     // copies existing properties
  scale: SCALE,
  tileScale: PH5.tileScale,
  geometries: false
});

// If we had to add 'pad', remove it to keep CSV clean
if (!HAS_METRICS && !HAS_HELPERS) {
  toExport = toExport.map(function(f){ return f.select(f.propertyNames().remove('pad')); });
}

/* 10) EXPORT — single merged CSV with explicit column order */
var predictors = HAS_METRICS && HAS_HELPERS ? predictorsImg.bandNames()
                 : HAS_METRICS ? metrics.bandNames()
                 : HAS_HELPERS ? helpersRen.bandNames()
                 : ee.List([]);

// Order: admin/meta first, then predictors
var selectors = ee.List([
  'tile_id','lat','lon','season','year','run_tag','label','weight'
]).cat(predictors);

// (Optional) Tiny preview (~1k rows) to inspect schema quickly
var preview = toExport.select(selectors).randomColumn('r', 99).sort('r').limit(300);
Export.table.toDrive({
  collection: preview,
  description: 'phase5_preview_' + RUN_TAG,
  fileNamePrefix: 'phase5_preview_' + RUN_TAG,
  folder: PH5.exportFolder,
  fileFormat: 'CSV'
});

// Main export
Export.table.toDrive({
  collection: toExport.select(selectors),
  description: 'phase5_samples_' + RUN_TAG,
  fileNamePrefix: 'phase5_samples_' + RUN_TAG,
  folder: PH5.exportFolder,
  fileFormat: 'CSV'
});

/* 11) LIGHT DIAGNOSTICS */
print('Tiles:', grid.size());
print('Balanced counts — target per class:', PH5.targetPerClass);
