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
var labelImg  = ee.Image(p4.label01).rename('label').toInt16();
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
  .rename('tile_id').toInt();

/* 5) lon/lat bands (avoid geometry in samples/export) */
var lonlat = ee.Image.pixelLonLat()
  .select(['latitude','longitude'], ['lat','lon']);

/* 6) BASE IMAGE FOR SAMPLING */
var baseImg = ee.Image.cat([labelImg, weightImg, tileIdImg, lonlat]).updateMask(cropMaskSafe);

/* 7) PER-TILE, PER-CLASS SAMPLING */
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

  return s0.merge(s1).map(function (f) {
    return f.set({
      season: ee.String(PARAMS.seasonStart).cat('_').cat(PARAMS.seasonEnd),
      year: ee.Date(PARAMS.seasonStart).get('year'),
      run_tag: RUN_TAG
    });
  });
}).flatten();

/* 8) GLOBAL BALANCE */
var withRand  = perTileSamples.randomColumn('rand', PH5.seed);
var stressed  = withRand.filter(ee.Filter.eq('label', 1)).sort('rand').limit(PH5.targetPerClass);
var healthy   = withRand.filter(ee.Filter.eq('label', 0)).sort('rand').limit(PH5.targetPerClass);
var samplesBalanced = stressed.merge(healthy);

/* 9) ATTACH PREDICTORS SAFELY */

// Helper: keep only items from `want` that actually exist in `have`
function keepExistingBands(want, have) {
  want = ee.List(want);
  have = ee.List(have);
  var haveDict = ee.Dictionary.fromLists(have, have.map(function(_) { return 1; }));
  var kept = want.map(function(b) {
    b = ee.String(b);
    return ee.Algorithms.If(haveDict.contains(b), b, null);
  });
  return ee.List(kept).removeAll([null]);
}

// --- Validation helper ---
function warnMissing(from, image, required){
  var have = ee.List(image.bandNames());
  var missing = ee.List(required).removeAll(have);
  print('⚠️ ' + from + ' missing bands:', missing);
}

// Slim predictors (metrics/helpers) with defensive selection
if (HAS_METRICS) {
  var wantM = ee.List([
    'NDVI_peak','NDVI_drop','NDVI_AUC','NDVI_slopeEarly','NDVI_datePeak_doy',
    'NDWI_drop','NDWI_AUC'
  ]);
  var haveM = metrics.bandNames();
  var keepM = keepExistingBands(wantM, haveM);
  metrics = ee.Image(ee.Algorithms.If(
    keepM.size().gt(0),
    metrics.select(keepM),
    metrics.select([haveM.get(0)])
  ));
}

if (HAS_HELPERS) {
  var wantH = ee.List(['h_AUC_z','h_NDVI_AUC_cur','h_sd_floor_applied']);
  var haveH = helpersRen.bandNames();
  var keepH = keepExistingBands(wantH, haveH);
  helpersRen = ee.Image(ee.Algorithms.If(
    keepH.size().gt(0),
    helpersRen.select(keepH),
    helpersRen.select([haveH.get(0)])
  ));
}

// --- Phase-specific validation (right before building predictorsImg) ---
var reqP2 = [
  'NDVI_peak','NDVI_drop','NDVI_AUC','NDVI_slopeEarly','NDVI_datePeak_doy',
  'NDWI_drop','NDWI_AUC'
];
var reqP3 = ['h_AUC_z','h_NDVI_AUC_cur','h_sd_floor_applied'];
var reqP4 = ['label','weight'];

if (HAS_METRICS)  warnMissing('PH2', metrics,    reqP2);
if (HAS_HELPERS)  warnMissing('PH3', helpersRen, reqP3);
warnMissing('PH4', baseImg.select(['label','weight']), reqP4);

// Build predictors image safely
var predictorsImg;
if (HAS_METRICS && HAS_HELPERS) {
  predictorsImg = metrics.addBands(helpersRen);
} else if (HAS_METRICS) {
  predictorsImg = metrics;
} else if (HAS_HELPERS) {
  predictorsImg = helpersRen;
} else {
  predictorsImg = ee.Image.constant(0).rename('pad');
}

// Ensure ≥1 band; cast to float to shrink payload
predictorsImg = ee.Image(ee.Algorithms.If(
  predictorsImg.bandNames().size().gt(0),
  predictorsImg,
  ee.Image.constant(0).rename('pad')
)).toFloat();

// Join predictors to points
var toExport = predictorsImg.sampleRegions({
  collection: samplesBalanced,
  scale: SCALE,
  tileScale: PH5.tileScale,
  geometries: false
});

if (!HAS_METRICS && !HAS_HELPERS) {
  toExport = toExport.map(function(f){ return f.select(f.propertyNames().remove('pad')); });
}

/* 10) EXPORT */
var predictors = predictorsImg.bandNames();
var selectorsEE = ee.List([
  'tile_id','lat','lon','season','year','run_tag','label','weight'
]).cat(predictors);
var selectorsClient = selectorsEE.getInfo();

// Ultra‑light preview (~5 rows). No random/sort to avoid heavy ops.
var preview = toExport.select(selectorsEE).limit(5);

Export.table.toDrive({
  collection: preview,
  description: 'phase5_preview_' + RUN_TAG,
  fileNamePrefix: 'phase5_preview_' + RUN_TAG,
  folder: PH5.exportFolder,
  fileFormat: 'CSV',
  selectors: selectorsClient
});

Export.table.toDrive({
  collection: toExport.select(selectorsEE),
  description: 'phase5_samples_' + RUN_TAG,
  fileNamePrefix: 'phase5_samples_' + RUN_TAG,
  folder: PH5.exportFolder,
  fileFormat: 'CSV',
  selectors: selectorsClient
});

/* 11) LIGHT DIAGNOSTICS */
print('Tiles:', grid.size());
print('Balanced counts — target per class:', PH5.targetPerClass);
