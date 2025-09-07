/**** PHASE 1 ****/

/**** PARAMS (non‑negotiable defaults) ****/
var PARAMS = {
  aoiSource: 'FAO/GAUL_SIMPLIFIED_500m/2015/level2', // GAUL L2
  country: 'India',
  state: 'Punjab',
  district: 'Ludhiana',
  crs: 'EPSG:32643',      // UTM 43N
  scale: 10,
  seasonStart: '2023-11-01',
  seasonEnd:   '2024-04-15',
  // Cloud/shadow masking
  s2sr: 'COPERNICUS/S2_SR_HARMONIZED',
  s2cloudless: 'COPERNICUS/S2_CLOUD_PROBABILITY',
  s2cloudlessMaxProb: 40,           // s2cloudless <= 40
  sclRemove: [3,6,8,9,10,11],       // SCL classes to remove
  // Cropland≈wheat mask
  dwCollection: 'GOOGLE/DYNAMICWORLD/V1',
  dwProbBand: 'crops',
  useWorldCover: true,              // optional WorldCover intersect
  worldCoverYear: 2021,             // REQUIRED
  worldCoverCollectionRoot: 'ESA/WorldCover/v200',
  worldCoverCroplandCode: 40,
  erodePixels: 1,                   // erode 1 pixel @ 10 m
  // Compositing
  windowDays: 10,                   // 10‑day windows
  // Preview limits (for Map layers only)
  previewLimitS2: 60,
  previewLimitS2Cloud: 40,
  previewLimitClean: 50,
  previewLimitDW: 90,   // ~3 months worth is plenty for a preview
  // Visuals (centralised presets)
  rgbVisRaw:  {min: 200, max: 3000, bands: ['B4','B3','B2']},
  rgbVisComp: {min: 200, max: 3000, bands: ['B4_med','B3_med','B2_med']},
  ndviVis:    {min: 0, max: 1, palette: ['#d7301f','#fdbb84','#addd8e','#1a9850']},
  cloudVis:   {min: 0, max: 100, palette: ['#00441b','#1b7837','#a6dba0','#ffffe5','#fdae61','#d7191c']},
  dwVis:      {min: 0, max: 1, palette: ['#f7fbff','#6baed6','#08306b']},
  // Peak-window mask (Option B)
  peakStart: '2024-02-01',
  peakEnd:   '2024-03-20',
  dwPeakPercentile: 80,   // use DW crops p80
  dwPeakThr: 0.5,         // require >= 0.5 during peak
  ndviPeakThr: 0.5,        // NDVI mean during peak must be >= 0.5

  showPreview: false,         // set false when importing from other phases
  runTag: 'rabi_2023_24'
};


/**** AOI: GAUL Level‑2 → India → Punjab → Ludhiana ****/
var gaulL2 = ee.FeatureCollection(PARAMS.aoiSource);
var aoi = gaulL2
  .filter(ee.Filter.eq('ADM0_NAME', PARAMS.country))
  .filter(ee.Filter.eq('ADM1_NAME', PARAMS.state))
  .filter(ee.Filter.eq('ADM2_NAME', PARAMS.district))
  .geometry();

if(PARAMS.showPreview){
Map.centerObject(aoi, 9);
Map.addLayer(aoi, {color: 'black'}, '01_AOI — Ludhiana', true);
}
/**** Season dates ****/
var start = ee.Date(PARAMS.seasonStart);
var end   = ee.Date(PARAMS.seasonEnd);

/**** Fetch Sentinel‑2 SR + s2cloudless ****/
var s2  = ee.ImageCollection(PARAMS.s2sr)
  .filterBounds(aoi)
  .filterDate(start, end);

var s2c = ee.ImageCollection(PARAMS.s2cloudless)
  .filterBounds(aoi)
  .filterDate(start, end);


/**** PREVIEW: raw S2 RGB median (capped + clipped) ****/
if(PARAMS.showPreview){

var s2_raw_tc = s2.select(['B4','B3','B2'])
  .limit(PARAMS.previewLimitS2)
  .map(function(img){ return img.clip(aoi); })
  .median()
  .clip(aoi);
Map.addLayer(s2_raw_tc, PARAMS.rgbVisRaw, '02_S2_SR median (raw, cloudy, capped+clipped)', true);


/**** PREVIEW: s2cloudless prob mean (capped + clipped) ****/
var cloudProbMean = s2c
  .limit(PARAMS.previewLimitS2Cloud)
  .select('probability')
  .mean()
  .clip(aoi);
Map.addLayer(cloudProbMean, PARAMS.cloudVis, '03_s2cloudless prob (mean, capped+clipped)', true);

  
}

/**** Join by system:index (paired only) ****/
var join = ee.Join.inner();
var filter = ee.Filter.equals({
  leftField: 'system:index',
  rightField: 'system:index'
});

var joined = join.apply(s2, s2c, filter)
  .map(function(f) {
    var img = ee.Image(ee.Feature(f).get('primary'));
    var prob = ee.Image(ee.Feature(f).get('secondary')).select('probability').rename('cloud_prob');
    return img.addBands(prob);
  });


if(PARAMS.showPreview){
print('S2_SR (season) count:', s2.size());
print('S2_CLOUD_PROBABILITY (season) count:', s2c.size());
print('Joined count (paired only):', ee.ImageCollection(joined).size());
}

/**** Cloud/Shadow/Snow mask per image ****/
function maskS2(img) {
  var scl = img.select('SCL');
  var qa60 = img.select('QA60');
  var cloudProb = img.select('cloud_prob');

  var m1 = cloudProb.lte(PARAMS.s2cloudlessMaxProb);

  var sclKeep = ee.Image(1);
  PARAMS.sclRemove.forEach(function(code){ sclKeep = sclKeep.and(scl.neq(code)); });

  var bit10 = qa60.bitwiseAnd(1 << 10).eq(0);
  var bit11 = qa60.bitwiseAnd(1 << 11).eq(0);
  var m2 = bit10.and(bit11);

  var mask = m1.and(sclKeep).and(m2);

  // Reflectance bilinear; leave other bands default resampling (don’t force 'nearest')
  var refl = img.select(['B2','B3','B4','B8','B11']).resample('bilinear');
  var others = img.select(img.bandNames().removeAll(['B2','B3','B4','B8','B11']));

  return refl.addBands(others)
    .updateMask(mask)
    .copyProperties(img, img.propertyNames());
}

/**** Spectral indices per image ****/
function addIndices(img) {
  var nir = img.select('B8');
  var red = img.select('B4');
  var green = img.select('B3');
  var swir1 = img.select('B11');

  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var ndwi = nir.subtract(swir1).divide(nir.add(swir1)).rename('NDWI'); // Gao (NIR,SWIR1)
  var gndvi = nir.subtract(green).divide(nir.add(green)).rename('GNDVI');
  var L = 0.5;
  var savi = nir.subtract(red).multiply(1 + L).divide(nir.add(red).add(L)).rename('SAVI');
  var msavi2 = nir.multiply(2).add(1).pow(2)
      .subtract(nir.subtract(red).multiply(8))
      .sqrt()
      .multiply(-1).add(nir.multiply(2)).add(1)
      .multiply(0.5).rename('MSAVI2');

  return img.addBands([ndvi, ndwi, gndvi, savi, msavi2]);
}

/**** Cleaned + indexed collection ****/
var s2Clean = ee.ImageCollection(joined).map(maskS2).map(addIndices);


if(PARAMS.showPreview){
/**** PREVIEW: masked RGB median (capped + clipped) ****/
var masked_tc_preview = s2Clean
  .limit(PARAMS.previewLimitClean)
  .select(['B4','B3','B2'])
  .map(function(img){ return img.clip(aoi); })
  .reduce(ee.Reducer.median())
  .rename(['B4_med','B3_med','B2_med'])
  .clip(aoi);
Map.addLayer(masked_tc_preview, PARAMS.rgbVisComp, '04_S2 median (masked, capped+clipped)', true);

/**** PREVIEW: NDVI median (capped + clipped) ****/
var sample_ndvi_preview = s2Clean
  .limit(PARAMS.previewLimitClean)
  .select('NDVI')
  .map(function(img){ return img.clip(aoi); })
  .median()
  .clip(aoi);
Map.addLayer(sample_ndvi_preview, PARAMS.ndviVis, '05_NDVI median (masked, capped+clipped)', true);
}


/**** Seasonal mean NDVI (for mask logic uses full; display uses capped+clipped) ****/
var seasonalMeanNDVI = s2Clean.select('NDVI').mean().rename('NDVI_mean');
var seasonalMeanNDVI_display = s2Clean.limit(PARAMS.previewLimitClean).select('NDVI').mean().rename('NDVI_mean').clip(aoi);
if(PARAMS.showPreview){
Map.addLayer(seasonalMeanNDVI_display, PARAMS.ndviVis, '06_Seasonal mean NDVI (capped+clipped)', true);
}
/**** Dynamic World crops probability mean ****/
var dw = ee.ImageCollection(PARAMS.dwCollection)
  .filterBounds(aoi)
  .filterDate(start, end)
  .select(PARAMS.dwProbBand);
var dwMeanCrops = dw.mean().rename('DW_crops_mean');

if(PARAMS.showPreview){
// Fast preview: cap + clip BEFORE reduce to keep tiles tiny
var dwMeanCrops_preview = dw
  .limit(PARAMS.previewLimitDW)
  .map(function(img){ return img.clip(aoi); })
  .mean()
  .rename('DW_crops_mean')
  .clip(aoi);

Map.addLayer(dwMeanCrops_preview, PARAMS.dwVis, '07_DW crops prob (mean, capped+clipped)', true);
}

/**** Optional WorldCover cropland ****/
var wcAsset = PARAMS.worldCoverCollectionRoot + '/' + PARAMS.worldCoverYear;
var worldCover = ee.Image(wcAsset).select('Map').eq(PARAMS.worldCoverCroplandCode).rename('WC_cropland');
if(PARAMS.showPreview){
Map.addLayer(worldCover.updateMask(worldCover).clip(aoi), {palette:['#ffcc00']}, '08_WorldCover cropland=40 (clipped)', true);
}

//////////////////////////////////////////////////////////////////////////////////////////
/**** Build cropland≈wheat mask — Option B (DW peak Pxx + NDVI peak mean) ****/
var peakStart = ee.Date(PARAMS.peakStart);
var peakEnd   = ee.Date(PARAMS.peakEnd);

// DW crops probability percentile over peak window
var dwPeakP = ee.ImageCollection(PARAMS.dwCollection)
  .filterBounds(aoi)
  .filterDate(peakStart, peakEnd)
  .select('crops')
  .reduce(ee.Reducer.percentile([PARAMS.dwPeakPercentile])); // band: 'crops_p80' if 80

var dwPeak = dwPeakP.select('crops_p' + PARAMS.dwPeakPercentile)
  .rename('DW_crops_p');

// NDVI peak mean over the same window
var ndviPeakMean = s2Clean
  .filterDate(peakStart, peakEnd)
  .select('NDVI')
  .mean();

// Build mask
var baseMask = dwPeak.gte(PARAMS.dwPeakThr);
if (PARAMS.useWorldCover) baseMask = baseMask.and(worldCover);
baseMask = baseMask.and(ndviPeakMean.gte(PARAMS.ndviPeakThr));

// Name band before erosion, then erode + selfMask for clean display
var cropMask_preErode = baseMask.rename('cropMask');

if(PARAMS.showPreview){

Map.addLayer(dwPeak.clip(aoi), PARAMS.dwVis, '07b_DW crops p'+PARAMS.dwPeakPercentile+' (peak)', true);
Map.addLayer(ndviPeakMean.clip(aoi), PARAMS.ndviVis, '07c_NDVI peak mean', true);
}

var kernel = ee.Kernel.square({radius: PARAMS.erodePixels, units: 'pixels'});
var cropMask = cropMask_preErode.focal_min({kernel: kernel, iterations: 1}).selfMask();

if(PARAMS.showPreview){

Map.addLayer(cropMask.updateMask(cropMask).clip(aoi), {palette:['#00ff88']}, '10_cropMask (Option B, eroded, clipped)', true);
}
/**** Record mask provenance in properties (applied to cropMask & composites) ****/
var maskProvenance = {
  mask_version: 'OptionB_DW_p' + PARAMS.dwPeakPercentile,
  peakStart: PARAMS.peakStart,
  peakEnd: PARAMS.peakEnd,
  dwPeakThr: PARAMS.dwPeakThr,
  ndviPeakThr: PARAMS.ndviPeakThr,
  useWorldCover: PARAMS.useWorldCover,
  worldCoverYear: PARAMS.worldCoverYear,
  windowDays: PARAMS.windowDays
};
cropMask = cropMask.set(maskProvenance);
cropMask = cropMask.set({run_tag: PARAMS.runTag});

//////////////////////////////////////////////////////////////////////////////////////////

/**** 10‑day compositing with ee.Reducer.median() and _med renaming ****/
function makeWindows(startDate, endDate, stepDays) {
  var n = endDate.difference(startDate, 'day').divide(stepDays).ceil();
  var list = ee.List.sequence(0, ee.Number(n).subtract(1));
  return list.map(function(i){
    i = ee.Number(i);
    var winStart = startDate.advance(i.multiply(stepDays), 'day');
    var winEnd = winStart.advance(stepDays, 'day');
    return ee.Dictionary({start: winStart, end: winEnd});
  });
}
var windows = ee.List(makeWindows(start, end, PARAMS.windowDays));

/**** Build composites while skipping empty windows ****/
function reduceWindow(winDict) {
  winDict = ee.Dictionary(winDict);
  var ws = ee.Date(winDict.get('start'));
  var we = ee.Date(winDict.get('end'));

  var ic = s2Clean.filterDate(ws, we).filterBounds(aoi);
  var reduced = ic.reduce(ee.Reducer.median());
  var oldNames = reduced.bandNames();
  var newNames = oldNames.map(function(b){ return ee.String(b).replace('_median', '_med'); });
  var mid = ws.advance(PARAMS.windowDays / 2, 'day');

  return reduced.rename(newNames)
    .updateMask(cropMask)
    .set({
      'run_tag': PARAMS.runTag,
      'win_start': ws.format('YYYY-MM-dd'),
      'win_end':   we.format('YYYY-MM-dd'),
      'system:index': ws.format('YYYYMMdd'),
      'system:time_start': mid.millis()
    })
    .set(maskProvenance);
}

// Map over all windows; return a sentinel image for empty ones, then filter it out.
var composites = ee.ImageCollection(
  ee.List(windows).map(function(w) {
    w = ee.Dictionary(w);
    var ws = ee.Date(w.get('start'));
    var we = ee.Date(w.get('end'));
    var icHasData = s2Clean.filterDate(ws, we).size().gt(0);

    return ee.Algorithms.If(
      icHasData,
      reduceWindow(w),
      ee.Image().set('skip', 1)  // placeholder to filter away
    );
  })
).filter(ee.Filter.neq('skip', 1));

if(PARAMS.showPreview){
print('Composites (10‑day) count:', composites.size());


// For map display only: clip (keeps tiles small & within AOI)
var firstComp = ee.Image(composites.first());
Map.addLayer(firstComp.select('NDVI_med').clip(aoi), PARAMS.ndviVis, '11_First window NDVI_med (clipped)', true);
Map.addLayer(firstComp.clip(aoi), PARAMS.rgbVisComp, '12_First composite RGB (clipped)', true);


print('First composite bands:', ee.Image(composites.first()).bandNames());
print('First comp window:', firstComp.get('win_start'), '→', firstComp.get('win_end'));

/**** Summary layers ****/
Map.addLayer(aoi, {color: 'red'}, '13_AOI (outline, top)', true);

/**** Area calculation (hectares) ****/
var areaHa = ee.Image.pixelArea()
  .updateMask(cropMask)
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: aoi,
    scale: PARAMS.scale,
    maxPixels: 1e13
  }).getNumber('area');
var areaHaVal = areaHa.divide(10000);
print('cropMask area (ha):', areaHaVal);
}

/**** Exposed outputs for chaining ****/
var outputs = {
  aoi: aoi,
  cropMask: cropMask,
  composites: composites,
  ndviPeakMean: ndviPeakMean,
  dwPeak: dwPeak
};

if(PARAMS.showPreview){
print('Phase-1 outputs ready:', Object.keys(outputs));
}

exports.PARAMS = PARAMS;
exports.aoi = aoi;
exports.cropMask = cropMask;
exports.composites = composites;
exports.ndviPeakMean = ndviPeakMean;
exports.dwPeak = dwPeak;
exports.maskProvenance = maskProvenance;

var compositesManifest = ee.FeatureCollection(
  composites.map(function(img){
    return ee.Feature(null, img.toDictionary(['system:index','win_start','win_end','run_tag']));
  })
);
exports.compositesManifest = compositesManifest;


// ============================================
// NEW SECTION: EXPORT CROP MASK FOR DISSERTATION
// ============================================
print('========== CROP MASK EXPORT ==========');

// Create a visualization version of the crop mask
var cropMaskVis = cropMask.selfMask().visualize({
  palette: ['#2E7D32'],  // Dark green for wheat
  opacity: 1
});

// Export crop mask as GeoTIFF for QGIS
Export.image.toDrive({
  image: cropMask.selfMask(),
  description: 'Ludhiana_Crop_Mask_Binary',
  folder: 'dissertation_figures',
  fileNamePrefix: 'ludhiana_wheat_mask_2023_24',
  region: aoi,
  scale: 10,
  crs: PARAMS.crs,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export visualization version
Export.image.toDrive({
  image: cropMaskVis,
  description: 'Ludhiana_Crop_Mask_Visual',
  folder: 'dissertation_figures',
  fileNamePrefix: 'ludhiana_wheat_mask_visual',
  region: aoi,
  scale: 10,
  crs: PARAMS.crs,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

print('Crop mask exports configured - check Tasks tab');