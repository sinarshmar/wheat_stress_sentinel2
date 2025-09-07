/**** PHASE 2 ****/
/* Temporal Smoothing & Metrics — GEE (JavaScript) */

/* ========= 0) IMPORT PHASE 1 ========= */
var p1 = require('users/lalutatu/wheatstress:Phase_1'); // <-- replace <myname>

/* Handy handles from Phase 1 */
var aoi        = p1.aoi;
var cropMask   = p1.cropMask;         // boolean, eroded, self-masked
var composites = p1.composites;       // 10-day med comps with *_med bands
var p1crs      = p1.PARAMS.crs || 'EPSG:32643';
var p1scale    = p1.PARAMS.scale || 10;

/* ========= 1) PHASE 2 CONFIG ========= */
var PARAMS2 = {
  method: 'median3',     // 'median3' (default) or 'sg5'
  indices: ['NDVI','NDWI','GNDVI','SAVI','MSAVI2'],

  // Post-peak DROP anchor window
  postPeakCenter: '2024-03-05',
  postPeakHalfWinDays: 10,

  // Early-season SLOPE windows
  slopeWin1Center: '2023-12-15',
  slopeWin2Center: '2024-01-10',
  slopeHalfWinDays: 10,

  // AUC integration
  stepDays: 10,
  scaleFactor: 1.0,

  // Preview
  showPreview: false,
  chartIndex: 'NDVI',

  // Spatial defaults (inherit Phase 1)
  crs: p1crs,
  scale: p1scale
};

/* ========= 2) HELPERS ========= */
function toDate(d) { return ee.Date(d); }

function withinDays(center, halfWinDays){
  center = ee.Date(center);
  var start = center.advance(ee.Number(halfWinDays).multiply(-1), 'day');
  // filterDate is [start, end); push end by 1 day to include boundary
  var end   = center.advance(ee.Number(halfWinDays).add(1), 'day');
  return ee.Filter.date(start, end);
}

function addTimeBands(img) {
  var d   = img.date();
  var tms = ee.Image.constant(d.millis()).rename('time').toInt64();
  var doy = ee.Image.constant(d.getRelative('day', 'year').add(1)).rename('doy').toInt16();
  return img.addBands([tms, doy]);
}

/* Median over a small IC (returns single-band image, renamed) */
function medianBand(ic, bandName, newName) {
  var medImg = ic.select(bandName).median().rename(newName);
  return medImg;
}

/* Median-3 smoother around the current image time */
function smoothAtTime_median3(icSorted, currentImg, idx) {
  var t = currentImg.date();
  // 3 windows total: t - 10d, t, t + 10d  (expand end by +1d for inclusivity)
  var winIC = icSorted.filterDate(t.advance(-PARAMS2.stepDays, 'day'),
                                  t.advance(PARAMS2.stepDays + 1, 'day'));
  return medianBand(winIC, idx + '_med', idx + '_sm');
}

/* Savitzky–Golay 5-point (poly=2) smoothing with robust fallback to median3.
   Coefficients for smoothing: [-3, 12, 17, 12, -3] / 35 */
function smoothAtTime_sg5(icSorted, currentImg, idx) {
  var t = currentImg.date();
  var candidates = icSorted.filterDate(t.advance(-PARAMS2.stepDays*2, 'day'),
                                       t.advance(PARAMS2.stepDays*2 + 1, 'day'))
                           .map(function(im){
                             var diff = ee.Number(ee.Date(im.get('system:time_start'))
                                         .difference(t, 'millisecond')).abs();
                             return im.set('tdiff', diff);
                           })
                           .sort('tdiff');
  var size = candidates.size();

  var sgImg = ee.Image(ee.Algorithms.If(
    size.gte(5),
    (function(){
      var list5 = candidates.toList(5);
      var weights = ee.List([-3, 12, 17, 12, -3]);
      var weightedSum = ee.ImageCollection(ee.List.sequence(0, 4).map(function(k){
        k = ee.Number(k);
        var im = ee.Image(list5.get(k));
        var w  = ee.Number(weights.get(k)).divide(35);
        return im.select(idx + '_med').multiply(w);
      })).sum().rename(idx + '_sm');
      return weightedSum;
    })(),
    // Fallback if insufficient neighbors
    smoothAtTime_median3(icSorted, currentImg, idx)
  ));
  return sgImg;
}

/* Apply smoothing (median3 or sg5) to ALL indices; returns IC with added *_sm bands */
function smoothCollection(ic, indices, method) {
  var icSorted = ic.sort('system:time_start');
  var smIC = icSorted.map(function(img){
    // Start from original image to preserve all properties
    var out = ee.Image(img);
    indices.forEach(function(idx){
      var smBand = (method === 'sg5')
        ? smoothAtTime_sg5(icSorted, img, idx)
        : smoothAtTime_median3(icSorted, img, idx);
      // Keep crop mask
      smBand = smBand.updateMask(cropMask);
      out = out.addBands(smBand);
    });
    return out.copyProperties(img, img.propertyNames());
  });
  return smIC;
}

/* Median of smoothed band in a specified window; returns single-band image (or fully masked) */
function medianSmInWindow(icSm, idx, centerStr, halfWinDays, outName) {
  var fil = withinDays(centerStr, halfWinDays);
  var winIC = icSm.filter(fil).select(idx + '_sm');
  var has = winIC.size().gt(0);
  var maskedEmpty = ee.Image.constant(0).updateMask(ee.Image(0));
  var img = ee.Image(ee.Algorithms.If(
    has, winIC.median(), maskedEmpty
  )).rename(outName || (idx + '_sm_median')).updateMask(cropMask);
  return img;
}

/* Compute all metrics for one index; returns ee.Image of 6 bands for this index */
function computeMetrics(icSm, idx) {
  var smBand = idx + '_sm';

  // 1) PEAK value and DATE at peak
  var peak = icSm.select(smBand).max().rename(idx + '_peak').updateMask(cropMask);

  var withTime = icSm.map(addTimeBands);
  var peakPick = withTime.qualityMosaic(smBand);
  var dateMillis = peakPick.select('time').rename(idx + '_datePeak_millis').toInt64();
  var dateDoy    = peakPick.select('doy').rename(idx + '_datePeak_doy').toInt16();

  // 2) DROP = peak − anchor (post-peak anchor window median)
  var anchor = medianSmInWindow(icSm, idx, PARAMS2.postPeakCenter,
                                PARAMS2.postPeakHalfWinDays, idx + '_anchor');
  var drop = peak.subtract(anchor).rename(idx + '_drop').updateMask(cropMask);

  // 3) SLOPEEARLY = (early2 − early1) / Δdays
  var early1 = medianSmInWindow(icSm, idx, PARAMS2.slopeWin1Center,
                                PARAMS2.slopeHalfWinDays, idx + '_early1');
  var early2 = medianSmInWindow(icSm, idx, PARAMS2.slopeWin2Center,
                                PARAMS2.slopeHalfWinDays, idx + '_early2');
  var deltaDays = ee.Number(ee.Date(PARAMS2.slopeWin2Center)
                      .difference(ee.Date(PARAMS2.slopeWin1Center), 'day'));
  var slopeEarly = early2.subtract(early1)
                         .divide(deltaDays)
                         .rename(idx + '_slopeEarly')
                         .updateMask(cropMask);

  // 4) AUC = scaleFactor * Σ( sm * stepDays )
  var auc = icSm.select(smBand)
                .map(function(im){ return im.multiply(PARAMS2.stepDays); })
                .sum()
                .multiply(PARAMS2.scaleFactor)
                .rename(idx + '_AUC')
                .updateMask(cropMask);

  // Stack (keep only required bands)
  return ee.Image.cat([
    peak,
    dateMillis,
    dateDoy,
    drop,
    slopeEarly,
    auc
  ]);
}

/* Build metric band names list (client-side) */
function buildMetricNames(indices){
  var suffixes = ['_peak','_datePeak_millis','_datePeak_doy','_drop','_slopeEarly','_AUC'];
  var names = [];
  indices.forEach(function(idx){
    suffixes.forEach(function(s){ names.push(idx + s); });
  });
  return names;
}

/* ========= 3) RUN: SMOOTHING ========= */
var compositesSm = smoothCollection(composites, PARAMS2.indices, PARAMS2.method);

/* ========= 4) RUN: METRICS ========= */
var perIndexImgs = PARAMS2.indices.map(function(idx){
  return computeMetrics(compositesSm, idx);
});
var metricsImage = ee.Image.cat(perIndexImgs).clip(aoi);

// Optional tiny asset cache (makes P5 faster to iterate)
// Export.image.toAsset({ image: m.select(['NDVI_AUC','NDVI_drop','NDWI_AUC']),
//   description: 'ph2_cache_core', assetId: 'users/<you>/ph2_cache_core',
//   region: p1.aoi, scale: 10, maxPixels: 1e13 });



// Flatten a dictionary property on an EE object onto the same object with a prefix.
// If the property is missing or not a dictionary, it simply returns the object unchanged.
function flattenDictProperty(eeObj, propName, prefix) {
  var dict = ee.Dictionary(eeObj.get(propName));
  // If dict is empty, size() is 0 and fromLists([]) is a no-op.
  var keys = dict.keys();
  var vals = keys.map(function(k){ return dict.get(k); });
  var prefKeys = keys.map(function(k){ return ee.String(prefix).cat(ee.String(k)); });
  var flat = ee.Dictionary.fromLists(prefKeys, vals);
  return eeObj.setMulti(flat);
}


// Attach provenance / params
var firstComp = ee.Image(composites.first());
metricsImage = ee.Image(metricsImage.set({
  run_tag: firstComp.get('run_tag'),
  phase1_module: 'users/lalutatu/wheatstress:Phase_1',
  smoothing_method: PARAMS2.method,
  indices: PARAMS2.indices.join(','),
  stepDays: PARAMS2.stepDays,
  scaleFactor: PARAMS2.scaleFactor,
  postPeakCenter: PARAMS2.postPeakCenter,
  postPeakHalfWinDays: PARAMS2.postPeakHalfWinDays,
  slopeWin1Center: PARAMS2.slopeWin1Center,
  slopeWin2Center: PARAMS2.slopeWin2Center,
  slopeHalfWinDays: PARAMS2.slopeHalfWinDays
}));

metricsImage = metricsImage.set({maskProvenance: firstComp.get('maskProvenance')});
metricsImage = ee.Image(flattenDictProperty(metricsImage, 'maskProvenance', 'mask_'));


var metricNames = buildMetricNames(PARAMS2.indices);

/* ========= 5) QA PREVIEW ========= */
if (PARAMS2.showPreview) {
  Map.centerObject(aoi, 9);

  // Crop mask tint
  Map.addLayer(cropMask.selfMask().clip(aoi), {palette:['66bb66'], opacity:0.2}, 'Crop mask');

  // Quicklooks
  Map.addLayer(metricsImage.select('NDVI_peak').clip(aoi),
               {min:0.2, max:0.9, palette:['2c7bb6','abd9e9','ffffbf','fdae61','d7191c']},
               'NDVI_peak');
  Map.addLayer(metricsImage.select('NDVI_drop').clip(aoi),
               {min:-0.3, max:0.3, palette:['313695','74add1','ffffff','f46d43','a50026']},
               'NDVI_drop (peak - anchor)');
  Map.addLayer(metricsImage.select('NDVI_AUC').clip(aoi),
               {min:3, max:25, palette:['00441b','1b7837','a6dba0','d9f0d3']},
               'NDVI_AUC', true);


Map.onClick(function(pt){
  var point = ee.Geometry.Point([pt.lon, pt.lat]);
  var idx   = PARAMS2.chartIndex || 'NDVI';
  var rawBand = idx + '_med';
  var smBand  = idx + '_sm';

  var list = compositesSm.toList(compositesSm.size());
  var n    = compositesSm.size();

  // Build a grouped table: two rows per date (series = 'raw' or 'sm')
  var feats = ee.List.sequence(0, n.subtract(1)).map(function(i){
    i = ee.Number(i);
    var im   = ee.Image(list.get(i));
    var date = ee.Date(im.get('system:time_start'));  // <-- Date, not millis
    var vals = im.select([rawBand, smBand]).reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: point,
      scale: PARAMS2.scale,
      maxPixels: 1e8,
      bestEffort: true
    });
    var fRaw = ee.Feature(null, {date: date, series: 'raw', value: vals.get(rawBand)});
    var fSm  = ee.Feature(null, {date: date, series: 'sm',  value: vals.get(smBand)});
    return [fRaw, fSm];
  }).flatten();

  var fc = ee.FeatureCollection(feats);

  var chart = ui.Chart.feature.groups(fc, 'date', 'value', 'series')
    .setChartType('LineChart')
    .setOptions({
      title: idx + ' — raw vs smoothed (' + PARAMS2.method + ')',
      hAxis: {title:'Date', format:'YYYY-MM-dd'},
      vAxis: {title:'Value'},
      pointSize: 2,
      lineWidth: 2,
      legend: {position:'bottom'},
      interpolateNulls: true
    });
  print(chart);

  // Robust peak-date readout (handles outside mask / nulls)
  var sample = metricsImage.sample(point, PARAMS2.scale).first();
  var peakDate = ee.Algorithms.If(
    sample,
    (function(){
      var v = ee.Feature(sample).get(idx + '_datePeak_millis');
      return ee.Algorithms.If(
        v,
        ee.Date(ee.Number(v)).format('YYYY-MM-dd'),
        'No data (masked pixel)'
      );
    })(),
    'Outside crop mask / no data'
  );
  print('Peak date (' + idx + '):', peakDate);
  print('Anchor center:', ee.Date(PARAMS2.postPeakCenter));
  print('Slope centers:', ee.Date(PARAMS2.slopeWin1Center), ee.Date(PARAMS2.slopeWin2Center));
});



  // Summary prints
  print('PARAMS2', PARAMS2);
  print('Composites count:', composites.size());
  print('metricNames (', metricNames.length, '):', metricNames);
  print('metricsImage band types:', metricsImage.bandTypes());
}

/* ========= 6) EXPORTS FOR PHASE 3 ========= */
exports.PARAMS2      = PARAMS2;
exports.compositesSm = compositesSm;
exports.metricsImage = metricsImage;
exports.metricNames  = metricNames;

