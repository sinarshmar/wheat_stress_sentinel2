/**** PHASE 3 — Historical Baseline & Anomaly Helpers (GEE JS, memory-lean) ****/
var p1 = require('users/lalutatu/wheatstress:Phase_1');
var p2 = require('users/lalutatu/wheatstress:Phase_2');

/* === Handles === */
var aoi        = p1.aoi;
var cropMask   = p1.cropMask;
var PARAMS     = p1.PARAMS;
var PARAMS2    = p2.PARAMS2;
var curMetrics = ee.Image(p2.metricsImage);
var curSmIC    = ee.ImageCollection(p2.compositesSm);
var maskProv   = ee.Dictionary(p1.maskProvenance);

print('Phase-2 centres (raw):', {
  postPeakCenter: PARAMS2.postPeakCenter,
  slopeWin1Center: PARAMS2.slopeWin1Center,
  slopeWin2Center: PARAMS2.slopeWin2Center
});

// window + AUC scale
var WIN_DAYS  = ee.Number(PARAMS.windowDays || PARAMS2.stepDays || 10);
var AUC_SCALE = ee.Number(PARAMS2.aucScale || PARAMS2.scaleFactor || 1);

/* === Phase-3 Config === */
var PARAMS3 = {
  histSeasons: [
    {start: '2020-11-01', end: '2021-04-15', tag: 'rabi_2020_21'},
    {start: '2021-11-01', end: '2022-04-15', tag: 'rabi_2021_22'},
    {start: '2022-11-01', end: '2023-04-15', tag: 'rabi_2022_23'}
  ],
  currentSeason: {start: PARAMS.seasonStart, end: PARAMS.seasonEnd},
  // We’ll still use the full list for current-season references/props,
  // but for historical we will compute NDVI only to stay within memory.
  indices: ['NDVI','NDWI','GNDVI','SAVI','MSAVI2'],
  histIndices: ['NDVI'],     // << memory-lean baseline
  eps: 1e-6,
  sdFloor: 1.0,     // minimum SD for z-score denom, in NDVI*AUC units (tweak 0.5–2.0 if needed)
  clipZ: 5,
  showPreview: false
};

/* === Console sanity prints === */
print('Use print(...) to write to this console.');
print('PARAMS2', PARAMS2);
print('Composites count (P1):', ee.ImageCollection(p1.composites).size());
print('metricNames size:', ee.List(p2.metricNames).size());
print('metricNames:', p2.metricNames);
print('metricsImage band types:', curMetrics.bandTypes());
print('PH3 PARAMS3:', PARAMS3);
print('Window/Step days → P1.windowDays =', PARAMS.windowDays, ', P2.stepDays =', PARAMS2.stepDays, ', using =', WIN_DAYS);

/* === Helpers === */
function prefixDict(prefix, dict){
  dict = ee.Dictionary(dict);
  var ks = dict.keys();
  var pks = ks.map(function(k){ return ee.String(prefix).cat(ee.String(k)); });
  return ee.Dictionary.fromLists(pks, ks.map(function(k){ return dict.get(k); }));
}

/* === Phase-1-style composites (lean) === */
function maskS2(img) {
  var scl = img.select('SCL');
  var qa60 = img.select('QA60');
  var cloudProb = img.select('cloud_prob');
  var m1 = cloudProb.lte(PARAMS.s2cloudlessMaxProb);
  var sclKeep = ee.Image(1);
  PARAMS.sclRemove.forEach(function(code){ sclKeep = sclKeep.and(scl.neq(code)); });
  var bit10 = qa60.bitwiseAnd(1 << 10).eq(0);
  var bit11 = qa60.bitwiseAnd(1 << 11).eq(0);
  var mask = m1.and(sclKeep).and(bit10.and(bit11));
  var refl = img.select(['B2','B3','B4','B8','B11']).resample('bilinear');
  var others = img.select(img.bandNames().removeAll(['B2','B3','B4','B8','B11']));
  return refl.addBands(others).updateMask(mask).copyProperties(img, img.propertyNames());
}
function addIndices(img) {
  var nir = img.select('B8'), red = img.select('B4'), grn = img.select('B3'), sw1 = img.select('B11');
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  // keep other indices available if you ever toggle to heavy mode
  var ndwi = nir.subtract(sw1).divide(nir.add(sw1)).rename('NDWI');
  var gndv = nir.subtract(grn).divide(nir.add(grn)).rename('GNDVI');
  var L = 0.5;
  var savi = nir.subtract(red).multiply(1+L).divide(nir.add(red).add(L)).rename('SAVI');
  var msavi2 = nir.multiply(2).add(1).pow(2).subtract(nir.subtract(red).multiply(8)).sqrt()
                .multiply(-1).add(nir.multiply(2)).add(1).multiply(0.5).rename('MSAVI2');
  return img.addBands([ndvi, ndwi, gndv, savi, msavi2]);
}

// SUPER-LEAN path: keep only what's needed to build NDVI, mask hard, clip early.
function prepS2LeanNDVI(img) {
  // Expect img has B4, B8, SCL, QA60, cloud_prob
  var scl = img.select('SCL');
  var qa60 = img.select('QA60');
  var cloudProb = img.select('cloud_prob');

  var m1 = cloudProb.lte(PARAMS.s2cloudlessMaxProb);
  var sclKeep = ee.Image(1);
  PARAMS.sclRemove.forEach(function(code){ sclKeep = sclKeep.and(scl.neq(code)); });
  var bit10 = qa60.bitwiseAnd(1 << 10).eq(0);
  var bit11 = qa60.bitwiseAnd(1 << 11).eq(0);
  var mask = m1.and(sclKeep).and(bit10.and(bit11));

  var ndvi = img.select('B8').subtract(img.select('B4'))
               .divide(img.select('B8').add(img.select('B4')))
               .rename('NDVI');

  // IMPORTANT: mask + clip BEFORE any aggregation; drop all other bands
  return ndvi.updateMask(mask)
             .updateMask(cropMask)
             .clip(aoi)
             .copyProperties(img, ['system:time_start','system:index']);
}


function buildComposites(season, indexList){  // indexList controls what we reduce
  var start = ee.Date(season.start), end = ee.Date(season.end);

  // Fetch SR + cloudprob but SELECT ONLY the bands needed downstream
  var s2  = ee.ImageCollection(PARAMS.s2sr)
              .filterBounds(aoi).filterDate(start, end)
              .select(['B4','B8','SCL','QA60']); // << keep minimal
  var s2c = ee.ImageCollection(PARAMS.s2cloudless)
              .filterBounds(aoi).filterDate(start, end)
              .select(['probability']);          // << one band

  // Join via saveFirst; attach cloud_prob; CLIP EARLY to AOI
  var joined = ee.Join.saveFirst('s2c').apply({
    primary: s2, secondary: s2c,
    condition: ee.Filter.equals({leftField:'system:index', rightField:'system:index'})
  });

  var s2WithProb = ee.ImageCollection(joined).map(function(img){
    img = ee.Image(img);
    var prob = ee.Image(img.get('s2c')).select('probability').rename('cloud_prob');
    // keep only minimal SR + QA + SCL + cloud_prob
    return img.addBands(prob).clip(aoi);
  });

  // Detect NDVI-only path
  var idxList = ee.List(indexList);
  var isNDVIonly = idxList.size().eq(1).and(ee.String(idxList.get(0)).compareTo('NDVI').eq(0));

  // Build a minimal cleaned collection:
  //  - NDVI-only path: use the SUPER-LEAN prep (no SR bands kept)
  //  - otherwise: fall back to maskS2 + addIndices (rare case)
  var cleaned = ee.ImageCollection(ee.Algorithms.If(
    isNDVIonly,
    s2WithProb.map(prepS2LeanNDVI),
    s2WithProb.map(maskS2).map(addIndices).select(idxList).map(function(im){
      return im.updateMask(cropMask).clip(aoi);
    })
  ));

  // 10-day windows identical to Phase 1
  function makeWindows(startDate, endDate, stepDays) {
    var n = endDate.difference(startDate, 'day').divide(stepDays).ceil();
    var list = ee.List.sequence(0, ee.Number(n).subtract(1));
    return list.map(function(i){
      i = ee.Number(i);
      var winStart = startDate.advance(i.multiply(stepDays), 'day');
      var winEnd   = winStart.advance(stepDays, 'day');
      return ee.Dictionary({start: winStart, end: winEnd});
    });
  }

  function reduceWindow(winDict) {
    winDict = ee.Dictionary(winDict);
    var ws = ee.Date(winDict.get('start')), we = ee.Date(winDict.get('end'));
    var ic = cleaned.filterDate(ws, we);

    // Median only of the requested index list
    var reduced = ic.reduce(ee.Reducer.median());
    var newNames = idxList.map(function(b){ return ee.String(b).cat('_med'); });
    var mid = ws.advance(WIN_DAYS.divide(2), 'day');

    return ee.Image(reduced.rename(newNames))
      .updateMask(cropMask)
      .clip(aoi)
      .set({
        'run_tag': (PARAMS.runTag || PARAMS.run_tag),
        'win_start': ws.format('YYYY-MM-dd'),
        'win_end':   we.format('YYYY-MM-dd'),
        'system:index': ws.format('YYYYMMdd'),
        'system:time_start': mid.millis(),
        'maskProvenance': maskProv
      });
  }

  var windows = ee.List(makeWindows(start, end, WIN_DAYS));
  var composites = ee.ImageCollection(
    windows.map(function(w){
      var ws = ee.Date(ee.Dictionary(w).get('start'));
      var we = ee.Date(ee.Dictionary(w).get('end'));
      var has = cleaned.filterDate(ws, we).size().gt(0);
      return ee.Image(ee.Algorithms.If(has, reduceWindow(w), ee.Image().set('skip',1)));
    })
  ).filter(ee.Filter.neq('skip', 1));

  return composites;
}


/* === Phase-2 smoothing (exact) === */
function medianBand(ic, bandName, newName){ return ic.select(bandName).median().rename(newName); }
function smoothAtTime_median3(icSorted, currentImg, idx){
  var t = ee.Date(currentImg.get('system:time_start'));
  var step = ee.Number(PARAMS2.stepDays || 10);
  var winIC = icSorted.filterDate(t.advance(step.multiply(-1),'day'), t.advance(step.add(1),'day'));
  return medianBand(winIC, idx + '_med', idx + '_sm');
}
function smoothAtTime_sg5(icSorted, currentImg, idx){
  var t = ee.Date(currentImg.get('system:time_start'));
  var step = ee.Number(PARAMS2.stepDays || 10);
  var candidates = icSorted.filterDate(t.advance(step.multiply(-2),'day'), t.advance(step.multiply(2).add(1),'day'))
     .map(function(im){
       im = ee.Image(im);
       var ts = ee.Date(im.get('system:time_start'));
       return im.set('tdiff', ee.Number(ts.difference(t,'millisecond')).abs());
     }).sort('tdiff');
  var size = candidates.size();
  return ee.Image(ee.Algorithms.If(size.gte(5),
    (function(){
      var list5 = candidates.toList(5);
      var weights = ee.List([-3,12,17,12,-3]);
      return ee.ImageCollection(ee.List.sequence(0,4).map(function(k){
        k = ee.Number(k);
        var im = ee.Image(list5.get(k));
        var w  = ee.Number(weights.get(k)).divide(35);
        return im.select(idx + '_med').multiply(w);
      })).sum().rename(idx + '_sm');
    })(),
    smoothAtTime_median3(icSorted, currentImg, idx)
  ));
}
function smoothCollection(ic, indices, method){
  var sorted = ic.sort('system:time_start');
  return sorted.map(function(img){
    var out = ee.Image(img);
    indices.forEach(function(idx){
      var sm = (method === 'sg5') ? smoothAtTime_sg5(sorted, img, idx)
                                  : smoothAtTime_median3(sorted, img, idx);
      out = out.addBands(sm.updateMask(cropMask));
    });
    return out.copyProperties(img, img.propertyNames());
  });
}

/* === AUC-only helper (historical, NDVI) === */
function aucOnly(smIC, idx){
  return smIC.select(idx+'_sm')
    .map(function(im){ return im.multiply(PARAMS2.stepDays); })
    .sum()
    .multiply(AUC_SCALE)
    .rename(idx+'_AUC')
    .updateMask(cropMask)
    .clip(aoi);
}

/* === Historical loop → NDVI_AUC_<tag> stack (lean) === */
var histAUCImages = [];
var histSmSeries  = {}; // keep for charts if preview

PARAMS3.histSeasons.forEach(function(season){
  // Only NDVI for historical baseline to avoid memory blow-ups
  var comp = buildComposites(season, PARAMS3.histIndices);      // ['NDVI'] only
  var sm   = ee.ImageCollection(smoothCollection(comp, PARAMS3.histIndices, PARAMS2.method));
  if (PARAMS3.showPreview) { histSmSeries[season.tag] = sm; }   // chart hook only when needed

  // AUC for NDVI only (no peak/drop/slope for history)
  var ndviAUC = aucOnly(sm, 'NDVI').rename('NDVI_AUC_' + season.tag);
  histAUCImages.push(ndviAUC);
});

var histAUCImage = ee.Image.cat(histAUCImages).updateMask(cropMask).clip(aoi);
print('Historical AUC bands:', histAUCImage.bandNames());

/* === Baseline stats === */
var stats = histAUCImage.reduce(
  ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs:true})
                   .combine({reducer2: ee.Reducer.count(),  sharedInputs:true})
);
var aucBaselineStats = stats.rename(['NDVI_AUC_hist_mean','NDVI_AUC_hist_sd','hist_valid_count'])
                           .updateMask(stats.select('count').gte(2))
                           .updateMask(cropMask)
                           .clip(aoi);

/* === Current-season helpers & z-score === */
var NDVI_drop    = curMetrics.select('NDVI_drop').updateMask(cropMask);
var NDWI_drop    = curMetrics.select('NDWI_drop').updateMask(cropMask);
var NDVI_AUC_cur = curMetrics.select('NDVI_AUC').rename('NDVI_AUC_cur').updateMask(cropMask);

var meanHist = aucBaselineStats.select('NDVI_AUC_hist_mean');
var sdRaw    = aucBaselineStats.select('NDVI_AUC_hist_sd');
var sdUsed   = sdRaw.max(PARAMS3.sdFloor).rename('NDVI_AUC_hist_sd_used');
var floorApplied = sdRaw.lt(PARAMS3.sdFloor).rename('sd_floor_applied');  // 1 where floored

var AUC_z_raw = NDVI_AUC_cur.subtract(meanHist).divide(sdRaw.max(PARAMS3.eps)).rename('AUC_z_raw');
var AUC_z     = NDVI_AUC_cur.subtract(meanHist).divide(sdUsed).rename('AUC_z');
var AUC_z_vis = AUC_z.clamp(-PARAMS3.clipZ, PARAMS3.clipZ).rename('AUC_z_vis');

var anomalyHelpers = ee.Image.cat([
  NDVI_drop, NDWI_drop, NDVI_AUC_cur,
  AUC_z_raw, AUC_z, AUC_z_vis,
  sdUsed, floorApplied
]).updateMask(cropMask).clip(aoi);

if (PARAMS3.showPreview) {
print('Baseline bands:', aucBaselineStats.bandNames());
print('Anomaly helpers:', anomalyHelpers.bandNames());
}
/* === Provenance (cast back to Image after setMulti) === */
var sharedProps = {
  run_tag: (PARAMS.runTag || PARAMS.run_tag),
  hist_seasons: PARAMS3.histSeasons.map(function(s){ return s.tag; }),
  phase2_method: PARAMS2.method,
  phase2_indices: PARAMS3.indices,
  postPeakCenter: PARAMS2.postPeakCenter,
  postPeakHalfWinDays: PARAMS2.postPeakHalfWinDays,
  slopeWin1Center: PARAMS2.slopeWin1Center,
  slopeWin2Center: PARAMS2.slopeWin2Center,
  slopeHalfWinDays: PARAMS2.slopeHalfWinDays,
  stepDays: PARAMS2.stepDays,
  crs: PARAMS.crs, scale: PARAMS.scale,
  sdFloor: PARAMS3.sdFloor,                 // <<< add
  auc_units: 'NDVI*day',                      // <<< add (NDVI is unitless; days from stepDays)
  baseline_years: 3 // number of historical seasons used
  
};


var maskFlat = prefixDict('mask_', maskProv);
aucBaselineStats = ee.Image(aucBaselineStats.set(sharedProps).set('maskProvenance', maskProv).setMulti(maskFlat));
anomalyHelpers   = ee.Image(anomalyHelpers  .set(sharedProps).set('maskProvenance', maskProv).setMulti(maskFlat));

print('Baseline bands after props:', aucBaselineStats.bandNames());
print('Helpers bands after props:', anomalyHelpers.bandNames());

/* === Optional sanity: aligned centres per hist season (safe prints) === */
function toDateSafe(centerAny, fb) {
  var local = (centerAny !== undefined && centerAny !== null && centerAny !== '') ? centerAny : fb;
  return ee.Date(local);
}
function alignDateToSeason(centerAny, season, prefer, fb) {
  var d      = toDateSafe(centerAny, fb);
  var m      = ee.Number(d.get('month'));
  var day    = ee.Number(d.get('day'));
  var startY = ee.Number(ee.Date(season.start).get('year'));
  var endY   = ee.Number(ee.Date(season.end).get('year'));
  var startDate = ee.Date.fromYMD(startY, m, day);
  var endDate   = ee.Date.fromYMD(endY,   m, day);
  var pref = ee.String(prefer || 'auto');
  var isStart = pref.compareTo('start').eq(0);
  var isEnd   = pref.compareTo('end').eq(0);
  return ee.Date(ee.Algorithms.If(isStart, startDate,
                    ee.Algorithms.If(isEnd, endDate,
                      ee.Algorithms.If(m.gte(7), startDate, endDate))));
}
PARAMS3.histSeasons.forEach(function(season) {
  var c1 = alignDateToSeason(PARAMS2.slopeWin1Center, season, 'start', '2023-12-15');
  var c2 = alignDateToSeason(PARAMS2.slopeWin2Center, season, 'end',   '2024-01-10');
  var ca = alignDateToSeason(PARAMS2.postPeakCenter,  season, 'end',   '2024-03-05');
  print('Aligned centres — ' + season.tag, ee.Dictionary({
    slopeWin1Center: c1.format('YYYY-MM-dd'),
    slopeWin2Center: c2.format('YYYY-MM-dd'),
    postPeakCenter:  ca.format('YYYY-MM-dd')
  }));
});

/* === QA Map (toggle) === */
if (PARAMS3.showPreview) {
  Map.centerObject(aoi, 9);
  Map.addLayer(AUC_z_vis, {min:-PARAMS3.clipZ, max:PARAMS3.clipZ,
    palette:['#2166ac','#67a9cf','#d1e5f0','#f7f7f7','#fddbc7','#ef8a62','#b2182b']}, 'AUC_z (clipped)');
  Map.addLayer(anomalyHelpers.select('NDVI_drop'),
               {min:-0.4, max:0.4, palette:['#313695','#74add1','#ffffff','#f46d43','#a50026']}, 'NDVI_drop');
  Map.addLayer(anomalyHelpers.select('NDWI_drop'),
               {min:-0.4, max:0.4, palette:['#313695','#74add1','#ffffff','#f46d43','#a50026']}, 'NDWI_drop');
  Map.addLayer(aucBaselineStats.select('NDVI_AUC_hist_sd'),
               {min:0, max:50, palette:['#f7f7f7','#cccccc','#969696','#636363','#252525']}, 'NDVI_AUC_hist_sd');
}

/* === Exports (for Phase 4) === */
exports.PARAMS3          = PARAMS3;
exports.histAUCImage     = histAUCImage;     // NDVI_AUC_<tag> stack
exports.aucBaselineStats = aucBaselineStats; // mean/sd/count
exports.anomalyHelpers   = anomalyHelpers;   // NDVI_drop, NDWI_drop, NDVI_AUC_cur, AUC_z, AUC_z_vis

/* === Prints === */
print('run_tag:', (PARAMS.runTag || PARAMS.run_tag));
print('Phase-2 method:', PARAMS2.method, 'stepDays:', PARAMS2.stepDays, 'P1 windowDays:', PARAMS.windowDays);
// Fast AUC_z sanity via sampling (avoids full-AOI timeouts)
var sampleScale = ee.Number(PARAMS.scale).multiply(5);   // e.g., 50 m if base is 10 m
var samplesFC = AUC_z.sample({
  region: aoi,
  scale: sampleScale,
  numPixels: 150000,         // lower if still slow; raise if AOI is small
  seed: 42,
  tileScale: 8,
  geometries: false
}).select(['AUC_z']);

// Compute stats via a single reduceColumns call
var stats = samplesFC.reduceColumns({
  reducer: ee.Reducer.min()
            .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
            .combine({reducer2: ee.Reducer.mean(), sharedInputs: true})
            .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
            .combine({reducer2: ee.Reducer.median(), sharedInputs: true}),
  selectors: ['AUC_z']
});

var quick = ee.Dictionary({
  count:  samplesFC.size(),
  min:    stats.get('min'),
  max:    stats.get('max'),
  mean:   stats.get('mean'),
  sd:     stats.get('stdDev'),
  median: stats.get('median')
});
print('AUC_z quick stats (sampled):', quick);

// Optional percentiles
var pct = samplesFC.reduceColumns({
  reducer: ee.Reducer.percentile([5,25,50,75,95]),
  selectors: ['AUC_z']
});
print('AUC_z percentiles (sampled):', pct);


// --- SD sanity (sampled) ---
var sdSampleScale = ee.Number(PARAMS.scale).multiply(5);
var sdSamples = aucBaselineStats.select('NDVI_AUC_hist_sd').sample({
  region: aoi, scale: sdSampleScale, numPixels: 60000, seed: 13, tileScale: 8, geometries: false
});

var sdStats = sdSamples.reduceColumns({
  reducer: ee.Reducer.min()
            .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
            .combine({reducer2: ee.Reducer.mean(), sharedInputs: true})
            .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
            .combine({reducer2: ee.Reducer.percentile([5,25,50,75,95]), sharedInputs: true}),
  selectors: ['NDVI_AUC_hist_sd']
});
print('NDVI_AUC_hist_sd quick stats (sampled):', sdStats);

var fracTinySD = sdSamples.filter(ee.Filter.lt('NDVI_AUC_hist_sd', PARAMS3.sdFloor))
                          .size().divide(sdSamples.size());
print('Fraction of pixels with sd < sdFloor:', fracTinySD);



