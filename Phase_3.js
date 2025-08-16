/**** PHASE 3 — Historical Baseline & Anomaly Helpers (GEE JS) ****/
/* Requires:
 *   Phase_1 exports: aoi, cropMask, composites, PARAMS, maskProvenance
 *   Phase_2 exports: PARAMS2, compositesSm, metricsImage, metricNames
 */
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

// Safe window size (prefer P1.windowDays, else P2.stepDays, else 10)
var WIN_DAYS  = ee.Number(PARAMS.windowDays || PARAMS2.stepDays || 10);
var AUC_SCALE = ee.Number(PARAMS2.aucScale || PARAMS2.scaleFactor || 1);

/* === Phase-3 Config (spec) === */
var PARAMS3 = {
  histSeasons: [
    {start: '2020-11-01', end: '2021-04-15', tag: 'rabi_2020_21'},
    {start: '2021-11-01', end: '2022-04-15', tag: 'rabi_2021_22'},
    {start: '2022-11-01', end: '2023-04-15', tag: 'rabi_2022_23'}
  ],
  currentSeason: {start: PARAMS.seasonStart, end: PARAMS.seasonEnd},
  indices: ['NDVI','NDWI','GNDVI','SAVI','MSAVI2'],
  usePhase2Params: true,
  eps: 1e-6,
  clipZ: 5,
  showPreview: true
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

/* === Tiny helpers === */
function prefixDict(prefix, dict){
  dict = ee.Dictionary(dict);
  var ks = dict.keys();
  var pks = ks.map(function(k){ return ee.String(prefix).cat(ee.String(k)); });
  return ee.Dictionary.fromLists(pks, ks.map(function(k){ return dict.get(k); }));
}

/* === Phase-1-exact masking & indices === */
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

  var refl = img.select(['B2','B3','B4','B8','B11']).resample('bilinear');
  var others = img.select(img.bandNames().removeAll(['B2','B3','B4','B8','B11']));

  return refl.addBands(others)
             .updateMask(mask)
             .copyProperties(img, img.propertyNames());
}
function addIndices(img) {
  var nir = img.select('B8'), red = img.select('B4'), grn = img.select('B3'), sw1 = img.select('B11');
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var ndwi = nir.subtract(sw1).divide(nir.add(sw1)).rename('NDWI'); // NIR–SWIR1
  var gndv = nir.subtract(grn).divide(nir.add(grn)).rename('GNDVI');
  var L = 0.5;
  var savi = nir.subtract(red).multiply(1+L).divide(nir.add(red).add(L)).rename('SAVI');
  var msavi2 = nir.multiply(2).add(1).pow(2)
      .subtract(nir.subtract(red).multiply(8)).sqrt()
      .multiply(-1).add(nir.multiply(2)).add(1).multiply(0.5).rename('MSAVI2');
  return img.addBands([ndvi, ndwi, gndv, savi, msavi2]);
}

/* === Phase-1-style 10-day composites for arbitrary season === */
function buildComposites(season){
  var start = ee.Date(season.start), end = ee.Date(season.end);
  var s2  = ee.ImageCollection(PARAMS.s2sr).filterBounds(aoi).filterDate(start, end);
  var s2c = ee.ImageCollection(PARAMS.s2cloudless).filterBounds(aoi).filterDate(start, end);

  // FIX: Build ImageCollection via saveFirst join (avoid List.map on a FeatureCollection).
  var joined = ee.Join.saveFirst('s2c').apply({
    primary: s2, secondary: s2c,
    condition: ee.Filter.equals({leftField:'system:index', rightField:'system:index'})
  });

  var s2WithProb = ee.ImageCollection(joined).map(function(img){
    img = ee.Image(img);
    var prob = ee.Image(img.get('s2c')).select('probability').rename('cloud_prob');
    return img.addBands(prob);
  });

  var s2Clean = s2WithProb.map(maskS2).map(addIndices);

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
    var ic = s2Clean.filterDate(ws, we).filterBounds(aoi);
    var reduced = ic.reduce(ee.Reducer.median());
    var oldNames = reduced.bandNames();
    var newNames = oldNames.map(function(b){ return ee.String(b).replace('_median', '_med'); });
    var mid = ws.advance(WIN_DAYS.divide(2), 'day');

    return reduced.rename(newNames)
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
      var has = s2Clean.filterDate(ws, we).size().gt(0);
      return ee.Image(ee.Algorithms.If(has, reduceWindow(w), ee.Image().set('skip',1)));
    })
  ).filter(ee.Filter.neq('skip', 1));

  return composites;
}

/* === Phase-2 smoothing & metrics (exact, including SG5) === */
function withinDays(center, halfWinDays){
  center = ee.Date(center);
  var start = center.advance(ee.Number(halfWinDays).multiply(-1), 'day');
  var end   = center.advance(ee.Number(halfWinDays).add(1), 'day'); // inclusive
  return ee.Filter.date(start, end);
}
function medianBand(ic, bandName, newName){ return ic.select(bandName).median().rename(newName); }

// NOTE: use ee.Number for step to avoid client/server arithmetic pitfalls.
function smoothAtTime_median3(icSorted, currentImg, idx){
  var t = ee.Date(currentImg.get('system:time_start'));
  var step = ee.Number(PARAMS2.stepDays || 10);
  var winIC = icSorted.filterDate(t.advance(step.multiply(-1),'day'),
                                  t.advance(step.add(1),'day'));
  return medianBand(winIC, idx + '_med', idx + '_sm');
}
function smoothAtTime_sg5(icSorted, currentImg, idx){
  var t = ee.Date(currentImg.get('system:time_start'));
  var step = ee.Number(PARAMS2.stepDays || 10);
  var candidates = icSorted.filterDate(t.advance(step.multiply(2).multiply(-1),'day'),
                                       t.advance(step.multiply(2).add(1),'day'))
     .map(function(im){
       im = ee.Image(im);
       var ts = ee.Date(im.get('system:time_start'));
       var diff = ee.Number(ts.difference(t,'millisecond')).abs();
       return im.set('tdiff', diff);
     }).sort('tdiff');
  var size = candidates.size();
  var sgImg = ee.Image(ee.Algorithms.If(size.gte(5),
    (function(){
      var list5 = candidates.toList(5);
      var weights = ee.List([-3,12,17,12,-3]);
      var weighted = ee.ImageCollection(ee.List.sequence(0,4).map(function(k){
        k = ee.Number(k);
        var im = ee.Image(list5.get(k));
        var w  = ee.Number(weights.get(k)).divide(35);
        return im.select(idx + '_med').multiply(w);
      })).sum().rename(idx + '_sm');
      return weighted;
    })(),
    smoothAtTime_median3(icSorted, currentImg, idx)
  ));
  return sgImg;
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
function medianSmInWindow(icSm, idx, centerStr, halfWinDays, outName){
  var winIC = icSm.filter(withinDays(centerStr, halfWinDays)).select(idx + '_sm');
  var has = winIC.size().gt(0);
  var empty = ee.Image.constant(0).updateMask(ee.Image(0));
  return ee.Image(ee.Algorithms.If(has, winIC.median(), empty))
           .rename(outName || (idx + '_sm_median'))
           .updateMask(cropMask);
}

// Coerce a possibly-missing centre to an ee.Date, with a safe default.
function toDateSafe(centerAny, fallbackStr) {
  var local = (centerAny !== undefined && centerAny !== null && centerAny !== '') ? centerAny : fallbackStr;
  return ee.Date(local);
}

// prefer: 'start' | 'end' | 'auto'
function alignDateToSeason(centerAny, season, prefer, fallbackStr) {
  var d      = toDateSafe(centerAny, fallbackStr);
  var m      = ee.Number(d.get('month'));
  var day    = ee.Number(d.get('day'));
  var startY = ee.Number(ee.Date(season.start).get('year'));
  var endY   = ee.Number(ee.Date(season.end).get('year'));

  var startDate = ee.Date.fromYMD(startY, m, day);
  var endDate   = ee.Date.fromYMD(endY,   m, day);

  var pref = ee.String(prefer || 'auto');
  var isStart = pref.compareTo('start').eq(0);
  var isEnd   = pref.compareTo('end').eq(0);

  return ee.Date(
    ee.Algorithms.If(
      isStart, startDate,
      ee.Algorithms.If(
        isEnd,   endDate,
        ee.Algorithms.If(
          m.gte(7), startDate, endDate   // auto: Jul–Dec → start year; Jan–Jun → end year
        )
      )
    )
  );
}

/* === Season metrics (P2-exact, with aligned centres) === */
function computeSeasonMetrics(smIC, indices, season){
  var c1 = alignDateToSeason(PARAMS2.slopeWin1Center, season, 'start', '2023-12-15'); // Dec → startY
  var c2 = alignDateToSeason(PARAMS2.slopeWin2Center, season, 'end',   '2024-01-10'); // Jan → endY
  var ca = alignDateToSeason(PARAMS2.postPeakCenter,  season, 'end',   '2024-03-05'); // Mar → endY

  var peak = ee.ImageCollection(indices.map(function(idx){
    return smIC.select(idx+'_sm').max().rename(idx+'_peak');
  })).toBands().rename(indices.map(function(i){return i+'_peak';})).updateMask(cropMask);

  var anchor = ee.ImageCollection(indices.map(function(idx){
    return medianSmInWindow(smIC, idx, ca, PARAMS2.postPeakHalfWinDays, idx+'_anchor');
  })).toBands().rename(indices.map(function(i){return i+'_anchor';})).updateMask(cropMask);

  var drop = ee.ImageCollection(indices.map(function(idx){
    return peak.select(idx+'_peak').subtract(anchor.select(idx+'_anchor')).rename(idx+'_drop');
  })).toBands().rename(indices.map(function(i){return i+'_drop';})).updateMask(cropMask);

  var early1 = ee.ImageCollection(indices.map(function(idx){
    return medianSmInWindow(smIC, idx, c1, PARAMS2.slopeHalfWinDays, idx+'_early1');
  })).toBands();
  var early2 = ee.ImageCollection(indices.map(function(idx){
    return medianSmInWindow(smIC, idx, c2, PARAMS2.slopeHalfWinDays, idx+'_early2');
  })).toBands();

  var deltaDays = ee.Number(ee.Date(c2).difference(ee.Date(c1), 'day'));
  var slopeEarly = early2.subtract(early1).divide(deltaDays.max(1))
                         .rename(indices.map(function(i){return i+'_slopeEarly';}))
                         .updateMask(cropMask);

  var auc = ee.ImageCollection(indices.map(function(idx){
    return smIC.select(idx+'_sm').map(function(im){ return im.multiply(PARAMS2.stepDays); }).sum()
               .multiply(AUC_SCALE)
               .rename(idx+'_AUC');
  })).toBands().rename(indices.map(function(i){return i+'_AUC';})).updateMask(cropMask);

  return ee.Image.cat([peak, anchor, drop, slopeEarly, auc]).updateMask(cropMask).clip(aoi)
           .set({'season_start': season.start, 'season_end': season.end});
}

/* === Historical loop → NDVI_AUC_<tag> stack === */
var histAUCImages = [];
var histSmSeries  = {}; // for charts and debugging

PARAMS3.histSeasons.forEach(function(season){
  var comp = buildComposites(season);
  var sm   = ee.ImageCollection(smoothCollection(comp, PARAMS3.indices, PARAMS2.method));
  histSmSeries[season.tag] = sm;
  var met  = computeSeasonMetrics(sm, PARAMS3.indices, season);
  var aucB = met.select('NDVI_AUC').rename('NDVI_AUC_' + season.tag).updateMask(cropMask).clip(aoi);
  histAUCImages.push(aucB);
});

var histAUCImage = ee.Image.cat(histAUCImages).updateMask(cropMask).clip(aoi);
print('Historical AUC bands:', histAUCImage.bandNames());

/* === Baseline stats (mean / sd / count over historical bands) === */
var stats = histAUCImage.reduce(
  ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs:true})
                   .combine({reducer2: ee.Reducer.count(),  sharedInputs:true})
);

// Rename to requested band names
var aucBaselineStats = stats.rename(['NDVI_AUC_hist_mean','NDVI_AUC_hist_sd','hist_valid_count'])
                           .updateMask(stats.select('count').gte(2))
                           .updateMask(cropMask)
                           .clip(aoi);

print('Baseline bands:', aucBaselineStats.bandNames());
print('Hist valid count min/max:',
  aucBaselineStats.select('hist_valid_count').reduceRegion({
    reducer: ee.Reducer.minMax(), geometry: aoi, scale: PARAMS.scale, bestEffort: true, maxPixels: 1e13
  })
);

/* === Current-season helpers & z-score === */
var NDVI_drop    = curMetrics.select('NDVI_drop').updateMask(cropMask);
var NDWI_drop    = curMetrics.select('NDWI_drop').updateMask(cropMask);
var NDVI_AUC_cur = curMetrics.select('NDVI_AUC').rename('NDVI_AUC_cur').updateMask(cropMask);

var meanHist = aucBaselineStats.select('NDVI_AUC_hist_mean');
var sdHist   = aucBaselineStats.select('NDVI_AUC_hist_sd').max(PARAMS3.eps);
var AUC_z    = NDVI_AUC_cur.subtract(meanHist).divide(sdHist).rename('AUC_z');
var AUC_z_vis= AUC_z.clamp(-PARAMS3.clipZ, PARAMS3.clipZ).rename('AUC_z_vis');

var anomalyHelpers = ee.Image.cat([NDVI_drop, NDWI_drop, NDVI_AUC_cur, AUC_z, AUC_z_vis])
                      .updateMask(cropMask).clip(aoi);

print('Anomaly helpers:', anomalyHelpers.bandNames());

/* === Provenance === */
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
  crs: PARAMS.crs, scale: PARAMS.scale
};
var maskFlat = prefixDict('mask_', maskProv);

// IMPORTANT: cast back to ee.Image after .set/.setMulti
aucBaselineStats = ee.Image(
  aucBaselineStats
    .set(sharedProps)
    .set('maskProvenance', maskProv)
    .setMulti(maskFlat)
);
anomalyHelpers = ee.Image(
  anomalyHelpers
    .set(sharedProps)
    .set('maskProvenance', maskProv)
    .setMulti(maskFlat)
);

// (Optional) quick sanity:
print('Baseline bands after props:', aucBaselineStats.bandNames());
print('Helpers bands after props:', anomalyHelpers.bandNames());

/* === Optional sanity: show aligned centres for each hist season === */
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
  Map.addLayer(NDVI_drop, {min:-0.4, max:0.4, palette:['#313695','#74add1','#ffffff','#f46d43','#a50026']}, 'NDVI_drop');
  Map.addLayer(NDWI_drop, {min:-0.4, max:0.4, palette:['#313695','#74add1','#ffffff','#f46d43','#a50026']}, 'NDWI_drop');
  Map.addLayer(aucBaselineStats.select('NDVI_AUC_hist_sd'),
               {min:0, max:50, palette:['#f7f7f7','#cccccc','#969696','#636363','#252525']}, 'NDVI_AUC_hist_sd');

  var uiPanel = ui.Panel({style:{width:'420px'}});
  ui.root.insert(0, uiPanel);
  Map.onClick(function(coords){
    uiPanel.clear();
    var pt = ee.Geometry.Point([coords.lon, coords.lat]);
    var chCur = ui.Chart.image.series({
      imageCollection: curSmIC.select('NDVI_sm'), region: pt, reducer: ee.Reducer.mean(),
      scale: PARAMS.scale
    }).setOptions({title:'NDVI_sm — current', hAxis:{title:'Date'}, vAxis:{title:'NDVI'}, lineWidth:2, pointSize:0});
    var tag0 = PARAMS3.histSeasons[0].tag;
    var histSm = ee.ImageCollection(histSmSeries[tag0]).select('NDVI_sm');
    var chHist = ui.Chart.image.series({
      imageCollection: histSm, region: pt, reducer: ee.Reducer.mean(), scale: PARAMS.scale
    }).setOptions({title:'Historical '+tag0, hAxis:{title:'Date'}, vAxis:{title:'NDVI'}, lineWidth:2, pointSize:0, colors:['#ef8a62']});

    var vals = ee.Image.cat([AUC_z.rename('AUC_z'), NDVI_AUC_cur,
                             meanHist.rename('NDVI_AUC_hist_mean'),
                             sdHist.rename('NDVI_AUC_hist_sd')])
      .reduceRegion({reducer: ee.Reducer.first(), geometry: pt, scale: PARAMS.scale, maxPixels: 1e13, bestEffort: true});
    vals.evaluate(function(v){
      uiPanel.add(ui.Label('AUC_z: ' + v.AUC_z));
      uiPanel.add(ui.Label('NDVI_AUC_cur: ' + v.NDVI_AUC_cur));
      uiPanel.add(ui.Label('NDVI_AUC_hist_mean: ' + v.NDVI_AUC_hist_mean));
      uiPanel.add(ui.Label('NDVI_AUC_hist_sd: ' + v.NDVI_AUC_hist_sd));
    });
    uiPanel.add(chCur); uiPanel.add(chHist);
  });
}

/* === Exports (for Phase 4) === */
exports.PARAMS3          = PARAMS3;
exports.histAUCImage     = histAUCImage;     // NDVI_AUC_<tag> stack
exports.aucBaselineStats = aucBaselineStats; // mean/sd/count
exports.anomalyHelpers   = anomalyHelpers;   // NDVI_drop, NDWI_drop, NDVI_AUC_cur, AUC_z, AUC_z_vis

/* === Prints === */
print('run_tag:', (PARAMS.runTag || PARAMS.run_tag));
print('Phase-2 method:', PARAMS2.method, 'stepDays:', PARAMS2.stepDays, 'P1 windowDays:', PARAMS.windowDays);
print('AUC_z basic stats:',
  AUC_z.reduceRegion({
    reducer: ee.Reducer.minMax().combine({reducer2: ee.Reducer.mean(), sharedInputs:true}),
    geometry: aoi, scale: PARAMS.scale, maxPixels: 1e13, bestEffort: true
  })
);
