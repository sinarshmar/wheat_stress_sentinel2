/**** PHASE 4 — Pseudo-Labels & Confidence Weights (GEE JS) ****/

var p1 = require('users/lalutatu/wheatstress:Phase_1');
var p2 = require('users/lalutatu/wheatstress:Phase_2');
var p3 = require('users/lalutatu/wheatstress:Phase_3');

/* Handles from prior phases */
var aoi        = p1.aoi;
var cropMask   = p1.cropMask;
var PARAMS     = p1.PARAMS;
var PARAMS2    = p2.PARAMS2;
var PARAMS3    = p3.PARAMS3;
var helpers    = p3.anomalyHelpers;   // NDVI_drop, NDWI_drop, AUC_z
var maskProv   = ee.Dictionary(p1.maskProvenance || {});

/* Phase-4 config (Option-B) */
var PARAMS4 = {
  ruleVersion: 'OptionB_v2_relaxed',
  ndviDropThr: 0.02,   // Capture top ~25% (p75 = 0.0175)
  ndwiDropThr: 0.01,   // Capture top ~20-25%
  aucZThr: 1.5,        // Capture bottom ~25% (p75 = 1.37)
  
  // Change the logic:
  useORLogic: true,     // New parameter

  weightCap: 3.0,
  showPreview: true,    // Turn on to verify
  showDistStats: false,
  weightGamma: 1.1,
};


/* Utils */
function prefixDict(prefix, dict){
  dict = ee.Dictionary(dict || {});
  var ks = dict.keys();
  var pks = ks.map(function(k){ return ee.String(prefix).cat(ee.String(k)); });
  return ee.Dictionary.fromLists(pks, ks.map(function(k){ return dict.get(k); }));
}
function guessRunTag(P){ return (P.run_tag || P.runTag || 'unknown_run'); }
function requireBands(img, needed){
  var have = ee.Image(img).bandNames();
  needed.forEach(function(b){
    var ok = have.contains(b);
    ok.evaluate(function(v){ if (!v) print('⚠️ PH4 missing band:', b); });
  });
}

// finite guard
// Guard: keep only finite (not NaN, not Inf) values
function finite(img) {
  img = ee.Image(img);
  var finiteMask = img.lt(1e20).and(img.gt(-1e20)).and(img.eq(img)); // mask out ±Inf and NaN
  return img.updateMask(finiteMask);
}

// Fetch helpers (strict mask + AOI clip)
requireBands(helpers, ['NDVI_drop','NDWI_drop','AUC_z']);
var NDVI_drop = finite(helpers.select('NDVI_drop')).updateMask(cropMask).clip(aoi);
var NDWI_drop = finite(helpers.select('NDWI_drop')).updateMask(cropMask).clip(aoi);
var AUC_z     = finite(helpers.select('AUC_z'))    .updateMask(cropMask).clip(aoi);

/* Label rule (Option-B): primary AND auxiliary */
var primary = NDVI_drop.gt(PARAMS4.ndviDropThr);
var auxiliary = NDWI_drop.gt(PARAMS4.ndwiDropThr).and(AUC_z.lt(PARAMS4.aucZThr));
var stressed = PARAMS4.useORLogic ? 
  primary.or(auxiliary) :  // OR logic - more inclusive
  primary.and(auxiliary);   // AND logic - original restrictive
  
// CRITICAL FIX: Create 0/1 labels for ALL pixels in cropMask
var label01   = stressed.unmask(0)  // Unmask to 0 for healthy pixels
  .updateMask(cropMask)              // Apply crop mask
  .rename('label01')
  .clip(aoi);
  
var labelMask = label01.eq(1).selfMask();  // Mask for stressed pixels only (for display)

/* Weight rule (Option A, anchored to thresholds) */
var n1 = NDVI_drop.divide(PARAMS4.ndviDropThr); // >=1 in positives
var n2 = NDWI_drop.divide(PARAMS4.ndwiDropThr);

// Anchor so (n1=1, n2=1) ⇒ zero signal; only the excess over threshold contributes
var m1 = n1.subtract(1).max(0);
var m2 = n2.subtract(1).max(0);
var prod = m1.multiply(m2);

// Ensure gamma exists (fallback)
PARAMS4.weightGamma = (PARAMS4.weightGamma !== undefined) ? PARAMS4.weightGamma : 1.1;

// Soft 0..1 curve: 1 - exp(-gamma * prod)
var soft01 = ee.Image(1).subtract(prod.multiply(PARAMS4.weightGamma).multiply(-1).exp());

// CRITICAL FIX: Assign weights to ALL pixels using pixel-wise operations
// Use where() or multiply/add for pixel-wise conditional assignment
var stressedWeight = soft01.multiply(PARAMS4.weightCap);
var defaultWeight = 0.1;  // Small weight for healthy pixels

// Pixel-wise weight assignment
var weight = stressed.multiply(stressedWeight)  // Where stressed=1, use calculated weight
  .add(stressed.not().multiply(defaultWeight))   // Where stressed=0, use default weight
  .updateMask(cropMask)
  .rename('weight')
  .clip(aoi);

var weight01 = stressed.multiply(soft01)  // Where stressed=1, use soft01
  .add(stressed.not().multiply(defaultWeight / PARAMS4.weightCap))  // Where stressed=0, use normalized default
  .updateMask(cropMask)
  .rename('weight01')
  .clip(aoi);

// Apply finite guard
weight = finite(weight);
weight01 = finite(weight01);

/* Stacks */
var labelStack  = ee.Image.cat([label01, labelMask]).updateMask(cropMask).clip(aoi);
var weightStack = ee.Image.cat([weight, weight01]).updateMask(cropMask).clip(aoi);
var labelWeight = ee.Image.cat([label01, weight, weight01]).updateMask(cropMask).clip(aoi);

/* Provenance */
var sharedProps = {
  run_tag: guessRunTag(PARAMS),
  phase4_rule: 'OptionB',
  phase4_rule_version: PARAMS4.ruleVersion,
  ndviDropThr: PARAMS4.ndviDropThr,
  ndwiDropThr: PARAMS4.ndwiDropThr,
  aucZThr:     PARAMS4.aucZThr,
  weightCap:   PARAMS4.weightCap,
  crs: PARAMS.crs,
  scale: PARAMS.scale,
  phase2_method: PARAMS2.method,
  hist_seasons: ee.List(PARAMS3.histSeasons || []).map(function(s){ return ee.Dictionary(s).get('tag'); }),
  weightGamma: PARAMS4.weightGamma
};
var maskFlat = prefixDict('mask_', maskProv);

label01     = ee.Image(label01    .set(sharedProps).set('maskProvenance', maskProv).setMulti(maskFlat));
labelMask   = ee.Image(labelMask  .set(sharedProps).set('maskProvenance', maskProv).setMulti(maskFlat));
weight      = ee.Image(weight     .set(sharedProps).set('maskProvenance', maskProv).setMulti(maskFlat));
weight01    = ee.Image(weight01   .set(sharedProps).set('maskProvenance', maskProv).setMulti(maskFlat));
labelStack  = ee.Image(labelStack .set(sharedProps).set('maskProvenance', maskProv).setMulti(maskFlat));
weightStack = ee.Image(weightStack.set(sharedProps).set('maskProvenance', maskProv).setMulti(maskFlat));
labelWeight = ee.Image(labelWeight.set(sharedProps).set('maskProvenance', maskProv).setMulti(maskFlat));

/* QA (toggle) */
if (PARAMS4.showPreview) {
  Map.centerObject(aoi, 9);
  
  // Validation: Check that we have both classes
  var class0Count = label01.eq(0).selfMask().reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: aoi,
    scale: 100,  // Coarse for speed
    maxPixels: 1e8,
    bestEffort: true
  }).get('label01');
  
  var class1Count = label01.eq(1).selfMask().reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: aoi,
    scale: 100,
    maxPixels: 1e8,
    bestEffort: true
  }).get('label01');
  
  print('VALIDATION - Class 0 (healthy) pixels:', class0Count);
  print('VALIDATION - Class 1 (stressed) pixels:', class1Count);
  print('VALIDATION - Ratio (stressed/healthy):', ee.Number(class1Count).divide(class0Count));

  // Keep only the mask ON by default; everything else OFF to save memory.
  Map.addLayer(labelMask, {palette:['#00ff88']}, 'PH4: label01 (mask)', true);
  Map.addLayer(weight, {min:0, max:PARAMS4.weightCap,
    palette:['#f7fbff','#c6dbef','#6baed6','#2171b5','#08306b']}, 'PH4: weight', false);
  Map.addLayer(NDVI_drop, {min:-0.4, max:0.6,
    palette:['#313695','#74add1','#ffffff','#f46d43','#a50026']}, 'NDVI_drop', false);
  Map.addLayer(NDWI_drop, {min:-0.3, max:0.5,
    palette:['#313695','#74add1','#ffffff','#f46d43','#a50026']}, 'NDWI_drop', false);
  Map.addLayer(AUC_z, {min:-4, max:4,
    palette:['#2166ac','#67a9cf','#d1e5f0','#f7f7f7','#fddbc7','#ef8a62','#b2182b']}, 'AUC_z', false);

  Map.onClick(function(pt){
    var g = ee.Geometry.Point([pt.lon, pt.lat]);
    var samp = ee.Image.cat([NDVI_drop, NDWI_drop, AUC_z, label01, weight, weight01])
      .sample(g, PARAMS.scale).first();
    print('PH4 @ click', ee.Dictionary({
      NDVI_drop: ee.Feature(samp).get('NDVI_drop'),
      NDWI_drop: ee.Feature(samp).get('NDWI_drop'),
      AUC_z:     ee.Feature(samp).get('AUC_z'),
      label01:   ee.Feature(samp).get('label01'),
      weight:    ee.Feature(samp).get('weight'),
      weight01:  ee.Feature(samp).get('weight01')
    }));
  });

  // --- Quick stats (fixed & lighter) ---
  var sampleScale = ee.Number(PARAMS.scale).multiply(10); // ~100 m

  // 1) True positive fraction (unbiased by weight mask)
  var labelSample = label01.sample({
    region: aoi, scale: sampleScale, numPixels: 20000, seed: 7, tileScale: 12
  });
  var posFrac = labelSample.reduceColumns({
    reducer: ee.Reducer.mean(), selectors: ['label01']
  }).get('mean');
  print('Phase 4 — positive fraction (sampled):', posFrac);

  // Area-weighted fraction (robust, coarse scale)
  var pxArea  = ee.Image.pixelArea()
                 .reproject({crs: PARAMS.crs, scale: sampleScale})
                 .updateMask(cropMask);
  var posArea = pxArea.updateMask(label01.eq(1)).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: aoi, scale: sampleScale, tileScale: 12, bestEffort: true
  }).get('area');
  var totArea = pxArea.reduceRegion({
    reducer: ee.Reducer.sum(), geometry: aoi, scale: sampleScale, tileScale: 12, bestEffort: true
  }).get('area');
  print('Phase 4 — positive fraction (area-weighted):', ee.Number(posArea).divide(totArea));

  // Gate diagnostics (sampled, unbiased)
  var gateSample = ee.Image.cat([
    NDVI_drop.gt(PARAMS4.ndviDropThr).rename('primary'),
    NDWI_drop.gt(PARAMS4.ndwiDropThr).and(AUC_z.lt(PARAMS4.aucZThr)).rename('aux'),
    label01.rename('both')
  ]).sample({
    region: aoi, scale: sampleScale, numPixels: 15000, seed: 5, tileScale: 12
  });
  var gateMeans = gateSample.reduceColumns({
    reducer: ee.Reducer.mean().repeat(3), selectors: ['primary','aux','both']
  });
  print('PH4 — gate coverage (primary, aux, both):', gateMeans);
  
  var posArea_ha = ee.Number(posArea).divide(10000);
  var totArea_ha = ee.Number(totArea).divide(10000);
  print('PH4 — positive area (ha):', posArea_ha);
  print('PH4 — total crop area (ha):', totArea_ha);

  // 2) Weight stats (all pixels)
  var weightSamp = weight.sample({
    region: aoi, scale: sampleScale, numPixels: 20000, seed: 7, tileScale: 12
  });
  var wStats = weightSamp.reduceColumns({
    reducer: ee.Reducer.min()
      .combine({reducer2: ee.Reducer.max(), sharedInputs:true})
      .combine({reducer2: ee.Reducer.mean(), sharedInputs:true})
      .combine({reducer2: ee.Reducer.stdDev(), sharedInputs:true})
      .combine({reducer2: ee.Reducer.percentile([5,25,50,75,95]), sharedInputs:true}),
    selectors: ['weight']
  });
  print('PH4 — weight stats (all pixels):', wStats);

  // 3) Helper distributions (optional; turn on only if needed)
  if (PARAMS4.showDistStats) {
    var distSamp = ee.Image.cat([
      NDVI_drop.rename('ndvi_drop'),
      NDWI_drop.rename('ndwi_drop'),
      AUC_z.rename('auc_z')
    ]).sample({
      region: aoi, scale: sampleScale, numPixels: 20000, seed: 11, tileScale: 12
    });
    
    var perc = ee.Reducer.percentile([5,25,50,75,95]);
    
    var ndviP = distSamp.reduceColumns({ reducer: perc, selectors: ['ndvi_drop'] });
    var ndwiP = distSamp.reduceColumns({ reducer: perc, selectors: ['ndwi_drop'] });
    var zP    = distSamp.reduceColumns({ reducer: perc, selectors: ['auc_z'] });
    
    print('PH4 — percentiles NDVI_drop:', ndviP);
    print('PH4 — percentiles NDWI_drop:', ndwiP);
    print('PH4 — percentiles AUC_z:', zP);
  }
  
  // Add this to Phase 4 showPreview section:
  var dropStats = ee.Image.cat([NDVI_drop, NDWI_drop, AUC_z])
    .reduceRegion({
      reducer: ee.Reducer.percentile([50, 75, 90, 95, 99]),
      geometry: aoi,
      scale: 100,
      maxPixels: 1e8,
      bestEffort: true
  });
  print('Stress indicator percentiles:', dropStats);
  
  
}

/* Exports */
exports.PARAMS4     = PARAMS4;
exports.label01     = label01;
exports.labelMask   = labelMask;
exports.weight      = weight;
exports.weight01    = weight01;
exports.labelStack  = labelStack;
exports.weightStack = weightStack;
exports.labelWeight = labelWeight;