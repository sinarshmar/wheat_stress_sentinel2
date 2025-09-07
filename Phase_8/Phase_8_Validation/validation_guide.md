# Phase 8: Interactive Validation Guide

## Key Concept: Visual Consistency
We're checking if areas the model classified as "stressed" LOOK stressed in imagery.
This is NOT about absolute truth - it's about visual consistency.

## Assessment Area: 100-Meter Radius
- **Focus zone**: Evaluate crop conditions within ~100m radius of center point
- **Zoom level**: Links open at zoom 15 for optimal 100m assessment  
- **Scale reference**: At zoom 15, 100m radius appears as ~21 pixels on screen
- **Flexibility**: You can zoom in/out as needed for better assessment

## Interactive Assessment Process

### Step 1: Open the HTML Interface
- Load the validation_interface.html file in your browser
- All 150 points are displayed in a table with interactive dropdowns

### Step 2: For Each Point
1. **Click "Open in EO Browser"** - Opens to Feb 2024 at zoom 15
2. **Assess the 100m radius area** around the center crosshair
3. **Check other dates** using EO Browser's calendar (Dec 2023, Mar 2024)
4. **Fill out the three dropdown columns:**
   - **Your Assessment**: Agree/Disagree/Uncertain with model
   - **Your Confidence**: High/Medium/Low in your assessment
   - **Visual Stress Level**: None/Low/Moderate/High stress you observe
5. **Add notes** if needed in the text area

### Step 3: Save Your Results
- Click "Download as CSV" button to save your assessment
- This creates a validation_results.csv file with all your entries

## Assessment Criteria

### When Model Says STRESSED, Look For:
- **Color**: Yellow-green instead of dark green (February)
- **Coverage**: Sparse canopy, soil visible between plants
- **Pattern**: Patchy, irregular growth within field
- **Timing**: Delayed emergence (December) or early senescence (March)

### When Model Says HEALTHY, Look For:
- **Color**: Dark green (February peak)
- **Coverage**: Dense canopy, no soil visible
- **Pattern**: Uniform growth across field
- **Timing**: Good emergence (December), normal maturation (March)

## Dropdown Options

### Your Assessment:
- **Agree** - Visual evidence supports model prediction
- **Disagree** - Visual evidence contradicts model prediction
- **Uncertain** - Mixed signals or poor imagery quality

### Your Confidence:
- **High** - Very clear visual evidence for your assessment
- **Medium** - Moderate visual evidence
- **Low** - Difficult to assess due to clouds, mixed signals, etc.

### Visual Stress Level (what YOU see):
- **None** - Area looks completely healthy
- **Low** - Minor stress indicators visible
- **Moderate** - Clear stress indicators present
- **High** - Severe stress clearly visible

## Expected Results
- 70-85% agreement is EXCELLENT
- Higher agreement expected for high-confidence predictions
- Some disagreement is normal and informative

## Time Estimate
- Quick scan: 30-60 seconds per point
- Detailed check: 1-2 minutes per point
- Total: 2-3 hours for 150 points

## Technical Notes
- All assessment data is saved in the browser until you download the CSV
- Use "Clear All" button to reset all entries if needed
- The CSV file includes all point metadata plus your assessments

Remember: You're the human expert providing visual validation of the model's patterns within the 100m assessment area!
