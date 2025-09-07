"""
Phase 8: Visual Validation Sampling (CLEANED VERSION)
Generates 150 validation points for GUIDED validation only
Model predictions are visible to help calibrate assessment
WITH COMPREHENSIVE LOGGING TO FILE
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from pathlib import Path
import json
from datetime import datetime
import random
import sys
import traceback

# ============================================================================
# LOGGING SETUP
# ============================================================================

class Logger:
    """Custom logger to write to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.log.write("=" * 80 + "\n")
        self.log.write("PHASE 8 LOG FILE - GUIDED VALIDATION\n")
        self.log.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write("=" * 80 + "\n\n")
        self.log.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.write("\n" + "=" * 80 + "\n")
        self.log.write(f"LOG END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write("=" * 80 + "\n")
        self.log.close()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Input files
    PROBABILITY_MAP = "ludhiana_stress_map_rf_2023_24_probability_pct.tif"
    
    # Output directory
    OUTPUT_DIR = Path("Phase_8_Validation")
    LOG_DIR = Path("logs")
    
    # Sampling parameters
    N_STRESSED = 75  # Points from stressed class
    N_HEALTHY = 75   # Points from healthy class
    TOTAL_POINTS = N_STRESSED + N_HEALTHY
    
    # Probability thresholds (INVERTED - high values = healthy)
    HEALTHY_THRESHOLD_MIN = 90  # Very confident healthy
    HEALTHY_THRESHOLD_MAX = 100
    STRESSED_THRESHOLD_MIN = 0   # Very confident stressed  
    STRESSED_THRESHOLD_MAX = 10
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Sentinel Hub EO Browser parameters
    SENTINEL_HUB_BASE = "https://apps.sentinel-hub.com/eo-browser/"
    ZOOM_LEVEL = 15  # Set to 15 for 100m radius assessment
    # At zoom 15: ~4.77m per pixel, 100m radius = ~21 pixels
    # At zoom 16: ~2.39m per pixel, 100m radius = ~42 pixels
    # At zoom 17: ~1.19m per pixel, 100m radius = ~84 pixels
    
    # Key dates for validation
    VALIDATION_DATES = [
        ("2023-12-15", "Early season"),
        ("2024-02-16", "Peak growth"),
        ("2024-03-15", "Late season")
    ]

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def setup_directories(config):
    """Create output directories"""
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    print(f"✓ Output directory ready: {config.OUTPUT_DIR}")
    print(f"✓ Log directory ready: {config.LOG_DIR}")

def load_probability_map(filepath):
    """Load and analyze probability map"""
    print("\n" + "="*60)
    print("LOADING PROBABILITY MAP")
    print("="*60)
    
    with rasterio.open(filepath) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        
        # Get valid pixels (non-zero)
        valid_mask = data > 0
        
        print(f"Map dimensions: {data.shape}")
        print(f"CRS: {crs}")
        print(f"Valid pixels: {valid_mask.sum():,}")
        
        return data, valid_mask, transform, crs

def sample_stressed_points(data, valid_mask, transform, n_samples, config):
    """
    Sample stressed points (LOW probability values = stressed)
    Remember: values are INVERTED (low = stressed, high = healthy)
    """
    print("\n" + "-"*40)
    print("SAMPLING STRESSED POINTS")
    print("-"*40)
    
    # Find stressed pixels (LOW healthy probability)
    stressed_mask = valid_mask & (data >= config.STRESSED_THRESHOLD_MIN) & (data <= config.STRESSED_THRESHOLD_MAX)
    stressed_pixels = np.where(stressed_mask)
    
    n_available = len(stressed_pixels[0])
    print(f"Available stressed pixels (0-10% healthy prob): {n_available:,}")
    
    if n_available < n_samples:
        print(f"⚠️ Warning: Only {n_available} stressed pixels available, requested {n_samples}")
        n_samples = n_available
    
    # Random sample
    np.random.seed(config.RANDOM_SEED)
    indices = np.random.choice(n_available, n_samples, replace=False)
    
    # Get selected pixels
    sample_rows = stressed_pixels[0][indices]
    sample_cols = stressed_pixels[1][indices]
    
    # Import pyproj for coordinate transformation
    from pyproj import Transformer
    # Create transformer from UTM to WGS84
    transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    
    # Convert to coordinates
    points = []
    for row, col in zip(sample_rows, sample_cols):
        # Get UTM coordinates
        utm_x, utm_y = xy(transform, row, col)
        # Convert to lat/lon
        lon, lat = transformer.transform(utm_x, utm_y)
        prob_value = data[row, col]
        
        points.append({
            'point_id': f'S_{len(points)+1:03d}',
            'class': 'stressed',
            'pixel_row': int(row),
            'pixel_col': int(col),
            'longitude': lon,
            'latitude': lat,
            'utm_x': utm_x,
            'utm_y': utm_y,
            'healthy_probability': float(prob_value),
            'stress_probability': float(100 - prob_value),  # Invert for clarity
            'confidence': 'high' if prob_value <= 5 else 'moderate'
        })
    
    print(f"✓ Sampled {len(points)} stressed points")
    return points

def sample_healthy_points(data, valid_mask, transform, n_samples, config):
    """
    Sample healthy points (HIGH probability values = healthy)
    """
    print("\n" + "-"*40)
    print("SAMPLING HEALTHY POINTS")
    print("-"*40)
    
    # Find healthy pixels (HIGH healthy probability)
    healthy_mask = valid_mask & (data >= config.HEALTHY_THRESHOLD_MIN) & (data <= config.HEALTHY_THRESHOLD_MAX)
    healthy_pixels = np.where(healthy_mask)
    
    n_available = len(healthy_pixels[0])
    print(f"Available healthy pixels (90-100% healthy prob): {n_available:,}")
    
    if n_available < n_samples:
        print(f"⚠️ Warning: Only {n_available} healthy pixels available, requested {n_samples}")
        n_samples = n_available
    
    # Random sample
    np.random.seed(config.RANDOM_SEED + 1)  # Different seed for healthy
    indices = np.random.choice(n_available, n_samples, replace=False)
    
    # Get selected pixels
    sample_rows = healthy_pixels[0][indices]
    sample_cols = healthy_pixels[1][indices]
    
    # Import pyproj for coordinate transformation
    from pyproj import Transformer
    # Create transformer from UTM to WGS84
    transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    
    # Convert to coordinates
    points = []
    for row, col in zip(sample_rows, sample_cols):
        # Get UTM coordinates
        utm_x, utm_y = xy(transform, row, col)
        # Convert to lat/lon
        lon, lat = transformer.transform(utm_x, utm_y)
        prob_value = data[row, col]
        
        points.append({
            'point_id': f'H_{len(points)+1:03d}',
            'class': 'healthy',
            'pixel_row': int(row),
            'pixel_col': int(col),
            'longitude': lon,
            'latitude': lat,
            'utm_x': utm_x,
            'utm_y': utm_y,
            'healthy_probability': float(prob_value),
            'stress_probability': float(100 - prob_value),
            'confidence': 'high' if prob_value >= 95 else 'moderate'
        })
    
    print(f"✓ Sampled {len(points)} healthy points")
    return points

def randomize_points(points):
    """Randomize point order for unbiased presentation"""
    random.seed(42)  # Fixed seed for reproducibility
    randomized = points.copy()
    random.shuffle(randomized)
    
    # Add presentation order
    for i, point in enumerate(randomized):
        point['presentation_order'] = i + 1
    
    return randomized

def create_sentinel_hub_links(points, config):
    """Create Sentinel Hub EO Browser links for each point with February 2024 as default"""
    print("\n" + "-"*40)
    print("CREATING BROWSER LINKS")
    print("-"*40)
    
    for point in points:
        lat = point['latitude']
        lon = point['longitude']
        
        # Updated Sentinel Hub EO Browser URL format with February 2024 as default
        base_params = (
            f"zoom={config.ZOOM_LEVEL}"
            f"&lat={lat:.6f}"
            f"&lng={lon:.6f}"
            f"&themeId=DEFAULT-THEME"
            f"&datasetId=S2L2A"
            f"&layerId=1_TRUE_COLOR"
            f"&fromTime=2024-02-16T00%3A00%3A00.000Z"
            f"&toTime=2024-02-16T23%3A59%3A59.999Z"
            f"&demSource3D=%22MAPZEN%22"
        )
        
        # Base link defaults to February 2024
        point['browser_link'] = f"{config.SENTINEL_HUB_BASE}?{base_params}"
        
        # Add date-specific links for all three validation dates
        point['date_links'] = {}
        for date, description in config.VALIDATION_DATES:
            # Date format for Sentinel Hub
            date_params = (
                f"zoom={config.ZOOM_LEVEL}"
                f"&lat={lat:.6f}"
                f"&lng={lon:.6f}"
                f"&themeId=DEFAULT-THEME"
                f"&datasetId=S2L2A"
                f"&layerId=1_TRUE_COLOR"
                f"&fromTime={date}T00%3A00%3A00.000Z"
                f"&toTime={date}T23%3A59%3A59.999Z"
                f"&demSource3D=%22MAPZEN%22"
            )
            point['date_links'][description] = f"{config.SENTINEL_HUB_BASE}?{date_params}"
    
    print(f"✓ Created browser links for {len(points)} points (default: Feb 2024, zoom {config.ZOOM_LEVEL})")
    return points

def create_html_interface(points, config):
    """Create interactive HTML interface with validation links, dropdowns, and save functionality"""
    print("\n" + "-"*40)
    print("CREATING INTERACTIVE HTML INTERFACE")
    print("-"*40)
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Phase 8: Interactive Validation Interface</title>
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }
        h1 { 
            color: #2E7D32; 
            border-bottom: 3px solid #2E7D32;
            padding-bottom: 10px;
        }
        .instructions { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .key-point {
            background: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }
        .sentinel-instructions {
            background: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin: 15px 0;
        }
        .zoom-guide {
            background: #f3e5f5;
            padding: 15px;
            border-left: 4px solid #9c27b0;
            margin: 15px 0;
        }
        .visual-circle {
            width: 42px;
            height: 42px;
            border: 3px solid #9c27b0;
            border-radius: 50%;
            margin: 10px auto;
            position: relative;
            background: rgba(156, 39, 176, 0.1);
        }
        .circle-label {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 9px;
            font-weight: bold;
            color: #9c27b0;
        }
        .save-section {
            background: #e8f5e9;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #4caf50;
            text-align: center;
        }
        .save-button {
            background: #4caf50;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin: 10px;
        }
        .save-button:hover { background: #45a049; }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin-top: 20px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
            font-size: 12px;
        }
        th { 
            background-color: #2E7D32; 
            color: white; 
            position: sticky;
            top: 0;
            z-index: 10;
            font-size: 13px;
        }
        tr:hover { background-color: #f5f5f5; }
        .link-button { 
            background: #2196F3; 
            color: white; 
            padding: 4px 8px; 
            text-decoration: none; 
            border-radius: 3px; 
            margin: 2px;
            display: inline-block;
            font-size: 11px;
        }
        .link-button:hover { background: #0b7dda; }
        .copy-button {
            background: #ff9800;
            color: white;
            padding: 2px 6px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
        }
        .copy-button:hover { background: #f57c00; }
        .dropdown {
            width: 100%;
            padding: 4px;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 11px;
        }
        .text-input {
            width: 100%;
            padding: 4px;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 11px;
            min-height: 40px;
            resize: vertical;
        }
        .stressed-row { background-color: #ffebee; }
        .healthy-row { background-color: #e8f5e9; }
        .stress-label { 
            background: #f44336; 
            color: white; 
            padding: 2px 6px; 
            border-radius: 3px;
            font-weight: bold;
            font-size: 11px;
        }
        .healthy-label { 
            background: #4caf50; 
            color: white; 
            padding: 2px 6px; 
            border-radius: 3px;
            font-weight: bold;
            font-size: 11px;
        }
        .confidence-high { color: #2E7D32; font-weight: bold; }
        .confidence-moderate { color: #ff9800; font-weight: bold; }
        .coords-cell {
            font-family: monospace;
            font-size: 11px;
        }
        .date-list {
            font-size: 10px;
            line-height: 1.3;
            color: #666;
        }
    </style>
    <script>
        function copyCoords(lat, lon) {
            const coords = lat + ', ' + lon;
            navigator.clipboard.writeText(coords).then(function() {
                alert('Coordinates copied: ' + coords);
            }, function(err) {
                alert('Could not copy coordinates. Please copy manually: ' + coords);
            });
        }
        
        function saveToCSV() {
            const table = document.getElementById('validationTable');
            const rows = table.querySelectorAll('tbody tr');
            
            // CSV headers
            let csvContent = 'point_id,order,longitude,latitude,browser_link,model_prediction,model_stress_prob,model_confidence,your_assessment,your_confidence,visual_stress_level,notes\\n';
            
            rows.forEach(function(row) {
                const cells = row.querySelectorAll('td');
                const pointId = cells[1].textContent.trim();
                const order = cells[0].textContent.trim();
                const coords = cells[2].textContent.trim().replace('Copy', '').trim();
                const [lat, lon] = coords.split(', ');
                const browserLink = cells[6].querySelector('.link-button').href;
                const modelPrediction = cells[3].textContent.trim();
                const stressProb = cells[4].textContent.trim();
                const modelConfidence = cells[5].textContent.trim();
                
                // Get dropdown and textarea values
                const assessment = cells[7].querySelector('.dropdown').value;
                const confidence = cells[8].querySelector('.dropdown').value;
                const stressLevel = cells[9].querySelector('.dropdown').value;
                const notes = cells[10].querySelector('.text-input').value.replace(/[\\n\\r]/g, ' ').replace(/"/g, '""');
                
                // Create CSV row
                csvContent += `"${pointId}","${order}","${lon}","${lat}","${browserLink}","${modelPrediction}","${stressProb}","${modelConfidence}","${assessment}","${confidence}","${stressLevel}","${notes}"\\n`;
            });
            
            // Create and download CSV file
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', 'validation_results.csv');
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            alert('Validation results saved to validation_results.csv');
        }
        
        function clearAll() {
            const dropdowns = document.querySelectorAll('.dropdown');
            const textInputs = document.querySelectorAll('.text-input');
            
            dropdowns.forEach(function(dropdown) {
                dropdown.selectedIndex = 0;
            });
            
            textInputs.forEach(function(input) {
                input.value = '';
            });
            
            alert('All entries cleared');
        }
    </script>
</head>
<body>
    <h1>Phase 8: Interactive Validation Interface</h1>
    
    <div class="instructions">
        <h2>Purpose: Visual Consistency Check</h2>
        <div class="key-point">
            <strong>Important:</strong> This is GUIDED validation. Model predictions are shown to help calibrate your assessment. 
            We're checking if predictions are <em>visually consistent</em> with imagery.
        </div>
        
        <div class="zoom-guide">
            <h3>🔍 100-Meter Assessment Area:</h3>
            <div class="visual-circle">
                <div class="circle-label">100m<br>radius</div>
            </div>
            <ul>
                <li><strong>Links open at zoom level 15</strong> - optimized for 100m radius assessment</li>
                <li><strong>Focus area:</strong> Assess crop conditions in roughly a <strong>100-meter radius</strong> around the center point</li>
                <li><strong>Visual reference:</strong> At this zoom, 100m radius appears as shown in the circle above</li>
                <li>You can zoom in/out as needed, but start assessment at the default zoom</li>
            </ul>
        </div>
        
        <div class="sentinel-instructions">
            <h3>🛰️ How to Use Sentinel Hub EO Browser:</h3>
            <ol>
                <li><strong>Click "Open in EO Browser"</strong> to open the location (defaults to Feb 2024 - peak growth, zoom level 15)</li>
                <li><strong>Once in EO Browser:</strong>
                    <ul>
                        <li>The link opens at <strong>February 16, 2024</strong> (peak growth period)</li>
                        <li><strong>Assessment area:</strong> Focus on the ~100m radius around the center crosshair</li>
                        <li>You can navigate to other dates using the calendar:
                            <ul>
                                <li><strong>December 15, 2023</strong> - Early season</li>
                                <li><strong>March 15, 2024</strong> - Late season</li>
                            </ul>
                        </li>
                        <li>Try both <strong>True Color</strong> and <strong>NDVI</strong> visualizations</li>
                        <li>If cloudy, check ±5 days from target date</li>
                    </ul>
                </li>
                <li><strong>Alternative:</strong> Copy coordinates and paste into EO Browser search box</li>
            </ol>
        </div>
    </div>
    
    <div class="save-section">
        <h3>💾 Save Your Assessment Results:</h3>
        <button class="save-button" onclick="saveToCSV()">📥 Download as CSV</button>
        <button class="save-button" onclick="clearAll()" style="background: #f44336;">🗑️ Clear All</button>
        <p><em>Fill out the assessment columns below, then click "Download as CSV" to save your results.</em></p>
    </div>
    
    <table id="validationTable">
        <thead>
            <tr>
                <th width="4%">#</th>
                <th width="7%">Point ID</th>
                <th width="12%">Coordinates</th>
                <th width="8%">Model Says</th>
                <th width="6%">Stress %</th>
                <th width="7%">Confidence</th>
                <th width="15%">EO Browser Link</th>
                <th width="12%">Your Assessment</th>
                <th width="10%">Your Confidence</th>
                <th width="11%">Visual Stress Level</th>
                <th width="8%">Notes</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Add each point as a table row with interactive dropdowns
    for point in sorted(points, key=lambda x: x['presentation_order']):
        row_class = 'stressed-row' if point['class'] == 'stressed' else 'healthy-row'
        label_class = 'stress-label' if point['class'] == 'stressed' else 'healthy-label'
        conf_class = 'confidence-high' if point['confidence'] == 'high' else 'confidence-moderate'
        
        # Get coordinates
        lat = point['latitude']
        lon = point['longitude']
        
        # Use the browser link that was already created (defaults to Feb 2024, zoom 15)
        browser_link = point.get('browser_link', f"https://apps.sentinel-hub.com/eo-browser/?zoom=15&lat={lat:.6f}&lng={lon:.6f}")
        
        html_content += f"""
            <tr class="{row_class}">
                <td>{point['presentation_order']}</td>
                <td><strong>{point['point_id']}</strong></td>
                <td class="coords-cell">
                    {lat:.6f}, {lon:.6f}
                    <button class="copy-button" onclick="copyCoords({lat:.6f}, {lon:.6f})">Copy</button>
                </td>
                <td><span class="{label_class}">{point['class'].upper()}</span></td>
                <td>{point['stress_probability']:.1f}%</td>
                <td><span class="{conf_class}">{point['confidence']}</span></td>
                <td>
                    <a href="{browser_link}" target="_blank" class="link-button">🛰️ Open in EO Browser (Zoom 15)</a>
                    <div class="date-list">Opens Feb 2024 (peak) at zoom 15. Assess 100m radius area.</div>
                </td>
                <td>
                    <select class="dropdown">
                        <option value="">-- Select --</option>
                        <option value="Agree">Agree</option>
                        <option value="Disagree">Disagree</option>
                        <option value="Uncertain">Uncertain</option>
                    </select>
                </td>
                <td>
                    <select class="dropdown">
                        <option value="">-- Select --</option>
                        <option value="High">High</option>
                        <option value="Medium">Medium</option>
                        <option value="Low">Low</option>
                    </select>
                </td>
                <td>
                    <select class="dropdown">
                        <option value="">-- Select --</option>
                        <option value="None">None</option>
                        <option value="Low">Low</option>
                        <option value="Moderate">Moderate</option>
                        <option value="High">High</option>
                    </select>
                </td>
                <td>
                    <textarea class="text-input" placeholder="Add notes here..."></textarea>
                </td>
            </tr>
"""
    
    html_content += """
        </tbody>
    </table>
    
    <div style="margin-top: 30px; padding: 20px; background: white; border-radius: 8px;">
        <h3>Visual Indicators Quick Reference:</h3>
        <table style="width: 100%; margin-top: 10px;">
            <tr>
                <th style="background: #4caf50; color: white;">🟢 HEALTHY (within 100m radius)</th>
                <th style="background: #f44336; color: white;">🔴 STRESSED (within 100m radius)</th>
            </tr>
            <tr>
                <td>
                    <ul>
                        <li><strong>Dec:</strong> Good emergence, uniform green</li>
                        <li><strong>Feb:</strong> Dark green, dense canopy</li>
                        <li><strong>Mar:</strong> Gradual yellowing (normal)</li>
                    </ul>
                </td>
                <td>
                    <ul>
                        <li><strong>Dec:</strong> Poor/patchy emergence</li>
                        <li><strong>Feb:</strong> Yellow-green, sparse</li>
                        <li><strong>Mar:</strong> Premature browning</li>
                    </ul>
                </td>
            </tr>
        </table>
        
        <h3 style="margin-top: 20px;">Assessment Guide:</h3>
        <div style="display: flex; gap: 20px; margin-top: 15px;">
            <div style="flex: 1;">
                <h4>Your Assessment:</h4>
                <ul>
                    <li><strong>Agree:</strong> Visual evidence supports model</li>
                    <li><strong>Disagree:</strong> Visual evidence contradicts model</li>
                    <li><strong>Uncertain:</strong> Mixed signals or poor imagery</li>
                </ul>
            </div>
            <div style="flex: 1;">
                <h4>Your Confidence:</h4>
                <ul>
                    <li><strong>High:</strong> Very clear visual evidence</li>
                    <li><strong>Medium:</strong> Moderate visual evidence</li>
                    <li><strong>Low:</strong> Difficult to assess</li>
                </ul>
            </div>
            <div style="flex: 1;">
                <h4>Visual Stress Level:</h4>
                <ul>
                    <li><strong>None:</strong> Looks completely healthy</li>
                    <li><strong>Low:</strong> Minor stress indicators</li>
                    <li><strong>Moderate:</strong> Clear stress indicators</li>
                    <li><strong>High:</strong> Severe stress visible</li>
                </ul>
            </div>
        </div>
        
        <h3 style="margin-top: 20px;">Assessment Tips:</h3>
        <ul>
            <li><strong>Focus area:</strong> Evaluate crop conditions within ~100m radius of the center point</li>
            <li><strong>Default zoom:</strong> Links open at zoom level 15 for optimal 100m assessment</li>
            <li><strong>Zoom flexibility:</strong> You can zoom in/out as needed for better assessment</li>
            <li>Use the calendar in EO Browser to check December 2023 and March 2024</li>
            <li>Compare with neighboring fields outside the 100m area for context</li>
            <li>Focus on overall pattern within the assessment area, not individual pixels</li>
            <li>70-85% agreement with model = excellent validation</li>
            <li><strong>Remember to save your work:</strong> Click "Download as CSV" when finished</li>
        </ul>
    </div>
    
    <div class="save-section">
        <h3>🎯 Finish Your Assessment</h3>
        <button class="save-button" onclick="saveToCSV()">📥 Download Results as CSV</button>
        <p><em>Don't forget to save your assessment results when you're done!</em></p>
    </div>
</body>
</html>
"""
    
    html_path = config.OUTPUT_DIR / 'validation_interface.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Created interactive HTML interface: {html_path}")
    return html_path

def create_validation_guide(config):
    """Create a concise validation guide"""
    guide = """# Phase 8: Interactive Validation Guide

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
"""
    
    guide_path = config.OUTPUT_DIR / 'validation_guide.md'
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"✓ Created validation guide: {guide_path}")

def create_summary_json(points, config):
    """Create a JSON summary of the validation setup"""
    summary = {
        'metadata': {
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'purpose': 'Interactive guided visual validation of model predictions',
            'total_points': len(points),
            'stressed_points': sum(1 for p in points if p['class'] == 'stressed'),
            'healthy_points': sum(1 for p in points if p['class'] == 'healthy'),
            'random_seed': config.RANDOM_SEED
        },
        'sampling': {
            'stressed_threshold': f"{config.STRESSED_THRESHOLD_MIN}-{config.STRESSED_THRESHOLD_MAX}% healthy prob",
            'healthy_threshold': f"{config.HEALTHY_THRESHOLD_MIN}-{config.HEALTHY_THRESHOLD_MAX}% healthy prob",
            'note': 'Probability values are INVERTED (high = healthy, low = stressed)'
        },
        'validation_approach': {
            'type': 'Interactive guided validation',
            'description': 'Model predictions visible with interactive HTML interface',
            'goal': 'Check visual consistency, not absolute accuracy',
            'assessment_area': '100-meter radius around each point',
            'zoom_level': config.ZOOM_LEVEL,
            'interface': 'HTML with dropdown menus and CSV export'
        },
        'assessment_options': {
            'your_assessment': ['Agree', 'Disagree', 'Uncertain'],
            'your_confidence': ['High', 'Medium', 'Low'],
            'visual_stress_level': ['None', 'Low', 'Moderate', 'High']
        },
        'expected_outcomes': {
            'good_agreement': '70-85%',
            'interpretation': 'Agreement shows visual consistency, not ground truth accuracy'
        },
        'files_generated': [
            'validation_interface.html (interactive assessment tool)',
            'validation_guide.md',
            'validation_summary.json'
        ],
        'output_file': 'validation_results.csv (generated by user via HTML interface)'
    }
    
    summary_path = config.OUTPUT_DIR / 'validation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Created summary JSON: {summary_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Setup configuration
    config = Config()
    
    # Setup directories
    setup_directories(config)
    
    # Setup logging to file
    log_filename = config.LOG_DIR / "phase_8_log.txt"
    sys.stdout = Logger(log_filename)
    
    try:
        print("="*60)
        print("PHASE 8: INTERACTIVE VALIDATION SETUP")
        print("="*60)
        print("\nPurpose: Generate validation points for visual consistency check")
        print("Approach: INTERACTIVE guided validation with HTML interface")
        print("Goal: Assess if model predictions are visually consistent")
        print(f"Assessment area: 100m radius at zoom level {config.ZOOM_LEVEL}\n")
        
        # Load probability map
        prob_path = Path(config.PROBABILITY_MAP)
        if not prob_path.exists():
            print(f"❌ Error: Probability map not found at {prob_path}")
            print("Please ensure Phase 7 exports are in the current directory")
            return
        
        data, valid_mask, transform, crs = load_probability_map(prob_path)
        
        # Sample points
        stressed_points = sample_stressed_points(data, valid_mask, transform, 
                                                config.N_STRESSED, config)
        healthy_points = sample_healthy_points(data, valid_mask, transform, 
                                              config.N_HEALTHY, config)
        
        # Combine and randomize for unbiased presentation
        all_points = stressed_points + healthy_points
        randomized_points = randomize_points(all_points)
        
        # Add browser links
        points_with_links = create_sentinel_hub_links(randomized_points, config)
        
        # Create validation materials (no CSV spreadsheet, only interactive HTML)
        html_path = create_html_interface(points_with_links, config)
        create_validation_guide(config)
        create_summary_json(points_with_links, config)
        
        # Save complete point data for analysis
        full_data_path = config.OUTPUT_DIR / 'validation_points_complete.json'
        with open(full_data_path, 'w') as f:
            json.dump(points_with_links, f, indent=2)
        print(f"\n✓ Saved complete point data: {full_data_path}")
        
        # Final summary
        print("\n" + "="*60)
        print("INTERACTIVE VALIDATION SETUP COMPLETE")
        print("="*60)
        
        print(f"\n✅ Generated {len(all_points)} validation points:")
        print(f"   - {len(stressed_points)} stressed (high confidence)")
        print(f"   - {len(healthy_points)} healthy (high confidence)")
        
        print(f"\n📁 Files created in {config.OUTPUT_DIR}/:")
        print(f"   1. validation_interface.html - Interactive assessment tool")
        print(f"   2. validation_guide.md - Quick reference guide")
        print(f"   3. validation_summary.json - Metadata summary")
        print(f"   4. validation_points_complete.json - Complete point data")
        
        print("\n📋 NEXT STEPS:")
        print("1. Open 'validation_interface.html' in your browser")
        print("2. For each point:")
        print("   - Click 'Open in EO Browser' to view imagery")
        print("   - Focus on 100m radius around center point")
        print("   - Check imagery for all three dates (Dec, Feb, Mar)")
        print("   - Use dropdown menus to record your assessment")
        print("   - Add notes in the text area if needed")
        print("3. Click 'Download as CSV' to save your results")
        print("4. Run Phase_8_Analysis.py on the downloaded CSV file")
        
        print(f"\n🔍 Assessment Details:")
        print(f"   - Zoom level: {config.ZOOM_LEVEL} (optimized for 100m radius)")
        print(f"   - Default date: February 16, 2024 (peak growth)")
        print(f"   - Assessment area: ~100m radius around each point")
        print(f"   - Interface: Interactive HTML with dropdown menus")
        
        print("\n⏱️  Time Estimate: 2-3 hours (1-2 minutes per point)")
        print("\n🎯 Remember: 70-85% agreement = EXCELLENT validation!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        traceback.print_exc()
    
    finally:
        print("\n" + "="*60)
        print("PHASE 8 COMPLETE")
        print("="*60)
        print(f"\n📝 Full log saved to: {log_filename}")
        
        # Close logger and restore stdout
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()