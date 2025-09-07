"""
Analyze Binary Classification Export
WITH COMPREHENSIVE LOGGING TO FILE
"""

import rasterio
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import json

class Logger:
    """Custom logger to write to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.log.write(f"=" * 80 + "\n")
        self.log.write(f"BINARY CLASSIFICATION ANALYSIS LOG\n")
        self.log.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"=" * 80 + "\n\n")
        self.log.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.write(f"\n" + "=" * 80 + "\n")
        self.log.write(f"LOG END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"=" * 80 + "\n")
        self.log.close()

def analyze_binary_classification(filepath='ludhiana_stress_map_rf_2023_24_binary.tif'):
    """
    Comprehensive analysis of the binary classification map
    """
    print("="*60)
    print("BINARY CLASSIFICATION ANALYSIS")
    print("="*60)
    print(f"\nAnalyzing: {filepath}")
    print("-"*60)
    
    # Check if file exists
    if not Path(filepath).exists():
        print(f"ERROR: File not found at {filepath}")
        return None
    
    results = {
        'filename': filepath,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'file_exists': True
    }
    
    try:
        with rasterio.open(filepath) as src:
            # 1. FILE METADATA
            print("\n1. FILE METADATA:")
            print(f"   - Driver: {src.driver}")
            print(f"   - Dimensions: {src.width} x {src.height} pixels")
            print(f"   - Bands: {src.count}")
            print(f"   - Data type: {src.dtypes[0]}")
            print(f"   - CRS: {src.crs}")
            print(f"   - Pixel size: {src.res[0]}m x {src.res[1]}m")
            print(f"   - NoData value: {src.nodata}")
            
            # Calculate area
            pixel_area_ha = (src.res[0] * src.res[1]) / 10000
            total_area_km2 = (src.width * src.height * pixel_area_ha) / 100
            print(f"   - Total area: {total_area_km2:.1f} km²")
            
            results['metadata'] = {
                'width': src.width,
                'height': src.height,
                'total_pixels': src.width * src.height,
                'pixel_size_m': src.res[0],
                'total_area_km2': round(total_area_km2, 1),
                'crs': str(src.crs)
            }
            
            # 2. READ AND ANALYZE DATA
            print("\n2. DATA ANALYSIS:")
            binary_data = src.read(1)
            
            # Find valid pixels (0 or 1)
            valid_mask = (binary_data == 0) | (binary_data == 1)
            valid_binary = binary_data[valid_mask]
            
            # Count stressed and healthy pixels
            stressed_pixels = np.sum(valid_binary == 1)
            healthy_pixels = np.sum(valid_binary == 0)
            total_valid = len(valid_binary)
            nodata_pixels = binary_data.size - total_valid
            
            print(f"\n   Classification Results:")
            print(f"   - Total pixels: {binary_data.size:,}")
            print(f"   - Valid pixels: {total_valid:,} ({total_valid/binary_data.size*100:.1f}%)")
            print(f"   - NoData pixels: {nodata_pixels:,} ({nodata_pixels/binary_data.size*100:.1f}%)")
            
            if total_valid > 0:
                print(f"\n   Binary Classification:")
                print(f"   - Stressed (1): {stressed_pixels:,} pixels ({stressed_pixels/total_valid*100:.1f}%)")
                print(f"   - Healthy (0): {healthy_pixels:,} pixels ({healthy_pixels/total_valid*100:.1f}%)")
                
                # Calculate areas
                stressed_area_ha = stressed_pixels * pixel_area_ha
                healthy_area_ha = healthy_pixels * pixel_area_ha
                total_crop_area_ha = total_valid * pixel_area_ha
                
                print(f"\n   Area Calculations:")
                print(f"   - Total cropland: {total_crop_area_ha:.1f} ha ({total_crop_area_ha/100:.1f} km²)")
                print(f"   - Stressed area: {stressed_area_ha:.1f} ha ({stressed_area_ha/100:.1f} km²)")
                print(f"   - Healthy area: {healthy_area_ha:.1f} ha ({healthy_area_ha/100:.1f} km²)")
                
                results['classification'] = {
                    'stressed_pixels': int(stressed_pixels),
                    'healthy_pixels': int(healthy_pixels),
                    'total_valid_pixels': int(total_valid),
                    'nodata_pixels': int(nodata_pixels),
                    'stressed_percentage': round(stressed_pixels/total_valid*100, 2),
                    'healthy_percentage': round(healthy_pixels/total_valid*100, 2),
                    'stressed_area_ha': round(stressed_area_ha, 1),
                    'healthy_area_ha': round(healthy_area_ha, 1),
                    'total_crop_area_ha': round(total_crop_area_ha, 1)
                }
                
                # 3. SPATIAL DISTRIBUTION CHECK
                print("\n3. SPATIAL DISTRIBUTION CHECK:")
                
                # Check edges vs center
                edge_width = min(100, src.width // 10, src.height // 10)
                edge_mask = np.zeros_like(binary_data, dtype=bool)
                edge_mask[:edge_width, :] = True  # Top
                edge_mask[-edge_width:, :] = True  # Bottom
                edge_mask[:, :edge_width] = True  # Left
                edge_mask[:, -edge_width:] = True  # Right
                
                edge_data = binary_data[edge_mask & valid_mask]
                center_data = binary_data[~edge_mask & valid_mask]
                
                if len(edge_data) > 0 and len(center_data) > 0:
                    edge_stress_pct = np.sum(edge_data == 1) / len(edge_data) * 100
                    center_stress_pct = np.sum(center_data == 1) / len(center_data) * 100
                    
                    print(f"   - Edge pixels stress: {edge_stress_pct:.1f}%")
                    print(f"   - Center pixels stress: {center_stress_pct:.1f}%")
                    
                    if abs(edge_stress_pct - center_stress_pct) > 10:
                        print("   ⚠️ WARNING: Significant difference between edge and center stress")
                        print("      This might indicate edge effects or processing artifacts")
                    else:
                        print("   ✅ Stress distribution appears spatially consistent")
                    
                    results['spatial_distribution'] = {
                        'edge_stress_percentage': round(edge_stress_pct, 2),
                        'center_stress_percentage': round(center_stress_pct, 2),
                        'spatial_consistency': bool(abs(edge_stress_pct - center_stress_pct) <= 10)
                    }
                
                # 4. COMPARISON WITH EXPECTED VALUES
                print("\n4. VALIDATION AGAINST EXPECTATIONS:")
                
                # From Phase 7 console output, we expect ~10.3% stress at 10m
                expected_stress_pct = 10.3
                actual_stress_pct = stressed_pixels/total_valid*100
                difference = abs(actual_stress_pct - expected_stress_pct)
                
                print(f"   - Expected stress: {expected_stress_pct:.1f}% (from Phase 7)")
                print(f"   - Actual stress: {actual_stress_pct:.1f}%")
                print(f"   - Difference: {difference:.1f}%")
                
                if difference < 2:
                    print("   ✅ EXCELLENT: Binary classification matches Phase 7 output")
                elif difference < 5:
                    print("   ✅ GOOD: Binary classification close to Phase 7 output")
                else:
                    print("   ⚠️ WARNING: Significant difference from Phase 7 output")
                    print("      Check if correct file or processing issues")
                
                results['validation'] = {
                    'expected_stress_pct': expected_stress_pct,
                    'actual_stress_pct': round(actual_stress_pct, 2),
                    'difference_pct': round(difference, 2),
                    'match_quality': 'EXCELLENT' if difference < 2 else 'GOOD' if difference < 5 else 'POOR'
                }
                
            else:
                print("   ❌ No valid classification data found!")
                results['classification'] = None
            
            # 5. FINAL ASSESSMENT
            print("\n" + "="*60)
            print("FINAL ASSESSMENT:")
            print("-"*60)
            
            if total_valid > 0 and results['validation']['match_quality'] in ['EXCELLENT', 'GOOD']:
                print("✅ BINARY CLASSIFICATION EXPORT SUCCESSFUL!")
                print(f"   - {stressed_pixels:,} stressed pixels ({stressed_pixels/total_valid*100:.1f}%)")
                print(f"   - {healthy_pixels:,} healthy pixels ({healthy_pixels/total_valid*100:.1f}%)")
                print(f"   - Total crop area: {total_crop_area_ha:.1f} ha")
                print(f"   - Matches Phase 7 output: {results['validation']['match_quality']}")
                results['export_status'] = 'SUCCESS'
            else:
                print("❌ ISSUES DETECTED WITH BINARY CLASSIFICATION")
                if total_valid == 0:
                    print("   - No valid classification data")
                else:
                    print(f"   - Large discrepancy with expected values")
                results['export_status'] = 'FAILED'
                
    except Exception as e:
        print(f"\n❌ ERROR reading file: {e}")
        results['export_status'] = f'ERROR - {str(e)}'
        import traceback
        traceback.print_exc()
    
    # Save results to JSON
    output_json = Path(filepath).parent / 'binary_classification_analysis.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to: {output_json}")
    
    return results

def compare_with_probability_map():
    """
    Compare binary classification with probability map if both exist
    """
    print("\n" + "="*60)
    print("COMPARISON WITH PROBABILITY MAP")
    print("="*60)
    
    prob_json = Path('probability_map_analysis.json')
    binary_json = Path('binary_classification_analysis.json')
    
    if not prob_json.exists() or not binary_json.exists():
        print("Cannot compare - run both probability and binary analyses first")
        return
    
    with open(prob_json) as f:
        prob_results = json.load(f)
    with open(binary_json) as f:
        binary_results = json.load(f)
    
    if prob_results.get('stress_categories') and binary_results.get('classification'):
        # Probability map stress (>80% probability)
        prob_high_stress_pct = prob_results['stress_categories']['high_stress_pct']
        
        # Binary classification stress
        binary_stress_pct = binary_results['classification']['stressed_percentage']
        
        print(f"\nStress Detection Comparison:")
        print(f"  - Probability map (>80% stress): {prob_high_stress_pct:.1f}%")
        print(f"  - Binary classification: {binary_stress_pct:.1f}%")
        print(f"  - Difference: {abs(prob_high_stress_pct - binary_stress_pct):.1f}%")
        
        if abs(prob_high_stress_pct - binary_stress_pct) < 2:
            print("\n  ✅ EXCELLENT agreement between probability and binary outputs")
        else:
            print("\n  ⚠️ Some discrepancy between probability and binary outputs")
            print("     This is expected if different thresholds were used")

if __name__ == "__main__":
    # Setup log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging to file - FIXED FILENAME (overwrites previous)
    log_filename = log_dir / "diagnose_binary_log.txt"
    sys.stdout = Logger(log_filename)
    
    try:
        # Analyze binary classification
        results = analyze_binary_classification()
        
        # Compare with probability map if available
        if results and results.get('export_status') == 'SUCCESS':
            compare_with_probability_map()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n📝 Full log saved to: {log_filename}")
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal  # Restore original stdout