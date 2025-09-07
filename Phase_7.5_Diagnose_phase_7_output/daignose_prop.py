"""
Phase 8: Analyze Probability Map Export
Check if the GEE export was successful or limited
WITH COMPREHENSIVE LOGGING TO FILE

"""

import numpy as np
import rasterio
from rasterio.plot import show_hist
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import sys

class Logger:
    """Custom logger to write to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.log.write(f"=" * 80 + "\n")
        self.log.write(f"PROBABILITY MAP ANALYSIS LOG\n")
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

def analyze_probability_map(filepath):
    """
    Comprehensive analysis of the probability map to check export quality
    """
    print("="*60)
    print("PHASE 8: PROBABILITY MAP ANALYSIS")
    print("="*60)
    print(f"\nAnalyzing: {filepath}")
    print("-"*60)
    
    # Check if file exists
    if not Path(filepath).exists():
        print(f"ERROR: File not found at {filepath}")
        return None
    
    # Dictionary to store all results
    results = {
        'filename': filepath,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'file_exists': True
    }
    
    try:
        with rasterio.open(filepath) as src:
            # 1. BASIC FILE INFORMATION
            print("\n1. FILE METADATA:")
            print(f"   - Driver: {src.driver}")
            print(f"   - Dimensions: {src.width} x {src.height} pixels")
            print(f"   - Bands: {src.count}")
            print(f"   - Data type: {src.dtypes[0]}")
            print(f"   - CRS: {src.crs}")
            print(f"   - Pixel size: {src.res}")
            print(f"   - NoData value: {src.nodata}")
            
            results['metadata'] = {
                'width': src.width,
                'height': src.height,
                'total_pixels': src.width * src.height,
                'bands': src.count,
                'dtype': str(src.dtypes[0]),
                'crs': str(src.crs),
                'pixel_size_m': src.res[0],
                'nodata_value': src.nodata
            }
            
            # Calculate approximate area
            pixel_area_ha = (src.res[0] * src.res[1]) / 10000
            total_area_km2 = (src.width * src.height * pixel_area_ha) / 100
            print(f"   - Approximate area: {total_area_km2:.1f} km²")
            results['metadata']['area_km2'] = total_area_km2
            
            # 2. READ AND ANALYZE DATA
            print("\n2. DATA ANALYSIS:")
            data = src.read(1)  # Read first (only) band
            
            # Check for data presence
            print(f"   - Shape: {data.shape}")
            print(f"   - Memory size: {data.nbytes / 1024 / 1024:.1f} MB")
            
            # Identify different data categories
            if src.nodata is not None:
                nodata_mask = (data == src.nodata)
                nodata_count = np.sum(nodata_mask)
                valid_mask = ~nodata_mask
            else:
                # Check for common nodata values
                nodata_values = [255, 0, -9999, -999, -1]
                nodata_mask = np.isin(data, nodata_values)
                nodata_count = np.sum(nodata_mask)
                valid_mask = ~nodata_mask
            
            valid_data = data[valid_mask]
            valid_count = len(valid_data)
            
            print(f"\n   Data Categories:")
            print(f"   - Total pixels: {data.size:,}")
            print(f"   - Valid pixels: {valid_count:,} ({100*valid_count/data.size:.1f}%)")
            print(f"   - NoData pixels: {nodata_count:,} ({100*nodata_count/data.size:.1f}%)")
            
            results['data_coverage'] = {
                'total_pixels': int(data.size),
                'valid_pixels': int(valid_count),
                'nodata_pixels': int(nodata_count),
                'valid_percentage': round(100*valid_count/data.size, 2),
                'nodata_percentage': round(100*nodata_count/data.size, 2)
            }
            
            # 3. CHECK IF EXPORT WAS SUCCESSFUL
            print("\n3. EXPORT QUALITY CHECK:")
            
            # Check 1: Is there any valid data?
            if valid_count == 0:
                print("   ❌ EXPORT FAILED: No valid data found!")
                results['export_status'] = 'FAILED - No valid data'
                return results
            
            # Check 2: Is coverage reasonable? (expecting ~30-40% of area to be cropland)
            coverage_pct = (valid_count / data.size) * 100
            if coverage_pct < 10:
                print(f"   ⚠️ WARNING: Very low coverage ({coverage_pct:.1f}%)")
                print("   Export may be incomplete or highly masked")
                results['export_status'] = 'PARTIAL - Low coverage'
            elif coverage_pct > 80:
                print(f"   ⚠️ WARNING: Very high coverage ({coverage_pct:.1f}%)")
                print("   Check if crop mask was applied correctly")
                results['export_status'] = 'CHECK - High coverage'
            else:
                print(f"   ✅ Coverage looks reasonable ({coverage_pct:.1f}%)")
                results['export_status'] = 'SUCCESS'
            
            # 4. ANALYZE PROBABILITY DISTRIBUTION
            print("\n4. PROBABILITY DISTRIBUTION:")
            
            if valid_count > 0:
                # Basic statistics
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                mean_val = np.mean(valid_data)
                median_val = np.median(valid_data)
                std_val = np.std(valid_data)
                
                print(f"   - Range: [{min_val:.1f}, {max_val:.1f}]")
                print(f"   - Mean: {mean_val:.1f}")
                print(f"   - Median: {median_val:.1f}")
                print(f"   - Std Dev: {std_val:.1f}")
                
                # CRITICAL: The file contains HEALTHY probability (0-100)
                # So we need to INVERT for stress probability
                print("\n   NOTE: File contains HEALTHY probability (GEE default)")
                print("   Converting to STRESS probability for analysis...")
                
                # Invert: stress_prob = 100 - healthy_prob
                stress_data = 100 - valid_data
                
                # Check if values are in expected range (0-100 for percentage)
                if min_val < 0 or max_val > 100:
                    print(f"   ⚠️ WARNING: Values outside 0-100 range!")
                    print(f"   This might indicate an export/scaling issue")
                
                # Percentiles (using stress probability)
                percentiles = [5, 25, 50, 75, 95]
                perc_values = np.percentile(stress_data, percentiles)
                print(f"\n   Stress Probability Percentiles:")
                for p, v in zip(percentiles, perc_values):
                    print(f"   - P{p:2d}: {v:.1f}%")
                
                # Stress categories (using stress probability)
                high_stress = np.sum(stress_data > 80)
                moderate_stress = np.sum((stress_data >= 50) & (stress_data <= 80))
                low_stress = np.sum((stress_data >= 20) & (stress_data < 50))
                healthy = np.sum(stress_data < 20)
                
                print(f"\n   Stress Categories (% of valid pixels):")
                print(f"   - High stress (>80%): {100*high_stress/valid_count:.1f}%")
                print(f"   - Moderate stress (50-80%): {100*moderate_stress/valid_count:.1f}%")
                print(f"   - Low stress (20-50%): {100*low_stress/valid_count:.1f}%")
                print(f"   - Healthy (<20%): {100*healthy/valid_count:.1f}%")
                
                results['statistics'] = {
                    'min': float(min_val),
                    'max': float(max_val),
                    'mean': float(mean_val),
                    'median': float(median_val),
                    'std': float(std_val),
                    'percentiles': {f'p{p}': float(v) for p, v in zip(percentiles, perc_values)},
                    'note': 'Original file contains HEALTHY probability, converted to STRESS for analysis'
                }
                
                results['stress_categories'] = {
                    'high_stress_pixels': int(high_stress),
                    'moderate_stress_pixels': int(moderate_stress),
                    'low_stress_pixels': int(low_stress),
                    'healthy_pixels': int(healthy),
                    'high_stress_pct': round(100*high_stress/valid_count, 2),
                    'moderate_stress_pct': round(100*moderate_stress/valid_count, 2),
                    'low_stress_pct': round(100*low_stress/valid_count, 2),
                    'healthy_pct': round(100*healthy/valid_count, 2)
                }
                
                # 5. CHECK FOR SAMPLING VIABILITY
                print("\n5. VALIDATION SAMPLING POTENTIAL:")
                
                # Check if we have enough pixels for each category
                min_pixels_needed = 50  # For validation
                
                print(f"   Can we sample 50 points from each category?")
                print(f"   - High stress (>80%): {'YES ✅' if high_stress >= min_pixels_needed else 'NO ❌'} ({high_stress:,} pixels available)")
                print(f"   - Healthy (<20%): {'YES ✅' if healthy >= min_pixels_needed else 'NO ❌'} ({healthy:,} pixels available)")
                print(f"   - Uncertain (40-60%): ", end="")
                uncertain = np.sum((stress_data >= 40) & (stress_data <= 60))
                print(f"{'YES ✅' if uncertain >= min_pixels_needed else 'NO ❌'} ({uncertain:,} pixels available)")
                
                results['sampling_viability'] = {
                    'high_stress_available': int(high_stress),
                    'healthy_available': int(healthy),
                    'uncertain_available': int(uncertain),
                    'can_sample_high_stress': bool(high_stress >= min_pixels_needed),
                    'can_sample_healthy': bool(healthy >= min_pixels_needed),
                    'can_sample_uncertain': bool(uncertain >= min_pixels_needed)
                }
                
                # 6. VISUALIZATIONS
                print("\n6. GENERATING VISUALIZATIONS...")
                
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Histogram (using stress probability)
                ax = axes[0, 0]
                hist, bins = np.histogram(stress_data, bins=50, range=(0, 100))
                ax.bar(bins[:-1], hist, width=2, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Stress Probability (%)')
                ax.set_ylabel('Pixel Count')
                ax.set_title('Distribution of Stress Probabilities')
                ax.axvline(x=50, color='red', linestyle='--', label='50% threshold')
                ax.legend()
                
                # Cumulative distribution
                ax = axes[0, 1]
                sorted_data = np.sort(stress_data)
                cumsum = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax.plot(sorted_data, cumsum * 100)
                ax.set_xlabel('Stress Probability (%)')
                ax.set_ylabel('Cumulative % of Pixels')
                ax.set_title('Cumulative Distribution')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
                ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
                
                # Box plot
                ax = axes[1, 0]
                bp = ax.boxplot(stress_data, vert=False, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                ax.set_xlabel('Stress Probability (%)')
                ax.set_title('Box Plot of Stress Probabilities')
                ax.set_xlim(0, 100)
                
                # Summary text
                ax = axes[1, 1]
                ax.axis('off')
                summary_text = f"""
                EXPORT SUMMARY
                
                Status: {results['export_status']}
                Valid pixels: {valid_count:,} ({coverage_pct:.1f}%)
                
                Stress Distribution:
                • High (>80%): {100*high_stress/valid_count:.1f}%
                • Moderate (50-80%): {100*moderate_stress/valid_count:.1f}%
                • Low (20-50%): {100*low_stress/valid_count:.1f}%
                • Healthy (<20%): {100*healthy/valid_count:.1f}%
                
                Validation Sampling:
                • High stress pixels: {high_stress:,}
                • Healthy pixels: {healthy:,}
                • Uncertain pixels: {uncertain:,}
                
                NOTE: Original file contained
                HEALTHY probability (inverted
                for this analysis)
                """
                ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
                
                plt.suptitle(f'Probability Map Analysis: {Path(filepath).name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Save figure
                output_fig = Path(filepath).parent / 'probability_map_analysis.png'
                plt.savefig(output_fig, dpi=150, bbox_inches='tight')
                print(f"   ✅ Saved visualization to: {output_fig}")
                plt.show()
                
            # 7. FINAL VERDICT
            print("\n" + "="*60)
            print("FINAL ASSESSMENT:")
            print("-"*60)
            
            if results['export_status'] == 'SUCCESS':
                print("✅ EXPORT SUCCESSFUL!")
                print(f"   - File contains valid data for {valid_count:,} pixels")
                print(f"   - Coverage is {coverage_pct:.1f}% of total area")
                print(f"   - Data range is appropriate (0-100)")
                print(f"   - Sufficient pixels for validation sampling")
                print("\n   RECOMMENDATION: Proceed with validation sampling from this file")
            elif results['export_status'] == 'PARTIAL - Low coverage':
                print("⚠️ PARTIAL EXPORT")
                print(f"   - Limited coverage ({coverage_pct:.1f}%)")
                print("   - May be due to strict crop masking or export issues")
                print("\n   RECOMMENDATION: Check if this matches expected cropland area")
            else:
                print("❌ EXPORT ISSUES DETECTED")
                print("   - File may be corrupted or incomplete")
                print("\n   RECOMMENDATION: Re-export from GEE or use alternative file")
            
    except Exception as e:
        print(f"\n❌ ERROR reading file: {e}")
        results['export_status'] = f'ERROR - {str(e)}'
        return results
    
    # Save results to JSON
    output_json = Path(filepath).parent / 'probability_map_analysis.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Full results saved to: {output_json}")
    
    return results

def generate_sampling_report(results):
    """
    Generate a report on sampling feasibility
    """
    print("\n" + "="*60)
    print("VALIDATION SAMPLING FEASIBILITY REPORT")
    print("="*60)
    
    if not results or 'sampling_viability' not in results:
        print("Cannot generate report - no valid data")
        return
    
    sv = results['sampling_viability']
    
    print("\nProposed 150-point validation sample:")
    print("-"*40)
    
    # Option 1: Ideal stratification
    print("\nOption 1: IDEAL STRATIFICATION")
    ideal = [
        ("High stress (>80%)", 50, sv['high_stress_available']),
        ("Healthy (<20%)", 50, sv['healthy_available']),
        ("Uncertain (40-60%)", 50, sv['uncertain_available'])
    ]
    
    total_available = sum([x[2] for x in ideal])
    can_do_ideal = all([x[2] >= x[1] for x in ideal])
    
    for category, needed, available in ideal:
        status = "✅" if available >= needed else "❌"
        print(f"   {category:20s}: {needed:3d} needed, {available:6,} available {status}")
    
    if can_do_ideal:
        print("   ✅ IDEAL STRATIFICATION IS POSSIBLE!")
    else:
        print("   ❌ Cannot achieve ideal stratification")
        
        # Option 2: Adjusted stratification
        print("\nOption 2: ADJUSTED STRATIFICATION")
        print("   Proportional to available pixels:")
        
        for category, _, available in ideal:
            if total_available > 0:
                proportion = available / total_available
                adjusted = int(150 * proportion)
                print(f"   {category:20s}: {adjusted:3d} points ({proportion*100:.1f}% of sample)")
    
    # Option 3: Simplified approach
    print("\nOption 3: SIMPLIFIED APPROACH")
    print("   Just high vs low probability:")
    high = sv['high_stress_available']
    low = sv['healthy_available']
    if high >= 75 and low >= 75:
        print(f"   High (>80%): 75 points ✅")
        print(f"   Low (<20%): 75 points ✅")
        print("   ✅ SIMPLIFIED APPROACH IS POSSIBLE!")
    else:
        print("   ❌ Insufficient pixels for even simplified approach")
        print(f"   Alternative: Use the 22 GEE-exported points for validation")

if __name__ == "__main__":
    # Setup log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging to file - FIXED FILENAME (overwrites previous)
    log_filename = log_dir / "diagnose_probability_log.txt"
    sys.stdout = Logger(log_filename)
    
    try:
        # File path - update this to your actual file location
        filepath = "ludhiana_stress_map_rf_2023_24_probability_pct.tif"
        
        # Run analysis
        results = analyze_probability_map(filepath)
        
        # Generate sampling report
        if results and results.get('export_status') != 'FAILED - No valid data':
            generate_sampling_report(results)
        
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