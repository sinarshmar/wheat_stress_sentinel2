"""
Phase 5.5: Merge multiple GeoTIFF tiles and perform spatial join
Updated version for TILED exports from Phase 5
WITH COMPREHENSIVE LOGGING TO FILE
"""

import pandas as pd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform
import glob
import os
from pathlib import Path
import sys
from datetime import datetime

# Configuration - UPDATED FOR TILED FILENAMES
INPUT_DIR = Path("GEE_exports")
OUTPUT_DIR = Path("Phase_5.5_results")
LOG_DIR = Path("logs")

# Create a custom logging class to capture all output
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.log.write(f"=" * 80 + "\n")
        self.log.write(f"PHASE 5.5 LOG FILE\n")
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

def setup_directories():
    """Create output and log directories if they don't exist"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}")
    print(f"✓ Log directory ready: {LOG_DIR}")

def validate_inputs():
    """Check that required files exist with new naming"""
    print("\nValidating input files...")
    
    # Updated filenames for tiled exports
    coords_file = INPUT_DIR / 'coordinates_labels_tiled_rabi_2023_24.csv'
    tif_pattern = str(INPUT_DIR / 'predictors_image_tiled_rabi_2023_24*.tif')
    summary_file = INPUT_DIR / 'sampling_summary_rabi_2023_24.csv'
    
    # Check coordinates file
    if not coords_file.exists():
        raise FileNotFoundError(
            f"Coordinates file not found: {coords_file}\n"
            f"Please ensure Phase 5 tiled export completed successfully."
        )
    
    # Check TIF files
    tif_files = glob.glob(tif_pattern)
    if len(tif_files) == 0:
        raise FileNotFoundError(
            f"No predictor TIF files found matching pattern: {tif_pattern}\n"
            f"Please ensure Phase 5 exports completed."
        )
    
    print(f"✓ Found coordinates file: {coords_file.name}")
    print(f"✓ Found {len(tif_files)} TIF file(s)")
    
    # Optional: Check summary file
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        print(f"✓ Found summary file with sampling statistics")
        print(f"  - Sampling method: {summary_df['sampling_method'].iloc[0]}")
        print(f"  - Actual total samples: {summary_df['actual_total'].iloc[0]}")
    
    return coords_file, tif_files

def merge_tiles_and_diagnose(tif_files):
    """
    Merge multiple GeoTIFF tiles and check data quality
    """
    print("\n" + "="*60)
    print("MERGING AND DIAGNOSING RASTER DATA")
    print("="*60)
    
    print(f"\n1. Found {len(tif_files)} raster file(s):")
    for f in tif_files:
        print(f"   - {os.path.basename(f)}")
    
    if len(tif_files) == 1:
        # Single file, just use it
        print("\n2. Single file - no merging needed")
        merged_file = OUTPUT_DIR / "predictors_image_MERGED.tif"
        
        # Copy to output directory
        import shutil
        shutil.copy2(tif_files[0], merged_file)
        print(f"   ✓ Copied to: {merged_file}")
        
    else:
        # Multiple tiles - need to merge
        print("\n2. Merging tiles...")
        
        # Open all tiles
        src_files = []
        for fp in tif_files:
            src = rasterio.open(fp)
            src_files.append(src)
        
        # Merge tiles
        mosaic, out_trans = merge(src_files)
        
        # Get metadata from first file
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        
        # Save merged file
        merged_file = OUTPUT_DIR / "predictors_image_MERGED.tif"
        with rasterio.open(merged_file, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Close source files
        for src in src_files:
            src.close()
        
        print(f"   ✓ Merged to: {merged_file}")
    
    # Now diagnose the merged/single file
    print("\n3. Analyzing raster data...")
    
    with rasterio.open(merged_file) as src:
        print(f"   - Shape: {src.height} x {src.width}")
        print(f"   - Bands: {src.count}")
        print(f"   - CRS: {src.crs}")
        print(f"   - NoData value: {src.nodata}")
        
        # Band names from Phase 5
        band_names = [
            'NDVI_AUC', 'NDVI_drop', 'NDVI_peak', 'NDVI_slopeEarly',
            'NDWI_AUC', 'NDWI_drop', 'NDWI_peak',
            'GNDVI_AUC', 'GNDVI_drop',
            'SAVI_AUC', 'MSAVI2_AUC',
            'h_NDVI_drop', 'h_NDWI_drop', 'h_AUC_z', 'h_NDVI_AUC_cur'
        ]
        
        print("\n4. Band Statistics (with unmask(-9999)):")
        print("-" * 80)
        
        band_stats = []
        has_minus9999 = False
        
        for i in range(1, min(src.count + 1, 16)):
            band = src.read(i)
            band_name = band_names[i-1] if i <= len(band_names) else f'Band_{i}'
            
            # Check for -9999 values (unmasked areas)
            minus9999_count = (band == -9999).sum()
            valid_mask = (band != -9999) & ~np.isnan(band)
            valid_pixels = valid_mask.sum()
            
            if minus9999_count > 0:
                has_minus9999 = True
            
            print(f"\nBand {i} ({band_name}):")
            print(f"  - Total pixels: {band.size:,}")
            print(f"  - Valid data pixels: {valid_pixels:,} ({100*valid_pixels/band.size:.1f}%)")
            print(f"  - NoData (-9999) pixels: {minus9999_count:,} ({100*minus9999_count/band.size:.1f}%)")
            
            if valid_pixels > 0:
                valid_data = band[valid_mask]
                
                stats = {
                    'band': i,
                    'name': band_name,
                    'valid_pixels': valid_pixels,
                    'nodata_pixels': minus9999_count,
                    'min': valid_data.min(),
                    'max': valid_data.max(),
                    'mean': valid_data.mean(),
                    'std': valid_data.std(),
                    'median': np.median(valid_data)
                }
                band_stats.append(stats)
                
                print(f"  - Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"  - Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        
        print("\n" + "="*80)
        print("5. SUMMARY:")
        print("-" * 80)
        
        if has_minus9999:
            print("✓ unmask(-9999) worked! NoData areas are now -9999")
            print("  This means sampling will work at ALL coordinates!")
        else:
            print("⚠️  No -9999 values found. The unmask might not have worked.")
        
        # Save stats
        stats_df = pd.DataFrame(band_stats)
        stats_file = OUTPUT_DIR / 'new_raster_statistics.csv'
        stats_df.to_csv(stats_file, index=False)
        print(f"\n📊 Statistics saved to: {stats_file}")
    
    return merged_file

def extract_all_points(raster_file, coords_file):
    """
    Extract values for all sample points from the raster
    """
    print("\n" + "="*60)
    print("EXTRACTING ALL SAMPLE POINTS")
    print("="*60)
    
    # Load coordinates - UPDATED FOR TILED FORMAT
    df = pd.read_csv(coords_file)
    print(f"\n1. Loaded {len(df)} sample points from {coords_file.name}")
    
    # Check what columns we have
    print(f"   Columns found: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    # Check class distribution if available
    if 'class_name' in df.columns:
        class_counts = df['class_name'].value_counts()
        print(f"   Class distribution:")
        for class_name, count in class_counts.items():
            print(f"     - {class_name}: {count} ({100*count/len(df):.1f}%)")
    
    # Check if tile_id exists (from tiled sampling)
    if 'tile_id' in df.columns:
        tile_counts = df['tile_id'].value_counts().sort_index()
        print(f"   Samples per tile: {dict(tile_counts)}")
    
    # Open raster and extract
    with rasterio.open(raster_file) as src:
        # Transform coordinates
        xs, ys = transform('EPSG:4326', src.crs, 
                          df['lon'].values, df['lat'].values)
        
        # Sample at all points
        print("\n2. Extracting predictor values...")
        coords = [(x, y) for x, y in zip(xs, ys)]
        values = list(src.sample(coords))
        values_array = np.array(values)
        
        print(f"   - Extracted shape: {values_array.shape}")
        
        # Count valid vs nodata
        nodata_mask = (values_array == -9999).any(axis=1)
        valid_mask = ~nodata_mask
        
        print(f"   - Points with valid data: {valid_mask.sum()} ({100*valid_mask.sum()/len(df):.1f}%)")
        print(f"   - Points in NoData areas: {nodata_mask.sum()} ({100*nodata_mask.sum()/len(df):.1f}%)")
        
        # Create final dataset
        band_names = [
            'NDVI_AUC', 'NDVI_drop', 'NDVI_peak', 'NDVI_slopeEarly',
            'NDWI_AUC', 'NDWI_drop', 'NDWI_peak',
            'GNDVI_AUC', 'GNDVI_drop',
            'SAVI_AUC', 'MSAVI2_AUC',
            'h_NDVI_drop', 'h_NDWI_drop', 'h_AUC_z', 'h_NDVI_AUC_cur'
        ][:src.count]
        
        predictor_df = pd.DataFrame(values_array, columns=band_names)
        final_df = pd.concat([df, predictor_df], axis=1)
        
        # Save all data (including -9999)
        output_file = OUTPUT_DIR / 'training_data_with_unmask.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\n3. Saved full dataset to: {output_file}")
        
        # Save filtered data (excluding -9999)
        final_df_clean = final_df[valid_mask]
        clean_file = OUTPUT_DIR / 'training_data_clean.csv'
        final_df_clean.to_csv(clean_file, index=False)
        print(f"4. Saved clean dataset to: {clean_file}")
        
        # Show summary
        print("\n5. Final Dataset Summary:")
        print(f"   - Total samples: {len(final_df)}")
        print(f"   - Clean samples: {len(final_df_clean)}")
        
        # Use 'label' column for class counts
        if 'label' in final_df_clean.columns:
            print(f"   - Healthy (class 0): {(final_df_clean['label'] == 0).sum()}")
            print(f"   - Stressed (class 1): {(final_df_clean['label'] == 1).sum()}")
            stress_pct = 100 * (final_df_clean['label'] == 1).sum() / len(final_df_clean)
            print(f"   - Stress percentage: {stress_pct:.1f}%")
        
        # Sample of clean data
        if len(final_df_clean) > 0:
            print("\n6. Sample of clean data (first 3 rows):")
            cols_to_show = ['label', 'class_name', 'NDVI_AUC', 'NDVI_drop', 'NDVI_peak', 'h_AUC_z']
            cols_to_show = [col for col in cols_to_show if col in final_df_clean.columns]
            print(final_df_clean[cols_to_show].head(3))
            
            # Key predictor statistics
            print("\n7. Key Predictor Statistics (clean data):")
            key_predictors = ['NDVI_AUC', 'NDVI_drop', 'NDWI_drop', 'h_AUC_z']
            for pred in key_predictors:
                if pred in final_df_clean.columns:
                    print(f"   {pred}:")
                    print(f"     - Mean: {final_df_clean[pred].mean():.3f}")
                    print(f"     - Std:  {final_df_clean[pred].std():.3f}")
                    print(f"     - Range: [{final_df_clean[pred].min():.3f}, {final_df_clean[pred].max():.3f}]")
        
        return final_df_clean

def validate_data_quality(df):
    """
    Additional data quality checks
    """
    print("\n" + "="*60)
    print("DATA QUALITY VALIDATION")
    print("="*60)
    
    if df is None or len(df) == 0:
        print("❌ No data to validate")
        return
    
    print("\n1. Missing Values Check:")
    missing = df.isnull().sum()
    if missing.any():
        print("   ⚠️  Found missing values:")
        print(missing[missing > 0])
    else:
        print("   ✓ No missing values")
    
    print("\n2. Value Range Checks:")
    
    # Check NDVI values (should be between -1 and 1)
    ndvi_cols = [col for col in df.columns if 'NDVI' in col and 'AUC' not in col and 'drop' not in col]
    for col in ndvi_cols:
        if col in df.columns:
            out_of_range = ((df[col] < -1) | (df[col] > 1)).sum()
            if out_of_range > 0:
                print(f"   ⚠️  {col}: {out_of_range} values outside [-1, 1]")
            else:
                print(f"   ✓ {col}: All values in valid range")
    
    print("\n3. Class Balance:")
    if 'label' in df.columns:
        class_counts = df['label'].value_counts()
        print(f"   - Class 0 (healthy): {class_counts.get(0, 0)}")
        print(f"   - Class 1 (stressed): {class_counts.get(1, 0)}")
        
        if len(class_counts) == 2:
            imbalance_ratio = class_counts.max() / class_counts.min()
            print(f"   - Imbalance ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 5:
                print("   ⚠️  High class imbalance detected")
    
    print("\n4. Coordinate Validation:")
    # Check if coordinates are within reasonable bounds for Punjab, India
    lon_range = (73, 77)  # Approximate longitude range for Punjab
    lat_range = (29, 33)  # Approximate latitude range for Punjab
    
    lon_valid = df['lon'].between(*lon_range).all()
    lat_valid = df['lat'].between(*lat_range).all()
    
    if lon_valid and lat_valid:
        print(f"   ✓ All coordinates within Punjab bounds")
    else:
        print(f"   ⚠️  Some coordinates outside expected range")
        print(f"      Lon range: [{df['lon'].min():.2f}, {df['lon'].max():.2f}]")
        print(f"      Lat range: [{df['lat'].min():.2f}, {df['lat'].max():.2f}]")
    
    # Check for tile distribution if present
    if 'tile_id' in df.columns:
        print("\n5. Tile Distribution:")
        tile_dist = df['tile_id'].value_counts().sort_index()
        print(f"   Tiles represented: {len(tile_dist)}")
        print(f"   Samples per tile range: [{tile_dist.min()}, {tile_dist.max()}]")
        if tile_dist.std() / tile_dist.mean() > 0.5:
            print("   ⚠️  High variation in samples per tile")

def save_summary_report(clean_df, log_file):
    """Save a summary report to file"""
    summary_file = OUTPUT_DIR / 'phase_5_5_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PHASE 5.5 SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        if clean_df is not None and len(clean_df) > 0:
            f.write("FINAL RESULTS:\n")
            f.write(f"- Clean samples obtained: {len(clean_df)}\n")
            f.write(f"- Healthy samples: {(clean_df['label'] == 0).sum()}\n")
            f.write(f"- Stressed samples: {(clean_df['label'] == 1).sum()}\n")
            f.write(f"- Stress percentage: {100*(clean_df['label'] == 1).mean():.1f}%\n")
            f.write(f"- Success rate: {100*len(clean_df)/3821:.1f}%\n")
        else:
            f.write("ERROR: No clean samples obtained\n")
        
        f.write(f"\nLog file saved to: {log_file}\n")
        f.write("="*60 + "\n")
    
    print(f"\n📄 Summary report saved to: {summary_file}")

if __name__ == "__main__":
    # Setup directories first
    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    # Setup logging to file - FIXED FILENAME (overwrites previous)
    log_filename = LOG_DIR / "phase_5_5_log.txt"
    sys.stdout = Logger(log_filename)
    
    try:
        print("PHASE 5.5: SPATIAL JOIN AND VALIDATION")
        print("Updated for TILED sampling outputs")
        print("=" * 60)
        
        # Validate inputs with new filenames
        coords_file, tif_files = validate_inputs()
        
        # Merge tiles and diagnose
        merged_file = merge_tiles_and_diagnose(tif_files)
        
        clean_df = None
        if merged_file and merged_file.exists():
            # Extract all points
            clean_df = extract_all_points(merged_file, coords_file)
            
            # Validate data quality
            if clean_df is not None:
                validate_data_quality(clean_df)
            
            if clean_df is not None and len(clean_df) > 100:
                print("\n" + "=" * 60)
                print("✅ SUCCESS! Training data ready!")
                print(f"   - Clean dataset: {OUTPUT_DIR / 'training_data_clean.csv'}")
                print(f"   - {len(clean_df)} samples ready for Phase 6")
                print("   Next step: Run Phase_6.py for model training")
            else:
                print("\n⚠️  Warning: Few clean samples. Check unmask values.")
        else:
            print("\n❌ Error: Merged file not created properly")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure Phase 5 exports are in GEE_exports/ folder:")
        print("  - coordinates_labels_tiled_rabi_2023_24.csv")
        print("  - predictors_image_tiled_rabi_2023_24*.tif")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("PHASE 5.5 COMPLETE")
    print("=" * 60)
    
    # Save summary report
    if 'clean_df' in locals():
        save_summary_report(clean_df, log_filename)
    
    # Close logger
    print(f"\n📝 Full log saved to: {log_filename}")
    sys.stdout.close()
    sys.stdout = sys.stdout.terminal  # Restore original stdout