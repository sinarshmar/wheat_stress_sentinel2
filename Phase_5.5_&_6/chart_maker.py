"""
Phase 5 Charts for Dissertation
Sample Distribution Analysis from Tiled Sampling
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set up paths
GEE_EXPORTS_DIR = Path("GEE_exports")

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Read the data from GEE_exports folder
coords_df = pd.read_csv(GEE_EXPORTS_DIR / 'coordinates_labels_tiled_rabi_2023_24.csv')
summary_df = pd.read_csv(GEE_EXPORTS_DIR / 'sampling_summary_rabi_2023_24.csv')

# Print summary statistics
print("="*50)
print("SAMPLING SUMMARY STATISTICS")
print("="*50)
print(f"Total samples collected: {len(coords_df)}")
print(f"Target samples per class: {summary_df['target_per_class'].iloc[0]}")
print(f"Actual healthy samples: {summary_df['actual_healthy'].iloc[0]}")
print(f"Actual stressed samples: {summary_df['actual_stressed'].iloc[0]}")
print(f"Number of tiles: {summary_df['tiles_count'].iloc[0]}")

# ============================================
# CHART 1: Overall Class Distribution
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
class_counts = coords_df['class_name'].value_counts()
colors = ['#2E7D32', '#D32F2F']  # Green for healthy, Red for stressed
axes[0].pie(class_counts.values, 
            labels=class_counts.index, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 11})
axes[0].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')

# Bar chart with counts
ax1 = axes[1]
bars = ax1.bar(class_counts.index, class_counts.values, color=colors, alpha=0.8)
ax1.set_ylabel('Number of Samples', fontsize=12)
ax1.set_xlabel('Class', fontsize=12)
ax1.set_title('Sample Counts by Class', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(class_counts.values) * 1.1)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('figure_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# CHART 2: Distribution by Tile
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Calculate samples per tile and class
tile_distribution = coords_df.groupby(['tile_id', 'class_name']).size().unstack(fill_value=0)

# Stacked bar chart
ax2 = axes[0]
tile_distribution.plot(kind='bar', 
                       stacked=True, 
                       color=['#2E7D32', '#D32F2F'],
                       ax=ax2,
                       width=0.8)
ax2.set_xlabel('Tile ID', fontsize=12)
ax2.set_ylabel('Number of Samples', fontsize=12)
ax2.set_title('Sample Distribution Across Tiles (Stacked)', fontsize=14, fontweight='bold')
ax2.legend(title='Class', loc='upper right', frameon=True)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

# Add horizontal line for expected samples per tile
expected_per_tile = summary_df['target_per_class'].iloc[0] * 2 / summary_df['tiles_count'].iloc[0]
ax2.axhline(y=expected_per_tile, color='black', linestyle='--', alpha=0.5, 
            label=f'Expected per tile: {int(expected_per_tile)}')

# Grouped bar chart
ax3 = axes[1]
tile_distribution.plot(kind='bar', 
                       color=['#2E7D32', '#D32F2F'],
                       ax=ax3,
                       width=0.8)
ax3.set_xlabel('Tile ID', fontsize=12)
ax3.set_ylabel('Number of Samples', fontsize=12)
ax3.set_title('Sample Distribution Across Tiles (Grouped)', fontsize=14, fontweight='bold')
ax3.legend(title='Class', loc='upper right', frameon=True)
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('figure_tile_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# CHART 3: Spatial Distribution Heatmap
# ============================================
# Create a 3x3 grid representation
tile_grid = np.zeros((3, 3))
healthy_grid = np.zeros((3, 3))
stressed_grid = np.zeros((3, 3))

for tile_id in range(9):
    row = tile_id // 3
    col = tile_id % 3
    tile_data = coords_df[coords_df['tile_id'] == tile_id]
    tile_grid[row, col] = len(tile_data)
    healthy_grid[row, col] = len(tile_data[tile_data['class_name'] == 'healthy'])
    stressed_grid[row, col] = len(tile_data[tile_data['class_name'] == 'stressed'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Total samples heatmap
im1 = axes[0].imshow(tile_grid, cmap='YlOrRd', aspect='equal')
axes[0].set_title('Total Samples per Tile', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Column')
axes[0].set_ylabel('Row')
plt.colorbar(im1, ax=axes[0])

# Add text annotations
for i in range(3):
    for j in range(3):
        text = axes[0].text(j, i, f'{int(tile_grid[i, j])}',
                           ha="center", va="center", color="black", fontsize=12)

# Healthy samples heatmap
im2 = axes[1].imshow(healthy_grid, cmap='Greens', aspect='equal')
axes[1].set_title('Healthy Samples per Tile', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Column')
axes[1].set_ylabel('Row')
plt.colorbar(im2, ax=axes[1])

for i in range(3):
    for j in range(3):
        text = axes[1].text(j, i, f'{int(healthy_grid[i, j])}',
                           ha="center", va="center", color="black", fontsize=12)

# Stressed samples heatmap
im3 = axes[2].imshow(stressed_grid, cmap='Reds', aspect='equal')
axes[2].set_title('Stressed Samples per Tile', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Column')
axes[2].set_ylabel('Row')
plt.colorbar(im3, ax=axes[2])

for i in range(3):
    for j in range(3):
        text = axes[2].text(j, i, f'{int(stressed_grid[i, j])}',
                           ha="center", va="center", color="black", fontsize=12)

plt.tight_layout()
plt.savefig('figure_spatial_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# CHART 4: Sample Balance Analysis
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Calculate class ratio per tile
tile_ratios = []
for tile_id in range(9):
    tile_data = coords_df[coords_df['tile_id'] == tile_id]
    healthy_count = len(tile_data[tile_data['class_name'] == 'healthy'])
    stressed_count = len(tile_data[tile_data['class_name'] == 'stressed'])
    if healthy_count > 0:
        ratio = stressed_count / healthy_count
    else:
        ratio = 0
    tile_ratios.append(ratio)

# Ratio bar chart
ax4 = axes[0]
bars = ax4.bar(range(9), tile_ratios, color='purple', alpha=0.7)
ax4.axhline(y=0.382, color='red', linestyle='--', label='Overall Ratio (0.382)')
ax4.set_xlabel('Tile ID', fontsize=12)
ax4.set_ylabel('Stressed/Healthy Ratio', fontsize=12)
ax4.set_title('Class Balance Ratio by Tile', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar, ratio in zip(bars, tile_ratios):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{ratio:.3f}',
             ha='center', va='bottom', fontsize=10)

# Weight distribution
ax5 = axes[1]
ax5.hist(coords_df['weight'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Sample Weight', fontsize=12)
ax5.set_ylabel('Frequency', fontsize=12)
ax5.set_title('Distribution of Sample Weights', fontsize=14, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figure_balance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# SUMMARY TABLE
# ============================================
print("\n" + "="*50)
print("DETAILED TILE STATISTICS")
print("="*50)

tile_stats = []
for tile_id in range(9):
    tile_data = coords_df[coords_df['tile_id'] == tile_id]
    healthy = len(tile_data[tile_data['class_name'] == 'healthy'])
    stressed = len(tile_data[tile_data['class_name'] == 'stressed'])
    total = len(tile_data)
    
    tile_stats.append({
        'Tile ID': tile_id,
        'Healthy': healthy,
        'Stressed': stressed,
        'Total': total,
        'Stressed %': f"{(stressed/total*100):.1f}%" if total > 0 else "0%"
    })

stats_df = pd.DataFrame(tile_stats)
print(stats_df.to_string(index=False))

print("\n" + "="*50)
print("OVERALL STATISTICS")
print("="*50)
print(f"Total Healthy: {class_counts.get('healthy', 0)}")
print(f"Total Stressed: {class_counts.get('stressed', 0)}")
print(f"Overall Stressed Percentage: {(class_counts.get('stressed', 0) / len(coords_df) * 100):.2f}%")
print(f"Average samples per tile: {len(coords_df) / 9:.1f}")
print(f"Standard deviation of samples per tile: {tile_distribution.sum(axis=1).std():.1f}")

# Save summary to CSV
stats_df.to_csv('tile_statistics_summary.csv', index=False)
print("\nTile statistics saved to 'tile_statistics_summary.csv'")

print("\n" + "="*50)
print("All charts have been generated and saved!")
print("="*50)