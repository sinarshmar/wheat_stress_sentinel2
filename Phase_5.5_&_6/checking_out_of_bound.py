import pandas as pd

# Load your training data
df = pd.read_csv('Phase_5.5_results/training_data_clean.csv')

# Check coordinate bounds
print("Lon range:", df['lon'].min(), "-", df['lon'].max())
print("Lat range:", df['lat'].min(), "-", df['lat'].max())

# Ludhiana approximate bounds:
# Lon: 75.7 - 76.1
# Lat: 30.7 - 31.1

# Check if samples fall outside
outside = df[(df['lon'] < 75.7) | (df['lon'] > 76.1) | 
             (df['lat'] < 30.7) | (df['lat'] > 31.1)]
print(f"Samples outside Ludhiana bounds: {len(outside)} / {len(df)}")