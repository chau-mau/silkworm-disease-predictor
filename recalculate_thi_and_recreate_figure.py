"""
Recalculate THI using proper NRC formula for 2025 data
NRC (1971) formula: THI = (1.8 × T + 32) - [(0.55 - 0.0055 × RH) × (1.8 × T - 58)]
Then recreate the figure with correct THI values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load 2025 cleaned data
df = pd.read_csv('results/cleaned_data_2025.csv')
df['Date_parsed'] = pd.to_datetime(df['Date_parsed'])
df = df.dropna(subset=['Date_parsed'])

print("2025 data loaded")
print(f"Original THI range: {df['THI'].min():.1f} - {df['THI'].max():.1f}")

# Calculate mean temperature from Tmax and Tmin
df['Temp_Mean'] = (df['Tmax'] + df['Tmin']) / 2

# Calculate THI using NRC (1971) formula - SAME as 2024 analysis
def calculate_thi_nrc(temp_c, rh):
    """
    Calculate Thermo-Humidity Index using NRC (1971) formula
    THI = (1.8 × T + 32) - [(0.55 - 0.0055 × RH) × (1.8 × T - 58)]
    """
    temp_f = (1.8 * temp_c) + 32
    thi = temp_f - ((0.55 - 0.0055 * rh) * (temp_f - 58))
    return thi

# Recalculate THI using mean temperature and humidity
df['THI_NRC'] = df.apply(lambda row: calculate_thi_nrc(row['Temp_Mean'], row['Humidity']), axis=1)

print(f"\nRecalculated THI (NRC formula) range: {df['THI_NRC'].min():.1f} - {df['THI_NRC'].max():.1f}")
print(f"Mean THI: {df['THI_NRC'].mean():.2f}")

# Classify THI stress levels
def classify_thi_stress(thi):
    """Classify THI into stress categories"""
    if thi < 70:
        return "No Stress"
    elif thi < 72:
        return "Mild Stress"
    elif thi < 80:
        return "Moderate Stress"
    elif thi < 90:
        return "Severe Stress"
    else:
        return "Emergency"

df['Stress_Level'] = df['THI_NRC'].apply(classify_thi_stress)

# Print stress level distribution
stress_counts = df['Stress_Level'].value_counts()
print("\nTHI Stress Level Distribution:")
for level, count in stress_counts.items():
    pct = count / len(df) * 100
    print(f"  {level}: {count} records ({pct:.1f}%)")

# Disease names (excluding Muscardine)
disease_names = {
    'Pebrine': 'Pebrine',
    'Virosis': 'Viriosis', 
    'Bacteriosis': 'Bacteriosis'
}

# Convert disease columns to binary
disease_cols = ['Pebrine', 'Virosis', 'Bacteriosis']
for col in disease_cols:
    df[f'{col}_binary'] = (df[col] > 0).astype(int)

# Daily aggregation using CORRECT THI
daily_data = df.groupby('Date_parsed').agg({
    'THI_NRC': 'mean',
    'Temp_Mean': 'mean',
    'Humidity': 'mean',
    'Pebrine': 'sum',
    'Virosis': 'sum',
    'Bacteriosis': 'sum'
}).reset_index()

daily_data.rename(columns={
    'Date_parsed': 'Date',
    'THI_NRC': 'THI_Mean',
    'Temp_Mean': 'Temp_Mean',
    'Humidity': 'RH_Mean'
}, inplace=True)

# Convert summed disease counts to binary
for col in disease_cols:
    daily_data[f'{col}_binary'] = (daily_data[col] > 0).astype(int)

print(f"\nDaily aggregation: {len(daily_data)} days")
print(f"Daily THI range: {daily_data['THI_Mean'].min():.1f} - {daily_data['THI_Mean'].max():.1f}")

# Set EXACT same high-quality plot parameters as original
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 22

# Disease colors - EXACT same as original
disease_colors = {
    'Pebrine': '#E74C3C',    # Red
    'Virosis': '#3498DB',    # Blue
    'Bacteriosis': '#F39C12' # Orange
}

# Create figure with EXACT same dimensions
fig = plt.figure(figsize=(22, 11))

# Create axes
ax1 = plt.subplot(111)

# Plot THI as line - EXACT same style
ax1.plot(daily_data['Date'], daily_data['THI_Mean'],
         color='black', linewidth=3.5, marker='o', markersize=8,
         label='THI', alpha=0.8, zorder=5)

# Add stress level zones - EXACT same as original
ax1.axhspan(0, 70, alpha=0.25, color='green', label='No Stress (<70)')
ax1.axhspan(72, 80, alpha=0.25, color='orange', label='Moderate Stress (72-80)')
ax1.axhspan(80, 100, alpha=0.25, color='red', label='Severe Stress (>80)')

# Labels - EXACT same
ax1.set_xlabel('Date', fontsize=22, fontweight='bold')
ax1.set_ylabel('Thermo-Humidity Index (THI)', fontsize=22, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
ax1.tick_params(axis='x', rotation=45, labelsize=18)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

# Set y-axis limits to match original figure style
ax1.set_ylim([0, 105])

# Secondary y-axis for disease occurrence
ax2 = ax1.twinx()

# Plot each disease - EXACT same style as original
variables = ['Pebrine', 'Virosis', 'Bacteriosis']
y_offset = 0

for var in variables:
    # Plot line connecting disease occurrences
    disease_plot = daily_data[f'{var}_binary'] + y_offset
    
    ax2.plot(daily_data['Date'], disease_plot,
             color=disease_colors[var], linewidth=3, alpha=0.7,
             linestyle='--', zorder=3)
    
    # Mark disease occurrences with large markers
    yes_dates = daily_data[daily_data[f'{var}_binary'] == 1]['Date']
    yes_values = [1 + y_offset] * len(yes_dates)
    
    ax2.scatter(yes_dates, yes_values,
                color=disease_colors[var], s=400, alpha=0.9,
                marker='s', edgecolor='black', linewidth=2.5,
                label=f'{disease_names[var]} ({var[:2]})', zorder=10)
    
    y_offset += 0.2

# Secondary axis settings - EXACT same
ax2.set_ylabel('Disease Occurrence', fontsize=22, fontweight='bold')
ax2.set_ylim([-0.3, 1.8])
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Absent', 'Present'], fontsize=18)
ax2.tick_params(axis='y', labelsize=18)

# Get all legend elements
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Create single legend ABOVE the plot - EXACT same
fig.legend(lines1 + lines2, labels1 + labels2,
          loc='upper center', fontsize=15,
          framealpha=0.98, edgecolor='black',
          ncol=4, bbox_to_anchor=(0.5, 0.97),
          columnspacing=2)

# Adjust layout - EXACT same
plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.92)

# Save - EXACT same settings
plt.savefig('figures/fig1_combined_timeseries_2025_corrected.png', dpi=600, bbox_inches='tight', pad_inches=0.3)
plt.savefig('figures/fig1_combined_timeseries_2025_corrected.pdf', bbox_inches='tight', pad_inches=0.3)

print("\nSaved: figures/fig1_combined_timeseries_2025_corrected.png")
print("Saved: figures/fig1_combined_timeseries_2025_corrected.pdf")

plt.close()

# Also save the corrected data
df.to_csv('results/cleaned_data_2025_corrected_thi.csv', index=False)
daily_data.to_csv('results/daily_data_2025_corrected_thi.csv', index=False)
print("\nSaved: results/cleaned_data_2025_corrected_thi.csv")
print("Saved: results/daily_data_2025_corrected_thi.csv")

# Print summary
print("\n" + "="*80)
print("FIGURE CREATED WITH CORRECTED THI")
print("="*80)
print(f"\nDiseases shown: {variables}")
print(f"Muscardine: EXCLUDED as requested")
print(f"\nDate range: {daily_data['Date'].min().date()} to {daily_data['Date'].max().date()}")
print(f"Total days: {len(daily_data)}")
print(f"\nCorrected THI Statistics:")
print(f"  Mean: {daily_data['THI_Mean'].mean():.2f}")
print(f"  Min: {daily_data['THI_Mean'].min():.2f}")
print(f"  Max: {daily_data['THI_Mean'].max():.2f}")
print(f"  Std: {daily_data['THI_Mean'].std():.2f}")

for var in variables:
    count = daily_data[f'{var}_binary'].sum()
    print(f"  {var}: {count} days with disease")

print("\nComparison with 2024 data:")
print("  2024 THI range: 69.19 - 76.75 (mean: 73.15)")
print(f"  2025 THI range: {daily_data['THI_Mean'].min():.2f} - {daily_data['THI_Mean'].max():.2f} (mean: {daily_data['THI_Mean'].mean():.2f})")
