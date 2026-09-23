"""
Create fig1_combined_timeseries for 2025 data
EXACT same style as THI_Diseases_analysis.ipynb
Excluding Muscardine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

# Load 2025 cleaned data
df = pd.read_csv('results/cleaned_data_2025.csv')
df['Date_parsed'] = pd.to_datetime(df['Date_parsed'])
df = df.dropna(subset=['Date_parsed'])

print("2025 data loaded")
print(f"Date range: {df['Date_parsed'].min().date()} to {df['Date_parsed'].max().date()}")
print(f"Total observations: {len(df)} records")

# Disease names (excluding Muscardine)
disease_names = {
    'Pebrine': 'Pebrine',
    'Virosis': 'Viriosis', 
    'Bacteriosis': 'Bacteriosis'
}

# Convert disease columns to binary (1/0) - any positive value = 1
disease_cols = ['Pebrine', 'Virosis', 'Bacteriosis']
for col in disease_cols:
    df[f'{col}_binary'] = (df[col] > 0).astype(int)

# Daily aggregation for THI
daily_data = df.groupby('Date_parsed').agg({
    'THI': 'mean',
    'Tmax': 'mean',
    'Tmin': 'mean',
    'Humidity': 'mean',
    'Pebrine': 'sum',
    'Virosis': 'sum',
    'Bacteriosis': 'sum'
}).reset_index()

daily_data.rename(columns={
    'Date_parsed': 'Date',
    'THI': 'THI_Mean',
    'Tmax': 'Temp_Max',
    'Tmin': 'Temp_Min',
    'Humidity': 'RH_Mean'
}, inplace=True)

# Convert summed disease counts to binary (if any case on that day = 1)
for col in disease_cols:
    daily_data[f'{col}_binary'] = (daily_data[col] > 0).astype(int)

print(f"Daily aggregation: {len(daily_data)} days")

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
ax2.set_ylim(-0.3, 1.8)
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
plt.savefig('figures/fig1_combined_timeseries_2025.png', dpi=600, bbox_inches='tight', pad_inches=0.3)
plt.savefig('figures/fig1_combined_timeseries_2025.pdf', bbox_inches='tight', pad_inches=0.3)

print("Saved: figures/fig1_combined_timeseries_2025.png")
print("Saved: figures/fig1_combined_timeseries_2025.pdf")

plt.close()

# Print summary
print("\n" + "="*80)
print("FIGURE CREATED SUCCESSFULLY")
print("="*80)
print(f"\nDiseases shown: {variables}")
print(f"Muscardine: EXCLUDED as requested")
print(f"\nDate range: {daily_data['Date'].min().date()} to {daily_data['Date'].max().date()}")
print(f"Total days: {len(daily_data)}")

for var in variables:
    count = daily_data[f'{var}_binary'].sum()
    print(f"  {var}: {count} days with disease")
