"""
Create time series figure for 2025 data (excluding Muscardine)
Similar to Previous_work/fig1_combined_timeseries (1).png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Load cleaned data
df = pd.read_csv('results/cleaned_data_2025.csv')
df['Date_parsed'] = pd.to_datetime(df['Date_parsed'])

# Filter out rows with valid dates
df = df.dropna(subset=['Date_parsed'])

print(f"Data points: {len(df)}")
print(f"Date range: {df['Date_parsed'].min()} to {df['Date_parsed'].max()}")

# Daily aggregation for climate variables
daily_climate = df.groupby('Date_parsed').agg({
    'Tmax': 'mean',
    'Tmin': 'mean',
    'Humidity': 'mean',
    'THI': 'mean',
    'Wind_Speed': 'mean'
}).reset_index()

# Daily aggregation for diseases (excluding Muscardine)
disease_cols = ['Pebrine', 'Virosis', 'Bacteriosis']
daily_disease = df.groupby('Date_parsed')[disease_cols].sum().reset_index()

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.patch.set_facecolor('white')

# Color scheme
colors = {
    'Tmax': '#d62728',      # red
    'Tmin': '#1f77b4',      # blue
    'Humidity': '#2ca02c',  # green
    'THI': '#ff7f0e',       # orange
    'Pebrine': '#9467bd',   # purple
    'Virosis': '#e377c2',   # pink
    'Bacteriosis': '#8c564b' # brown
}

# Plot 1: Temperature (Tmax and Tmin)
ax1 = axes[0]
ax1_twin = ax1.twinx()

ax1.plot(daily_climate['Date_parsed'], daily_climate['Tmax'], 
         color=colors['Tmax'], linewidth=2, marker='o', markersize=4, label='Tmax')
ax1.plot(daily_climate['Date_parsed'], daily_climate['Tmin'], 
         color=colors['Tmin'], linewidth=2, marker='s', markersize=4, label='Tmin')

ax1.set_ylabel('Temperature (°C)', fontsize=11)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([15, 40])

# Plot 2: Humidity
ax2 = axes[1]
ax2.plot(daily_climate['Date_parsed'], daily_climate['Humidity'], 
         color=colors['Humidity'], linewidth=2, marker='o', markersize=4)
ax2.set_ylabel('Humidity (%)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([50, 100])

# Plot 3: THI
ax3 = axes[2]
ax3.plot(daily_climate['Date_parsed'], daily_climate['THI'], 
         color=colors['THI'], linewidth=2, marker='o', markersize=4)
ax3.set_ylabel('THI', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([28, 38])

# Add THI reference lines
ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax3.axhline(y=32, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax3.axhline(y=34, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Plot 4: Disease occurrence (Pebrine, Virosis, Bacteriosis - NO Muscardine)
ax4 = axes[3]

# Stack the disease data
bottom_pebrine = daily_disease['Pebrine']
bottom_virosis = bottom_pebrine + daily_disease['Virosis']

ax4.bar(daily_disease['Date_parsed'], daily_disease['Pebrine'], 
        color=colors['Pebrine'], label='Pebrine', alpha=0.8, width=0.8)
ax4.bar(daily_disease['Date_parsed'], daily_disease['Virosis'], 
        bottom=bottom_pebrine,
        color=colors['Virosis'], label='Virosis', alpha=0.8, width=0.8)
ax4.bar(daily_disease['Date_parsed'], daily_disease['Bacteriosis'], 
        bottom=bottom_virosis,
        color=colors['Bacteriosis'], label='Bacteriosis', alpha=0.8, width=0.8)

ax4.set_ylabel('Disease Count', fontsize=11)
ax4.set_xlabel('Date', fontsize=11)
ax4.legend(loc='upper left', framealpha=0.9)
ax4.grid(True, alpha=0.3, axis='y')

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)

# Save figure
plt.savefig('figures/timeseries_2025_no_muscardine.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('figures/timeseries_2025_no_muscardine.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Saved: figures/timeseries_2025_no_muscardine.png")
print("Saved: figures/timeseries_2025_no_muscardine.pdf")

plt.close()

# Create a second version with lines instead of bars for diseases
fig2, axes2 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig2.patch.set_facecolor('white')

# Plot 1-3: Same as before
ax1 = axes2[0]
ax1.plot(daily_climate['Date_parsed'], daily_climate['Tmax'], 
         color=colors['Tmax'], linewidth=2, marker='o', markersize=4, label='Tmax')
ax1.plot(daily_climate['Date_parsed'], daily_climate['Tmin'], 
         color=colors['Tmin'], linewidth=2, marker='s', markersize=4, label='Tmin')
ax1.set_ylabel('Temperature (°C)', fontsize=11)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([15, 40])

ax2 = axes2[1]
ax2.plot(daily_climate['Date_parsed'], daily_climate['Humidity'], 
         color=colors['Humidity'], linewidth=2, marker='o', markersize=4)
ax2.set_ylabel('Humidity (%)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([50, 100])

ax3 = axes2[2]
ax3.plot(daily_climate['Date_parsed'], daily_climate['THI'], 
         color=colors['THI'], linewidth=2, marker='o', markersize=4)
ax3.set_ylabel('THI', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([28, 38])
ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax3.axhline(y=32, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax3.axhline(y=34, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Plot 4: Disease as lines (alternative visualization)
ax4 = axes2[3]
ax4.plot(daily_disease['Date_parsed'], daily_disease['Pebrine'], 
         color=colors['Pebrine'], linewidth=2, marker='o', markersize=5, label='Pebrine')
ax4.plot(daily_disease['Date_parsed'], daily_disease['Virosis'], 
         color=colors['Virosis'], linewidth=2, marker='s', markersize=5, label='Virosis')
ax4.plot(daily_disease['Date_parsed'], daily_disease['Bacteriosis'], 
         color=colors['Bacteriosis'], linewidth=2, marker='^', markersize=5, label='Bacteriosis')

ax4.set_ylabel('Disease Count', fontsize=11)
ax4.set_xlabel('Date', fontsize=11)
ax4.legend(loc='upper left', framealpha=0.9)
ax4.grid(True, alpha=0.3)

# Format x-axis
for ax in axes2:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)

plt.savefig('figures/timeseries_2025_lines_no_muscardine.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: figures/timeseries_2025_lines_no_muscardine.png")

plt.close()

print("\nTime series figures created successfully!")
print(f"Diseases shown: {disease_cols}")
print(f"Muscardine: EXCLUDED as requested")
