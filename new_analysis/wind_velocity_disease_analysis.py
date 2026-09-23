"""
Wind VELOCITY (Vector) vs Disease Incidence Analysis - Plot-wise
Using NASA POWER data with both speed and direction
October 7-31, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set high-resolution output
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

print("=" * 80)
print("WIND VELOCITY (VECTOR) vs DISEASE INCIDENCE ANALYSIS")
print("Ranchi, Jharkhand - October 7-31, 2025")
print("=" * 80)

# Load NASA wind velocity data
wind_df = pd.read_csv('data/nasa_wind_velocity_daily.csv')
wind_df['date'] = pd.to_datetime(wind_df['date'])

print("\nNASA Wind Velocity Data Loaded:")
print(f"  Period: {wind_df['date'].min()} to {wind_df['date'].max()}")
print(f"  Records: {len(wind_df)} days")

# Load disease data
df = pd.read_csv('../dmc-ctrti-2025.csv')
df = df[df['Plot_No'].notna()]
df['Plot_No'] = df['Plot_No'].astype(int)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

for col in ['Pebrine', 'Virosis', 'Bacteriosis']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df = df[df['Wind_Speed'].notna()]

# Aggregate disease data by date (sum across all plots)
disease_daily = df.groupby('Date').agg({
    'Pebrine': 'sum',
    'Virosis': 'sum',
    'Bacteriosis': 'sum',
    'Wind_Speed': 'mean'  # Original wind speed from CSV
}).reset_index()

# Merge with NASA wind velocity data
merged = pd.merge(disease_daily, wind_df, left_on='Date', right_on='date', how='inner')

print(f"\nMerged dataset: {len(merged)} days")

# Calculate additional velocity metrics
# Wind velocity magnitude (same as speed but from NASA data)
merged['velocity_magnitude'] = merged['wind_speed_ms_mean']

# Wind velocity components (already calculated in NASA data)
# Vx: East-West component (positive = East)
# Vy: North-South component (positive = North)
merged['vx_mean'] = merged['wind_vx_mean']
merged['vy_mean'] = merged['wind_vy_mean']

# Resultant velocity (net daily movement)
merged['resultant_velocity'] = np.sqrt(merged['resultant_vx']**2 + merged['resultant_vy']**2)

# Wind variability (standard deviation of direction)
merged['direction_variability'] = merged['wind_direction_deg_std']

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS: WIND VELOCITY vs DISEASES")
print("=" * 80)

diseases = ['Pebrine', 'Virosis', 'Bacteriosis']
wind_vars = [
    ('velocity_magnitude', 'Wind Velocity Magnitude (m/s)'),
    ('resultant_velocity', 'Resultant Wind Velocity (m/s)'),
    ('vx_mean', 'East-West Component Vx (m/s)'),
    ('vy_mean', 'North-South Component Vy (m/s)'),
    ('wind_direction_deg_mean', 'Wind Direction (degrees)'),
    ('direction_variability', 'Direction Variability (std)')
]

correlation_results = []

for disease in diseases:
    if merged[disease].sum() == 0:
        print(f"\n{disease}: No cases recorded")
        continue
    
    print(f"\n{'='*60}")
    print(f"{disease}")
    print(f"{'='*60}")
    
    for var_code, var_name in wind_vars:
        r, p = pearsonr(merged[var_code], merged[disease])
        rho, p_rho = spearmanr(merged[var_code], merged[disease])
        
        correlation_results.append({
            'Disease': disease,
            'Variable': var_name,
            'Variable_Code': var_code,
            'Pearson_r': r,
            'Pearson_p': p,
            'Spearman_rho': rho,
            'Spearman_p': p_rho,
            'Significant': 'Yes' if p < 0.05 else 'No'
        })
        
        sig_marker = "***" if p < 0.05 else ""
        print(f"  {var_name}:")
        print(f"    Pearson r = {r:.4f} (p = {p:.4f}) {sig_marker}")
        print(f"    Spearman rho = {rho:.4f} (p = {p_rho:.4f})")

# Save correlation results
corr_df = pd.DataFrame(correlation_results)
corr_df.to_csv('results/wind_velocity_disease_correlations.csv', index=False)
print("\nSaved: results/wind_velocity_disease_correlations.csv")

# Create comprehensive visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. Wind velocity components time series with disease overlay
fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)

# Wind velocity magnitude
ax = axes[0]
ax.plot(merged['Date'], merged['velocity_magnitude'], 'g-', linewidth=2, marker='o', label='Wind Speed')
ax.set_ylabel('Wind Speed (m/s)', fontsize=14, fontweight='bold', color='green')
ax.tick_params(axis='y', labelcolor='green')
ax.set_title('Wind Velocity Magnitude', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(loc='upper left')

# Wind direction
ax = axes[1]
ax.plot(merged['Date'], merged['wind_direction_deg_mean'], 'purple', linewidth=2, marker='s', label='Wind Direction')
ax.set_ylabel('Wind Direction (degrees)', fontsize=14, fontweight='bold', color='purple')
ax.tick_params(axis='y', labelcolor='purple')
ax.set_title('Wind Direction (0°=N, 90°=E, 180°=S, 270°=W)', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(loc='upper left')

# Vx and Vy components
ax = axes[2]
ax.plot(merged['Date'], merged['vx_mean'], 'b-', linewidth=2, marker='o', label='Vx (East-West)')
ax.plot(merged['Date'], merged['vy_mean'], 'r-', linewidth=2, marker='s', label='Vy (North-South)')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('Velocity Component (m/s)', fontsize=14, fontweight='bold')
ax.set_title('Wind Velocity Components (Vx: East+, Vy: North+)', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(loc='upper left')

# Disease overlay - Viriosis only (has data)
ax = axes[3]
ax.bar(merged['Date'], merged['Virosis'], color='blue', alpha=0.7, width=0.8, label='Viriosis')
ax.set_ylabel('Viriosis Cases', fontsize=14, fontweight='bold', color='blue')
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelcolor='blue')
ax.set_title('Viriosis Incidence', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(loc='upper left')

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/wind_velocity_timeseries_disease.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/wind_velocity_timeseries_disease.png")

# 2. Scatter plots for significant correlations
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, (var_code, var_name) in enumerate(wind_vars):
    ax = axes[idx]
    
    # Viriosis (only disease with data)
    ax.scatter(merged[var_code], merged['Virosis'], 
               s=200, c='blue', alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add trend line
    z = np.polyfit(merged[var_code], merged['Virosis'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(merged[var_code].min(), merged[var_code].max(), 100)
    ax.plot(x_line, p_line(x_line), 'r--', linewidth=2)
    
    # Calculate correlation
    r, p = pearsonr(merged[var_code], merged['Virosis'])
    
    ax.set_xlabel(var_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Viriosis Cases', fontsize=12, fontweight='bold', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_title(f'{var_name}\nr = {r:.3f}, p = {p:.4f}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

plt.suptitle('Wind Velocity Parameters vs Viriosis Incidence', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/wind_velocity_vs_viriosis_scatter.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/wind_velocity_vs_viriosis_scatter.png")

# 3. Wind rose showing disease occurrence
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

# Create wind rose
wind_dirs = merged['wind_direction_deg_mean'].values
wind_speeds = merged['velocity_magnitude'].values
disease_cases = merged['Virosis'].values

# Normalize disease cases for color mapping
norm_cases = (disease_cases - disease_cases.min()) / (disease_cases.max() - disease_cases.min() + 0.001)

# Convert to radians
theta = np.radians(wind_dirs)

# Create scatter plot on polar axis
scatter = ax.scatter(theta, wind_speeds, c=disease_cases, s=300, 
                     cmap='Reds', alpha=0.7, edgecolors='black', linewidth=2)

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_title('Wind Velocity Rose with Viriosis Incidence\n(Darker = More Cases)', 
             fontsize=18, fontweight='bold', pad=30)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
cbar.set_label('Viriosis Cases', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/wind_rose_viriosis.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/wind_rose_viriosis.png")

# 4. Comparison: Original Speed vs Velocity Vector
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Speed vs Viriosis
ax = axes[0, 0]
ax.scatter(merged['Wind_Speed'], merged['Virosis'], s=200, c='green', alpha=0.6, edgecolors='black')
r, p = pearsonr(merged['Wind_Speed'], merged['Virosis'])
z = np.polyfit(merged['Wind_Speed'], merged['Virosis'], 1)
p_line = np.poly1d(z)
x_line = np.linspace(merged['Wind_Speed'].min(), merged['Wind_Speed'].max(), 100)
ax.plot(x_line, p_line(x_line), 'r--', linewidth=2)
ax.set_xlabel('Original Wind Speed (m/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Viriosis Cases', fontsize=14, fontweight='bold', color='blue')
ax.set_title(f'Original Speed vs Viriosis\nr = {r:.3f}, p = {p:.4f}', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)

# Velocity Magnitude vs Viriosis
ax = axes[0, 1]
ax.scatter(merged['velocity_magnitude'], merged['Virosis'], s=200, c='blue', alpha=0.6, edgecolors='black')
r, p = pearsonr(merged['velocity_magnitude'], merged['Virosis'])
z = np.polyfit(merged['velocity_magnitude'], merged['Virosis'], 1)
p_line = np.poly1d(z)
x_line = np.linspace(merged['velocity_magnitude'].min(), merged['velocity_magnitude'].max(), 100)
ax.plot(x_line, p_line(x_line), 'r--', linewidth=2)
ax.set_xlabel('NASA Velocity Magnitude (m/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Viriosis Cases', fontsize=14, fontweight='bold', color='blue')
ax.set_title(f'NASA Velocity vs Viriosis\nr = {r:.3f}, p = {p:.4f}', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)

# Resultant Velocity vs Viriosis
ax = axes[1, 0]
ax.scatter(merged['resultant_velocity'], merged['Virosis'], s=200, c='red', alpha=0.6, edgecolors='black')
r, p = pearsonr(merged['resultant_velocity'], merged['Virosis'])
z = np.polyfit(merged['resultant_velocity'], merged['Virosis'], 1)
p_line = np.poly1d(z)
x_line = np.linspace(merged['resultant_velocity'].min(), merged['resultant_velocity'].max(), 100)
ax.plot(x_line, p_line(x_line), 'k--', linewidth=2)
ax.set_xlabel('Resultant Wind Velocity (m/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Viriosis Cases', fontsize=14, fontweight='bold', color='blue')
ax.set_title(f'Resultant Velocity vs Viriosis\nr = {r:.3f}, p = {p:.4f}', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)

# Direction vs Viriosis
ax = axes[1, 1]
ax.scatter(merged['wind_direction_deg_mean'], merged['Virosis'], s=200, c='purple', alpha=0.6, edgecolors='black')
r, p = pearsonr(merged['wind_direction_deg_mean'], merged['Virosis'])
ax.set_xlabel('Wind Direction (degrees)', fontsize=14, fontweight='bold')
ax.set_ylabel('Viriosis Cases', fontsize=14, fontweight='bold', color='blue')
ax.set_title(f'Direction vs Viriosis\nr = {r:.3f}, p = {p:.4f}', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)

plt.suptitle('Wind Speed vs Wind Velocity Comparison', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/speed_vs_velocity_comparison.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/speed_vs_velocity_comparison.png")

# Save merged data
merged.to_csv('results/wind_velocity_disease_merged.csv', index=False)
print("Saved: results/wind_velocity_disease_merged.csv")

# Print final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\nKey Findings:")
print("-" * 80)

# Find strongest correlations
virosis_corr = corr_df[corr_df['Disease'] == 'Viriosis'].sort_values('Pearson_r', key=abs, ascending=False)

print("\nStrongest correlations with Viriosis:")
for _, row in virosis_corr.head(3).iterrows():
    sig = "*** SIGNIFICANT ***" if row['Pearson_p'] < 0.05 else "(not significant)"
    print(f"  {row['Variable']}: r = {row['Pearson_r']:.4f} {sig}")

print("\nData Comparison:")
print(f"  Original Wind Speed (field): {merged['Wind_Speed'].mean():.2f} ± {merged['Wind_Speed'].std():.2f} m/s")
print(f"  NASA Velocity Magnitude: {merged['velocity_magnitude'].mean():.2f} ± {merged['velocity_magnitude'].std():.2f} m/s")
print(f"  Resultant Velocity: {merged['resultant_velocity'].mean():.2f} ± {merged['resultant_velocity'].std():.2f} m/s")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  - data/nasa_wind_velocity_hourly.csv")
print("  - data/nasa_wind_velocity_daily.csv")
print("  - results/wind_velocity_disease_correlations.csv")
print("  - results/wind_velocity_disease_merged.csv")
print("  - figures/wind_velocity_timeseries_disease.png")
print("  - figures/wind_velocity_vs_viriosis_scatter.png")
print("  - figures/wind_rose_viriosis.png")
print("  - figures/speed_vs_velocity_comparison.png")
