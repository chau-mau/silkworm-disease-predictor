"""
Wind Velocity vs Pebrine and Viriosis Analysis - Plot-wise
Correlates wind speed with disease incidence at plot level
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
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

print("=" * 80)
print("WIND VELOCITY vs DISEASE INCIDENCE ANALYSIS - PLOT-WISE")
print("=" * 80)

# Load original data with wind speed
df = pd.read_csv('../dmc-ctrti-2025.csv')

# Clean and prepare data
df = df[df['Plot_No'].notna()]  # Remove empty rows
df['Plot_No'] = df['Plot_No'].astype(int)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

# Convert disease columns to numeric (handle 'X' and empty values)
for col in ['Pebrine', 'Virosis', 'Bacteriosis', 'Muscardine']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Convert Wind_Speed to numeric
df['Wind_Speed'] = pd.to_numeric(df['Wind_Speed'], errors='coerce')

# Remove rows with missing wind speed
df = df[df['Wind_Speed'].notna()]

print(f"\nData loaded: {len(df)} observations")
print(f"Plots: {sorted(df['Plot_No'].unique())}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Summary by plot
print("\n" + "=" * 80)
print("PLOT-WISE SUMMARY")
print("=" * 80)

plot_summary = df.groupby('Plot_No').agg({
    'Wind_Speed': ['mean', 'std', 'min', 'max'],
    'Pebrine': 'sum',
    'Virosis': 'sum',
    'Bacteriosis': 'sum',
    'Muscardine': 'sum',
    'Date': 'count'
}).round(3)

plot_summary.columns = ['Wind_Mean', 'Wind_Std', 'Wind_Min', 'Wind_Max', 
                        'Pebrine_Total', 'Virosis_Total', 'Bacteriosis_Total', 
                        'Muscardine_Total', 'Days_Observed']

print(plot_summary.to_string())

# Calculate correlation between wind speed and diseases at plot level
print("\n" + "=" * 80)
print("PLOT-LEVEL CORRELATION ANALYSIS")
print("=" * 80)

# Use plot summary for correlation
plot_data = plot_summary.reset_index()

correlations = []

# Pebrine correlation
r_pb, p_pb = pearsonr(plot_data['Wind_Mean'], plot_data['Pebrine_Total'])
correlations.append({
    'Disease': 'Pebrine',
    'Correlation': r_pb,
    'P_value': p_pb,
    'Significant': 'Yes' if p_pb < 0.05 else 'No'
})
print(f"\nPebrine vs Wind Speed:")
print(f"  Pearson r = {r_pb:.4f}, p = {p_pb:.4f}")
if p_pb < 0.05:
    print(f"  *** SIGNIFICANT ***")

# Viriosis correlation
r_vr, p_vr = pearsonr(plot_data['Wind_Mean'], plot_data['Virosis_Total'])
correlations.append({
    'Disease': 'Viriosis',
    'Correlation': r_vr,
    'P_value': p_vr,
    'Significant': 'Yes' if p_vr < 0.05 else 'No'
})
print(f"\nViriosis vs Wind Speed:")
print(f"  Pearson r = {r_vr:.4f}, p = {p_vr:.4f}")
if p_vr < 0.05:
    print(f"  *** SIGNIFICANT ***")

# Spearman correlation (non-parametric)
print("\n--- Spearman Rank Correlation ---")
rho_pb, p_rho_pb = spearmanr(plot_data['Wind_Mean'], plot_data['Pebrine_Total'])
rho_vr, p_rho_vr = spearmanr(plot_data['Wind_Mean'], plot_data['Virosis_Total'])
print(f"Pebrine: rho = {rho_pb:.4f}, p = {p_rho_pb:.4f}")
print(f"Viriosis: rho = {rho_vr:.4f}, p = {p_rho_vr:.4f}")

# Save plot summary
plot_data.to_csv('results/wind_disease_plot_summary.csv', index=False)
print("\nSaved: results/wind_disease_plot_summary.csv")

# Create visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. Scatter plot: Wind Speed vs Disease Incidence by Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Pebrine
axes[0].scatter(plot_data['Wind_Mean'], plot_data['Pebrine_Total'], 
                s=200, c='red', alpha=0.6, edgecolors='black', linewidth=2)
for i, row in plot_data.iterrows():
    axes[0].annotate(f"Plot {int(row['Plot_No'])}", 
                     (row['Wind_Mean'], row['Pebrine_Total']),
                     xytext=(5, 5), textcoords='offset points', fontsize=11)

# Add trend line
z = np.polyfit(plot_data['Wind_Mean'], plot_data['Pebrine_Total'], 1)
p = np.poly1d(z)
axes[0].plot(plot_data['Wind_Mean'], p(plot_data['Wind_Mean']), 
             "r--", alpha=0.8, linewidth=2, label=f'Trend (r={r_pb:.3f})')

axes[0].set_xlabel('Mean Wind Speed (m/s)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Pebrine Incidence (Total Cases)', fontsize=14, fontweight='bold')
axes[0].set_title(f'Wind Speed vs Pebrine Incidence by Plot\np = {p_pb:.4f}', 
                  fontsize=16, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(alpha=0.3)

# Viriosis
axes[1].scatter(plot_data['Wind_Mean'], plot_data['Virosis_Total'], 
                s=200, c='blue', alpha=0.6, edgecolors='black', linewidth=2)
for i, row in plot_data.iterrows():
    axes[1].annotate(f"Plot {int(row['Plot_No'])}", 
                     (row['Wind_Mean'], row['Virosis_Total']),
                     xytext=(5, 5), textcoords='offset points', fontsize=11)

# Add trend line
z = np.polyfit(plot_data['Wind_Mean'], plot_data['Virosis_Total'], 1)
p = np.poly1d(z)
axes[1].plot(plot_data['Wind_Mean'], p(plot_data['Wind_Mean']), 
             "b--", alpha=0.8, linewidth=2, label=f'Trend (r={r_vr:.3f})')

axes[1].set_xlabel('Mean Wind Speed (m/s)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Viriosis Incidence (Total Cases)', fontsize=14, fontweight='bold')
axes[1].set_title(f'Wind Speed vs Viriosis Incidence by Plot\np = {p_vr:.4f}', 
                  fontsize=16, fontweight='bold')
axes[1].legend(fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/wind_vs_disease_scatter_plots.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/wind_vs_disease_scatter_plots.png")

# 2. Bar chart comparing wind speed and disease by plot
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

x = np.arange(len(plot_data))
width = 0.35

# Wind speed by plot
axes[0].bar(x, plot_data['Wind_Mean'], width, label='Mean Wind Speed', 
            color='skyblue', edgecolor='black', alpha=0.8)
axes[0].errorbar(x, plot_data['Wind_Mean'], yerr=plot_data['Wind_Std'], 
                 fmt='none', color='black', capsize=5)
axes[0].set_ylabel('Wind Speed (m/s)', fontsize=14, fontweight='bold')
axes[0].set_title('Mean Wind Speed by Plot', fontsize=16, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"Plot {int(p)}" for p in plot_data['Plot_No']], rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Disease incidence by plot
axes[1].bar(x - width/2, plot_data['Pebrine_Total'], width, label='Pebrine', 
            color='red', alpha=0.7, edgecolor='black')
axes[1].bar(x + width/2, plot_data['Virosis_Total'], width, label='Viriosis', 
            color='blue', alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Total Disease Cases', fontsize=14, fontweight='bold')
axes[1].set_title('Disease Incidence by Plot', fontsize=16, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"Plot {int(p)}" for p in plot_data['Plot_No']], rotation=45)
axes[1].legend(fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/wind_and_disease_by_plot.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/wind_and_disease_by_plot.png")

# 3. Time series of wind speed with disease occurrence
fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

# Aggregate by date
daily_data = df.groupby('Date').agg({
    'Wind_Speed': 'mean',
    'Pebrine': 'sum',
    'Virosis': 'sum'
}).reset_index()

# Wind speed
axes[0].plot(daily_data['Date'], daily_data['Wind_Speed'], 'g-', linewidth=2, marker='o')
axes[0].set_ylabel('Wind Speed (m/s)', fontsize=14, fontweight='bold', color='green')
axes[0].tick_params(axis='y', labelcolor='green')
axes[0].set_title('Daily Mean Wind Speed', fontsize=16, fontweight='bold')
axes[0].grid(alpha=0.3)

# Pebrine
axes[1].bar(daily_data['Date'], daily_data['Pebrine'], color='red', alpha=0.7, width=0.8)
axes[1].set_ylabel('Pebrine Cases', fontsize=14, fontweight='bold', color='red')
axes[1].tick_params(axis='y', labelcolor='red')
axes[1].set_title('Daily Pebrine Incidence', fontsize=16, fontweight='bold')
axes[1].grid(alpha=0.3)

# Viriosis
axes[2].bar(daily_data['Date'], daily_data['Virosis'], color='blue', alpha=0.7, width=0.8)
axes[2].set_ylabel('Viriosis Cases', fontsize=14, fontweight='bold', color='blue')
axes[2].tick_params(axis='y', labelcolor='blue')
axes[2].set_title('Daily Viriosis Incidence', fontsize=16, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/wind_disease_timeseries.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/wind_disease_timeseries.png")

# 4. Heatmap of wind speed vs disease by plot and date
pivot_wind = df.pivot_table(values='Wind_Speed', index='Plot_No', columns='Date', aggfunc='mean')
pivot_pebrine = df.pivot_table(values='Pebrine', index='Plot_No', columns='Date', aggfunc='sum')
pivot_virosis = df.pivot_table(values='Virosis', index='Plot_No', columns='Date', aggfunc='sum')

fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# Wind speed heatmap
sns.heatmap(pivot_wind, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[0], cbar_kws={'label': 'Wind Speed (m/s)'})
axes[0].set_title('Wind Speed by Plot and Date', fontsize=16, fontweight='bold')
axes[0].set_xlabel('')

# Pebrine heatmap
sns.heatmap(pivot_pebrine, annot=True, fmt='.0f', cmap='Reds', ax=axes[1], cbar_kws={'label': 'Pebrine Cases'})
axes[1].set_title('Pebrine Incidence by Plot and Date', fontsize=16, fontweight='bold')
axes[1].set_xlabel('')

# Viriosis heatmap
sns.heatmap(pivot_virosis, annot=True, fmt='.0f', cmap='Blues', ax=axes[2], cbar_kws={'label': 'Viriosis Cases'})
axes[2].set_title('Viriosis Incidence by Plot and Date', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/wind_disease_heatmap_by_plot_date.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/wind_disease_heatmap_by_plot_date.png")

# Daily-level correlation
print("\n" + "=" * 80)
print("DAILY-LEVEL CORRELATION ANALYSIS")
print("=" * 80)

r_daily_pb, p_daily_pb = pearsonr(daily_data['Wind_Speed'], daily_data['Pebrine'])
r_daily_vr, p_daily_vr = pearsonr(daily_data['Wind_Speed'], daily_data['Virosis'])

print(f"\nDaily Wind Speed vs Pebrine:")
print(f"  r = {r_daily_pb:.4f}, p = {p_daily_pb:.4f}")

print(f"\nDaily Wind Speed vs Viriosis:")
print(f"  r = {r_daily_vr:.4f}, p = {p_daily_vr:.4f}")

# Save correlation results
corr_results = pd.DataFrame([
    {'Level': 'Plot', 'Disease': 'Pebrine', 'Pearson_r': r_pb, 'P_value': p_pb, 
     'Spearman_rho': rho_pb, 'Spearman_p': p_rho_pb},
    {'Level': 'Plot', 'Disease': 'Viriosis', 'Pearson_r': r_vr, 'P_value': p_vr,
     'Spearman_rho': rho_vr, 'Spearman_p': p_rho_vr},
    {'Level': 'Daily', 'Disease': 'Pebrine', 'Pearson_r': r_daily_pb, 'P_value': p_daily_pb,
     'Spearman_rho': None, 'Spearman_p': None},
    {'Level': 'Daily', 'Disease': 'Viriosis', 'Pearson_r': r_daily_vr, 'P_value': p_daily_vr,
     'Spearman_rho': None, 'Spearman_p': None}
])

corr_results.to_csv('results/wind_disease_correlations.csv', index=False)
print("\nSaved: results/wind_disease_correlations.csv")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nWind Speed Statistics:")
print(df['Wind_Speed'].describe().round(3).to_string())

print("\nDisease Incidence Statistics:")
print(df[['Pebrine', 'Virosis']].describe().round(3).to_string())

# Key findings
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("\n1. PLOT-LEVEL CORRELATIONS:")
print(f"   Pebrine: r = {r_pb:.4f} (p = {p_pb:.4f}) {'*** SIGNIFICANT ***' if p_pb < 0.05 else '(not significant)'}")
print(f"   Viriosis: r = {r_vr:.4f} (p = {p_vr:.4f}) {'*** SIGNIFICANT ***' if p_vr < 0.05 else '(not significant)'}")

print("\n2. DAILY-LEVEL CORRELATIONS:")
print(f"   Pebrine: r = {r_daily_pb:.4f} (p = {p_daily_pb:.4f}) {'*** SIGNIFICANT ***' if p_daily_pb < 0.05 else '(not significant)'}")
print(f"   Viriosis: r = {r_daily_vr:.4f} (p = {p_daily_vr:.4f}) {'*** SIGNIFICANT ***' if p_daily_vr < 0.05 else '(not significant)'}")

print("\n3. WIND SPEED RANGE BY PLOT:")
for _, row in plot_data.iterrows():
    print(f"   Plot {int(row['Plot_No'])}: {row['Wind_Mean']:.2f} ± {row['Wind_Std']:.2f} m/s")

print("\n4. DISEASE INCIDENCE BY PLOT:")
for _, row in plot_data.iterrows():
    print(f"   Plot {int(row['Plot_No'])}: Pebrine={int(row['Pebrine_Total'])}, Viriosis={int(row['Virosis_Total'])}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  Data:")
print("    - results/wind_disease_plot_summary.csv")
print("    - results/wind_disease_correlations.csv")
print("\n  Figures (600 DPI):")
print("    - figures/wind_vs_disease_scatter_plots.png")
print("    - figures/wind_and_disease_by_plot.png")
print("    - figures/wind_disease_timeseries.png")
print("    - figures/wind_disease_heatmap_by_plot_date.png")
