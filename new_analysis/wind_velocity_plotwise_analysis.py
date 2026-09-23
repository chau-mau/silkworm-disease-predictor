"""
Wind VELOCITY vs Disease Incidence - PLOT-WISE Analysis
Using NASA POWER wind data + plot-level disease observations
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
print("WIND VELOCITY vs DISEASE INCIDENCE - PLOT-WISE ANALYSIS")
print("Ranchi, Jharkhand - October 7-31, 2025")
print("=" * 80)

# Load NASA wind velocity data (daily)
wind_df = pd.read_csv('data/nasa_wind_velocity_daily.csv')
wind_df['date'] = pd.to_datetime(wind_df['date'])

print("\nNASA Wind Velocity Data Loaded:")
print(f"  Period: {wind_df['date'].min()} to {wind_df['date'].max()}")

# Load original plot-wise disease data
df = pd.read_csv('../dmc-ctrti-2025.csv')
df = df[df['Plot_No'].notna()]
df['Plot_No'] = df['Plot_No'].astype(int)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

# Convert disease columns to numeric
for col in ['Pebrine', 'Virosis', 'Bacteriosis']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Filter to dates with wind data
df = df[df['Date'].isin(wind_df['date'])]

print(f"\nPlot-wise disease data: {len(df)} observations")
print(f"Plots: {sorted(df['Plot_No'].unique())}")

# Create plot-wise summary with wind velocity
plot_summary = []

for plot_no in sorted(df['Plot_No'].unique()):
    plot_data = df[df['Plot_No'] == plot_no]
    
    # Get dates for this plot
    plot_dates = plot_data['Date'].unique()
    
    # Get corresponding wind data
    plot_wind = wind_df[wind_df['date'].isin(plot_dates)]
    
    if len(plot_wind) == 0:
        continue
    
    # Calculate plot-wise wind velocity statistics
    summary = {
        'Plot_No': plot_no,
        'Days_Observed': len(plot_data),
        'Wind_Speed_Mean': plot_wind['wind_speed_ms_mean'].mean(),
        'Wind_Speed_Std': plot_wind['wind_speed_ms_mean'].std(),
        'Wind_Direction_Mean': plot_wind['wind_direction_deg_mean'].mean(),
        'Velocity_Magnitude_Mean': plot_wind['wind_speed_ms_mean'].mean(),
        'Resultant_Velocity_Mean': plot_wind['resultant_speed'].mean(),
        'Vx_Mean': plot_wind['wind_vx_mean'].mean(),
        'Vy_Mean': plot_wind['wind_vy_mean'].mean(),
        'Pebrine_Total': plot_data['Pebrine'].sum(),
        'Virosis_Total': plot_data['Virosis'].sum(),
        'Bacteriosis_Total': plot_data['Bacteriosis'].sum(),
        'Spacing': plot_data['Spacing'].iloc[0] if 'Spacing' in plot_data.columns else 'Unknown',
        'Instar': plot_data['Instar'].iloc[0] if 'Instar' in plot_data.columns else 'Unknown'
    }
    plot_summary.append(summary)

plot_df = pd.DataFrame(plot_summary)

print("\n" + "=" * 80)
print("PLOT-WISE SUMMARY")
print("=" * 80)
print(plot_df.round(3).to_string(index=False))

# Save plot summary
plot_df.to_csv('results/wind_velocity_plotwise_summary.csv', index=False)
print("\nSaved: results/wind_velocity_plotwise_summary.csv")

# Correlation analysis
print("\n" + "=" * 80)
print("PLOT-WISE CORRELATION ANALYSIS")
print("=" * 80)

diseases = ['Pebrine_Total', 'Virosis_Total', 'Bacteriosis_Total']
wind_vars = [
    ('Wind_Speed_Mean', 'Wind Speed (m/s)'),
    ('Velocity_Magnitude_Mean', 'Velocity Magnitude (m/s)'),
    ('Resultant_Velocity_Mean', 'Resultant Velocity (m/s)'),
    ('Vx_Mean', 'East-West Component Vx (m/s)'),
    ('Vy_Mean', 'North-South Component Vy (m/s)'),
    ('Wind_Direction_Mean', 'Wind Direction (degrees)')
]

correlation_results = []

for disease in diseases:
    if plot_df[disease].sum() == 0:
        print(f"\n{disease.replace('_Total', '')}: No cases recorded")
        continue
    
    print(f"\n{'='*60}")
    print(f"{disease.replace('_Total', '').upper()}")
    print(f"{'='*60}")
    
    for var_code, var_name in wind_vars:
        r, p = pearsonr(plot_df[var_code], plot_df[disease])
        rho, p_rho = spearmanr(plot_df[var_code], plot_df[disease])
        
        correlation_results.append({
            'Disease': disease.replace('_Total', ''),
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
corr_df.to_csv('results/wind_velocity_plotwise_correlations.csv', index=False)
print("\nSaved: results/wind_velocity_plotwise_correlations.csv")

# Generate visualizations
print("\n" + "=" * 80)
print("GENERATING PLOT-WISE VISUALIZATIONS")
print("=" * 80)

# 1. Plot-wise scatter: Wind Velocity vs Viriosis (only disease with data)
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']

for idx, (var_code, var_name) in enumerate(wind_vars):
    ax = axes[idx]
    
    # Scatter plot with plot numbers
    scatter = ax.scatter(plot_df[var_code], plot_df['Virosis_Total'], 
                        s=300, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add plot number labels
    for i, row in plot_df.iterrows():
        ax.annotate(f"Plot {int(row['Plot_No'])}", 
                   (row[var_code], row['Virosis_Total']),
                   xytext=(8, 8), textcoords='offset points', 
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add trend line
    if len(plot_df) > 2:
        z = np.polyfit(plot_df[var_code], plot_df['Virosis_Total'], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(plot_df[var_code].min(), plot_df[var_code].max(), 100)
        ax.plot(x_line, p_line(x_line), 'r--', linewidth=2)
    
    # Calculate correlation
    r, p = pearsonr(plot_df[var_code], plot_df['Virosis_Total'])
    
    ax.set_xlabel(var_name, fontsize=13, fontweight='bold')
    ax.set_ylabel('Viriosis Cases (Total)', fontsize=13, fontweight='bold', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_title(f'{var_name}\nr = {r:.3f}, p = {p:.4f}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Plot-wise Wind Velocity vs Viriosis Incidence', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/plotwise_wind_velocity_vs_viriosis.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/plotwise_wind_velocity_vs_viriosis.png")

# 2. Bar chart: Wind velocity and disease by plot
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

x = np.arange(len(plot_df))
width = 0.6

# Wind velocity by plot
ax = axes[0]
bars1 = ax.bar(x, plot_df['Velocity_Magnitude_Mean'], width, 
               label='Mean Wind Velocity', color='skyblue', edgecolor='black', alpha=0.8)
ax.errorbar(x, plot_df['Velocity_Magnitude_Mean'], yerr=plot_df['Wind_Speed_Std'], 
            fmt='none', color='black', capsize=5, capthick=2)
ax.set_ylabel('Wind Velocity (m/s)', fontsize=16, fontweight='bold', color='steelblue')
ax.tick_params(axis='y', labelcolor='steelblue')
ax.set_title('Mean Wind Velocity by Plot', fontsize=18, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"Plot {int(p)}" for p in plot_df['Plot_No']], rotation=45)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, plot_df['Velocity_Magnitude_Mean'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Disease by plot
ax2 = axes[1]
bars2 = ax2.bar(x, plot_df['Virosis_Total'], width, 
                label='Viriosis Cases', color='blue', edgecolor='black', alpha=0.7)
ax2.set_ylabel('Viriosis Cases (Total)', fontsize=16, fontweight='bold', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_title('Viriosis Incidence by Plot', fontsize=18, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f"Plot {int(p)}" for p in plot_df['Plot_No']], rotation=45)
ax2.set_xlabel('Plot Number', fontsize=16, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, plot_df['Virosis_Total'])):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/plotwise_wind_and_disease_bars.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/plotwise_wind_and_disease_bars.png")

# 3. Wind direction analysis by plot
fig, ax = plt.subplots(figsize=(14, 10))

# Create wind direction plot
for i, row in plot_df.iterrows():
    # Convert direction to radians
    theta = np.radians(row['Wind_Direction_Mean'])
    r = row['Velocity_Magnitude_Mean']
    
    # Plot arrow
    ax.annotate('', xy=(theta, r), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors[i], lw=3))
    
    # Add plot label
    ax.text(theta, r + 0.3, f"Plot {int(row['Plot_No'])}", 
            fontsize=12, fontweight='bold', ha='center')

# Create polar plot
ax = plt.subplot(111, projection='polar')

# Plot each plot's mean wind
for i, row in plot_df.iterrows():
    theta = np.radians(row['Wind_Direction_Mean'])
    r = row['Velocity_Magnitude_Mean']
    size = row['Virosis_Total'] * 50 + 100  # Size based on disease
    
    ax.scatter(theta, r, s=size, c=colors[i], alpha=0.7, 
               edgecolors='black', linewidth=2, label=f"Plot {int(row['Plot_No'])}")

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_title('Plot-wise Wind Direction and Velocity\n(Bubble size = Viriosis cases)', 
             fontsize=18, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig('figures/plotwise_wind_direction_polar.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/plotwise_wind_direction_polar.png")

# 4. Comprehensive plot-wise comparison
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Wind speed vs Viriosis
ax = axes[0, 0]
ax.scatter(plot_df['Wind_Speed_Mean'], plot_df['Virosis_Total'], 
           s=400, c='green', alpha=0.6, edgecolors='black', linewidth=2)
for i, row in plot_df.iterrows():
    ax.annotate(f"P{int(row['Plot_No'])}", 
               (row['Wind_Speed_Mean'], row['Virosis_Total']),
               xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
r, p = pearsonr(plot_df['Wind_Speed_Mean'], plot_df['Virosis_Total'])
ax.set_xlabel('Mean Wind Speed (m/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Viriosis Cases', fontsize=14, fontweight='bold', color='blue')
ax.set_title(f'Wind Speed vs Viriosis\nr = {r:.3f}, p = {p:.4f}', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)

# Vx vs Viriosis
ax = axes[0, 1]
ax.scatter(plot_df['Vx_Mean'], plot_df['Virosis_Total'], 
           s=400, c='red', alpha=0.6, edgecolors='black', linewidth=2)
for i, row in plot_df.iterrows():
    ax.annotate(f"P{int(row['Plot_No'])}", 
               (row['Vx_Mean'], row['Virosis_Total']),
               xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
r, p = pearsonr(plot_df['Vx_Mean'], plot_df['Virosis_Total'])
ax.set_xlabel('East-West Component Vx (m/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Viriosis Cases', fontsize=14, fontweight='bold', color='blue')
ax.set_title(f'Vx vs Viriosis\nr = {r:.3f}, p = {p:.4f}', fontsize=16, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)

# Vy vs Viriosis
ax = axes[1, 0]
ax.scatter(plot_df['Vy_Mean'], plot_df['Virosis_Total'], 
           s=400, c='purple', alpha=0.6, edgecolors='black', linewidth=2)
for i, row in plot_df.iterrows():
    ax.annotate(f"P{int(row['Plot_No'])}", 
               (row['Vy_Mean'], row['Virosis_Total']),
               xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
r, p = pearsonr(plot_df['Vy_Mean'], plot_df['Virosis_Total'])
ax.set_xlabel('North-South Component Vy (m/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Viriosis Cases', fontsize=14, fontweight='bold', color='blue')
ax.set_title(f'Vy vs Viriosis\nr = {r:.3f}, p = {p:.4f}', fontsize=16, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)

# Direction vs Viriosis
ax = axes[1, 1]
ax.scatter(plot_df['Wind_Direction_Mean'], plot_df['Virosis_Total'], 
           s=400, c='orange', alpha=0.6, edgecolors='black', linewidth=2)
for i, row in plot_df.iterrows():
    ax.annotate(f"P{int(row['Plot_No'])}", 
               (row['Wind_Direction_Mean'], row['Virosis_Total']),
               xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
r, p = pearsonr(plot_df['Wind_Direction_Mean'], plot_df['Virosis_Total'])
ax.set_xlabel('Wind Direction (degrees)', fontsize=14, fontweight='bold')
ax.set_ylabel('Viriosis Cases', fontsize=14, fontweight='bold', color='blue')
ax.set_title(f'Direction vs Viriosis\nr = {r:.3f}, p = {p:.4f}', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)

plt.suptitle('Plot-wise Wind Velocity Components vs Viriosis', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/plotwise_velocity_components_analysis.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/plotwise_velocity_components_analysis.png")

# Print final summary
print("\n" + "=" * 80)
print("PLOT-WISE INTERPRETATION SUMMARY")
print("=" * 80)

print("\nPlot Characteristics:")
print("-" * 80)
for _, row in plot_df.iterrows():
    print(f"\nPlot {int(row['Plot_No'])}:")
    print(f"  Wind: {row['Velocity_Magnitude_Mean']:.2f} m/s from {row['Wind_Direction_Mean']:.1f}°")
    print(f"  Vx (East+): {row['Vx_Mean']:.2f}, Vy (North+): {row['Vy_Mean']:.2f}")
    print(f"  Viriosis: {int(row['Virosis_Total'])} cases")
    if row['Virosis_Total'] > 0:
        print(f"  *** DISEASE PRESENT ***")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Find strongest correlations
virosis_corr = corr_df[corr_df['Disease'] == 'Virosis'].sort_values('Pearson_r', key=abs, ascending=False)

print("\nStrongest correlations with Viriosis (Plot-wise):")
for _, row in virosis_corr.head(3).iterrows():
    sig = "*** SIGNIFICANT ***" if row['Pearson_p'] < 0.05 else "(not significant)"
    print(f"  {row['Variable']}: r = {row['Pearson_r']:.4f} {sig}")

print("\nPlot-wise Patterns:")
disease_plots = plot_df[plot_df['Virosis_Total'] > 0]
no_disease_plots = plot_df[plot_df['Virosis_Total'] == 0]

if len(disease_plots) > 0:
    print(f"\n  Plots WITH disease (n={len(disease_plots)}):")
    print(f"    Mean wind speed: {disease_plots['Wind_Speed_Mean'].mean():.2f} m/s")
    print(f"    Mean direction: {disease_plots['Wind_Direction_Mean'].mean():.1f}°")
    print(f"    Mean Vx: {disease_plots['Vx_Mean'].mean():.2f} m/s")
    print(f"    Mean Vy: {disease_plots['Vy_Mean'].mean():.2f} m/s")

if len(no_disease_plots) > 0:
    print(f"\n  Plots WITHOUT disease (n={len(no_disease_plots)}):")
    print(f"    Mean wind speed: {no_disease_plots['Wind_Speed_Mean'].mean():.2f} m/s")
    print(f"    Mean direction: {no_disease_plots['Wind_Direction_Mean'].mean():.1f}°")
    print(f"    Mean Vx: {no_disease_plots['Vx_Mean'].mean():.2f} m/s")
    print(f"    Mean Vy: {no_disease_plots['Vy_Mean'].mean():.2f} m/s")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  - results/wind_velocity_plotwise_summary.csv")
print("  - results/wind_velocity_plotwise_correlations.csv")
print("  - figures/plotwise_wind_velocity_vs_viriosis.png")
print("  - figures/plotwise_wind_and_disease_bars.png")
print("  - figures/plotwise_wind_direction_polar.png")
print("  - figures/plotwise_velocity_components_analysis.png")
