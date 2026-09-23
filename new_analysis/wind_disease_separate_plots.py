"""
Wind Velocity vs Disease Incidence - Separate Plots for Each Disease
Clear, individual visualizations for Pebrine, Viriosis, and Bacteriosis
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
print("WIND VELOCITY vs DISEASE INCIDENCE - SEPARATE ANALYSES")
print("=" * 80)

# Load original data with wind speed
df = pd.read_csv('../dmc-ctrti-2025.csv')

# Clean and prepare data
df = df[df['Plot_No'].notna()]
df['Plot_No'] = df['Plot_No'].astype(int)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

# Convert disease columns to numeric
for col in ['Pebrine', 'Virosis', 'Bacteriosis', 'Muscardine']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Convert Wind_Speed to numeric
df['Wind_Speed'] = pd.to_numeric(df['Wind_Speed'], errors='coerce')

# Remove rows with missing wind speed
df = df[df['Wind_Speed'].notna()]

print(f"\nData loaded: {len(df)} observations")
print(f"Plots: {sorted(df['Plot_No'].unique())}")

# Create plot-wise summary
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

plot_data = plot_summary.reset_index()

# Define colors for each disease
disease_colors = {
    'Pebrine': '#e74c3c',      # Red
    'Viriosis': '#3498db',     # Blue  
    'Bacteriosis': '#f39c12'   # Orange
}

# Function to create individual disease plot
def create_disease_wind_plot(disease_col, disease_name, color):
    """Create a separate, clear visualization for one disease vs wind speed"""
    
    # Check if there's variance in disease data
    disease_values = plot_data[disease_col]
    
    if disease_values.max() == 0:
        print(f"\n{disease_name}: No cases recorded - cannot calculate correlation")
        return None
    
    # Calculate correlation
    r, p = pearsonr(plot_data['Wind_Mean'], plot_data[disease_col])
    rho, p_rho = spearmanr(plot_data['Wind_Mean'], plot_data[disease_col])
    
    print(f"\n{disease_name}:")
    print(f"  Pearson r = {r:.4f}, p = {p:.4f}")
    print(f"  Spearman rho = {rho:.4f}, p = {p_rho:.4f}")
    
    # Create figure with two subplots (scatter + bar)
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.3)
    
    # Top subplot: Scatter plot with trend line
    ax1 = fig.add_subplot(gs[0])
    
    # Create bubble sizes based on observation days
    sizes = plot_data['Days_Observed'] * 50
    
    # Scatter plot
    scatter = ax1.scatter(plot_data['Wind_Mean'], plot_data[disease_col], 
                         s=sizes, c=color, alpha=0.7, edgecolors='black', 
                         linewidth=2, label=f'{disease_name} Cases')
    
    # Add plot number labels
    for i, row in plot_data.iterrows():
        ax1.annotate(f"Plot {int(row['Plot_No'])}", 
                     (row['Wind_Mean'], row[disease_col]),
                     xytext=(10, 10), textcoords='offset points', 
                     fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                              edgecolor=color, alpha=0.8))
    
    # Add trend line if enough variance
    if len(plot_data) > 2 and disease_values.var() > 0:
        z = np.polyfit(plot_data['Wind_Mean'], plot_data[disease_col], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(plot_data['Wind_Mean'].min(), plot_data['Wind_Mean'].max(), 100)
        ax1.plot(x_line, p_line(x_line), '--', color='darkred', linewidth=3, 
                label=f'Trend Line (r={r:.3f})')
    
    # Formatting
    ax1.set_xlabel('Mean Wind Speed (m/s)', fontsize=18, fontweight='bold')
    ax1.set_ylabel(f'{disease_name} Cases (Total)', fontsize=18, fontweight='bold', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(f'Wind Speed vs {disease_name} Incidence by Plot\n' + 
                  f'Pearson r = {r:.4f}, p = {p:.4f} | Spearman rho = {rho:.4f}', 
                  fontsize=20, fontweight='bold', pad=20)
    ax1.legend(fontsize=14, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add significance indicator
    if p < 0.05:
        sig_text = '*** SIGNIFICANT (p < 0.05) ***'
        ax1.text(0.5, 0.95, sig_text, transform=ax1.transAxes, fontsize=16, 
                fontweight='bold', color='green', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        sig_text = 'Not Significant (p ≥ 0.05)'
        ax1.text(0.5, 0.95, sig_text, transform=ax1.transAxes, fontsize=16, 
                color='gray', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Bottom subplot: Side-by-side bar comparison
    ax2 = fig.add_subplot(gs[1])
    
    x = np.arange(len(plot_data))
    width = 0.35
    
    # Wind speed bars
    bars1 = ax2.bar(x - width/2, plot_data['Wind_Mean'], width, 
                    label='Mean Wind Speed (m/s)', color='skyblue', 
                    edgecolor='black', alpha=0.8)
    ax2.errorbar(x - width/2, plot_data['Wind_Mean'], yerr=plot_data['Wind_Std'], 
                 fmt='none', color='black', capsize=5, capthick=2)
    
    # Disease bars on secondary axis
    ax2b = ax2.twinx()
    bars2 = ax2b.bar(x + width/2, plot_data[disease_col], width, 
                     label=f'{disease_name} Cases', color=color, 
                     edgecolor='black', alpha=0.8)
    
    # Formatting
    ax2.set_xlabel('Plot Number', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Wind Speed (m/s)', fontsize=16, fontweight='bold', color='steelblue')
    ax2b.set_ylabel(f'{disease_name} Cases', fontsize=16, fontweight='bold', color=color)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Plot {int(p)}" for p in plot_data['Plot_No']], rotation=45, ha='right')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2b.tick_params(axis='y', labelcolor=color)
    ax2.set_title(f'Comparison: Wind Speed and {disease_name} by Plot', 
                  fontsize=18, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
    
    # Save figure
    filename = f'figures/wind_vs_{disease_name.lower()}_plotwise.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")
    
    return {'disease': disease_name, 'r': r, 'p': p, 'rho': rho, 'p_rho': p_rho}

# Create separate plots for each disease
print("\n" + "=" * 80)
print("GENERATING SEPARATE PLOTS FOR EACH DISEASE")
print("=" * 80)

results = []

# Pebrine
result_pb = create_disease_wind_plot('Pebrine_Total', 'Pebrine', disease_colors['Pebrine'])
if result_pb:
    results.append(result_pb)

# Viriosis
result_vr = create_disease_wind_plot('Virosis_Total', 'Viriosis', disease_colors['Viriosis'])
if result_vr:
    results.append(result_vr)

# Bacteriosis
result_bt = create_disease_wind_plot('Bacteriosis_Total', 'Bacteriosis', disease_colors['Bacteriosis'])
if result_bt:
    results.append(result_bt)

# Create summary comparison plot
print("\n" + "=" * 80)
print("GENERATING SUMMARY COMPARISON PLOT")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

diseases_to_plot = [
    ('Pebrine_Total', 'Pebrine', disease_colors['Pebrine']),
    ('Virosis_Total', 'Viriosis', disease_colors['Viriosis']),
    ('Bacteriosis_Total', 'Bacteriosis', disease_colors['Bacteriosis'])
]

for idx, (col, name, color) in enumerate(diseases_to_plot):
    ax = axes[idx]
    
    # Check if data has variance
    if plot_data[col].max() == 0:
        ax.text(0.5, 0.5, f'{name}\nNo Cases Recorded', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=16, fontweight='bold', color='gray')
        ax.set_title(name, fontsize=18, fontweight='bold')
        continue
    
    # Calculate correlation
    r, p = pearsonr(plot_data['Wind_Mean'], plot_data[col])
    
    # Scatter with trend line
    ax.scatter(plot_data['Wind_Mean'], plot_data[col], 
               s=300, c=color, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add plot labels
    for i, row in plot_data.iterrows():
        ax.annotate(f"{int(row['Plot_No'])}", 
                   (row['Wind_Mean'], row[col]),
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=11, fontweight='bold')
    
    # Trend line
    z = np.polyfit(plot_data['Wind_Mean'], plot_data[col], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(plot_data['Wind_Mean'].min(), plot_data['Wind_Mean'].max(), 100)
    ax.plot(x_line, p_line(x_line), '--', color='darkred', linewidth=2)
    
    # Formatting
    ax.set_xlabel('Mean Wind Speed (m/s)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{name} Cases', fontsize=14, fontweight='bold', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title(f'{name}\nr = {r:.3f}, p = {p:.4f}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Wind Speed vs Disease Incidence - Summary Comparison (Plot-wise)', 
             fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/wind_vs_all_diseases_summary.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/wind_vs_all_diseases_summary.png")

# Save results to CSV
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/wind_disease_correlations_detailed.csv', index=False)
    print("\nSaved: results/wind_disease_correlations_detailed.csv")

# Print final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\nPlot-wise Wind Speed Statistics:")
print(plot_data[['Plot_No', 'Wind_Mean', 'Wind_Std', 'Days_Observed']].to_string(index=False))

print("\nDisease Incidence by Plot:")
print(plot_data[['Plot_No', 'Pebrine_Total', 'Virosis_Total', 'Bacteriosis_Total']].to_string(index=False))

print("\nCorrelation Summary:")
for r in results:
    sig = "*** SIGNIFICANT ***" if r['p'] < 0.05 else "Not significant"
    print(f"\n{r['disease']}:")
    print(f"  Pearson r = {r['r']:.4f} (p = {r['p']:.4f}) - {sig}")
    print(f"  Spearman rho = {r['rho']:.4f} (p = {r['p_rho']:.4f})")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  - figures/wind_vs_pebrine_plotwise.png")
print("  - figures/wind_vs_viriosis_plotwise.png")
print("  - figures/wind_vs_bacteriosis_plotwise.png")
print("  - figures/wind_vs_all_diseases_summary.png")
print("  - results/wind_disease_correlations_detailed.csv")
