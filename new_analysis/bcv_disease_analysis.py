"""
BCV 1-19 and Disease Analysis for 2025 Data
Combines Bioclimatic Variables with Disease Occurrence Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
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
print("BCV 1-19 AND DISEASE ANALYSIS FOR 2025 DATA")
print("=" * 80)

# Load data
print("\nLoading datasets...")
bcv_df = pd.read_csv('data/bioclimatic_variables.csv')
disease_df = pd.read_csv('results/merged_disease_weather_data_2025.csv')

print(f"BCV data shape: {bcv_df.shape}")
print(f"Disease data shape: {disease_df.shape}")

# Parse dates
bcv_df['Date'] = pd.to_datetime(bcv_df['Date'])
disease_df['Date'] = pd.to_datetime(disease_df['Date'])

# Aggregate BCV data by date (mean across all plots)
bcv_daily = bcv_df.groupby('Date').agg({
    'BIO1': 'mean', 'BIO2': 'mean', 'BIO3': 'mean', 'BIO4': 'mean', 'BIO5': 'mean',
    'BIO6': 'mean', 'BIO7': 'mean', 'BIO8': 'mean', 'BIO9': 'mean', 'BIO10': 'mean',
    'BIO11': 'mean', 'BIO12': 'mean', 'BIO13': 'mean', 'BIO14': 'mean', 'BIO15': 'mean',
    'BIO16': 'mean', 'BIO17': 'mean', 'BIO18': 'mean', 'BIO19': 'mean'
}).reset_index()

print(f"Daily aggregated BCV shape: {bcv_daily.shape}")

# Merge datasets
merged_df = pd.merge(disease_df, bcv_daily, on='Date', how='inner')
print(f"Merged dataset shape: {merged_df.shape}")

# BCV 1-19 descriptions
bcv_descriptions = {
    'BIO1': 'Annual Mean Temperature',
    'BIO2': 'Mean Diurnal Range',
    'BIO3': 'Isothermality',
    'BIO4': 'Temperature Seasonality',
    'BIO5': 'Max Temperature of Warmest Month',
    'BIO6': 'Min Temperature of Coldest Month',
    'BIO7': 'Annual Temperature Range',
    'BIO8': 'Mean Temperature of Wettest Quarter',
    'BIO9': 'Mean Temperature of Driest Quarter',
    'BIO10': 'Mean Temperature of Warmest Quarter',
    'BIO11': 'Mean Temperature of Coldest Quarter',
    'BIO12': 'Annual Precipitation',
    'BIO13': 'Precipitation of Wettest Month',
    'BIO14': 'Precipitation of Driest Month',
    'BIO15': 'Precipitation Seasonality',
    'BIO16': 'Precipitation of Wettest Quarter',
    'BIO17': 'Precipitation of Driest Quarter',
    'BIO18': 'Precipitation of Warmest Quarter',
    'BIO19': 'Precipitation of Coldest Quarter'
}

# Diseases to analyze
diseases = ['PB_binary', 'VR_binary', 'BT_binary']
disease_names = {'PB_binary': 'Pebrine', 'VR_binary': 'Viriosis', 'BT_binary': 'Bacteriosis'}

# BCV columns
bcv_cols = [f'BIO{i}' for i in range(1, 20)]

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS: BCV vs DISEASES")
print("=" * 80)

# Calculate correlations
correlation_results = []

for disease in diseases:
    print(f"\n--- {disease_names[disease]} ---")
    for bcv in bcv_cols:
        # Point-biserial correlation (Pearson for binary vs continuous)
        r, p = pearsonr(merged_df[disease], merged_df[bcv])
        correlation_results.append({
            'Disease': disease_names[disease],
            'BCV': bcv,
            'Description': bcv_descriptions[bcv],
            'Correlation': r,
            'P_value': p,
            'Significant': 'Yes' if p < 0.05 else 'No'
        })
        
        if p < 0.05:
            print(f"  {bcv}: r = {r:.4f}, p = {p:.4f} *")

# Convert to DataFrame
corr_df = pd.DataFrame(correlation_results)

# Create correlation matrix for visualization
corr_matrix = corr_df.pivot(index='BCV', columns='Disease', values='Correlation')

print("\n" + "=" * 80)
print("TOP CORRELATIONS (|r| > 0.3)")
print("=" * 80)

significant = corr_df[corr_df['Significant'] == 'Yes'].sort_values('P_value')
if len(significant) > 0:
    print(significant.to_string(index=False))
else:
    print("No statistically significant correlations found (p < 0.05)")

# Top absolute correlations
print("\n--- Top 10 Correlations by Absolute Value ---")
top_corr = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index).head(10)
print(top_corr[['Disease', 'BCV', 'Correlation', 'P_value']].to_string(index=False))

# Save correlation results
corr_df.to_csv('results/bcv_disease_correlations.csv', index=False)
print("\nSaved: results/bcv_disease_correlations.csv")

# Create visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. Correlation Heatmap
fig, ax = plt.subplots(figsize=(14, 20))
mask = np.abs(corr_matrix.values) < 0.1  # Mask very weak correlations
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            vmin=-1, vmax=1, fmt='.3f', linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
            ax=ax)
ax.set_title('BCV 1-19 vs Silkworm Diseases Correlation Matrix\n(2025 Data)', 
             fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('Disease', fontsize=16, fontweight='bold')
ax.set_ylabel('Bioclimatic Variables (BIO1-19)', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/bcv_disease_correlation_heatmap.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/bcv_disease_correlation_heatmap.png")

# 2. Bar plot of top correlations
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for idx, disease in enumerate(diseases):
    disease_corr = corr_df[corr_df['Disease'] == disease_names[disease]].sort_values('Correlation')
    colors = ['red' if x < 0 else 'blue' for x in disease_corr['Correlation']]
    
    axes[idx].barh(range(len(disease_corr)), disease_corr['Correlation'], color=colors, alpha=0.7)
    axes[idx].set_yticks(range(len(disease_corr)))
    axes[idx].set_yticklabels(disease_corr['BCV'], fontsize=10)
    axes[idx].set_xlabel('Correlation Coefficient', fontsize=14, fontweight='bold')
    axes[idx].set_title(f'{disease_names[disease]}', fontsize=16, fontweight='bold')
    axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[idx].set_xlim(-1, 1)
    axes[idx].grid(axis='x', alpha=0.3)

plt.suptitle('BCV Correlations with Silkworm Diseases (2025)', fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/bcv_disease_correlation_bars.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/bcv_disease_correlation_bars.png")

# 3. Time series of key BCV variables with disease overlay
key_bcv = ['BIO1', 'BIO5', 'BIO12', 'BIO15']  # Temperature and precipitation related

fig, axes = plt.subplots(4, 1, figsize=(16, 20))

for idx, bcv in enumerate(key_bcv):
    ax = axes[idx]
    ax2 = ax.twinx()
    
    # Plot BCV
    ax.plot(merged_df['Date'], merged_df[bcv], 'b-', linewidth=2, label=bcv)
    ax.set_ylabel(f'{bcv}\n({bcv_descriptions[bcv]})', fontsize=12, fontweight='bold', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    # Plot disease occurrence (sum of all diseases)
    disease_sum = merged_df['PB_binary'] + merged_df['VR_binary'] + merged_df['BT_binary']
    ax2.fill_between(merged_df['Date'], 0, disease_sum, alpha=0.3, color='red', label='Disease Count')
    ax2.set_ylabel('Disease Count', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 3.5)
    
    ax.set_title(f'{bcv}: {bcv_descriptions[bcv]}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Key BCV Variables and Disease Occurrence Over Time (2025)', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/bcv_timeseries_disease.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/bcv_timeseries_disease.png")

# 4. Box plots: BCV values when disease present vs absent
fig, axes = plt.subplots(3, 6, figsize=(24, 18))
axes = axes.flatten()

for idx, bcv in enumerate(bcv_cols):
    if idx < len(axes):
        ax = axes[idx]
        
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        
        for disease in diseases:
            present = merged_df[merged_df[disease] == 1][bcv]
            absent = merged_df[merged_df[disease] == 0][bcv]
            data_to_plot.extend([present, absent])
            labels.extend([f'{disease_names[disease]}\nPresent', f'{disease_names[disease]}\nAbsent'])
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = ['lightcoral', 'lightblue'] * 3
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{bcv}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)

plt.suptitle('BCV Distribution: Disease Present vs Absent (2025)', fontsize=20, fontweight='bold', y=0.999)
plt.tight_layout()
plt.savefig('figures/bcv_boxplots_disease.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: figures/bcv_boxplots_disease.png")

# Statistical summary
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)

summary_stats = []
for bcv in bcv_cols:
    stats_dict = {'BCV': bcv, 'Description': bcv_descriptions[bcv]}
    stats_dict.update(merged_df[bcv].describe().to_dict())
    summary_stats.append(stats_dict)

summary_df = pd.DataFrame(summary_stats)
print("\nBCV Summary Statistics:")
print(summary_df[['BCV', 'mean', 'std', 'min', 'max']].to_string(index=False))

summary_df.to_csv('results/bcv_summary_statistics.csv', index=False)
print("\nSaved: results/bcv_summary_statistics.csv")

# Disease occurrence summary
print("\n" + "=" * 80)
print("DISEASE OCCURRENCE SUMMARY")
print("=" * 80)

for disease in diseases:
    name = disease_names[disease]
    present = merged_df[disease].sum()
    total = len(merged_df)
    pct = (present / total) * 100
    print(f"{name}: {present}/{total} days ({pct:.1f}%)")

# Key findings
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("\n1. STRONGEST CORRELATIONS:")
for disease in diseases:
    disease_data = corr_df[corr_df['Disease'] == disease_names[disease]]
    strongest = disease_data.loc[disease_data['Correlation'].abs().idxmax()]
    print(f"   {disease_names[disease]}: {strongest['BCV']} (r = {strongest['Correlation']:.4f})")

print("\n2. SIGNIFICANT CORRELATIONS (p < 0.05):")
sig_corr = corr_df[corr_df['Significant'] == 'Yes']
if len(sig_corr) > 0:
    for _, row in sig_corr.iterrows():
        print(f"   {row['Disease']} - {row['BCV']}: r = {row['Correlation']:.4f}, p = {row['P_value']:.4f}")
else:
    print("   No statistically significant correlations found")

print("\n3. BCV PATTERNS:")
print(f"   - Highest variability: {summary_df.loc[summary_df['std'].idxmax(), 'BCV']}")
print(f"   - Most stable: {summary_df.loc[summary_df['std'].idxmin(), 'BCV']}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  Data:")
print("    - results/bcv_disease_correlations.csv")
print("    - results/bcv_summary_statistics.csv")
print("\n  Figures (600 DPI):")
print("    - figures/bcv_disease_correlation_heatmap.png")
print("    - figures/bcv_disease_correlation_bars.png")
print("    - figures/bcv_timeseries_disease.png")
print("    - figures/bcv_boxplots_disease.png")
