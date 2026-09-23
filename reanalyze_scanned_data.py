"""
Reanalyze the scanned data with proper THI calculation
Create Excel sheets and updated visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pointbiserialr
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("REANALYSIS OF SCANNED DATA")
print("="*80)

# Load the compiled data
df = pd.read_csv('results/scanned_data_compiled.csv')
df['Date_parsed'] = pd.to_datetime(df['Date_parsed'])

print(f"\nData loaded: {len(df)} records")
print(f"Plots: {sorted(df['Plot_No'].unique())}")

# Calculate proper THI using NRC formula
def calculate_thi_nrc(temp_c, rh):
    """NRC (1971) formula for THI"""
    temp_f = (1.8 * temp_c) + 32
    thi = temp_f - ((0.55 - 0.0055 * rh) * (temp_f - 58))
    return thi

# Calculate mean temperature
df['Temp_Mean'] = (df['Tmax'] + df['Tmin']) / 2

# Recalculate THI
df['THI_NRC'] = df.apply(lambda row: calculate_thi_nrc(row['Temp_Mean'], row['Humidity']), axis=1)

# Classify THI stress
def classify_thi_stress(thi):
    if pd.isna(thi):
        return "Unknown"
    elif thi < 70:
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

# Clean disease data - convert to numeric
disease_cols = ['Pebrine', 'Virosis', 'Bacteriosis', 'Muscardine']
for col in disease_cols:
    # Convert X or any non-zero to 1, missing or 0 to 0
    df[col] = df[col].apply(lambda x: 1 if pd.notna(x) and str(x).strip() not in ['', '0', '0000', '00000'] else 0)

print("\nTHI Statistics (NRC Formula):")
print(f"  Mean: {df['THI_NRC'].mean():.2f}")
print(f"  Min: {df['THI_NRC'].min():.2f}")
print(f"  Max: {df['THI_NRC'].max():.2f}")
print(f"  Std: {df['THI_NRC'].std():.2f}")

print("\nStress Level Distribution:")
print(df['Stress_Level'].value_counts())

# ============================================================================
# EXCEL SHEET 1: Raw Data
# ============================================================================
print("\nCreating Excel sheets...")

with pd.ExcelWriter('results/Scanned_Data_Analysis_2025.xlsx', engine='openpyxl') as writer:
    
    # Sheet 1: Raw Data
    df.to_excel(writer, sheet_name='Raw_Data', index=False)
    
    # Sheet 2: Daily Summary
    daily_summary = df.groupby('Date_parsed').agg({
        'Plot_No': 'count',
        'Temp_Mean': 'mean',
        'Humidity': 'mean',
        'THI_NRC': 'mean',
        'Pebrine': 'sum',
        'Virosis': 'sum',
        'Bacteriosis': 'sum',
        'Muscardine': 'sum'
    }).round(2)
    daily_summary.columns = ['Observations', 'Avg_Temp', 'Avg_Humidity', 'Avg_THI', 
                             'Pebrine', 'Virosis', 'Bacteriosis', 'Muscardine']
    daily_summary.to_excel(writer, sheet_name='Daily_Summary')
    
    # Sheet 3: Plot Summary
    plot_summary = df.groupby('Plot_No').agg({
        'Date_parsed': 'count',
        'Spacing': 'first',
        'Instar': 'first',
        'Temp_Mean': 'mean',
        'Humidity': 'mean',
        'THI_NRC': 'mean',
        'Pebrine': 'sum',
        'Virosis': 'sum',
        'Bacteriosis': 'sum',
        'Muscardine': 'sum'
    }).round(2)
    plot_summary.columns = ['Observations', 'Spacing', 'Instar', 'Avg_Temp', 
                           'Avg_Humidity', 'Avg_THI', 'Pebrine', 'Virosis', 
                           'Bacteriosis', 'Muscardine']
    plot_summary.to_excel(writer, sheet_name='Plot_Summary')
    
    # Sheet 4: Spacing Analysis
    spacing_summary = df.groupby('Spacing').agg({
        'Plot_No': 'count',
        'THI_NRC': 'mean',
        'Pebrine': 'sum',
        'Virosis': 'sum',
        'Bacteriosis': 'sum',
        'Muscardine': 'sum'
    }).round(2)
    spacing_summary.to_excel(writer, sheet_name='Spacing_Analysis')
    
    # Sheet 5: Correlation Analysis
    corr_data = []
    for disease in disease_cols:
        valid_data = df[[disease, 'THI_NRC']].dropna()
        if len(valid_data) > 3:
            corr, p_val = pointbiserialr(valid_data[disease], valid_data['THI_NRC'])
            corr_data.append({
                'Disease': disease,
                'Correlation_with_THI': round(corr, 4),
                'P_value': round(p_val, 6),
                'Significant': 'Yes' if p_val < 0.05 else 'No',
                'N': len(valid_data)
            })
    
    corr_df = pd.DataFrame(corr_data)
    corr_df.to_excel(writer, sheet_name='Correlation_Analysis', index=False)

print("Saved: results/Scanned_Data_Analysis_2025.xlsx")

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

# Correlation analysis
print("\nDisease-THI Correlations:")
for disease in disease_cols:
    valid_data = df[[disease, 'THI_NRC']].dropna()
    if len(valid_data) > 3:
        corr, p_val = pointbiserialr(valid_data[disease], valid_data['THI_NRC'])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {disease}: r = {corr:.4f}, p = {p_val:.6f} {sig}")

# ANOVA for spacing effect
print("\nSpacing Effect (ANOVA):")
for disease in disease_cols:
    groups = [group[disease].values for name, group in df.groupby('Spacing') if len(group) > 1]
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {disease}: F = {f_stat:.4f}, p = {p_val:.6f} {sig}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\nCreating visualizations...")

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# 1. Time series with corrected THI
fig, ax1 = plt.subplots(figsize=(14, 6))

# Daily THI
daily_thi = df.groupby('Date_parsed')['THI_NRC'].mean()
ax1.plot(daily_thi.index, daily_thi.values, 'k-o', linewidth=2, markersize=6, label='THI')

# Stress zones
ax1.axhspan(0, 70, alpha=0.2, color='green', label='No Stress')
ax1.axhspan(72, 80, alpha=0.2, color='orange', label='Moderate Stress')
ax1.axhspan(80, 100, alpha=0.2, color='red', label='Severe Stress')

ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('THI (NRC Formula)', fontsize=12)
ax1.set_title('Daily THI - October 2025 (Scanned Data)', fontsize=14)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/scanned_data_thi_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/scanned_data_thi_timeseries.png")

# 2. Disease by plot
disease_by_plot = df.groupby('Plot_No')[disease_cols].sum()
fig, ax = plt.subplots(figsize=(12, 6))
disease_by_plot.plot(kind='bar', ax=ax, width=0.8)
ax.set_xlabel('Plot Number', fontsize=12)
ax.set_ylabel('Disease Count', fontsize=12)
ax.set_title('Disease Incidence by Plot', fontsize=14)
ax.legend(title='Disease')
plt.tight_layout()
plt.savefig('figures/scanned_data_disease_by_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/scanned_data_disease_by_plot.png")

# 3. THI vs Disease scatter
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, disease in enumerate(disease_cols):
    ax = axes[i]
    
    # Box plot
    df.boxplot(column='THI_NRC', by=disease, ax=ax)
    ax.set_xlabel(f'{disease} Present')
    ax.set_ylabel('THI')
    
    # T-test
    yes_group = df[df[disease] == 1]['THI_NRC'].dropna()
    no_group = df[df[disease] == 0]['THI_NRC'].dropna()
    if len(yes_group) > 1 and len(no_group) > 1:
        t_stat, p_val = stats.ttest_ind(yes_group, no_group)
        ax.set_title(f'{disease} (p={p_val:.4f})')

plt.suptitle('')
plt.tight_layout()
plt.savefig('figures/scanned_data_thi_disease_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/scanned_data_thi_disease_boxplot.png")

# 4. Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
analysis_vars = ['Tmax', 'Tmin', 'Humidity', 'THI_NRC'] + disease_cols
corr_matrix = df[analysis_vars].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Matrix - Climate vs Diseases', fontsize=14)
plt.tight_layout()
plt.savefig('figures/scanned_data_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figures/scanned_data_correlation_heatmap.png")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
with open('results/scanned_data_reanalysis_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("REANALYSIS OF SCANNED PDF DATA - OCTOBER 2025\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATA SOURCE:\n")
    f.write("  - Data_scanned.pdf (5 pages)\n")
    f.write("  - Data_scanned_2.pdf (4 pages)\n")
    f.write("  - Total records: {}\n".format(len(df)))
    f.write("  - Date range: {} to {}\n\n".format(
        df['Date_parsed'].min().strftime('%Y-%m-%d'),
        df['Date_parsed'].max().strftime('%Y-%m-%d')))
    
    f.write("PLOTS INCLUDED:\n")
    for plot in sorted(df['Plot_No'].unique()):
        spacing = df[df['Plot_No'] == plot]['Spacing'].iloc[0]
        instar = df[df['Plot_No'] == plot]['Instar'].iloc[0]
        f.write(f"  Plot {int(plot)}: {spacing} spacing, {instar} instar\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("CLIMATE STATISTICS (NRC THI Formula)\n")
    f.write("="*80 + "\n\n")
    f.write(f"THI Mean: {df['THI_NRC'].mean():.2f}\n")
    f.write(f"THI Range: {df['THI_NRC'].min():.2f} - {df['THI_NRC'].max():.2f}\n")
    f.write(f"Temperature Mean: {df['Temp_Mean'].mean():.2f}°C\n")
    f.write(f"Humidity Mean: {df['Humidity'].mean():.2f}%\n\n")
    
    f.write("Stress Level Distribution:\n")
    for level, count in df['Stress_Level'].value_counts().items():
        pct = count / len(df) * 100
        f.write(f"  {level}: {count} ({pct:.1f}%)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("DISEASE STATISTICS\n")
    f.write("="*80 + "\n\n")
    for disease in disease_cols:
        total = df[disease].sum()
        mean = df[disease].mean()
        f.write(f"{disease}:\n")
        f.write(f"  Total cases: {total}\n")
        f.write(f"  Mean per observation: {mean:.3f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("CORRELATION WITH THI\n")
    f.write("="*80 + "\n\n")
    for disease in disease_cols:
        valid_data = df[[disease, 'THI_NRC']].dropna()
        if len(valid_data) > 3:
            corr, p_val = pointbiserialr(valid_data[disease], valid_data['THI_NRC'])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            f.write(f"{disease}:\n")
            f.write(f"  Correlation: {corr:.4f}\n")
            f.write(f"  P-value: {p_val:.6f} {sig}\n\n")
    
    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print("\nSaved: results/scanned_data_reanalysis_report.txt")

print("\n" + "="*80)
print("REANALYSIS COMPLETE")
print("="*80)
print("\nGenerated Files:")
print("  - results/Scanned_Data_Analysis_2025.xlsx (Excel with multiple sheets)")
print("  - results/scanned_data_reanalysis_report.txt")
print("  - figures/scanned_data_thi_timeseries.png")
print("  - figures/scanned_data_disease_by_plot.png")
print("  - figures/scanned_data_thi_disease_boxplot.png")
print("  - figures/scanned_data_correlation_heatmap.png")
