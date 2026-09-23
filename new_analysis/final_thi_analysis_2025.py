"""
THI Correlation Analysis with Silkworm Diseases - 2025 Data
Based on THI_Diseases_analysis.ipynb structure
================================================
Analyzes correlation between Thermo-Humidity Index and:
- PB (Pebrine)
- VR (Viriosis)
- BT (Bacteriosis)

Date Range: October 7-31, 2025
Location: Ranchi, Jharkhand, India
"""

# ============================================================================
# SECTION 1: Installation and Imports
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import pointbiserialr
import warnings
warnings.filterwarnings('ignore')

# Set high-quality plot parameters for 600 DPI
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 22

print("Libraries imported successfully!")
print("High-resolution settings (600 DPI) configured")

# ============================================================================
# SECTION 2: Input Data - From Scanned PDFs 2025
# ============================================================================

# Disease observation data from scanned PDFs (Data_scanned.pdf and Data_scanned_2.pdf)
# Combined data from all plots (5, 7, 13, 14, 17, 18, 20, 21)

data = {
    'Date': ['07-10-2025', '08-10-2025', '09-10-2025', '10-10-2025', '11-10-2025', 
             '12-10-2025', '13-10-2025', '14-10-2025', '15-10-2025', '16-10-2025',
             '17-10-2025', '18-10-2025', '19-10-2025', '20-10-2025', '21-10-2025',
             '22-10-2025', '23-10-2025', '24-10-2025', '25-10-2025', '26-10-2025',
             '27-10-2025', '28-10-2025', '29-10-2025', '30-10-2025', '31-10-2025'],
    # Disease presence aggregated across all plots per day
    'PB': ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes',
           'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no',
           'no', 'no', 'no', 'no', 'no'],
    'VR': ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes',
           'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no',
           'no', 'no', 'no', 'no', 'no'],
    'BT': ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes',
           'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no',
           'no', 'no', 'no', 'no', 'no']
}

# Disease full names
disease_names = {
    'PB': 'Pebrine',
    'VR': 'Viriosis',
    'BT': 'Bacteriosis'
}

# Create DataFrame
df_disease = pd.DataFrame(data)

# Convert date to proper format
df_disease['Date'] = pd.to_datetime(df_disease['Date'], format='%d-%m-%Y')

# Convert yes/no to binary (1/0)
for col in ['PB', 'VR', 'BT']:
    df_disease[f'{col}_binary'] = (df_disease[col] == 'yes').astype(int)

print("Disease data loaded")
print(f"Date range: {df_disease['Date'].min().date()} to {df_disease['Date'].max().date()}")
print(f"Total observations: {len(df_disease)} days")
print("\nDisease Data Summary:")
for code, name in disease_names.items():
    count = df_disease[f'{code}_binary'].sum()
    print(f"  {code} ({name}): {count} positive cases")

# ============================================================================
# SECTION 3: Weather Data - From Scanned PDFs (Daily Means)
# ============================================================================

print("\nFetching weather data from scanned observations...")

# Weather data extracted from scanned PDFs (daily averages across all plots)
weather_data = {
    'Date': ['07-10-2025', '08-10-2025', '09-10-2025', '10-10-2025', '11-10-2025',
             '12-10-2025', '13-10-2025', '14-10-2025', '15-10-2025', '16-10-2025',
             '17-10-2025', '18-10-2025', '19-10-2025', '20-10-2025', '21-10-2025',
             '22-10-2025', '23-10-2025', '24-10-2025', '25-10-2025', '26-10-2025',
             '27-10-2025', '28-10-2025', '29-10-2025', '30-10-2025', '31-10-2025'],
    'Temp_Max': [29.8, 28.7, 28.5, 29.0, 28.0, 29.0, 28.0, 28.0, 28.0, 28.0,
                 28.0, 28.0, 29.0, 30.0, 29.0, 28.0, 27.0, 28.0, 29.0, 24.0,
                 26.0, 27.0, 27.0, 27.0, 25.0],
    'Temp_Min': [21.0, 21.0, 23.0, 20.0, 19.0, 21.0, 20.0, 18.0, 17.0, 18.0,
                 18.0, 18.0, 19.0, 20.0, 21.0, 20.0, 23.0, 21.0, 21.0, 21.0,
                 25.0, 26.0, 26.0, 26.0, 23.0],
    'Humidity': [75.0, 80.0, 79.7, 75.5, 70.6, 78.0, 65.8, 71.1, 75.0, 73.6,
                 75.0, 80.1, 86.6, 80.1, 75.0, 75.5, 69.6, 61.0, 72.4, 70.8,
                 79.2, 80.5, 60.5, 60.7, 63.7]
}

df_weather = pd.DataFrame(weather_data)
df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%d-%m-%Y')

print(f"Weather data loaded: {len(df_weather)} daily records")

# ============================================================================
# SECTION 4: Calculate THI
# ============================================================================

def calculate_thi_nrc(temp_c, rh):
    """
    Calculate Thermo-Humidity Index using NRC (1971) formula
    """
    temp_f = (1.8 * temp_c) + 32
    thi = temp_f - ((0.55 - 0.0055 * rh) * (temp_f - 58))
    return thi

def classify_thi_stress(thi):
    """
    Classify THI into stress categories
    """
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

# Calculate daily statistics
df_weather['Temp_Mean'] = (df_weather['Temp_Max'] + df_weather['Temp_Min']) / 2
df_weather['THI'] = df_weather.apply(
    lambda row: calculate_thi_nrc(row['Temp_Mean'], row['Humidity']),
    axis=1
)
df_weather['Stress_Level'] = df_weather['THI'].apply(classify_thi_stress)

print("\nTHI calculated for all records")
print(f"  THI Range: {df_weather['THI'].min():.2f} - {df_weather['THI'].max():.2f}")
print(f"  Mean THI: {df_weather['THI'].mean():.2f}")

# ============================================================================
# SECTION 5: Merge Disease and Weather Data
# ============================================================================

df_merged = pd.merge(df_disease, df_weather, on='Date', how='left')

print("\nDisease and weather data merged")
print(f"> Final dataset: {len(df_merged)} observations")

# Save merged data
df_merged.to_csv('results/merged_disease_weather_data_2025.csv', index=False)
print("\nMerged data saved to 'results/merged_disease_weather_data_2025.csv'")

# ============================================================================
# SECTION 6: Statistical Analysis
# ============================================================================

def perform_correlation_analysis(df, variable_name):
    """
    Perform comprehensive correlation analysis for a disease variable
    """
    results = {
        'variable': variable_name,
        'disease_name': disease_names[variable_name],
        'observations': len(df),
        'positive_cases': df[f'{variable_name}_binary'].sum(),
        'negative_cases': len(df) - df[f'{variable_name}_binary'].sum()
    }
    
    # Point-biserial correlation (for binary vs continuous)
    valid_data = df[[f'{variable_name}_binary', 'THI']].dropna()
    
    if len(valid_data) > 0:
        corr_thi, p_value_thi = pointbiserialr(valid_data[f'{variable_name}_binary'],
                                                valid_data['THI'])
        results['correlation_thi'] = corr_thi
        results['p_value_thi'] = p_value_thi
        results['significant_thi'] = 'Yes' if p_value_thi < 0.05 else 'No'
        
        # Also correlate with temperature and humidity
        corr_temp, p_value_temp = pointbiserialr(valid_data[f'{variable_name}_binary'],
                                                  df.loc[valid_data.index, 'Temp_Mean'])
        results['correlation_temp'] = corr_temp
        results['p_value_temp'] = p_value_temp
        
        corr_rh, p_value_rh = pointbiserialr(valid_data[f'{variable_name}_binary'],
                                             df.loc[valid_data.index, 'Humidity'])
        results['correlation_rh'] = corr_rh
        results['p_value_rh'] = p_value_rh
        
        # Mean THI for positive vs negative cases
        results['thi_when_yes'] = df[df[f'{variable_name}_binary'] == 1]['THI'].mean()
        results['thi_when_no'] = df[df[f'{variable_name}_binary'] == 0]['THI'].mean()
        results['thi_difference'] = results['thi_when_yes'] - results['thi_when_no']
        
        # T-test
        yes_group = df[df[f'{variable_name}_binary'] == 1]['THI'].dropna()
        no_group = df[df[f'{variable_name}_binary'] == 0]['THI'].dropna()
        
        if len(yes_group) > 0 and len(no_group) > 0:
            t_stat, t_pvalue = stats.ttest_ind(yes_group, no_group)
            results['t_statistic'] = t_stat
            results['t_test_pvalue'] = t_pvalue
    
    return results

print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

analysis_results = {}
variables = ['PB', 'VR', 'BT']

for var in variables:
    print(f"\n--- Analysis for {var} ({disease_names[var]}) ---")
    results = perform_correlation_analysis(df_merged, var)
    analysis_results[var] = results
    
    print(f"Observations: {results['observations']}")
    print(f"Positive cases (yes): {results['positive_cases']}")
    print(f"Negative cases (no): {results['negative_cases']}")
    print(f"\nCorrelation with THI: {results.get('correlation_thi', 'N/A'):.4f}")
    print(f"P-value: {results.get('p_value_thi', 'N/A'):.4f}")
    print(f"Significant (p<0.05): {results.get('significant_thi', 'N/A')}")
    print(f"\nMean THI when {var}=Yes: {results.get('thi_when_yes', 'N/A'):.2f}")
    print(f"Mean THI when {var}=No: {results.get('thi_when_no', 'N/A'):.2f}")
    print(f"Difference: {results.get('thi_difference', 'N/A'):.2f}")

# ============================================================================
# SECTION 7: Save Statistical Results
# ============================================================================

with open('results/statistical_analysis_results_2025.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("THERMO-HUMIDITY INDEX AND SILKWORM DISEASE CORRELATION ANALYSIS\n")
    f.write("2025 DATA - RANCHI, JHARKHAND\n")
    f.write("="*80 + "\n")
    f.write("Location: Ranchi, Jharkhand, India\n")
    f.write(f"Date Range: {df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}\n")
    f.write(f"Total Observations: {len(df_merged)} days\n")
    f.write("="*80 + "\n\n")
    
    f.write("SILKWORM DISEASES:\n")
    for code, name in disease_names.items():
        f.write(f"  {code} - {name}\n")
    f.write("\n")
    
    f.write("THI STRESS CATEGORIES:\n")
    f.write("  < 70  : No Stress\n")
    f.write("  70-72 : Mild Stress\n")
    f.write("  72-80 : Moderate Stress\n")
    f.write("  80-90 : Severe Stress\n")
    f.write("  > 90  : Emergency\n\n")
    
    f.write("="*80 + "\n")
    f.write("STATISTICAL RESULTS\n")
    f.write("="*80 + "\n\n")
    
    for var in variables:
        results = analysis_results[var]
        f.write(f"\n{'='*80}\n")
        f.write(f"Disease: {results['disease_name']} ({var})\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Sample Size:\n")
        f.write(f"  Total observations: {results['observations']}\n")
        f.write(f"  Positive cases (disease present): {results['positive_cases']}\n")
        f.write(f"  Negative cases (disease absent): {results['negative_cases']}\n\n")
        
        f.write(f"Point-Biserial Correlation:\n")
        f.write(f"  Correlation with THI: {results.get('correlation_thi', 'N/A'):.4f}\n")
        f.write(f"  P-value: {results.get('p_value_thi', 'N/A'):.6f}\n")
        f.write(f"  Statistically Significant (p<0.05): {results.get('significant_thi', 'N/A')}\n\n")
        
        f.write(f"  Correlation with Temperature: {results.get('correlation_temp', 'N/A'):.4f}\n")
        f.write(f"  P-value: {results.get('p_value_temp', 'N/A'):.6f}\n\n")
        
        f.write(f"  Correlation with Humidity: {results.get('correlation_rh', 'N/A'):.4f}\n")
        f.write(f"  P-value: {results.get('p_value_rh', 'N/A'):.6f}\n\n")
        
        f.write(f"Mean THI Comparison:\n")
        f.write(f"  Mean THI when {results['disease_name']} present: {results.get('thi_when_yes', 'N/A'):.2f}\n")
        f.write(f"  Mean THI when {results['disease_name']} absent: {results.get('thi_when_no', 'N/A'):.2f}\n")
        f.write(f"  Difference: {results.get('thi_difference', 'N/A'):.2f}\n\n")
        
        if 't_test_pvalue' in results:
            f.write(f"Independent T-Test:\n")
            f.write(f"  T-statistic: {results['t_statistic']:.4f}\n")
            f.write(f"  P-value: {results['t_test_pvalue']:.6f}\n")
            f.write(f"  Significant difference (p<0.05): {'Yes' if results['t_test_pvalue'] < 0.05 else 'No'}\n\n")
        
        # Interpretation
        f.write(f"Interpretation:\n")
        corr = results.get('correlation_thi', 0)
        if abs(corr) < 0.1:
            strength = "negligible"
        elif abs(corr) < 0.3:
            strength = "weak"
        elif abs(corr) < 0.5:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction = "positive" if corr > 0 else "negative"
        
        f.write(f"  There is a {strength} {direction} correlation between {results['disease_name']} and THI.\n")
        
        if results.get('significant_thi') == 'Yes':
            f.write(f"  This correlation is statistically significant (p<0.05).\n")
        else:
            f.write(f"  This correlation is NOT statistically significant (p>=0.05).\n")
        
        f.write("\n")

print("\nStatistical results saved to 'results/statistical_analysis_results_2025.txt'")

# ============================================================================
# SECTION 8: VISUALIZATION 1 - Combined Time Series
# ============================================================================

print("\nGenerating visualizations...")

fig = plt.figure(figsize=(22, 11))

# Create axes with space at top for legend
ax1 = plt.subplot(111)

# Define colors for each disease
disease_colors = {
    'PB': '#E74C3C',
    'VR': '#3498DB',
    'BT': '#F39C12'
}

# Plot THI as line
ax1.plot(df_merged['Date'], df_merged['THI'],
         color='black', linewidth=3.5, marker='o', markersize=8,
         label='THI', alpha=0.8, zorder=5)

# Add stress level zones with stronger colors (no Mild Stress zone)
ax1.axhspan(0, 70, alpha=0.25, color='green', label='No Stress (<70)')
ax1.axhspan(72, 80, alpha=0.25, color='orange', label='Moderate Stress (72-80)')
ax1.axhspan(80, 100, alpha=0.25, color='red', label='Severe Stress (>80)')

ax1.set_xlabel('Date', fontsize=22, fontweight='bold')
ax1.set_ylabel('Thermo-Humidity Index (THI)', fontsize=22, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
ax1.tick_params(axis='x', rotation=45, labelsize=18)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

# Secondary y-axis for disease occurrence
ax2 = ax1.twinx()

# Plot each disease with more opaque lines
y_offset = 0
for var in variables:
    disease_data = df_merged[[f'{var}_binary', 'Date']].copy()
    disease_data[f'{var}_plot'] = disease_data[f'{var}_binary'] + y_offset
    
    # Plot line connecting disease occurrences - MORE OPAQUE
    ax2.plot(df_merged['Date'], disease_data[f'{var}_plot'],
             color=disease_colors[var], linewidth=3, alpha=0.7,
             linestyle='--', zorder=3)
    
    # Mark disease occurrences with large markers
    yes_dates = df_merged[df_merged[f'{var}_binary'] == 1]['Date']
    yes_values = [1 + y_offset] * len(yes_dates)
    
    ax2.scatter(yes_dates, yes_values,
                color=disease_colors[var], s=400, alpha=0.9,
                marker='s', edgecolor='black', linewidth=2.5,
                label=f'{disease_names[var]} ({var})', zorder=10)
    
    y_offset += 0.2

ax2.set_ylabel('Disease Occurrence', fontsize=22, fontweight='bold')
ax2.set_ylim(-0.3, 1.8)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Absent', 'Present'], fontsize=18)
ax2.tick_params(axis='y', labelsize=18)

# Get all legend elements
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Create single legend ABOVE the plot
fig.legend(lines1 + lines2, labels1 + labels2,
          loc='upper center', fontsize=15,
          framealpha=0.98, edgecolor='black',
          ncol=4, bbox_to_anchor=(0.5, 0.97),
          columnspacing=2)

# Adjust layout to make room for legend at top
plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.92)

plt.savefig('figures/fig1_combined_timeseries_2025.png', dpi=600, bbox_inches='tight', pad_inches=0.3)
plt.close()

print("Saved: figures/fig1_combined_timeseries_2025.png")

# ============================================================================
# SECTION 9: VISUALIZATION 2 - 3x3 Correlation Matrix
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 9))

# Create 3x3 correlation matrix
climate_vars = ['THI', 'Temp_Mean', 'Humidity']
disease_vars = ['PB_binary', 'VR_binary', 'BT_binary']

# Calculate correlations
corr_matrix_3x3 = np.zeros((3, 3))

for i, climate_var in enumerate(climate_vars):
    for j, disease_var in enumerate(disease_vars):
        valid_data = df_merged[[climate_var, disease_var]].dropna()
        if len(valid_data) > 0:
            corr, _ = pointbiserialr(valid_data[disease_var], valid_data[climate_var])
            corr_matrix_3x3[i, j] = corr

# Create DataFrame with short names only
corr_df = pd.DataFrame(
    corr_matrix_3x3,
    index=['THI', 'Temperature (°C)', 'Humidity (%)'],
    columns=['PB', 'VR', 'BT']
)

# Create colormap
cmap = sns.diverging_palette(250, 10, as_cmap=True)

# Create heatmap
im = ax.imshow(corr_df.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(corr_df.columns, fontsize=24, fontweight='bold')
ax.set_yticklabels(corr_df.index, fontsize=24, fontweight='bold')

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Add correlation values and colored borders
for i in range(3):
    for j in range(3):
        corr_val = corr_df.iloc[i, j]
        
        # Text color
        if abs(corr_val) > 0.5:
            text_color = "white"
        else:
            text_color = "black"
        
        # Add value
        ax.text(j, i, f'{corr_val:.3f}',
                ha="center", va="center", color=text_color,
                fontsize=26, fontweight='bold')
        
        # Add colored border based on correlation strength
        if abs(corr_val) > 0.3:
            # Use the actual cell color for the border
            cell_color = cmap((corr_val + 1) / 2)
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                  fill=False, edgecolor=cell_color,
                                  linewidth=6, linestyle='-')
            ax.add_patch(rect)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20)
cbar.set_label('Correlation Coefficient', fontsize=22, fontweight='bold',
               rotation=270, labelpad=40)

# Title - simple and clean
ax.set_title('Climate Variables vs Silkworm Diseases (2025 Data)\nColored borders: |r| > 0.3',
             fontsize=24, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figures/fig2_correlation_matrix_2025.png', dpi=600, bbox_inches='tight')
plt.close()

print("Saved: figures/fig2_correlation_matrix_2025.png")

# ============================================================================
# SECTION 10: Summary Statistics Table
# ============================================================================

# Create summary table
summary_data = []
for var in variables:
    res = analysis_results[var]
    summary_data.append({
        'Disease': disease_names[var],
        'Code': var,
        'Positive Cases': res['positive_cases'],
        'Negative Cases': res['negative_cases'],
        'Correlation (r)': f"{res.get('correlation_thi', 0):.4f}",
        'P-value': f"{res.get('p_value_thi', 1):.6f}",
        'Significant': res.get('significant_thi', 'N/A'),
        'THI (Disease Present)': f"{res.get('thi_when_yes', 0):.2f}",
        'THI (Disease Absent)': f"{res.get('thi_when_no', 0):.2f}",
        'Difference': f"{res.get('thi_difference', 0):.2f}"
    })

summary_df = pd.DataFrame(summary_data)

# Save to CSV
summary_df.to_csv('results/summary_statistics_table_2025.csv', index=False)
print("\nSaved: results/summary_statistics_table_2025.csv")

# ============================================================================
# SECTION 11: Final Summary Report
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print("\nGenerated Files:")
print("\nData Files:")
print("  1. results/merged_disease_weather_data_2025.csv - Complete dataset")
print("  2. results/summary_statistics_table_2025.csv - Statistical summary")
print("  3. results/statistical_analysis_results_2025.txt - Detailed text report")

print("\nVisualization Files (600 DPI, Large Fonts):")
print("  1. figures/fig1_combined_timeseries_2025.png - Time series (legends outside)")
print("  2. figures/fig2_correlation_matrix_2025.png - 3x3 correlation heatmap")

print("\n" + "="*80)
print("KEY FINDINGS SUMMARY")
print("="*80)

# Overall THI statistics
print(f"\nTHI Statistics for Study Period:")
print(f"  Mean THI: {df_merged['THI'].mean():.2f}")
print(f"  Min THI: {df_merged['THI'].min():.2f}")
print(f"  Max THI: {df_merged['THI'].max():.2f}")
print(f"  Standard Deviation: {df_merged['THI'].std():.2f}")

print("\nDisease-THI Correlations:")
print("-" * 80)
for var in variables:
    res = analysis_results[var]
    corr = res.get('correlation_thi', 0)
    p_val = res.get('p_value_thi', 1)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    
    print(f"\n{disease_names[var]} ({var}):")
    print(f"  Positive cases: {res['positive_cases']}/{res['observations']} days ({res['positive_cases']/res['observations']*100:.1f}%)")
    print(f"  Correlation: r = {corr:.4f} (p = {p_val:.6f}) {sig}")
    
    if abs(corr) < 0.1:
        strength = "Negligible"
    elif abs(corr) < 0.3:
        strength = "Weak"
    elif abs(corr) < 0.5:
        strength = "Moderate"
    else:
        strength = "Strong"
    
    direction = "positive" if corr > 0 else "negative"
    print(f"  {strength} {direction} correlation")
    
    print(f"  Mean THI when disease present: {res.get('thi_when_yes', 0):.2f}")
    print(f"  Mean THI when disease absent: {res.get('thi_when_no', 0):.2f}")
    print(f"  Difference: {res.get('thi_difference', 0):.2f}")
    
    if res.get('significant_thi') == 'Yes':
        print(f"  > Statistically SIGNIFICANT relationship")
    else:
        print(f"  X NOT statistically significant")

print("\n" + "="*80)
print("Legend:")
print("  *** p < 0.001 (Highly significant)")
print("  **  p < 0.01  (Very significant)")
print("  *   p < 0.05  (Significant)")
print("  ns  p >= 0.05  (Not significant)")
print("="*80)

# Climate summary
print("\nClimate Conditions During Study Period:")
print(f"  Temperature: {df_merged['Temp_Mean'].mean():.2f}°C (±{df_merged['Temp_Mean'].std():.2f})")
print(f"  Range: {df_merged['Temp_Mean'].min():.2f}°C to {df_merged['Temp_Mean'].max():.2f}°C")
print(f"  Humidity: {df_merged['Humidity'].mean():.2f}% (±{df_merged['Humidity'].std():.2f})")
print(f"  Range: {df_merged['Humidity'].min():.2f}% to {df_merged['Humidity'].max():.2f}%")

# Stress level distribution
print("\nTHI Stress Level Distribution:")
stress_counts = df_merged['Stress_Level'].value_counts()
for level in ['No Stress', 'Mild Stress', 'Moderate Stress', 'Severe Stress', 'Emergency']:
    count = stress_counts.get(level, 0)
    pct = count / len(df_merged) * 100
    print(f"  {level}: {count} days ({pct:.1f}%)")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)

recommendations = []
for var in variables:
    res = analysis_results[var]
    if res.get('significant_thi') == 'Yes':
        corr = res.get('correlation_thi', 0)
        if corr > 0:
            recommendations.append(
                f"• {disease_names[var]} shows significant positive correlation with THI. "
                f"Increased monitoring recommended when THI > {res.get('thi_when_yes', 70):.1f}."
            )
        else:
            recommendations.append(
                f"• {disease_names[var]} shows significant negative correlation with THI. "
                f"Increased monitoring recommended when THI < {res.get('thi_when_yes', 70):.1f}."
            )

if len(recommendations) > 0:
    for rec in recommendations:
        print(f"\n{rec}")
else:
    print("\n• No statistically significant correlations found between THI and diseases.")
    print("• Continue regular monitoring regardless of THI levels.")

print("\n• Maintain optimal rearing conditions: Temperature 24-28°C, Humidity 70-85%")
print("• Implement preventive measures during periods of environmental stress")
print("• Regular disinfection and hygiene protocols are essential regardless of THI")

print("\n" + "="*80)

# Detailed Statistics Table
print("\nDetailed Statistics Table:")
print("="*80)
print(summary_df.to_string(index=False))

# Save enhanced summary
with open('results/analysis_summary_2025.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SILKWORM DISEASE AND THI CORRELATION ANALYSIS - SUMMARY\n")
    f.write("2025 DATA\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Study Location: Ranchi, Jharkhand, India\n")
    f.write(f"Coordinates: 23.3441°N, 85.3096°E\n")
    f.write(f"Study Period: {df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}\n")
    f.write(f"Total Observations: {len(df_merged)} days\n\n")
    
    f.write("="*80 + "\n")
    f.write("DISEASE DEFINITIONS\n")
    f.write("="*80 + "\n")
    for code, name in disease_names.items():
        f.write(f"{code} - {name}\n")
    f.write("\n")
    
    f.write("="*80 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("="*80 + "\n\n")
    
    for var in variables:
        res = analysis_results[var]
        f.write(f"{disease_names[var]} ({var}):\n")
        f.write(f"  Incidence: {res['positive_cases']}/{res['observations']} days\n")
        f.write(f"  Correlation with THI: r = {res.get('correlation_thi', 0):.4f}\n")
        f.write(f"  P-value: {res.get('p_value_thi', 1):.6f}\n")
        f.write(f"  Statistical Significance: {res.get('significant_thi', 'N/A')}\n")
        f.write(f"  Mean THI (disease present): {res.get('thi_when_yes', 0):.2f}\n")
        f.write(f"  Mean THI (disease absent): {res.get('thi_when_no', 0):.2f}\n")
        f.write(f"  Difference: {res.get('thi_difference', 0):.2f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("CLIMATE STATISTICS\n")
    f.write("="*80 + "\n\n")
    f.write(f"THI (Thermo-Humidity Index):\n")
    f.write(f"  Mean: {df_merged['THI'].mean():.2f}\n")
    f.write(f"  Standard Deviation: {df_merged['THI'].std():.2f}\n")
    f.write(f"  Range: {df_merged['THI'].min():.2f} - {df_merged['THI'].max():.2f}\n\n")
    
    f.write(f"Temperature:\n")
    f.write(f"  Mean: {df_merged['Temp_Mean'].mean():.2f}°C\n")
    f.write(f"  Standard Deviation: {df_merged['Temp_Mean'].std():.2f}°C\n")
    f.write(f"  Range: {df_merged['Temp_Mean'].min():.2f}°C - {df_merged['Temp_Mean'].max():.2f}°C\n\n")
    
    f.write(f"Relative Humidity:\n")
    f.write(f"  Mean: {df_merged['Humidity'].mean():.2f}%\n")
    f.write(f"  Standard Deviation: {df_merged['Humidity'].std():.2f}%\n")
    f.write(f"  Range: {df_merged['Humidity'].min():.2f}% - {df_merged['Humidity'].max():.2f}%\n\n")

print("\nSaved: results/analysis_summary_2025.txt")

print("\n" + "="*80)
print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
print("="*80)

print("\nFiles ready for use:")
print("  • 2 high-resolution visualizations (600 DPI)")
print("  • 3 data files (CSV format)")
print("  • 2 detailed text reports")

print("\nAll figures have large, legible fonts suitable for presentations and publications.")
print("="*80)
