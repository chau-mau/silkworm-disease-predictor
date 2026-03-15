"""
Silkworm Disease Analysis with Bioclimatic Variables
Ranchi, Jharkhand - October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, ttest_ind
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for plots - no titles in images
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Create output directories
import os
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("="*80)
print("SILKWORM DISEASE ANALYSIS - RANCHI 2025")
print("="*80)

# =====================
# 1. LOAD AND CLEAN DATA
# =====================
print("\n1. LOADING AND CLEANING DATA...")

# Load 2025 data
df = pd.read_csv('dmc-ctrti-2025.csv')
print(f"Original shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Clean the data
df = df[df['Plot_No'].notna()].copy()
df = df[df['Plot_No'] != ''].copy()

# Convert Plot_No to integer
df['Plot_No'] = pd.to_numeric(df['Plot_No'], errors='coerce')
df = df[df['Plot_No'].notna()].copy()
df['Plot_No'] = df['Plot_No'].astype(int)

# Clean and parse dates
df['Date'] = df['Date'].astype(str).str.strip()

# Handle different date formats
def parse_date(date_str):
    try:
        # Try DD-MM-YYYY format
        return pd.to_datetime(date_str, format='%d-%m-%Y')
    except:
        try:
            # Try other formats
            return pd.to_datetime(date_str, dayfirst=True)
        except:
            return pd.NaT

df['Date_parsed'] = df['Date'].apply(parse_date)

# Extract numeric values from columns that might have text
def extract_numeric(val):
    if pd.isna(val) or val == '' or str(val).strip() == '':
        return np.nan
    try:
        return float(val)
    except:
        # Try to extract number from string
        import re
        numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', str(val))
        if numbers:
            return float(numbers[0])
        return np.nan

numeric_cols = ['Tmax', 'Tmin', 'Humidity', 'Rainfall', 'Wind_Speed', 'THI', 'Dry_Bulb', 'Wet_Bulb']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(extract_numeric)

# Convert disease columns to numeric (0/1 or counts)
disease_cols = ['Pebrine', 'Virosis', 'Bacteriosis', 'Muscardine']
for col in disease_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'x'] else (0 if str(x).lower() in ['no', '0', 'false', ''] else extract_numeric(x)))
        df[col] = df[col].fillna(0)

# Handle Net_Tech column
if 'Net_Tech' in df.columns:
    df['Net_Tech'] = df['Net_Tech'].apply(lambda x: 'Yes' if str(x).lower() in ['yes', '1', 'true'] else ('No' if str(x).lower() in ['no', '0', 'false', ''] else 'Unknown'))

# Handle Pests column - extract individual pests
if 'Pests' in df.columns:
    df['Pests'] = df['Pests'].fillna('')
    df['Has_Uzi'] = df['Pests'].str.contains('U', case=False, na=False).astype(int)
    df['Has_Mites'] = df['Pests'].str.contains('M', case=False, na=False).astype(int)
    df['Has_Ants'] = df['Pests'].str.contains('A', case=False, na=False).astype(int)
    df['Has_Spiders'] = df['Pests'].str.contains('S', case=False, na=False).astype(int)
    df['Has_Athropoda'] = df['Pests'].str.contains('At', case=False, na=False).astype(int)
    df['Total_Pests'] = df['Has_Uzi'] + df['Has_Mites'] + df['Has_Ants'] + df['Has_Spiders'] + df['Has_Athropoda']

# Create any disease indicator
df['Any_Disease'] = ((df['Pebrine'] > 0) | (df['Virosis'] > 0) | (df['Bacteriosis'] > 0) | (df['Muscardine'] > 0)).astype(int)

# Calculate derived climate variables
df['Temp_Range'] = df['Tmax'] - df['Tmin']
df['Mean_Temp'] = (df['Tmax'] + df['Tmin']) / 2

# Create THI categories
def thi_category(thi):
    if pd.isna(thi):
        return 'Unknown'
    if thi < 70:
        return 'No Stress'
    elif thi < 72:
        return 'Mild Stress'
    elif thi < 80:
        return 'Moderate Stress'
    elif thi < 90:
        return 'Severe Stress'
    else:
        return 'Emergency'

df['THI_Category'] = df['THI'].apply(thi_category)

print(f"Cleaned shape: {df.shape}")
print(f"Date range: {df['Date_parsed'].min()} to {df['Date_parsed'].max()}")
print(f"Plots: {sorted(df['Plot_No'].unique())}")
print(f"Spacing types: {df['Spacing'].unique()}")
print(f"Instar stages: {df['Instar'].unique()}")

# Save cleaned data
df.to_csv('results/cleaned_data_2025.csv', index=False)
print("Cleaned data saved to results/cleaned_data_2025.csv")

# =====================
# 2. SUMMARY STATISTICS
# =====================
print("\n2. GENERATING SUMMARY STATISTICS...")

summary_stats = df[numeric_cols + disease_cols + ['Total_Pests', 'Any_Disease']].describe()
summary_stats.to_csv('results/summary_statistics.csv')
print("Summary statistics saved to results/summary_statistics.csv")

# Summary by Plot
plot_summary = df.groupby('Plot_No')[disease_cols + ['Total_Pests']].sum()
plot_summary.to_csv('results/summary_by_plot.csv')
print("Plot summary saved to results/summary_by_plot.csv")

# Summary by Spacing
spacing_summary = df.groupby('Spacing')[disease_cols + ['Total_Pests']].agg(['sum', 'mean', 'count'])
spacing_summary.to_csv('results/summary_by_spacing.csv')
print("Spacing summary saved to results/summary_by_spacing.csv")

# Summary by Instar
instar_summary = df.groupby('Instar')[disease_cols + ['Total_Pests']].agg(['sum', 'mean', 'count'])
instar_summary.to_csv('results/summary_by_instar.csv')
print("Instar summary saved to results/summary_by_instar.csv")

# =====================
# 3. VISUALIZATIONS - EXPLORATORY ANALYSIS
# =====================
print("\n3. GENERATING EXPLORATORY VISUALIZATIONS...")

# 3.1 Disease incidence by plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_disease = df.groupby('Plot_No')[disease_cols].sum()
plot_disease.plot(kind='bar', ax=ax)
ax.set_xlabel('Plot Number')
ax.set_ylabel('Disease Count')
ax.legend(title='Disease')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('figures/01_disease_by_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/01_disease_by_plot.png")

# 3.2 Disease incidence by spacing
fig, ax = plt.subplots(figsize=(10, 6))
spacing_disease = df.groupby('Spacing')[disease_cols].mean()
spacing_disease.plot(kind='bar', ax=ax)
ax.set_xlabel('Spacing')
ax.set_ylabel('Mean Disease Incidence')
ax.legend(title='Disease')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('figures/02_disease_by_spacing.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/02_disease_by_spacing.png")

# 3.3 Disease by instar stage
fig, ax = plt.subplots(figsize=(10, 6))
instar_disease = df.groupby('Instar')[disease_cols].mean()
instar_disease.plot(kind='bar', ax=ax)
ax.set_xlabel('Instar Stage')
ax.set_ylabel('Mean Disease Incidence')
ax.legend(title='Disease')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('figures/03_disease_by_instar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/03_disease_by_instar.png")

# 3.4 THI distribution
fig, ax = plt.subplots(figsize=(10, 6))
df['THI'].hist(bins=15, ax=ax, edgecolor='black')
ax.set_xlabel('THI (Thermo-Humidity Index)')
ax.set_ylabel('Frequency')
ax.axvline(df['THI'].mean(), color='red', linestyle='--', label=f'Mean: {df["THI"].mean():.2f}')
ax.legend()
plt.tight_layout()
plt.savefig('figures/04_thi_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/04_thi_distribution.png")

# 3.5 Temperature variables over time
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Tmax over time
df_time = df.dropna(subset=['Date_parsed'])
if len(df_time) > 0:
    for plot in df_time['Plot_No'].unique():
        plot_data = df_time[df_time['Plot_No'] == plot]
        axes[0, 0].plot(plot_data['Date_parsed'], plot_data['Tmax'], marker='o', label=f'Plot {plot}', alpha=0.7)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Tmax (°C)')

# Tmin over time
for plot in df_time['Plot_No'].unique():
    plot_data = df_time[df_time['Plot_No'] == plot]
    axes[0, 1].plot(plot_data['Date_parsed'], plot_data['Tmin'], marker='o', label=f'Plot {plot}', alpha=0.7)
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Tmin (°C)')

# Humidity over time
for plot in df_time['Plot_No'].unique():
    plot_data = df_time[df_time['Plot_No'] == plot]
    axes[1, 0].plot(plot_data['Date_parsed'], plot_data['Humidity'], marker='o', label=f'Plot {plot}', alpha=0.7)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Humidity (%)')

# THI over time
for plot in df_time['Plot_No'].unique():
    plot_data = df_time[df_time['Plot_No'] == plot]
    axes[1, 1].plot(plot_data['Date_parsed'], plot_data['THI'], marker='o', label=f'Plot {plot}', alpha=0.7)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('THI')

plt.tight_layout()
plt.savefig('figures/05_climate_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/05_climate_timeseries.png")

# 3.6 Disease occurrence over time
fig, ax = plt.subplots(figsize=(12, 6))
if len(df_time) > 0:
    daily_disease = df_time.groupby('Date_parsed')[disease_cols].sum()
    for disease in disease_cols:
        ax.plot(daily_disease.index, daily_disease[disease], marker='o', label=disease)
    ax.set_xlabel('Date')
    ax.set_ylabel('Disease Count')
    ax.legend(title='Disease')
    plt.tight_layout()
    plt.savefig('figures/06_disease_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/06_disease_timeseries.png")

# 3.7 Pest occurrence
fig, ax = plt.subplots(figsize=(10, 6))
pest_cols = ['Has_Uzi', 'Has_Mites', 'Has_Ants', 'Has_Spiders', 'Has_Athropoda']
pest_counts = [df[p].sum() for p in pest_cols]
pest_names = ['Uzi', 'Mites', 'Ants', 'Spiders', 'Athropoda']
ax.bar(pest_names, pest_counts, color='coral', edgecolor='black')
ax.set_xlabel('Pest Type')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('figures/07_pest_occurrence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/07_pest_occurrence.png")

# 3.8 Box plot - THI by disease presence
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, disease in enumerate(disease_cols):
    df.boxplot(column='THI', by=disease, ax=axes[i])
    axes[i].set_xlabel(f'{disease} Present')
    axes[i].set_ylabel('THI')
plt.suptitle('')
plt.tight_layout()
plt.savefig('figures/08_thi_by_disease_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/08_thi_by_disease_boxplot.png")

# =====================
# 4. CORRELATION ANALYSIS
# =====================
print("\n4. PERFORMING CORRELATION ANALYSIS...")

# Climate variables
climate_vars = ['Tmax', 'Tmin', 'Temp_Range', 'Mean_Temp', 'Humidity', 'Rainfall', 'Wind_Speed', 'THI', 'Dry_Bulb', 'Wet_Bulb']
analysis_vars = climate_vars + disease_cols + ['Total_Pests']

# Create correlation matrix
corr_data = df[analysis_vars].corr()
corr_data.to_csv('results/correlation_matrix.csv')

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
plt.tight_layout()
plt.savefig('figures/09_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/09_correlation_heatmap.png")

# Specific climate-disease correlations
correlation_results = []
for disease in disease_cols:
    for climate in climate_vars:
        valid_data = df[[disease, climate]].dropna()
        if len(valid_data) > 3:
            r, p = pearsonr(valid_data[climate], valid_data[disease])
            correlation_results.append({
                'Disease': disease,
                'Climate_Variable': climate,
                'Correlation': r,
                'P_value': p,
                'Significant': 'Yes' if p < 0.05 else 'No',
                'N': len(valid_data)
            })

corr_df = pd.DataFrame(correlation_results)
corr_df.to_csv('results/climate_disease_correlations.csv', index=False)
print("Climate-disease correlations saved to results/climate_disease_correlations.csv")

# Plot climate-disease correlations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, disease in enumerate(disease_cols):
    disease_corr = corr_df[corr_df['Disease'] == disease]
    colors = ['red' if s == 'Yes' else 'gray' for s in disease_corr['Significant']]
    axes[i].barh(disease_corr['Climate_Variable'], disease_corr['Correlation'], color=colors)
    axes[i].set_xlabel('Correlation Coefficient')
    axes[i].axvline(0, color='black', linestyle='-', linewidth=0.5)
    axes[i].set_xlim(-1, 1)
plt.tight_layout()
plt.savefig('figures/10_climate_disease_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/10_climate_disease_correlations.png")

# Scatter plots for significant correlations
sig_corr = corr_df[corr_df['Significant'] == 'Yes']
if len(sig_corr) > 0:
    for _, row in sig_corr.iterrows():
        disease = row['Disease']
        climate = row['Climate_Variable']
        fig, ax = plt.subplots(figsize=(8, 6))
        valid_data = df[[disease, climate]].dropna()
        ax.scatter(valid_data[climate], valid_data[disease], alpha=0.6)
        ax.set_xlabel(climate)
        ax.set_ylabel(disease)
        # Add trend line
        z = np.polyfit(valid_data[climate], valid_data[disease], 1)
        p = np.poly1d(z)
        ax.plot(valid_data[climate], p(valid_data[climate]), "r--", alpha=0.8)
        plt.tight_layout()
        plt.savefig(f'figures/11_scatter_{disease}_{climate}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: figures/11_scatter_{disease}_{climate}.png")

# =====================
# 5. SPACING/INSTAR EFFECT ANALYSIS
# =====================
print("\n5. ANALYZING SPACING AND INSTAR EFFECTS...")

# ANOVA for spacing effect
spacing_anova_results = []
for disease in disease_cols:
    groups = [group[disease].values for name, group in df.groupby('Spacing') if len(group) > 1]
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups)
        spacing_anova_results.append({
            'Disease': disease,
            'F_statistic': f_stat,
            'P_value': p_val,
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })

spacing_anova_df = pd.DataFrame(spacing_anova_results)
spacing_anova_df.to_csv('results/spacing_anova.csv', index=False)
print("Spacing ANOVA results saved to results/spacing_anova.csv")

# ANOVA for instar effect
instar_anova_results = []
for disease in disease_cols:
    groups = [group[disease].values for name, group in df.groupby('Instar') if len(group) > 1]
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups)
        instar_anova_results.append({
            'Disease': disease,
            'F_statistic': f_stat,
            'P_value': p_val,
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })

instar_anova_df = pd.DataFrame(instar_anova_results)
instar_anova_df.to_csv('results/instar_anova.csv', index=False)
print("Instar ANOVA results saved to results/instar_anova.csv")

# Box plots for spacing
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, disease in enumerate(disease_cols):
    df.boxplot(column=disease, by='Spacing', ax=axes[i])
    axes[i].set_xlabel('Spacing')
    axes[i].set_ylabel(disease)
plt.suptitle('')
plt.tight_layout()
plt.savefig('figures/12_disease_by_spacing_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/12_disease_by_spacing_boxplot.png")

# Box plots for instar
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, disease in enumerate(disease_cols):
    df.boxplot(column=disease, by='Instar', ax=axes[i])
    axes[i].set_xlabel('Instar')
    axes[i].set_ylabel(disease)
    axes[i].tick_params(axis='x', rotation=45)
plt.suptitle('')
plt.tight_layout()
plt.savefig('figures/13_disease_by_instar_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/13_disease_by_instar_boxplot.png")

# =====================
# 6. PEST-DISEASE INTERACTION
# =====================
print("\n6. ANALYZING PEST-DISEASE INTERACTIONS...")

# Correlation between pests and diseases
pest_disease_corr = []
pest_cols = ['Has_Uzi', 'Has_Mites', 'Has_Ants', 'Has_Spiders', 'Has_Athropoda', 'Total_Pests']
for pest in pest_cols:
    for disease in disease_cols:
        valid_data = df[[pest, disease]].dropna()
        if len(valid_data) > 3:
            r, p = pearsonr(valid_data[pest], valid_data[disease])
            pest_disease_corr.append({
                'Pest': pest,
                'Disease': disease,
                'Correlation': r,
                'P_value': p,
                'Significant': 'Yes' if p < 0.05 else 'No'
            })

pest_disease_df = pd.DataFrame(pest_disease_corr)
pest_disease_df.to_csv('results/pest_disease_correlations.csv', index=False)
print("Pest-disease correlations saved to results/pest_disease_correlations.csv")

# Heatmap of pest-disease correlations
pivot_corr = pest_disease_df.pivot(index='Pest', columns='Disease', values='Correlation')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
plt.tight_layout()
plt.savefig('figures/14_pest_disease_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/14_pest_disease_heatmap.png")

# Stacked bar: Disease by pest presence
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, pest in enumerate(pest_cols):
    if i < 6:
        crosstab = pd.crosstab(df[pest], df['Any_Disease'])
        crosstab.plot(kind='bar', stacked=True, ax=axes[i], color=['lightblue', 'salmon'])
        axes[i].set_xlabel(f'{pest} (0=No, 1=Yes)')
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].legend(['No Disease', 'Disease Present'])
plt.tight_layout()
plt.savefig('figures/15_disease_by_pest_stacked.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/15_disease_by_pest_stacked.png")

# =====================
# 7. NET TECHNOLOGY EFFICACY
# =====================
print("\n7. ANALYZING NET TECHNOLOGY EFFICACY...")

if 'Net_Tech' in df.columns and df['Net_Tech'].nunique() > 1:
    # T-test for net tech effect
    net_tech_results = []
    for disease in disease_cols:
        yes_group = df[df['Net_Tech'] == 'Yes'][disease].dropna()
        no_group = df[df['Net_Tech'] == 'No'][disease].dropna()
        if len(yes_group) > 1 and len(no_group) > 1:
            t_stat, p_val = ttest_ind(yes_group, no_group)
            net_tech_results.append({
                'Disease': disease,
                'Mean_With_Net': yes_group.mean(),
                'Mean_Without_Net': no_group.mean(),
                'Difference': yes_group.mean() - no_group.mean(),
                'T_statistic': t_stat,
                'P_value': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })
    
    net_tech_df = pd.DataFrame(net_tech_results)
    net_tech_df.to_csv('results/net_technology_efficacy.csv', index=False)
    print("Net technology efficacy saved to results/net_technology_efficacy.csv")
    
    # Box plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, disease in enumerate(disease_cols):
        df.boxplot(column=disease, by='Net_Tech', ax=axes[i])
        axes[i].set_xlabel('Net Technology')
        axes[i].set_ylabel(disease)
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('figures/16_net_tech_efficacy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/16_net_tech_efficacy.png")

# =====================
# 8. TIME-SERIES ANALYSIS
# =====================
print("\n8. PERFORMING TIME-SERIES ANALYSIS...")

df_time = df.dropna(subset=['Date_parsed']).copy()
if len(df_time) > 5:
    # Daily aggregation
    daily_data = df_time.groupby('Date_parsed').agg({
        'Tmax': 'mean', 'Tmin': 'mean', 'Humidity': 'mean', 'THI': 'mean',
        'Pebrine': 'sum', 'Virosis': 'sum', 'Bacteriosis': 'sum', 'Muscardine': 'sum',
        'Total_Pests': 'sum'
    }).reset_index()
    
    # Calculate 3-day rolling averages
    for col in ['Tmax', 'THI', 'Humidity']:
        daily_data[f'{col}_3day'] = daily_data[col].rolling(window=3, min_periods=1).mean()
    
    # Lag correlation (climate today vs disease tomorrow)
    lag_corr_results = []
    for disease in disease_cols:
        for climate in ['THI_3day', 'Tmax_3day', 'Humidity_3day']:
            if climate in daily_data.columns:
                # Lag 1
                valid = daily_data[[climate, disease]].dropna()
                if len(valid) > 3:
                    r, p = pearsonr(valid[climate][:-1], valid[disease][1:])
                    lag_corr_results.append({
                        'Disease': disease,
                        'Climate': climate,
                        'Lag': 1,
                        'Correlation': r,
                        'P_value': p
                    })
    
    lag_corr_df = pd.DataFrame(lag_corr_results)
    if len(lag_corr_df) > 0:
        lag_corr_df.to_csv('results/lag_correlation_analysis.csv', index=False)
        print("Lag correlation analysis saved to results/lag_correlation_analysis.csv")
    
    # Plot rolling averages with disease
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    ax1 = axes[0]
    ax1.plot(daily_data['Date_parsed'], daily_data['THI_3day'], 'b-', label='THI (3-day avg)')
    ax1.set_ylabel('THI', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(daily_data['Date_parsed'], daily_data['Virosis'], 'r-o', label='Virosis')
    ax2.set_ylabel('Virosis Count', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel('Date')
    
    ax1 = axes[1]
    ax1.plot(daily_data['Date_parsed'], daily_data['Tmax_3day'], 'g-', label='Tmax (3-day avg)')
    ax1.set_ylabel('Tmax (°C)', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax2 = ax1.twinx()
    ax2.plot(daily_data['Date_parsed'], daily_data['Pebrine'], 'r-o', label='Pebrine')
    ax2.set_ylabel('Pebrine Count', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel('Date')
    
    ax1 = axes[2]
    ax1.plot(daily_data['Date_parsed'], daily_data['Humidity_3day'], 'c-', label='Humidity (3-day avg)')
    ax1.set_ylabel('Humidity (%)', color='c')
    ax1.tick_params(axis='y', labelcolor='c')
    ax2 = ax1.twinx()
    ax2.plot(daily_data['Date_parsed'], daily_data['Bacteriosis'], 'r-o', label='Bacteriosis')
    ax2.set_ylabel('Bacteriosis Count', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('figures/17_timeseries_with_disease.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/17_timeseries_with_disease.png")

# =====================
# 9. PCA AND MULTIVARIATE ANALYSIS
# =====================
print("\n9. PERFORMING PCA AND MULTIVARIATE ANALYSIS...")

# Prepare data for PCA
pca_vars = ['Tmax', 'Tmin', 'Humidity', 'THI', 'Wind_Speed', 'Temp_Range']
pca_data = df[pca_vars].dropna()

if len(pca_data) > 5:
    scaler = StandardScaler()
    pca_scaled = scaler.fit_transform(pca_data)
    
    pca = PCA()
    pca_result = pca.fit_transform(pca_scaled)
    
    # Explained variance
    explained_var = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(pca_vars))],
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    explained_var.to_csv('results/pca_explained_variance.csv', index=False)
    print("PCA explained variance saved to results/pca_explained_variance.csv")
    
    # Plot explained variance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(explained_var['Component'], explained_var['Explained_Variance_Ratio'], alpha=0.7)
    ax.plot(explained_var['Component'], explained_var['Cumulative_Variance'], 'ro-')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.axhline(y=0.9, color='g', linestyle='--', label='90% threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/18_pca_explained_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/18_pca_explained_variance.png")
    
    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca_vars))],
        index=pca_vars
    )
    loadings.to_csv('results/pca_loadings.csv')
    print("PCA loadings saved to results/pca_loadings.csv")
    
    # Plot loadings heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(loadings.iloc[:, :3], annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    plt.tight_layout()
    plt.savefig('figures/19_pca_loadings.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/19_pca_loadings.png")
    
    # PCA scatter plot colored by disease
    for disease in disease_cols:
        fig, ax = plt.subplots(figsize=(10, 8))
        disease_values = df[disease].values[:len(pca_result)]
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=disease_values, 
                           cmap='Reds', alpha=0.6, s=50)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(disease)
        plt.tight_layout()
        plt.savefig(f'figures/20_pca_scatter_{disease}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: figures/20_pca_scatter_{disease}.png")

# =====================
# 10. PREDICTIVE MODELING
# =====================
print("\n10. PERFORMING PREDICTIVE MODELING...")

# Prepare features
feature_cols = ['Tmax', 'Tmin', 'Humidity', 'THI', 'Wind_Speed', 'Temp_Range', 
                'Has_Uzi', 'Has_Mites', 'Has_Ants', 'Has_Spiders', 'Has_Athropoda']

# Add encoded categorical variables
if 'Spacing' in df.columns:
    spacing_encoded = pd.get_dummies(df['Spacing'], prefix='Spacing')
    df = pd.concat([df, spacing_encoded], axis=1)
    feature_cols.extend(spacing_encoded.columns.tolist())

if 'Net_Tech' in df.columns:
    df['Net_Tech_Binary'] = (df['Net_Tech'] == 'Yes').astype(int)
    feature_cols.append('Net_Tech_Binary')

model_results = {}

for disease in disease_cols:
    print(f"\n  Modeling {disease}...")
    
    # Prepare data
    model_data = df[feature_cols + [disease]].dropna()
    if len(model_data) < 10:
        continue
        
    X = model_data[feature_cols]
    y = (model_data[disease] > 0).astype(int)  # Binary classification
    
    if y.nunique() < 2:
        continue
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    importance_df.to_csv(f'results/feature_importance_{disease}.csv', index=False)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
    ax.set_xlabel('Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'figures/21_feature_importance_{disease}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/21_feature_importance_{disease}.png")
    
    # ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    auc_lr = auc(fpr_lr, tpr_lr)
    ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})')
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
    auc_rf = auc(fpr_rf, tpr_rf)
    ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'figures/22_roc_curve_{disease}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/22_roc_curve_{disease}.png")
    
    model_results[disease] = {
        'LR_AUC': auc_lr,
        'RF_AUC': auc_rf,
        'N_samples': len(model_data),
        'N_positive': y.sum()
    }

# Save model summary
model_summary = pd.DataFrame(model_results).T
model_summary.to_csv('results/model_performance_summary.csv')
print("\nModel performance summary saved to results/model_performance_summary.csv")

# =====================
# 11. THRESHOLD ANALYSIS
# =====================
print("\n11. PERFORMING THRESHOLD ANALYSIS...")

threshold_results = []

for disease in disease_cols:
    for climate in ['THI', 'Tmax', 'Humidity']:
        valid_data = df[[disease, climate]].dropna()
        if len(valid_data) < 10:
            continue
            
        # Find optimal threshold using Youden's J statistic
        y_true = (valid_data[disease] > 0).astype(int)
        
        if y_true.nunique() < 2:
            continue
            
        # Try different thresholds
        thresholds = np.linspace(valid_data[climate].min(), valid_data[climate].max(), 50)
        best_j = 0
        best_threshold = None
        
        for thresh in thresholds:
            y_pred = (valid_data[climate] > thresh).astype(int)
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            
            if (tp + fn) > 0 and (tn + fp) > 0:
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                j = sensitivity + specificity - 1
                
                if j > best_j:
                    best_j = j
                    best_threshold = thresh
        
        if best_threshold is not None:
            threshold_results.append({
                'Disease': disease,
                'Climate_Variable': climate,
                'Optimal_Threshold': best_threshold,
                'Youden_J': best_j,
                'Mean_Disease_Above': valid_data[valid_data[climate] > best_threshold][disease].mean(),
                'Mean_Disease_Below': valid_data[valid_data[climate] <= best_threshold][disease].mean()
            })

threshold_df = pd.DataFrame(threshold_results)
if len(threshold_df) > 0:
    threshold_df.to_csv('results/threshold_analysis.csv', index=False)
    print("Threshold analysis saved to results/threshold_analysis.csv")
    
    # Plot threshold effects
    for _, row in threshold_df.iterrows():
        disease = row['Disease']
        climate = row['Climate_Variable']
        thresh = row['Optimal_Threshold']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_data = df[[disease, climate]].dropna()
        
        below = valid_data[valid_data[climate] <= thresh][disease]
        above = valid_data[valid_data[climate] > thresh][disease]
        
        ax.boxplot([below, above], labels=[f'{climate} ≤ {thresh:.1f}', f'{climate} > {thresh:.1f}'])
        ax.set_ylabel(disease)
        plt.tight_layout()
        plt.savefig(f'figures/23_threshold_{disease}_{climate}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: figures/23_threshold_{disease}_{climate}.png")

# =====================
# 12. SUMMARY REPORT
# =====================
print("\n12. GENERATING SUMMARY REPORT...")

with open('results/analysis_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SILKWORM DISEASE ANALYSIS - RANCHI 2025\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATA OVERVIEW:\n")
    f.write(f"  Total observations: {len(df)}\n")
    f.write(f"  Date range: {df['Date_parsed'].min()} to {df['Date_parsed'].max()}\n")
    f.write(f"  Number of plots: {df['Plot_No'].nunique()}\n")
    f.write(f"  Spacing types: {', '.join(df['Spacing'].dropna().unique())}\n")
    f.write(f"  Instar stages: {', '.join(df['Instar'].dropna().unique())}\n\n")
    
    f.write("DISEASE SUMMARY:\n")
    for disease in disease_cols:
        total = df[disease].sum()
        mean = df[disease].mean()
        f.write(f"  {disease}: Total={total}, Mean={mean:.3f}\n")
    f.write("\n")
    
    f.write("CLIMATE SUMMARY:\n")
    for var in ['Tmax', 'Tmin', 'Humidity', 'THI']:
        f.write(f"  {var}: Mean={df[var].mean():.2f}, Range={df[var].min():.2f}-{df[var].max():.2f}\n")
    f.write("\n")
    
    f.write("KEY CORRELATIONS (p < 0.05):\n")
    sig_corr = corr_df[corr_df['Significant'] == 'Yes']
    if len(sig_corr) > 0:
        for _, row in sig_corr.iterrows():
            f.write(f"  {row['Disease']} vs {row['Climate_Variable']}: r={row['Correlation']:.3f}, p={row['P_value']:.4f}\n")
    else:
        f.write("  No significant correlations found\n")
    f.write("\n")
    
    f.write("SPACING EFFECTS:\n")
    for _, row in spacing_anova_df.iterrows():
        sig = "Significant" if row['Significant'] == 'Yes' else "Not significant"
        f.write(f"  {row['Disease']}: F={row['F_statistic']:.3f}, p={row['P_value']:.4f} ({sig})\n")
    f.write("\n")
    
    f.write("INSTAR EFFECTS:\n")
    for _, row in instar_anova_df.iterrows():
        sig = "Significant" if row['Significant'] == 'Yes' else "Not significant"
        f.write(f"  {row['Disease']}: F={row['F_statistic']:.3f}, p={row['P_value']:.4f} ({sig})\n")
    f.write("\n")
    
    f.write("PEST-DISEASE CORRELATIONS:\n")
    sig_pest = pest_disease_df[pest_disease_df['Significant'] == 'Yes']
    if len(sig_pest) > 0:
        for _, row in sig_pest.iterrows():
            f.write(f"  {row['Pest']} vs {row['Disease']}: r={row['Correlation']:.3f}, p={row['P_value']:.4f}\n")
    else:
        f.write("  No significant pest-disease correlations found\n")
    f.write("\n")
    
    if 'Net_Tech' in df.columns and len(net_tech_df) > 0:
        f.write("NET TECHNOLOGY EFFICACY:\n")
        for _, row in net_tech_df.iterrows():
            sig = "Significant" if row['Significant'] == 'Yes' else "Not significant"
            f.write(f"  {row['Disease']}: Mean with net={row['Mean_With_Net']:.3f}, "
                   f"without={row['Mean_Without_Net']:.3f}, p={row['P_value']:.4f} ({sig})\n")
        f.write("\n")
    
    f.write("MODEL PERFORMANCE:\n")
    for disease, results in model_results.items():
        f.write(f"  {disease}:\n")
        f.write(f"    Logistic Regression AUC: {results['LR_AUC']:.3f}\n")
        f.write(f"    Random Forest AUC: {results['RF_AUC']:.3f}\n")
        f.write(f"    Samples: {results['N_samples']}, Positive cases: {results['N_positive']}\n")
    f.write("\n")
    
    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print("Summary report saved to results/analysis_summary.txt")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nResults saved in 'results/' directory")
print("Figures saved in 'figures/' directory")
print(f"\nTotal figures generated: {len([f for f in os.listdir('figures') if f.endswith('.png')])}")
