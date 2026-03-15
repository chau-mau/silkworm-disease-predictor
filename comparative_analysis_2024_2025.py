"""
Comparative Analysis: 2024 vs 2025
Silkworm Disease Analysis - Ranchi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

import os
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("="*80)
print("COMPARATIVE ANALYSIS: 2024 vs 2025")
print("="*80)

# Load 2024 data
df_2024 = pd.read_excel('plots (1).xlsx')
df_2024['Date'] = pd.to_datetime(df_2024['Date'], format='mixed', dayfirst=True, errors='coerce')
df_2024['Year'] = 2024

# Convert yes/no to 1/0
for col in ['PB', 'VR', 'BT', 'FG']:
    df_2024[col] = df_2024[col].map({'yes': 1, 'no': 0})

# Rename columns to match 2025
df_2024 = df_2024.rename(columns={'PB': 'Pebrine', 'VR': 'Virosis', 'BT': 'Bacteriosis', 'FG': 'Flacherie'})

print(f"\n2024 Data: {len(df_2024)} observations")
print(f"Date range: {df_2024['Date'].min()} to {df_2024['Date'].max()}")

# Load 2025 cleaned data
df_2025 = pd.read_csv('results/cleaned_data_2025.csv')
df_2025['Date_parsed'] = pd.to_datetime(df_2025['Date_parsed'])
df_2025['Year'] = 2025

print(f"\n2025 Data: {len(df_2025)} observations")
print(f"Date range: {df_2025['Date_parsed'].min()} to {df_2025['Date_parsed'].max()}")

# Summary statistics comparison
disease_cols = ['Pebrine', 'Virosis', 'Bacteriosis']

print("\n" + "="*80)
print("DISEASE INCIDENCE COMPARISON")
print("="*80)

comparison_data = []
for disease in disease_cols:
    count_2024 = df_2024[disease].sum()
    mean_2024 = df_2024[disease].mean()
    count_2025 = df_2025[disease].sum()
    mean_2025 = df_2025[disease].mean()
    
    comparison_data.append({
        'Disease': disease,
        'Count_2024': count_2024,
        'Mean_2024': mean_2024,
        'Count_2025': count_2025,
        'Mean_2025': mean_2025,
        'Difference_Count': count_2025 - count_2024,
        'Difference_Mean': mean_2025 - mean_2024
    })
    
    print(f"\n{disease}:")
    print(f"  2024: {count_2024} cases (mean: {mean_2024:.3f})")
    print(f"  2025: {count_2025} cases (mean: {mean_2025:.3f})")
    print(f"  Difference: {count_2025 - count_2024} cases")

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('results/comparison_2024_2025.csv', index=False)
print("\nComparison saved to results/comparison_2024_2025.csv")

# Visualization 1: Disease incidence comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(disease_cols))
width = 0.35
bars1 = ax.bar(x - width/2, comparison_df['Mean_2024'], width, label='2024', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, comparison_df['Mean_2025'], width, label='2025', color='salmon', edgecolor='black')
ax.set_xlabel('Disease')
ax.set_ylabel('Mean Incidence')
ax.set_xticks(x)
ax.set_xticklabels(disease_cols)
ax.legend()
plt.tight_layout()
plt.savefig('figures/24_disease_comparison_2024_2025.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/24_disease_comparison_2024_2025.png")

# Visualization 2: Daily disease pattern (2024)
if 'Date' in df_2024.columns and df_2024['Date'].notna().sum() > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    for disease in disease_cols:
        daily = df_2024.groupby('Date')[disease].sum()
        ax.plot(daily.index, daily.values, marker='o', label=disease)
    ax.set_xlabel('Date')
    ax.set_ylabel('Disease Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/25_disease_pattern_2024.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/25_disease_pattern_2024.png")

# Visualization 3: Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2024
disease_counts_2024 = [df_2024[d].sum() for d in disease_cols]
axes[0].bar(disease_cols, disease_counts_2024, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Disease')
axes[0].set_ylabel('Total Count')

# 2025
disease_counts_2025 = [df_2025[d].sum() for d in disease_cols]
axes[1].bar(disease_cols, disease_counts_2025, color='salmon', edgecolor='black')
axes[1].set_xlabel('Disease')
axes[1].set_ylabel('Total Count')

plt.tight_layout()
plt.savefig('figures/26_side_by_side_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/26_side_by_side_comparison.png")

# Statistical test for differences
print("\n" + "="*80)
print("STATISTICAL TESTS FOR YEAR DIFFERENCES")
print("="*80)

test_results = []
for disease in disease_cols:
    # Chi-square test
    data_2024 = df_2024[disease].fillna(0)
    data_2025 = df_2025[disease].fillna(0)
    
    # Create contingency table
    table = [[(data_2024 == 1).sum(), (data_2024 == 0).sum()],
             [(data_2025 == 1).sum(), (data_2025 == 0).sum()]]
    
    chi2, p_val, dof, expected = stats.chi2_contingency(table)
    
    test_results.append({
        'Disease': disease,
        'Chi2': chi2,
        'P_value': p_val,
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })
    
    print(f"\n{disease}:")
    print(f"  Chi-square: {chi2:.4f}")
    print(f"  P-value: {p_val:.4f}")
    print(f"  Significant difference: {'Yes' if p_val < 0.05 else 'No'}")

test_df = pd.DataFrame(test_results)
test_df.to_csv('results/statistical_tests_2024_2025.csv', index=False)
print("\nStatistical tests saved to results/statistical_tests_2024_2025.csv")

# Generate summary report
with open('results/comparative_analysis_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPARATIVE ANALYSIS: 2024 vs 2025\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATA OVERVIEW:\n")
    f.write(f"  2024: {len(df_2024)} observations\n")
    f.write(f"  2025: {len(df_2025)} observations\n\n")
    
    f.write("DISEASE INCIDENCE COMPARISON:\n")
    for _, row in comparison_df.iterrows():
        f.write(f"  {row['Disease']}:\n")
        f.write(f"    2024: {row['Count_2024']} cases (mean: {row['Mean_2024']:.3f})\n")
        f.write(f"    2025: {row['Count_2025']} cases (mean: {row['Mean_2025']:.3f})\n")
        f.write(f"    Difference: {row['Difference_Count']} cases\n\n")
    
    f.write("STATISTICAL TESTS:\n")
    for _, row in test_df.iterrows():
        sig = "Significant" if row['Significant'] == 'Yes' else "Not significant"
        f.write(f"  {row['Disease']}: Chi2={row['Chi2']:.4f}, p={row['P_value']:.4f} ({sig})\n")
    
    f.write("\n" + "="*80 + "\n")

print("\nComparative analysis summary saved to results/comparative_analysis_summary.txt")

print("\n" + "="*80)
print("COMPARATIVE ANALYSIS COMPLETE!")
print("="*80)
