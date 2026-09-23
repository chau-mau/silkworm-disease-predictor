"""
Fast ML Analysis with Bioclimatic Variables
Simplified version without extensive hyperparameter tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score)
import warnings
import json
import pickle
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("FAST ML ANALYSIS WITH BIOCLIMATIC VARIABLES")
print("="*80)

# Load data
df = pd.read_csv('../results/scanned_data_compiled.csv')
df['Date_parsed'] = pd.to_datetime(df['Date_parsed'])

# Calculate THI
def calculate_thi_nrc(temp_c, rh):
    temp_f = (1.8 * temp_c) + 32
    thi = temp_f - ((0.55 - 0.0055 * rh) * (temp_f - 58))
    return thi

df['Temp_Mean'] = (df['Tmax'] + df['Tmin']) / 2
df['THI_NRC'] = df.apply(lambda row: calculate_thi_nrc(row['Temp_Mean'], row['Humidity']), axis=1)

# Clean disease data
disease_cols = ['Pebrine', 'Virosis', 'Bacteriosis', 'Muscardine']
for col in disease_cols:
    df[col] = df[col].apply(lambda x: 1 if pd.notna(x) and str(x).strip() not in ['', '0', '0000', '00000'] else 0)

# Load bioclimatic variables
bio_vars = pd.read_csv('data/bioclimatic_variables.csv')
df_full = pd.concat([df.reset_index(drop=True), bio_vars], axis=1)

# Features
bioclimatic_features = [f'BIO{i}' for i in range(1, 20)]
climate_features = ['Tmax', 'Tmin', 'Temp_Mean', 'Humidity', 'THI_NRC']
df_full['Spacing_encoded'] = LabelEncoder().fit_transform(df_full['Spacing'].astype(str))
df_full['Instar_encoded'] = LabelEncoder().fit_transform(df_full['Instar'].astype(str))
management_features = ['Plot_No', 'Spacing_encoded', 'Instar_encoded']
all_features = bioclimatic_features + climate_features + management_features

X = df_full[all_features].fillna(df_full[all_features].mean())

print(f"\nFeature matrix: {X.shape}")

# Define models (simplified - no hyperparameter tuning)
models_dict = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate
results = {}
ensemble_models = {}

for disease in disease_cols:
    print(f"\n{'='*60}")
    print(f"Disease: {disease}")
    print('='*60)
    
    y = df_full[disease].values
    
    if y.sum() < 5:
        print(f"  Skipping - insufficient cases ({y.sum()})")
        continue
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    disease_results = {}
    trained_models = {}
    
    for name, model in models_dict.items():
        try:
            # Use scaled data for some models
            if name in ['LogisticRegression', 'SVM', 'KNN']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'cohen_kappa': cohen_kappa_score(y_test, y_pred)
            }
            
            disease_results[name] = metrics
            trained_models[name] = model
            
            print(f"  {name:20s} - ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"  {name:20s} - Error: {e}")
    
    results[disease] = disease_results
    
    # Select top 3 by ROC-AUC
    top_3 = sorted(disease_results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)[:3]
    top_3_names = [x[0] for x in top_3]
    print(f"\n  Top 3 models: {top_3_names}")
    
    # Create ensemble
    estimators = [(name, trained_models[name]) for name in top_3_names]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    # Fit ensemble
    if sum(1 for n in top_3_names if n in ['LogisticRegression', 'SVM', 'KNN']) >= 2:
        ensemble.fit(X_train_scaled, y_train)
        y_pred = ensemble.predict(X_test_scaled)
        y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]
    else:
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_prob = ensemble.predict_proba(X_test)[:, 1]
    
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred)
    }
    
    ensemble_models[disease] = {
        'ensemble': ensemble,
        'top_3': top_3_names,
        'metrics': ensemble_metrics,
        'scaler': scaler
    }
    
    print(f"\n  ENSEMBLE Performance:")
    for k, v in ensemble_metrics.items():
        print(f"    {k}: {v:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{disease} - Ensemble Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'figures/confusion_matrix_{disease}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Save results
pickle.dump(results, open('results/all_model_results.pkl', 'wb'))
pickle.dump(ensemble_models, open('results/ensemble_models.pkl', 'wb'))

# Create comparison table
comparison_rows = []
for disease, model_results in results.items():
    for model_name, metrics in model_results.items():
        row = {'Disease': disease, 'Model': model_name}
        row.update(metrics)
        comparison_rows.append(row)
    
    # Add ensemble
    row = {'Disease': disease, 'Model': 'ENSEMBLE'}
    row.update(ensemble_models[disease]['metrics'])
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv('results/model_comparison.csv', index=False)

# Visualizations
# 1. ROC-AUC comparison
plt.figure(figsize=(14, 8))
pivot = comparison_df.pivot_table(index='Model', columns='Disease', values='roc_auc')
pivot.plot(kind='bar', width=0.8)
plt.title('ROC-AUC Comparison', fontsize=14)
plt.ylabel('ROC-AUC')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.legend(title='Disease', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('figures/roc_auc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. F1-Score comparison
plt.figure(figsize=(14, 8))
pivot = comparison_df.pivot_table(index='Model', columns='Disease', values='f1')
pivot.plot(kind='bar', width=0.8)
plt.title('F1-Score Comparison', fontsize=14)
plt.ylabel('F1-Score')
plt.legend(title='Disease', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('figures/f1_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Feature importance from Random Forest
plt.figure(figsize=(12, 8))
for i, disease in enumerate(disease_cols):
    if disease in results and 'RandomForest' in results[disease]:
        rf_model = models_dict['RandomForest']
        # Re-train to get feature importances
        y = df_full[disease].values
        rf_model.fit(X, y)
        importances = pd.Series(rf_model.feature_importances_, index=all_features).nlargest(10)
        plt.subplot(2, 2, i+1)
        importances.plot(kind='barh')
        plt.title(f'{disease} - Top 10 Features')

plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Generate report
with open('results/analysis_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ML ANALYSIS WITH BIOCLIMATIC VARIABLES - RESULTS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset: {len(df_full)} records\n")
    f.write(f"Features: {len(all_features)} (19 bioclimatic + climate + management)\n\n")
    
    f.write("DISEASE DISTRIBUTION:\n")
    for disease in disease_cols:
        count = df_full[disease].sum()
        f.write(f"  {disease}: {count} cases ({count/len(df_full)*100:.1f}%)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("MODEL PERFORMANCE\n")
    f.write("="*80 + "\n\n")
    
    for disease in results.keys():
        f.write(f"\n{disease.upper()}:\n")
        f.write("-"*60 + "\n")
        
        for model_name, metrics in results[disease].items():
            f.write(f"\n{model_name}:\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
        
        f.write(f"\n*** ENSEMBLE (Top 3) ***\n")
        f.write(f"Models: {ensemble_models[disease]['top_3']}\n")
        for k, v in ensemble_models[disease]['metrics'].items():
            f.write(f"  {k}: {v:.4f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("BEST MODELS BY DISEASE (by ROC-AUC)\n")
    f.write("="*80 + "\n\n")
    
    for disease in results.keys():
        best = max(results[disease].items(), key=lambda x: x[1]['roc_auc'])
        f.write(f"{disease}: {best[0]} (ROC-AUC: {best[1]['roc_auc']:.4f})\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - results/model_comparison.csv")
print("  - results/all_model_results.pkl")
print("  - results/ensemble_models.pkl")
print("  - results/analysis_report.txt")
print("  - figures/roc_auc_comparison.png")
print("  - figures/f1_comparison.png")
print("  - figures/feature_importance.png")
print("  - figures/confusion_matrix_*.png")
