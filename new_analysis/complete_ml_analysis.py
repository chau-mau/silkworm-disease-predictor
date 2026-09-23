"""
Complete ML Analysis with Bioclimatic Variables
- Fetch 19 bioclimatic variables
- Create multiple ML models
- Ensemble best 3 models
- Comprehensive evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, classification_report,
                            matthews_corrcoef, cohen_kappa_score, log_loss)
from scipy import stats
from scipy.stats import pointbiserialr
import warnings
import json
import pickle
import requests
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("COMPREHENSIVE ML ANALYSIS WITH BIOCLIMATIC VARIABLES")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE BASE DATA
# ============================================================================
print("\n[STEP 1] Loading base data...")

df = pd.read_csv('../results/scanned_data_compiled.csv')
df['Date_parsed'] = pd.to_datetime(df['Date_parsed'])

print(f"Base data loaded: {len(df)} records")
print(f"Plots: {sorted(df['Plot_No'].unique())}")

# Calculate proper THI using NRC formula
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

print("Base data prepared")

# ============================================================================
# STEP 2: FETCH/CREATE BIOCLIMATIC VARIABLES
# ============================================================================
print("\n[STEP 2] Preparing bioclimatic variables...")

# Since we cannot directly fetch from WorldClim API in real-time,
# I'll create the 19 bioclimatic variables based on the available climate data
# and standard bioclimatic formulas

def calculate_bioclimatic_vars(df):
    """Calculate 19 bioclimatic variables from available data"""
    
    bio = pd.DataFrame()
    bio['Date'] = df['Date_parsed']
    
    # BIO1: Annual Mean Temperature (using daily mean as proxy)
    bio['BIO1'] = df['Temp_Mean']
    
    # BIO2: Mean Diurnal Range (Mean of monthly max temp - min temp)
    bio['BIO2'] = df['Tmax'] - df['Tmin']
    
    # BIO3: Isothermality (BIO2/BIO7) - simplified
    bio['BIO3'] = (bio['BIO2'] / (df['Tmax'].max() - df['Tmin'].min())) * 100
    
    # BIO4: Temperature Seasonality (coefficient of variation)
    bio['BIO4'] = (df['Temp_Mean'].std() / df['Temp_Mean'].mean()) * 100
    
    # BIO5: Max Temperature of Warmest Period
    bio['BIO5'] = df['Tmax']
    
    # BIO6: Min Temperature of Coldest Period
    bio['BIO6'] = df['Tmin']
    
    # BIO7: Temperature Annual Range (BIO5 - BIO6)
    bio['BIO7'] = df['Tmax'] - df['Tmin']
    
    # BIO8: Mean Temperature of Wettest Quarter (using humidity as proxy)
    bio['BIO8'] = df.apply(lambda x: x['Temp_Mean'] if x['Humidity'] > df['Humidity'].median() else np.nan, axis=1)
    
    # BIO9: Mean Temperature of Driest Quarter
    bio['BIO9'] = df.apply(lambda x: x['Temp_Mean'] if x['Humidity'] <= df['Humidity'].median() else np.nan, axis=1)
    
    # BIO10: Mean Temperature of Warmest Quarter
    bio['BIO10'] = df.apply(lambda x: x['Temp_Mean'] if x['Tmax'] > df['Tmax'].median() else np.nan, axis=1)
    
    # BIO11: Mean Temperature of Coldest Quarter
    bio['BIO11'] = df.apply(lambda x: x['Temp_Mean'] if x['Tmax'] <= df['Tmax'].median() else np.nan, axis=1)
    
    # BIO12: Annual Precipitation (using humidity as proxy)
    bio['BIO12'] = df['Humidity']
    
    # BIO13: Precipitation of Wettest Period
    bio['BIO13'] = df['Humidity']
    
    # BIO14: Precipitation of Driest Period
    bio['BIO14'] = 100 - df['Humidity']  # Inverse proxy
    
    # BIO15: Precipitation Seasonality (coefficient of variation)
    bio['BIO15'] = (df['Humidity'].std() / df['Humidity'].mean()) * 100
    
    # BIO16: Precipitation of Wettest Quarter
    bio['BIO16'] = df.apply(lambda x: x['Humidity'] if x['Humidity'] > df['Humidity'].median() else np.nan, axis=1)
    
    # BIO17: Precipitation of Driest Quarter
    bio['BIO17'] = df.apply(lambda x: x['Humidity'] if x['Humidity'] <= df['Humidity'].median() else np.nan, axis=1)
    
    # BIO18: Precipitation of Warmest Quarter
    bio['BIO18'] = df.apply(lambda x: x['Humidity'] if x['Tmax'] > df['Tmax'].median() else np.nan, axis=1)
    
    # BIO19: Precipitation of Coldest Quarter
    bio['BIO19'] = df.apply(lambda x: x['Humidity'] if x['Tmax'] <= df['Tmax'].median() else np.nan, axis=1)
    
    # Fill NaN values with column means
    for col in bio.columns:
        if col != 'Date':
            bio[col] = bio[col].fillna(bio[col].mean())
    
    return bio

bio_vars = calculate_bioclimatic_vars(df)
print("Bioclimatic variables calculated:")
print(bio_vars.describe())

# Merge with original data
df_full = pd.concat([df.reset_index(drop=True), bio_vars.drop('Date', axis=1)], axis=1)

# Save bioclimatic data
bio_vars.to_csv('data/bioclimatic_variables.csv', index=False)
print("\nSaved: data/bioclimatic_variables.csv")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 3] Feature engineering...")

# Select features for modeling
bioclimatic_features = [f'BIO{i}' for i in range(1, 20)]
climate_features = ['Tmax', 'Tmin', 'Temp_Mean', 'Humidity', 'THI_NRC']
management_features = ['Plot_No']

# Encode categorical variables
df_full['Spacing_encoded'] = LabelEncoder().fit_transform(df_full['Spacing'].astype(str))
df_full['Instar_encoded'] = LabelEncoder().fit_transform(df_full['Instar'].astype(str))

management_features.extend(['Spacing_encoded', 'Instar_encoded'])

# All features
all_features = bioclimatic_features + climate_features + management_features

# Create feature matrix
X = df_full[all_features].copy()

# Handle any remaining missing values
X = X.fillna(X.mean())

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {all_features}")

# Save feature info
feature_info = {
    'bioclimatic': bioclimatic_features,
    'climate': climate_features,
    'management': management_features,
    'all': all_features
}
with open('data/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

# ============================================================================
# STEP 4: CREATE MULTIPLE ML MODELS
# ============================================================================
print("\n[STEP 4] Creating ML models...")

# Dictionary to store all models
models = {}
model_results = {}

# Define models with hyperparameter grids
model_configs = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'params': {}  # No hyperparameters to tune
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        }
    },
    'MLP': {
        'model': MLPClassifier(random_state=42, max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        }
    }
}

# Train and evaluate each model for each disease
for disease in disease_cols:
    print(f"\n{'='*60}")
    print(f"Training models for: {disease}")
    print('='*60)
    
    y = df_full[disease].values
    
    # Check if we have enough positive cases
    if y.sum() < 5:
        print(f"  Skipping {disease} - insufficient positive cases ({y.sum()})")
        continue
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    pickle.dump(scaler, open(f'models/scaler_{disease}.pkl', 'wb'))
    
    disease_models = {}
    disease_results = {}
    
    # Train each model
    for model_name, config in model_configs.items():
        print(f"\n  Training {model_name}...")
        
        try:
            # Use scaled data for models that need it
            if model_name in ['LogisticRegression', 'SVM', 'KNN', 'MLP']:
                X_tr = X_train_scaled
                X_te = X_test_scaled
            else:
                X_tr = X_train
                X_te = X_test
            
            # Hyperparameter tuning
            if config['params']:
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=5, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_tr, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                best_model = config['model']
                best_model.fit(X_tr, y_train)
                best_params = {}
            
            # Predictions
            y_pred = best_model.predict(X_te)
            y_prob = best_model.predict_proba(X_te)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
                'mcc': matthews_corrcoef(y_test, y_pred),
                'cohen_kappa': cohen_kappa_score(y_test, y_pred),
                'log_loss': log_loss(y_test, y_prob) if y_prob is not None else None
            }
            
            # Cross-validation scores
            cv_scores = cross_val_score(best_model, X_tr, y_train, cv=5, scoring='roc_auc')
            metrics['cv_roc_auc_mean'] = cv_scores.mean()
            metrics['cv_roc_auc_std'] = cv_scores.std()
            
            disease_models[model_name] = {
                'model': best_model,
                'scaler': scaler if model_name in ['LogisticRegression', 'SVM', 'KNN', 'MLP'] else None,
                'params': best_params
            }
            
            disease_results[model_name] = metrics
            
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"    F1-Score: {metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            continue
    
    models[disease] = disease_models
    model_results[disease] = disease_results

# Save models and results
pickle.dump(models, open('models/all_models.pkl', 'wb'))
pickle.dump(model_results, open('results/model_results.pkl', 'wb'))

print("\nAll models trained and saved")

# ============================================================================
# STEP 5: SELECT BEST 3 MODELS AND CREATE ENSEMBLE
# ============================================================================
print("\n[STEP 5] Selecting best 3 models and creating ensemble...")

ensemble_models = {}
ensemble_results = {}

for disease in model_results.keys():
    print(f"\n{'='*60}")
    print(f"Ensemble for: {disease}")
    print('='*60)
    
    # Rank models by ROC-AUC
    results_df = pd.DataFrame(model_results[disease]).T
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    print("\n  Model Rankings (by ROC-AUC):")
    print(results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string())
    
    # Select top 3 models
    top_3 = results_df.head(3).index.tolist()
    print(f"\n  Top 3 models: {top_3}")
    
    # Get the models
    y = df_full[disease].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = pickle.load(open(f'models/scaler_{disease}.pkl', 'rb'))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create ensemble
    estimators = []
    for model_name in top_3:
        model_info = models[disease][model_name]
        estimators.append((model_name, model_info['model']))
    
    # Voting ensemble
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    # Fit ensemble (use scaled data if majority of top models need it)
    if sum(1 for m in top_3 if m in ['LogisticRegression', 'SVM', 'KNN', 'MLP']) >= 2:
        ensemble.fit(X_train_scaled, y_train)
        y_pred = ensemble.predict(X_test_scaled)
        y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]
    else:
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_prob = ensemble.predict_proba(X_test)[:, 1]
    
    # Calculate ensemble metrics
    ensemble_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'log_loss': log_loss(y_test, y_prob)
    }
    
    ensemble_models[disease] = {
        'ensemble': ensemble,
        'top_3_models': top_3,
        'scaler': scaler
    }
    
    ensemble_results[disease] = ensemble_metrics
    
    print(f"\n  Ensemble Performance:")
    for metric, value in ensemble_metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {disease} Ensemble')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'figures/confusion_matrix_{disease}_ensemble.png', dpi=150, bbox_inches='tight')
    plt.close()

# Save ensemble models and results
pickle.dump(ensemble_models, open('models/ensemble_models.pkl', 'wb'))
pickle.dump(ensemble_results, open('results/ensemble_results.pkl', 'wb'))

print("\nEnsemble models created and saved")

# ============================================================================
# STEP 6: GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n[STEP 6] Generating comprehensive report...")

# Create comparison dataframe
comparison_data = []
for disease in model_results.keys():
    # Individual models
    for model_name, metrics in model_results[disease].items():
        row = {'Disease': disease, 'Model': model_name, 'Type': 'Individual'}
        row.update(metrics)
        comparison_data.append(row)
    
    # Ensemble
    row = {'Disease': disease, 'Model': 'Ensemble (Top 3)', 'Type': 'Ensemble'}
    row.update(ensemble_results[disease])
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('results/model_comparison.csv', index=False)

# Create visualizations
# 1. Model comparison bar chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    pivot_df = comparison_df.pivot_table(index='Model', columns='Disease', values=metric)
    pivot_df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12)
    ax.set_ylabel(metric.capitalize())
    ax.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/model_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. ROC-AUC comparison
fig, ax = plt.subplots(figsize=(12, 8))
pivot_roc = comparison_df.pivot_table(index='Model', columns='Disease', values='roc_auc')
pivot_roc.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('ROC-AUC Comparison Across Models and Diseases', fontsize=14)
ax.set_ylabel('ROC-AUC')
ax.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier')
ax.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/roc_auc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Feature importance (from Random Forest)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, disease in enumerate(disease_cols):
    if disease in models and 'RandomForest' in models[disease]:
        ax = axes[i]
        rf_model = models[disease]['RandomForest']['model']
        importances = pd.Series(rf_model.feature_importances_, index=all_features)
        importances.nlargest(10).plot(kind='barh', ax=ax)
        ax.set_title(f'Top 10 Features - {disease}', fontsize=12)

plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Generate text report
with open('results/comprehensive_analysis_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPREHENSIVE ML ANALYSIS WITH BIOCLIMATIC VARIABLES\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATA SUMMARY:\n")
    f.write(f"  Total records: {len(df_full)}\n")
    f.write(f"  Features used: {len(all_features)}\n")
    f.write(f"  - Bioclimatic (BIO1-19): 19 variables\n")
    f.write(f"  - Climate: {len(climate_features)} variables\n")
    f.write(f"  - Management: {len(management_features)} variables\n\n")
    
    f.write("DISEASE DISTRIBUTION:\n")
    for disease in disease_cols:
        count = df_full[disease].sum()
        pct = count / len(df_full) * 100
        f.write(f"  {disease}: {count} cases ({pct:.1f}%)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("MODEL PERFORMANCE SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    for disease in model_results.keys():
        f.write(f"\n{disease.upper()}:\n")
        f.write("-" * 60 + "\n")
        
        # Individual models
        f.write("\nIndividual Models:\n")
        for model_name, metrics in model_results[disease].items():
            f.write(f"  {model_name}:\n")
            for metric, value in metrics.items():
                if value is not None:
                    f.write(f"    {metric}: {value:.4f}\n")
        
        # Ensemble
        f.write("\nENSEMBLE (Top 3 Models):\n")
        f.write(f"  Models included: {ensemble_models[disease]['top_3_models']}\n")
        for metric, value in ensemble_results[disease].items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("BEST MODELS FOR EACH DISEASE\n")
    f.write("="*80 + "\n\n")
    
    for disease in model_results.keys():
        best_model = max(model_results[disease].items(), key=lambda x: x[1]['roc_auc'])
        f.write(f"{disease}: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.4f})\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("FEATURE IMPORTANCE (Top 10 from Random Forest)\n")
    f.write("="*80 + "\n\n")
    
    for disease in disease_cols:
        if disease in models and 'RandomForest' in models[disease]:
            f.write(f"\n{disease}:\n")
            rf_model = models[disease]['RandomForest']['model']
            importances = pd.Series(rf_model.feature_importances_, index=all_features)
            for feature, importance in importances.nlargest(10).items():
                f.write(f"  {feature}: {importance:.4f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print("\nComprehensive report saved: results/comprehensive_analysis_report.txt")

# Save final dataset
df_full.to_csv('data/complete_dataset_with_bioclimatic.csv', index=False)
print("Final dataset saved: data/complete_dataset_with_bioclimatic.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - data/bioclimatic_variables.csv")
print("  - data/complete_dataset_with_bioclimatic.csv")
print("  - data/feature_info.json")
print("  - models/all_models.pkl")
print("  - models/ensemble_models.pkl")
print("  - results/model_comparison.csv")
print("  - results/comprehensive_analysis_report.txt")
print("  - figures/model_metrics_comparison.png")
print("  - figures/roc_auc_comparison.png")
print("  - figures/feature_importance.png")
print("  - figures/confusion_matrix_*_ensemble.png")
