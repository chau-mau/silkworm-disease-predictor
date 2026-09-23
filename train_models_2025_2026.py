"""
Train and save models on the combined 2025 + 2026 dataset.

Changes vs train_models.py:
  - Training data: results/merged_data_2025_2026.csv (2025 field data +
    2026 daily records from typed August observations and 'Disease 2026.xlsx',
    climate from NASA POWER / Open-Meteo ERA5).
  - THI uses the corrected NRC (1971) formula for BOTH years, consistent with
    recalculate_thi_and_recreate_figure.py.
  - 2026 rows have unknown plot attributes -> all spacing dummies 0,
    Net_Tech_Binary 0, pest binaries 0 (documented limitation).

Outputs: models.pkl, model_info.json (root + app/ copies)
"""

import pandas as pd
import numpy as np
import pickle
import json
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Training models on combined 2025 + 2026 dataset...")

# Load combined dataset
df = pd.read_csv('results/merged_data_2025_2026.csv', parse_dates=['Date'])

# Define features (THI = NRC-corrected, consistent across years)
feature_cols = ['Tmax', 'Tmin', 'Humidity', 'THI', 'Wind_Speed',
                'Has_Uzi', 'Has_Mites', 'Has_Ants', 'Has_Spiders', 'Has_Athropoda']

# Spacing one-hot (2026 'Unknown' -> all-zero dummies, no Unknown column)
spacing_dummies = pd.get_dummies(df['Spacing'], prefix='Spacing')
spacing_dummies = spacing_dummies[[c for c in spacing_dummies.columns if c != 'Spacing_Unknown']]
df = pd.concat([df, spacing_dummies], axis=1)
feature_cols.extend(spacing_dummies.columns.tolist())

# Net technology binary ('Yes' -> 1; 'No'/'Unknown' -> 0)
df['Net_Tech_Binary'] = (df['Net_Tech'] == 'Yes').astype(int)
feature_cols.append('Net_Tech_Binary')

disease_cols = ['Pebrine', 'Virosis', 'Bacteriosis', 'Muscardine']

# Model metadata
model_info = {
    'features': feature_cols,
    'diseases': disease_cols,
    'spacing_options': sorted([c.replace('Spacing_', '') for c in spacing_dummies.columns]),
    'training_data': 'results/merged_data_2025_2026.csv (2025 field data + 2026 Aug-Sep records)',
    'thi_formula': 'NRC (1971): THI = (1.8T+32) - (0.55-0.0055*RH)*(1.8T-58), T = mean of Tmax/Tmin',
    'climate_source_2026': 'NASA POWER daily (Aug 1 - Sep 17, 2026); Open-Meteo ERA5 backfill (Sep 18-22, 2026)',
    'feature_descriptions': {
        'Tmax': 'Maximum Temperature (°C)',
        'Tmin': 'Minimum Temperature (°C)',
        'Humidity': 'Relative Humidity (%)',
        'THI': 'Temperature-Humidity Index (NRC 1971 formula)',
        'Wind_Speed': 'Wind Speed (m/s)',
        'Has_Uzi': 'Uzi Fly Present (0/1)',
        'Has_Mites': 'Mites Present (0/1)',
        'Has_Ants': 'Ants Present (0/1)',
        'Has_Spiders': 'Spiders Present (0/1)',
        'Has_Athropoda': 'Athropoda Present (0/1)',
        'Net_Tech_Binary': 'Net Technology Used (0/1)'
    }
}

trained_models = {}
scaler = StandardScaler()

for disease in disease_cols:
    print(f"\nTraining model for {disease}...")

    model_data = df[feature_cols + [disease, 'Year']].dropna()
    X = model_data[feature_cols].astype(float)
    y = (model_data[disease] > 0).astype(int)

    if y.nunique() < 2 or len(model_data) < 10:
        print(f"  Skipping {disease} - insufficient data variation")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))

    # Sanity check: 2026-only accuracy (temporal holdout)
    yr = model_data['Year']
    mask26 = (yr == 2026).values
    if mask26.sum() > 0 and y[mask26].nunique() == 2:
        acc26 = accuracy_score(y[mask26], rf.predict(X[mask26]))
        print(f"  RF holdout accuracy on 2026 rows: {acc26:.3f} ({mask26.sum()} rows)")
    else:
        acc26 = None

    importance = dict(zip(feature_cols, rf.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

    trained_models[disease] = {
        'random_forest': rf,
        'logistic_regression': lr,
        'rf_accuracy': rf_acc,
        'lr_accuracy': lr_acc,
        'rf_accuracy_2026_holdout': acc26,
        'feature_importance': importance_sorted,
        'n_samples': len(model_data),
        'n_positive': int(y.sum())
    }
    model_info[f'{disease}_accuracy'] = {'RF': rf_acc, 'LR': lr_acc}

    print(f"  Samples: {len(model_data)} (positive: {int(y.sum())})")
    print(f"  RF Accuracy: {rf_acc:.3f}")
    print(f"  LR Accuracy: {lr_acc:.3f}")

# Save models and scaler
with open('models.pkl', 'wb') as f:
    pickle.dump({
        'models': trained_models,
        'scaler': scaler,
        'feature_cols': feature_cols
    }, f)

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

# Update web app copies
shutil.copy('models.pkl', 'app/models.pkl')
shutil.copy('model_info.json', 'app/model_info.json')

print("\n" + "=" * 60)
print("Models saved successfully!")
print("=" * 60)
print(f"Models trained for: {list(trained_models.keys())}")
print(f"Total features: {len(feature_cols)}")
print("Updated: models.pkl, model_info.json, app/models.pkl, app/model_info.json")
