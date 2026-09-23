"""
Train weather-only models for the silkworm disease forecast app.

Targets : Virosis, Bacteriosis  (Pebrine and Muscardine excluded)
Features: Tmax, Tmin, Humidity, THI (NRC 1971, derived from weather),
          Wind_Speed  (all from weather data only -
          no plot, pest, net-tech or other management inputs)

Data    : results/merged_data_2025_2026.csv
          (2025 field data + 2026 records; climate from NASA POWER /
          Open-Meteo ERA5 for 2026)

Outputs : models.pkl, model_info.json (root + app/ copies)
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

print("Training WEATHER-ONLY models (Virosis, Bacteriosis)...")

df = pd.read_csv('results/merged_data_2025_2026.csv', parse_dates=['Date'])

# Rainfall excluded: 75% of 2025 rows lack rainfall (dry season, always 0)
feature_cols = ['Tmax', 'Tmin', 'Humidity', 'THI', 'Wind_Speed']
disease_cols = ['Virosis', 'Bacteriosis']

model_info = {
    'features': feature_cols,
    'diseases': disease_cols,
    'training_data': ('results/merged_data_2025_2026.csv '
                      '(2025 field data + 2026 Aug-Sep records, weather only)'),
    'thi_formula': 'NRC (1971): THI = (1.8T+32) - (0.55-0.0055*RH)*(1.8T-58), T = mean of Tmax/Tmin',
    'climate_source_2026': 'NASA POWER daily (Aug 1 - Sep 17, 2026); Open-Meteo ERA5 backfill (Sep 18-22, 2026)',
    'feature_descriptions': {
        'Tmax': 'Maximum Temperature (°C)',
        'Tmin': 'Minimum Temperature (°C)',
        'Humidity': 'Relative Humidity (%)',
        'THI': 'Temperature-Humidity Index (NRC 1971, derived from weather)',
        'Wind_Speed': 'Wind Speed (m/s)'
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

    # Sanity check: accuracy on 2026 rows only (temporal holdout)
    yr = model_data['Year']
    mask26 = (yr == 2026).values
    if mask26.sum() > 0 and y[mask26].nunique() == 2:
        acc26 = accuracy_score(y[mask26], rf.predict(X[mask26]))
        print(f"  RF holdout accuracy on 2026 rows: {acc26:.3f} ({mask26.sum()} rows)")
    else:
        acc26 = None

    importance = dict(zip(feature_cols, rf.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

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
    print(f"  Feature importance: { {k: round(v, 3) for k, v in list(importance_sorted.items())[:3]} }")

with open('models.pkl', 'wb') as f:
    pickle.dump({
        'models': trained_models,
        'scaler': scaler,
        'feature_cols': feature_cols
    }, f)

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

shutil.copy('models.pkl', 'app/models.pkl')
shutil.copy('model_info.json', 'app/model_info.json')

print("\n" + "=" * 60)
print("Weather-only models saved (Virosis, Bacteriosis)!")
print("Updated: models.pkl, model_info.json, app/models.pkl, app/model_info.json")
