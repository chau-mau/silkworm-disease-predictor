"""
Train and save models for the silkworm disease prediction web app
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Training models for web application...")

# Load cleaned data
df = pd.read_csv('results/cleaned_data_2025.csv')

# Define features
feature_cols = ['Tmax', 'Tmin', 'Humidity', 'THI', 'Wind_Speed', 
                'Has_Uzi', 'Has_Mites', 'Has_Ants', 'Has_Spiders', 'Has_Athropoda']

# Add spacing encoding
spacing_dummies = pd.get_dummies(df['Spacing'], prefix='Spacing')
df = pd.concat([df, spacing_dummies], axis=1)
feature_cols.extend(spacing_dummies.columns.tolist())

# Add Net Tech
if 'Net_Tech' in df.columns:
    df['Net_Tech_Binary'] = (df['Net_Tech'] == 'Yes').astype(int)
    feature_cols.append('Net_Tech_Binary')

disease_cols = ['Pebrine', 'Virosis', 'Bacteriosis', 'Muscardine']

# Store model info
model_info = {
    'features': feature_cols,
    'diseases': disease_cols,
    'spacing_options': df['Spacing'].unique().tolist(),
    'feature_descriptions': {
        'Tmax': 'Maximum Temperature (°C)',
        'Tmin': 'Minimum Temperature (°C)',
        'Humidity': 'Relative Humidity (%)',
        'THI': 'Temperature-Humidity Index',
        'Wind_Speed': 'Wind Speed (m/s)',
        'Has_Uzi': 'Uzi Fly Present (0/1)',
        'Has_Mites': 'Mites Present (0/1)',
        'Has_Ants': 'Ants Present (0/1)',
        'Has_Spiders': 'Spiders Present (0/1)',
        'Has_Athropoda': 'Athropoda Present (0/1)',
        'Net_Tech_Binary': 'Net Technology Used (0/1)'
    }
}

# Train models for each disease
trained_models = {}
scaler = StandardScaler()

for disease in disease_cols:
    print(f"\nTraining model for {disease}...")
    
    # Prepare data
    model_data = df[feature_cols + [disease]].dropna()
    X = model_data[feature_cols]
    y = (model_data[disease] > 0).astype(int)
    
    if y.nunique() < 2 or len(model_data) < 10:
        print(f"  Skipping {disease} - insufficient data variation")
        continue
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    print(f"  RF Accuracy: {rf_acc:.3f}")
    print(f"  LR Accuracy: {lr_acc:.3f}")
    
    # Feature importance
    importance = dict(zip(feature_cols, rf.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Store model
    trained_models[disease] = {
        'random_forest': rf,
        'logistic_regression': lr,
        'rf_accuracy': rf_acc,
        'lr_accuracy': lr_acc,
        'feature_importance': importance_sorted,
        'n_samples': len(model_data),
        'n_positive': int(y.sum())
    }
    
    model_info[f'{disease}_accuracy'] = {'RF': rf_acc, 'LR': lr_acc}

# Save models and scaler
with open('models.pkl', 'wb') as f:
    pickle.dump({
        'models': trained_models,
        'scaler': scaler,
        'feature_cols': feature_cols
    }, f)

# Save model info as JSON
with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n" + "="*60)
print("Models saved successfully!")
print("="*60)
print(f"\nModels trained for: {list(trained_models.keys())}")
print(f"Total features: {len(feature_cols)}")
