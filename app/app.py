"""
Silkworm Disease Prediction Web Application
Flask backend for disease prediction
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
import os

app = Flask(__name__)

# Load models
with open('models.pkl', 'rb') as f:
    model_data = pickle.load(f)

models = model_data['models']
scaler = model_data['scaler']
feature_cols = model_data['feature_cols']

# Load model info
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

@app.route('/')
def index():
    """Home page with prediction form"""
    return render_template('index.html', 
                         spacing_options=model_info['spacing_options'],
                         features=model_info['feature_descriptions'])

@app.route('/predict', methods=['POST'])
def predict():
    """Make disease predictions"""
    try:
        # Get form data
        data = request.form
        
        # Extract values
        tmax = float(data.get('tmax', 30))
        tmin = float(data.get('tmin', 22))
        humidity = float(data.get('humidity', 75))
        thi = float(data.get('thi', 32))
        wind_speed = float(data.get('wind_speed', 1.5))
        spacing = data.get('spacing', '6x6')
        net_tech = int(data.get('net_tech', 0))
        
        # Pests
        has_uzi = int(data.get('has_uzi', 0))
        has_mites = int(data.get('has_mites', 0))
        has_ants = int(data.get('has_ants', 0))
        has_spiders = int(data.get('has_spiders', 0))
        has_athropoda = int(data.get('has_athropoda', 0))
        
        # Create feature vector
        features = {
            'Tmax': tmax,
            'Tmin': tmin,
            'Humidity': humidity,
            'THI': thi,
            'Wind_Speed': wind_speed,
            'Has_Uzi': has_uzi,
            'Has_Mites': has_mites,
            'Has_Ants': has_ants,
            'Has_Spiders': has_spiders,
            'Has_Athropoda': has_athropoda,
            'Net_Tech_Binary': net_tech
        }
        
        # Add spacing dummies
        for sp in model_info['spacing_options']:
            features[f'Spacing_{sp}'] = 1 if sp == spacing else 0
        
        # Create input array
        input_array = np.array([[features.get(col, 0) for col in feature_cols]])
        input_scaled = scaler.transform(input_array)
        
        # Make predictions
        predictions = {}
        for disease, model_dict in models.items():
            rf = model_dict['random_forest']
            lr = model_dict['logistic_regression']
            
            rf_prob = rf.predict_proba(input_array)[0][1]
            lr_prob = lr.predict_proba(input_scaled)[0][1]
            
            # Average probability
            avg_prob = (rf_prob + lr_prob) / 2
            
            predictions[disease] = {
                'probability': round(avg_prob * 100, 2),
                'risk_level': get_risk_level(avg_prob),
                'rf_prob': round(rf_prob * 100, 2),
                'lr_prob': round(lr_prob * 100, 2)
            }
        
        # Calculate overall risk
        overall_risk = np.mean([p['probability'] for p in predictions.values()])
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'overall_risk': round(overall_risk, 2),
            'input_summary': {
                'Temperature Range': f"{tmin}°C - {tmax}°C",
                'Humidity': f"{humidity}%",
                'THI': thi,
                'Spacing': spacing,
                'Net Technology': 'Yes' if net_tech else 'No',
                'Pests Present': sum([has_uzi, has_mites, has_ants, has_spiders, has_athropoda])
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability < 0.2:
        return {'level': 'Low', 'color': 'green', 'class': 'success'}
    elif probability < 0.4:
        return {'level': 'Moderate', 'color': 'yellow', 'class': 'warning'}
    elif probability < 0.6:
        return {'level': 'High', 'color': 'orange', 'class': 'warning'}
    else:
        return {'level': 'Very High', 'color': 'red', 'class': 'danger'}

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Similar prediction logic as above
        features = {
            'Tmax': float(data.get('tmax', 30)),
            'Tmin': float(data.get('tmin', 22)),
            'Humidity': float(data.get('humidity', 75)),
            'THI': float(data.get('thi', 32)),
            'Wind_Speed': float(data.get('wind_speed', 1.5)),
            'Has_Uzi': int(data.get('has_uzi', 0)),
            'Has_Mites': int(data.get('has_mites', 0)),
            'Has_Ants': int(data.get('has_ants', 0)),
            'Has_Spiders': int(data.get('has_spiders', 0)),
            'Has_Athropoda': int(data.get('has_athropoda', 0)),
            'Net_Tech_Binary': int(data.get('net_tech', 0))
        }
        
        spacing = data.get('spacing', '6x6')
        for sp in model_info['spacing_options']:
            features[f'Spacing_{sp}'] = 1 if sp == spacing else 0
        
        input_array = np.array([[features.get(col, 0) for col in feature_cols]])
        input_scaled = scaler.transform(input_array)
        
        predictions = {}
        for disease, model_dict in models.items():
            rf = model_dict['random_forest']
            lr = model_dict['logistic_regression']
            
            rf_prob = rf.predict_proba(input_array)[0][1]
            lr_prob = lr.predict_proba(input_scaled)[0][1]
            avg_prob = (rf_prob + lr_prob) / 2
            
            predictions[disease] = {
                'probability': round(avg_prob * 100, 2),
                'risk_level': get_risk_level(avg_prob)['level']
            }
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', model_info=model_info)

@app.route('/model-info')
def model_info_page():
    """Model information page"""
    return render_template('model_info.html', 
                         model_info=model_info,
                         models=models)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
