"""
Silkworm Disease Forecast Web Application
Flask backend - weather-driven Virosis & Bacteriosis risk prediction

Weather data: Open-Meteo forecast API (primary), NASA POWER (fallback).
Predictions use weather-only features (Tmax, Tmin, Humidity, THI, Wind_Speed).
THI is computed internally (NRC 1971); no manual parameter entry needed.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
import forecast

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


def _predict_diseases(tmax, tmin, humidity, wind_speed, rainfall=None):
    """Run the RF+LR ensemble for all diseases from weather parameters.

    THI is derived internally (NRC 1971). Returns predictions dict.
    """
    features = {
        'Tmax': float(tmax),
        'Tmin': float(tmin),
        'Humidity': float(humidity),
        'THI': forecast.thi_nrc(tmax, tmin, humidity),
        'Wind_Speed': float(wind_speed),
    }
    input_array = np.array([[features.get(col, 0) for col in feature_cols]])
    input_scaled = scaler.transform(input_array)

    predictions = {}
    for disease, model_dict in models.items():
        rf_prob = model_dict['random_forest'].predict_proba(input_array)[0][1]
        lr_prob = model_dict['logistic_regression'].predict_proba(input_scaled)[0][1]
        avg_prob = (rf_prob + lr_prob) / 2
        predictions[disease] = {
            'probability': round(avg_prob * 100, 2),
            'risk_level': get_risk_level(avg_prob)['level']
        }
    return predictions


@app.route('/')
def index():
    """Home page - live 7-day forecast with disease risk"""
    return render_template('forecast.html')


@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    """Fetch real-time weather forecast and predict disease risk per day.

    Optional JSON body: days (1-16), lat, lon (default: Ranchi)
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        days = int(data.get('days', 7))
        lat = float(data.get('lat', forecast.DEFAULT_LAT))
        lon = float(data.get('lon', forecast.DEFAULT_LON))

        result = forecast.get_forecast(lat=lat, lon=lon, days=days)
        if not result['success']:
            return jsonify({'success': False, 'error': result['error']})

        for day in result['daily']:
            day['predictions'] = _predict_diseases(
                day['tmax'], day['tmin'], day['humidity'], day['wind_speed'])

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Direct prediction from weather parameters (JSON API).

    Body: {"tmax": 30, "tmin": 22, "humidity": 75, "wind_speed": 1.5}
    """
    try:
        data = request.get_json(force=True)
        predictions = _predict_diseases(
            data.get('tmax', 30), data.get('tmin', 22),
            data.get('humidity', 75), data.get('wind_speed', 1.5))
        return jsonify({'success': True, 'predictions': predictions})
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
