# Silkworm Disease Predictor - Deployment Guide

## Quick Deploy Options

### Option 1: Render (Recommended - Free)
1. Go to https://render.com and sign up
2. Click "New +" -> "Web Service"
3. Connect your GitHub repo or use manual deploy
4. Settings:
   - **Build Command**: `pip install -r app/requirements.txt`
   - **Start Command**: `cd app && gunicorn app:app`
   - **Python Version**: 3.11
5. Deploy!

### Option 2: PythonAnywhere (Free)
1. Sign up at https://www.pythonanywhere.com
2. Upload all files
3. Create web app with manual configuration
4. WSGI configuration provided in templates

### Option 3: Run Locally
```bash
cd app
pip install -r requirements.txt
python app.py
```
Then open http://localhost:5000

## Files Structure
```
app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── models.pkl            # Trained ML models
├── model_info.json       # Model metadata
├── Procfile              # Deployment configuration
├── runtime.txt           # Python version
└── templates/
    ├── base.html         # Base template
    ├── index.html        # Home page with prediction form
    ├── about.html        # About page
    └── model_info.html   # Model documentation
```

## API Usage Example
```bash
curl -X POST https://your-app-url.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tmax": 30,
    "tmin": 22,
    "humidity": 75,
    "thi": 32,
    "wind_speed": 1.5,
    "spacing": "6x6",
    "net_tech": 0,
    "has_uzi": 0,
    "has_mites": 0,
    "has_ants": 0,
    "has_spiders": 0,
    "has_athropoda": 0
  }'
```

## Features
- Predict 4 silkworm diseases: Pebrine, Virosis, Bacteriosis, Muscardine
- Input climate parameters (temperature, humidity, THI, wind speed)
- Management options (spacing, net technology)
- Pest presence indicators
- Risk level classification (Low, Moderate, High, Very High)
- Personalized recommendations
- REST API for programmatic access
