# Silkworm Disease Predictor

[![Website](https://img.shields.io/badge/Website-Live-green)](https://chau-mau.github.io/silkworm-disease-predictor/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Author:** Nidhi Sukhija, chau-mau · she/her  
**Developed at:** CSB-Central Tasar Research and Training Institute (CTRTI), Ranchi, Jharkhand

A machine learning web application to predict silkworm disease occurrence based on climate and management parameters.

![Silkworm Disease Predictor](docs/screenshot.png)

## Features

- **Disease Prediction**: Predicts 2 major silkworm diseases:
  - Virosis (Viral infections)
  - Bacteriosis (Bacterial infections)

- **Weather-Driven**: No manual data entry - the app fetches a live 7-day
  weather forecast for Ranchi from public-domain sources:
  - Open-Meteo Forecast API (primary, free, no API key)
  - NASA POWER (fallback)
  - (IMD/Mausam APIs require IP whitelisting; see `app/forecast.py` notes)

- **Weather Features**: Tmax, Tmin, Humidity, THI (NRC 1971, auto-derived), Wind Speed
- **ML Models**: Ensemble of Random Forest and Logistic Regression
- **Risk Levels**: Low, Moderate, High, Very High with daily outlook chart

## Live Demo

**Disease Predictor**: [https://chau-mau.github.io/silkworm-disease-predictor/](https://chau-mau.github.io/silkworm-disease-predictor/)  
**Disease Calendar**: [https://chau-mau.github.io/silkworm-disease-predictor/calendar.html](https://chau-mau.github.io/silkworm-disease-predictor/calendar.html)  
**Mother Moth Tutorial**: [https://chau-mau.github.io/maada-shalabh-parikshan/](https://chau-mau.github.io/maada-shalabh-parikshan/)

## Research Background

- **Location**: Ranchi, Jharkhand, India (23.3441°N, 85.3096°E)
- **Study Period**: October 2025 + August-September 2026
- **Data Points**: 81 plot-level observations (2025) + 53 daily records (2026)
- **2026 Climate**: NASA POWER daily data, backfilled with Open-Meteo ERA5
- **Models**: Weather-only features (5), trained on combined 2025-2026 data

## Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/chau-mau/silkworm-disease-predictor.git
cd silkworm-disease-predictor

# Install dependencies
pip install -r app/requirements.txt

# Run the application (use Python 3; on Windows: py -3 app.py or app\run_app.bat)
cd app
python app.py

# Open http://localhost:5000 in your browser
```

### Forecast API

```bash
curl -X POST http://localhost:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"days": 7}'
```

Returns the live 7-day forecast (Tmax, Tmin, RH, wind, rainfall, THI) with a
Virosis/Bacteriosis risk prediction for each day.

### Direct Prediction API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"tmax": 30, "tmin": 22, "humidity": 75, "wind_speed": 1.5}'
```

THI is computed automatically (NRC 1971 formula).

## Project Structure

```
.
├── app/                      # Flask application
│   ├── app.py               # Main backend
│   ├── forecast.py          # Real-time forecast (Open-Meteo + NASA POWER)
│   ├── requirements.txt     # Dependencies
│   ├── models.pkl          # Trained ML models
│   ├── model_info.json     # Model metadata
│   └── templates/          # HTML templates
│       ├── base.html
│       ├── index.html
│       ├── forecast.html
│       ├── about.html
│       └── model_info.html
├── docs/                    # Static website (GitHub Pages)
│   ├── index.html
│   ├── calendar.html        # Ranchi disease calendar heatmap
│   └── calendar_data.json   # Pre-computed historical risk data
├── generate_calendar.py     # Build calendar_data.json from models
├── figures/                 # Analysis visualizations
├── results/                 # Analysis results
├── analysis_code.py         # Data analysis script
├── train_models.py          # Model training script
└── README.md               # This file
```

## Analysis Results

The project includes comprehensive statistical analysis:

- **Correlation Analysis**: Climate-disease relationships
- **ANOVA**: Spacing and instar effects
- **Pest-Disease Interactions**: Significant correlations found
- **Predictive Modeling**: Random Forest + Logistic Regression
- **Threshold Analysis**: Optimal climate thresholds for disease

See `results/` folder for detailed outputs.

## Deployment

### GitHub Pages (Static Site)
1. Push code to GitHub
2. Go to Settings → Pages
3. Source: Deploy from branch → main → docs folder
4. Site will be live at `https://USERNAME.github.io/silkworm-disease-predictor`

### Render (Full Flask App)
1. Create account at [render.com](https://render.com)
2. New Web Service → Connect GitHub repo
3. Build Command: `pip install -r app/requirements.txt`
4. Start Command: `cd app && gunicorn app:app`
5. Deploy!

## Technologies Used

- **Backend**: Python, Flask, scikit-learn
- **Frontend**: HTML, CSS, Bootstrap 5, JavaScript
- **ML Models**: Random Forest, Logistic Regression
- **Deployment**: GitHub Pages, Render

## Citation

If you use this tool in your research, please cite:

```
Silkworm Disease Predictor (2025)
Machine Learning-based Disease Prediction for Sericulture
Ranchi, Jharkhand, India
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaborations, please open an issue on GitHub.

---

**Developed for Silkworm Disease Research - Ranchi, Jharkhand**
