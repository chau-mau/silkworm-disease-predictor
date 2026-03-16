# Silkworm Disease Predictor

[![Website](https://img.shields.io/badge/Website-Live-green)](https://silkworm-disease-predictor.github.io)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A machine learning web application to predict silkworm disease occurrence based on climate and management parameters.

## Features

- **Disease Prediction**: Predicts 4 major silkworm diseases:
  - Pebrine (Microsporidian disease)
  - Virosis (Viral infections)
  - Bacteriosis (Bacterial infections)
  - Muscardine (Fungal disease)

- **Input Parameters**:
  - Climate: Temperature, Humidity, THI, Wind Speed
  - Management: Plot spacing, Net technology
  - Pest monitoring: Uzi fly, Mites, Ants, Spiders, Athropoda

- **ML Models**: Ensemble of Random Forest and Logistic Regression
- **Accuracy**: 95%+ prediction accuracy
- **Risk Levels**: Low, Moderate, High, Very High with recommendations

## Live Demo

**Website**: [https://chau-mau.github.io/silkworm-disease-predictor/](https://chau-mau.github.io/silkworm-disease-predictor/)

## Research Background

- **Location**: Ranchi, Jharkhand, India (23.3441°N, 85.3096°E)
- **Study Period**: October 2025
- **Data Points**: 81 observations from 8 plots
- **Models**: Trained on field data with 16 features

## Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/silkworm-disease-predictor.git
cd silkworm-disease-predictor

# Install dependencies
pip install -r app/requirements.txt

# Run the application
cd app
python app.py

# Open http://localhost:5000 in your browser
```

### API Usage

```bash
curl -X POST http://localhost:5000/api/predict \
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

## Project Structure

```
.
├── app/                      # Flask application
│   ├── app.py               # Main backend
│   ├── requirements.txt     # Dependencies
│   ├── models.pkl          # Trained ML models
│   ├── model_info.json     # Model metadata
│   └── templates/          # HTML templates
│       ├── base.html
│       ├── index.html
│       ├── about.html
│       └── model_info.html
├── docs/                    # Static website (GitHub Pages)
│   └── index.html
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
Machine Learning-based Disease Prediction for Tasar Sericulture
Ranchi, Jharkhand, India
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaborations, please open an issue on GitHub.

---

**Developed for Tasar Silkworm Disease Monitoring - Ranchi, Jharkhand**
