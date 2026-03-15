"""
Deploy the Silkworm Disease Predictor to Render
"""

import subprocess
import os
import json

print("="*70)
print("SILKWORM DISEASE PREDICTOR - DEPLOYMENT TO RENDER")
print("="*70)

# Check if git is initialized
if not os.path.exists('.git'):
    print("\n1. Initializing Git repository...")
    subprocess.run(['git', 'init'], check=True)
    subprocess.run(['git', 'config', 'user.email', 'research@silkworm-disease-predictor.com'], check=True)
    subprocess.run(['git', 'config', 'user.name', 'Silkworm Research'], check=True)
else:
    print("\n1. Git repository already initialized")

# Create .gitignore
with open('.gitignore', 'w') as f:
    f.write("""__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.env
.venv
venv/
ENV/
.ipynb_checkpoints/
*.log
.DS_Store
""")

print("\n2. Adding files to Git...")
subprocess.run(['git', 'add', '.'], check=True)

print("\n3. Committing changes...")
result = subprocess.run(['git', 'commit', '-m', 'Initial deployment of Silkworm Disease Predictor'], 
                       capture_output=True, text=True)
if result.returncode != 0:
    print("Note:", result.stderr)
else:
    print("Changes committed successfully")

print("\n" + "="*70)
print("DEPLOYMENT INSTRUCTIONS")
print("="*70)
print("""
To deploy this application to Render (FREE hosting):

1. Create a free account at: https://render.com

2. Create a new Web Service:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository OR use "Deploy from Git URL"

3. Configure the service:
   - Name: silkworm-disease-predictor
   - Environment: Python 3
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn app:app
   - Plan: Free

4. Click "Create Web Service"

5. Your app will be deployed at:
   https://silkworm-disease-predictor.onrender.com

ALTERNATIVE: Deploy to PythonAnywhere (FREE)
1. Create account at: https://www.pythonanywhere.com
2. Upload files via Files tab
3. Create a new web app
4. Configure WSGI file to import app from app.py

ALTERNATIVE: Deploy to Heroku (FREE tier available)
1. Create account at: https://heroku.com
2. Install Heroku CLI
3. Run: heroku create silkworm-disease-predictor
4. Run: git push heroku main

""")

# Create a README for deployment
with open('DEPLOYMENT_README.md', 'w') as f:
    f.write("""# Silkworm Disease Predictor - Deployment Guide

## Quick Deploy Options

### Option 1: Render (Recommended - Free)
1. Go to https://render.com and sign up
2. Click "New +" → "Web Service"
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
4. WSGI configuration:
```python
import sys
path = '/home/yourusername/silkworm-predictor/app'
if path not in sys.path:
    sys.path.append(path)
from app import app as application
```

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

## API Usage
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
""")

print("Deployment README saved to DEPLOYMENT_README.md")
print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("""
1. Create a GitHub repository
2. Push this code to GitHub:
   git remote add origin https://github.com/YOUR_USERNAME/silkworm-disease-predictor.git
   git push -u origin main

3. Go to https://render.com and deploy from GitHub

4. Your app will be live in minutes!
""")
