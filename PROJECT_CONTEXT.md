# PROJECT CONTEXT & HANDOFF NOTES
**Silkworm Disease Predictor + Mother Moth Exam Game**
Last updated: 2026-09-22 (night session)
Purpose: Continue work on the office computer.

================================================================================
1. PROJECT OVERVIEW
================================================================================
- Research: ML prediction of silkworm diseases at CSB-Central Tasar Research and
  Training Institute (CTRTI), Ranchi, Jharkhand (23.3441 N, 85.3096 E).
- Two deliverables:
  A) Flask web app "Silkworm Disease Predictor" (folder: app/) — 7-day disease
     risk forecast driven by live public weather data. Predicts ONLY Virosis and
     Bacteriosis (Pebrine & Muscardine REMOVED per user request).
  B) Standalone Hindi web game "मदर मॉथ परीक्षण खेल" (folder: moth_exam_game/)
     — teaches mother moth pebrine examination. Completely separate website.

================================================================================
2. CRITICAL ENVIRONMENT NOTE (Windows)
================================================================================
- `python` on PATH = Python 2.7 (MGLTools) — WILL FAIL (no f-strings support).
- ALWAYS use:  py -3 script.py   (Python 3.11 at
  C:\Users\kkoka\AppData\Local\Programs\Python\Python311\)
- Flask app launcher: app\run_app.bat
- Installed in py -3: pandas 2.1.0, scikit-learn 1.2.2, numpy 1.25.0, openpyxl,
  requests, python-docx 1.2.0.

================================================================================
3. WORK COMPLETED THIS SESSION
================================================================================
3.1 2026 data integration
- New script: build_2026_dataset.py
  * Compiled user's typed August 2026 records (bacteriosis Aug 2,3,11,20 = 1 larva
    each; Aug 22 = 2; Aug 24 = 10; virosis Aug 4 = 2 larvae)
  * Parsed 'Disease 2026.xlsx' (September 2026 records; mixed date formats —
    ALL interpreted as SEPTEMBER 2026; duplicate rows summed per day)
  * Days with no records in Aug 1–Sep 23 window = zero incidence (54 days)
- Climate from authentic sources:
  * NASA POWER daily API (T2M_MAX, T2M_MIN, RH2M, WS10M, PRECTOTCORR) for
    Aug 1–Sep 17, 2026
  * Open-Meteo ERA5 archive backfill for Sep 18–22 (NASA has ~5-day latency)
  * Sep 23 record EXCLUDED from training (no climate data available yet)
- THI computed with NRC (1971) formula for BOTH years:
  THI = (1.8T+32) − (0.55−0.0055×RH)×(1.8T−58), T = mean(Tmax,Tmin)
- Outputs: results/disease_data_2026.csv, results/climate_data_2026.csv,
  results/merged_data_2025_2026.csv (134 rows = 81 of 2025 + 53 of 2026)

3.2 Model retraining (weather-only, 2 diseases)
- Script: train_models_weather.py (canonical; train_models_2025_2026.py is the
  older 4-disease/management-feature version, kept for history)
- Features (5, weather only): Tmax, Tmin, Humidity, THI, Wind_Speed
- EXCLUDED: Pebrine, Muscardine (no 2026 positives); management features
  (spacing/net-tech/pests — not recorded in 2026); Rainfall (75% missing in
  2025, rest all zero)
- 116 training rows after dropping 18 rows with missing wind speed
- RF(100 trees, depth 10) + LogisticRegression(max_iter=1000), 80/20 stratified
  split (random_state=42), probabilities averaged
- Results: Virosis RF/LR acc 0.792, 2026 holdout 0.943 | Bacteriosis 0.792,
  holdout 0.981. Risk separation on 2026: Virosis 37.2% vs 8.3%,
  Bacteriosis 41.3% vs 13.9% (disease vs clean days)
- Saved: models.pkl, model_info.json (root + app/ copies)

3.3 Forecast integration (real-time, no manual entry)
- app/forecast.py: Open-Meteo Forecast API (primary, no key) → NASA POWER
  (fallback); 30-min in-memory cache; THI computed internally
- IMD/Mausam APIs all return 401 (IP whitelisting required) — documented in
  forecast.py; can be added later if IMD grants access
- Endpoints: /api/forecast (POST, days/lat/lon), /api/predict (weather-only JSON)
- Home page (/) = live 7-day forecast; manual form DELETED

3.4 Branding
- Footer: "Developed by CSB-Central Tasar Research and Training Institute -
  Ranchi, Jharkhand"
- app/static/: logo-4.png (CTRTI logo), tasar-silkworm.png (photo cropped from
  ctrti.res.in header), favicon.png
- Navbar: silkworm photo before title (replaced beetle-like bug icon)

3.5 Methodology document
- Methodology_Silkworm_Disease_Predictor.docx (Times New Roman 12, 1.5 spacing,
  NO Word Heading styles → no collapse/expand buttons)
- Regenerate/edit via: py -3 generate_methodology_doc.py

3.6 Mother Moth Exam Game (separate website)
- moth_exam_game/index.html — single self-contained Hindi game, no dependencies
- Flow: मॉथ चयन (male vs female — FEMALE is correct, cloud justification) →
  औज़ार चयन (6 tools) → कटाई-पिसाई (cut moth, mortar-pestle grind) →
  स्लाइड तैयारी (PVS drop, smear, cover slip) → माइक्रोस्कोप 600x (focus knob,
  find 4 spores, avoid debris) → निपटान (discard positive slide in
  alcohol/propanol tank). Max score 160.
- Run: cd moth_exam_game; py -3 -m http.server 8080 → http://localhost:8080
- FULLY REMOVED from Flask app (route, nav link, template deleted)
- Deployable standalone to GitHub Pages/Netlify/Render static

================================================================================
4. HOW TO RUN (office computer)
================================================================================
1. Copy the whole DMR folder (or git clone if pushed to GitHub — repo has
   push_to_github.bat/.sh scripts; see section 7)
2. Install deps:  py -3 -m pip install -r app/requirements.txt
   (flask, scikit-learn, numpy, pandas, gunicorn, requests)
3. Predictor app:  cd app; py -3 app.py  (or run_app.bat) → localhost:5000
4. Game:  cd moth_exam_game; py -3 -m http.server 8080 → localhost:8080
5. Retrain (optional): py -3 build_2026_dataset.py && py -3 train_models_weather.py
6. Methodology doc: py -3 generate_methodology_doc.py

================================================================================
5. KEY ASSUMPTIONS (verify with field records if possible)
================================================================================
1. All dates in Disease 2026.xlsx = September 2026 (Nov/Dec stored dates were
   read day-first). If any are actually Nov/Dec, remap in build_2026_dataset.py.
2. Duplicate Excel rows (e.g., three identical 9/12 rows) = separate trays,
   summed per day.
3. No disease record on a day in Aug 1–Sep 23, 2026 = zero incidence.
4. 2026 rows lack plot/spacing/instar/net-tech/pest data → excluded from model.
5. Wind speed differs systematically between NASA (low) and Open-Meteo/ERA5
   (high); wind is a minor feature, predictions dominated by temp/humidity.
6. THI = NRC formula now used everywhere (original 2025 THI column superseded).

================================================================================
6. KNOWN ISSUES / TODO
================================================================================
- [ ] USER TO VERIFY: Excel date interpretation (Sept vs Nov/Dec 2026)
- [ ] IMD API: apply for whitelisting → add as primary forecast provider
- [ ] Deployment: push to GitHub → Render (Flask) + GitHub Pages (game)
- [ ] Sept 23, 2026 disease row can be added once NASA data catches up
      (edit build_2026_dataset.py date range, re-run both scripts)
- [ ] Old train_models_2025_2026.py, analysis_code.py kept for history; not
      part of the current pipeline
- [ ] Weather-only test accuracy (~0.79) lower than old management model —
      expected trade-off for a fully automated forecast product

================================================================================
7. GITHUB / TRANSFER
================================================================================
- Repo has 4 commits; push scripts exist (push_to_github.bat, push_to_github.sh).
- NOTE: I did NOT commit/push anything (no permission asked). To transfer to
  office PC, either copy the folder or commit+push yourself.
- Suggested: git add -A; git commit -m "2026 data, forecast app, game, docs";
  git push — then clone on office PC.

================================================================================
8. FILE MAP (current pipeline only)
================================================================================
build_2026_dataset.py          → builds 2026 + merged datasets (calls APIs)
train_models_weather.py        → trains Virosis/Bacteriosis models → models.pkl
app/app.py                     → Flask backend (/, /api/forecast, /api/predict)
app/forecast.py                → weather fetch (Open-Meteo→NASA, cache, THI)
app/templates/forecast.html    → home page UI
app/static/                    → logo-4.png, tasar-silkworm.png, favicon.png
moth_exam_game/index.html      → standalone Hindi game
generate_methodology_doc.py    → creates Methodology_Silkworm_Disease_Predictor.docx
results/merged_data_2025_2026.csv  → training data (134 rows)
results/disease_data_2026.csv  → 2026 daily disease counts
results/climate_data_2026.csv  → 2026 daily NASA+ERA5 climate
Disease 2026.xlsx              → raw September 2026 field records
================================================================================
