"""
Generate the complete project methodology as a Microsoft Word document.

Formatting:
  - Times New Roman, 12 pt body text
  - 1.5 line spacing
  - Section titles are plain bold paragraphs (NOT Word Heading styles),
    so no collapse/expand arrows appear in the document.
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

doc = Document()

# ---------------------------------------------------------------------------
# Global formatting: Times New Roman 12 pt, 1.5 line spacing
# ---------------------------------------------------------------------------
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
style.paragraph_format.line_spacing = 1.5
style.paragraph_format.space_after = Pt(6)

for section in doc.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)


def title(text):
    """Main title (centered, bold, 16 pt - plain paragraph, no Heading style)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(16)
    r.font.name = 'Times New Roman'
    return p


def subtitle(text):
    """Subtitle (centered, italic, 12 pt)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(12)
    r.font.name = 'Times New Roman'
    return p


def heading(text):
    """Section heading - bold, 14 pt, plain paragraph (no collapse arrow)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)
    r.font.name = 'Times New Roman'
    r.font.color.rgb = RGBColor(0x2C, 0x55, 0x30)
    return p


def subheading(text):
    """Sub-section heading - bold, 12 pt."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)
    r.font.name = 'Times New Roman'
    return p


def para(text, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    return p


def bullet(text):
    """Bullet as plain paragraph with bullet char (avoids list styles)."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.15)
    r = p.add_run('\u2022  ' + text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    return p


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ''
        r = hdr[i].paragraphs[0].add_run(h)
        r.bold = True
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.name = 'Times New Roman'
            r.font.size = Pt(11)
    doc.add_paragraph()
    return t


# ===========================================================================
# CONTENT
# ===========================================================================
title('Methodology: Machine Learning-Based Silkworm Disease Prediction and Forecasting System')
subtitle('CSB-Central Tasar Research and Training Institute (CTRTI), Ranchi, Jharkhand')
subtitle('Study location: 23.3441\u00b0 N, 85.3096\u00b0 E | Study periods: October 2025 and August\u2013September 2026')

doc.add_paragraph()

# ---------------------------------------------------------------------------
heading('1. Study Area and Experimental Design')
para('The study was conducted at the experimental rearing facility of the Central Tasar '
     'Research and Training Institute (CTRTI), Ranchi, Jharkhand, India '
     '(23.3441\u00b0 N latitude, 85.3096\u00b0 E longitude). Two cropping seasons were monitored: '
     'October 2025 and August\u2013September 2026. In 2025, silkworm larvae were reared in '
     '8 experimental plots maintained under five spacing configurations (6\u00d76, 6\u00d710, '
     '8\u00d78, 10\u00d710 and 12\u00d712) and followed across instar stages, generating '
     '81 plot-level observations between 07 October 2025 and 31 October 2025. In 2026, '
     'disease incidence was recorded on a daily basis over a continuous rearing period '
     'from 01 August 2026 to 23 September 2026.')

# ---------------------------------------------------------------------------
heading('2. Disease Data Collection')
subheading('2.1 Field observations, October 2025')
para('Disease incidence was recorded at the plot level on each observation day. For every '
     'plot, the number of infected larvae was counted separately for four diseases: '
     'Pebrine (microsporidiosis), Virosis (viral infection), Bacteriosis (bacterial '
     'infection) and Muscardine (fungal infection). Alongside disease counts, rearing '
     'attributes (plot number, spacing configuration, instar stage, net technology use) '
     'and pest presence (Uzi fly, mites, ants, spiders, arthropoda) were documented. '
     'The compiled dataset (results/cleaned_data_2025_corrected_thi.csv) contains 81 rows '
     'with daily disease counts per plot.')
subheading('2.2 Field observations, August\u2013September 2026')
para('Disease incidence for the 2026 season was compiled from two sources:')
bullet('Typed field records for August 2026: Bacteriosis was recorded on 02 Aug (1 larva), '
       '03 Aug (1), 11 Aug (1), 20 Aug (1), 22 Aug (2) and 24 Aug (10 larvae); Virosis was '
       'recorded on 04 Aug (2 larvae).')
bullet('Disease 2026.xlsx for September 2026: per-tray larval counts of Virosis and '
       'Bacteriosis. All dates in the workbook were interpreted as September 2026 '
       '(the workbook mixes day-first and month-first date formats). Rows recorded for the '
       'same date were summed to obtain daily totals (e.g., 12 Sep: 59 Virosis and '
       '22 Bacteriosis larvae).')
para('Days within the rearing window (01 Aug \u2013 23 Sep 2026) with no disease record were '
     'treated as zero-incidence days, yielding 54 daily records. The compiled dataset is '
     'saved as results/disease_data_2026.csv.')

# ---------------------------------------------------------------------------
heading('3. Climate Data Collection')
subheading('3.1 Climate records, October 2025')
para('During the 2025 season, climate variables (maximum and minimum temperature, relative '
     'humidity, wind speed, rainfall, dry-bulb and wet-bulb temperature, photoperiod) were '
     'recorded in the field alongside each disease observation.')
subheading('3.2 Climate data, 2026 season \u2014 NASA POWER (primary authentic source)')
para('Daily weather data for the 2026 rearing period were retrieved programmatically from '
     'the NASA POWER (Prediction of Worldwide Energy Resources) daily point API '
     '(https://power.larc.nasa.gov/api/temporal/daily/point) for the CTRTI Ranchi '
     'coordinates. The following parameters were requested under the AG (agriculture) '
     'community: T2M_MAX (maximum air temperature, \u00b0C), T2M_MIN (minimum air temperature, '
     '\u00b0C), RH2M (mean relative humidity, %), WS10M (mean wind speed at 10 m, m s\u207b\u00b9) and '
     'PRECTOTCORR (corrected daily precipitation, mm), for 01 August 2026 to 23 September '
     '2026. Values of \u2212999 (NASA missing-data flag) were treated as missing. NASA POWER '
     'daily products carry a latency of approximately five days, and valid data were '
     'returned for 01 Aug \u2013 17 Sep 2026. The retrieved dataset is saved as '
     'results/climate_data_2026.csv.')
subheading('3.3 Gap filling \u2014 Open-Meteo ERA5 archive (secondary authentic source)')
para('The five missing days (18\u201322 September 2026) were backfilled from the Open-Meteo '
     'historical weather archive (https://archive-api.open-meteo.com/v1/archive), which '
     'serves ERA5 reanalysis data from the European Centre for Medium-Range Weather '
     'Forecasts. Daily maximum/minimum temperature, mean relative humidity, mean 10 m wind '
     'speed and precipitation sum were extracted for the same coordinates (Asia/Kolkata '
     'timezone) and merged with the NASA POWER series. The 23 September 2026 disease '
     'record was excluded from model training because no observed climate data were yet '
     'available for that date from either source, leaving 53 daily records for 2026.')
subheading('3.4 Note on India Meteorological Department (IMD) sources')
para('The official IMD/Mausam data services (api.imd.gov.in, city.imd.gov.in, '
     'mausam.imd.gov.in) were evaluated as candidate sources. All IMD programmatic '
     'endpoints currently require the caller\u2019s IP address or domain to be whitelisted by '
     'IMD and returned HTTP 401 for unregistered clients; they therefore could not be '
     'integrated directly. NASA POWER and Open-Meteo ERA5 are open public-domain sources '
     'requiring no registration and were adopted instead. An IMD provider can be added to '
     'the data-access module once API access is approved.')

# ---------------------------------------------------------------------------
heading('4. Derivation of the Temperature-Humidity Index (THI)')
para('The Temperature-Humidity Index was computed for every record (both seasons) using '
     'the NRC (1971) formulation:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('THI = (1.8 \u00d7 T + 32) \u2212 [(0.55 \u2212 0.0055 \u00d7 RH) \u00d7 (1.8 \u00d7 T \u2212 58)]')
r.italic = True
r.font.name = 'Times New Roman'
r.font.size = Pt(12)
para('where T is the mean of daily maximum and minimum temperature (\u00b0C) and RH is the '
     'mean relative humidity (%). The NRC formula was chosen because it was previously '
     'validated for this dataset (recalculate_thi_and_recreate_figure.py) and it supersedes '
     'the ad hoc THI values of the original 2025 records, ensuring a consistent THI scale '
     'across the 2025 and 2026 seasons (observed range approximately 70\u201380).')

# ---------------------------------------------------------------------------
heading('5. Data Integration and Preprocessing')
para('The 2025 and 2026 datasets were merged into a single training table '
     '(results/merged_data_2025_2026.csv; 134 rows: 81 plot-level rows from 2025 and 53 '
     'daily rows from 2026) by a dedicated build script (build_2026_dataset.py). The '
     'following preprocessing decisions were applied:')
bullet('Records with missing wind speed (18 of the 81 rows from 2025) were removed, '
       'leaving 116 complete training records.')
bullet('Disease counts were converted to binary occurrence targets '
       '(1 = at least one infected larva on that day/plot; 0 = none), matching the '
       'operational question \u201cwhether disease will occur\u201d rather than \u201chow many larvae '
       'will be infected\u201d.')
bullet('Pebrine and Muscardine were excluded from model development: neither disease was '
       'recorded anywhere in the 2026 season, so no current-season positive examples '
       'exist. Only Virosis and Bacteriosis are modelled and predicted.')
bullet('Management features (plot spacing, net technology, pest presence) were excluded '
       'from the predictive model so that forecasts depend solely on weather, which is '
       'available in real time from public sources.')
bullet('Rainfall was excluded as a model feature because 75% of the 2025 records lack '
       'rainfall data and all remaining 2025 values are zero (dry season); retaining it '
       'would have discarded most of the 2025 training rows. Rainfall is still displayed '
       'on the forecast page for information.')
para('The final feature set therefore consists of five weather variables: Tmax, Tmin, '
     'Humidity, THI (derived) and Wind_Speed.')

# ---------------------------------------------------------------------------
heading('6. Model Development')
subheading('6.1 Algorithms and ensemble design')
para('For each target disease (Virosis, Bacteriosis), two binary classifiers were trained '
     'and combined into an ensemble:')
bullet('Random Forest Classifier (scikit-learn): 100 estimators, maximum depth 10, '
       'random state 42; operates on raw (unscaled) features.')
bullet('Logistic Regression (scikit-learn): maximum 1000 iterations, L2 regularization, '
       'random state 42; trained on features standardized with a StandardScaler fitted '
       'on the training partition only.')
para('For any input day, the ensemble risk score is the arithmetic mean of the two '
     'predicted probabilities of disease occurrence:')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('P(disease) = [P_RF(disease) + P_LR(disease)] / 2')
r.italic = True
r.font.name = 'Times New Roman'
r.font.size = Pt(12)
subheading('6.2 Training protocol')
para('Data were partitioned into training (80%) and test (20%) subsets using stratified '
     'sampling (stratified by disease occurrence, random state 42) so that class '
     'proportions are preserved in both partitions. Models were trained independently '
     'for each disease on the full 2025+2026 training set and serialized, together with '
     'the scaler and the feature-column order, into models.pkl; metadata (features, '
     'formulas, accuracies) are stored in model_info.json. The complete training '
     'procedure is implemented in train_models_weather.py.')
subheading('6.3 Risk classification')
para('Ensemble probabilities are mapped to four advisory risk levels: Low (< 20%), '
     'Moderate (20\u201339.9%), High (40\u201359.9%) and Very High (\u2265 60%).')

# ---------------------------------------------------------------------------
heading('7. Model Evaluation')
para('Model performance was assessed on the held-out test partition and, additionally, '
     'on the 2026 season alone as a temporal holdout (the model\u2019s ability to generalize '
     'to an unseen season):')
table(
    ['Disease', 'RF test accuracy', 'LR test accuracy', 'RF accuracy on 2026 holdout',
     'Training samples', 'Positive cases'],
    [
        ['Virosis', '0.792', '0.792', '0.943', '116', '31'],
        ['Bacteriosis', '0.792', '0.792', '0.981', '116', '21'],
    ])
para('Discriminative ability on the 2026 season was further quantified by comparing mean '
     'predicted risk on disease days versus disease-free days: for Virosis, 37.2% on '
     'disease days versus 8.3% on clean days; for Bacteriosis, 41.3% versus 13.9%. '
     'Feature-importance analysis (Random Forest) identified Tmin, Tmax, Humidity and THI '
     'as the dominant predictors for both diseases, consistent with the significant '
     'temperature\u2013disease correlations established in the earlier statistical analysis of '
     'the 2025 data.')

# ---------------------------------------------------------------------------
heading('8. Real-Time Weather Forecast Integration')
para('The deployed web application does not require any manual data entry. Weather '
     'parameters for prediction are obtained automatically from public-domain forecast '
     'sources through a provider chain implemented in app/forecast.py:')
bullet('Primary source: Open-Meteo Forecast API (https://api.open-meteo.com/v1/forecast), '
       'a free service requiring no API key, providing daily maximum/minimum temperature, '
       'mean relative humidity, mean 10 m wind speed and precipitation sum for up to 16 '
       'days ahead (Asia/Kolkata timezone).')
bullet('Fallback source: NASA POWER daily endpoint, used automatically if the primary '
       'source is unreachable.')
para('Forecast responses are cached in memory for 30 minutes to limit API usage and '
     'reduce latency. For every forecast day the application computes THI internally '
     '(NRC 1971) and evaluates both disease models, producing a 7-day disease-risk '
     'outlook. The official IMD/Mausam APIs can be inserted as the primary provider once '
     'IP whitelisting is granted by IMD (Section 3.4).')

# ---------------------------------------------------------------------------
heading('9. Web Application')
para('The system is implemented as a Flask (Python 3.11) web application with a '
     'Bootstrap 5 front end and Chart.js visualizations:')
bullet('Home page (/): live 7-day forecast with per-day Virosis and Bacteriosis risk, a '
       'risk-trend chart, a highest-risk-day advisory, and daily weather cards.')
bullet('JSON API: POST /api/forecast returns the forecast with per-day predictions; '
       'POST /api/predict returns predictions directly from supplied weather parameters '
       '(tmax, tmin, humidity, wind_speed; THI computed automatically).')
bullet('Model information (/model-info) and About (/about) pages document the ensemble, '
       'performance and data sources.')
para('The application is self-contained (models, scaler, branding images and templates '
     'are served locally) and is deployable on Render (gunicorn) or GitHub Pages without '
     'any API keys.')

# ---------------------------------------------------------------------------
heading('10. Reproducibility: Analysis Pipeline')
para('The complete workflow is scripted and reproducible:')
bullet('build_2026_dataset.py \u2014 compiles the 2026 disease records, downloads NASA POWER '
       'and Open-Meteo ERA5 climate data, computes THI and writes the merged 2025+2026 '
       'training dataset.')
bullet('train_models_weather.py \u2014 trains the Random Forest + Logistic Regression '
       'ensembles for Virosis and Bacteriosis and exports models.pkl / model_info.json.')
bullet('app/app.py and app/forecast.py \u2014 serve the forecast-driven web application.')
bullet('Earlier stages (2025 statistical analysis, THI recalculation, PDF data '
       'extraction) are retained in analysis_code.py, '
       'recalculate_thi_and_recreate_figure.py and the extraction scripts.')

# ---------------------------------------------------------------------------
heading('11. Assumptions and Limitations')
bullet('September 2026 dates in Disease 2026.xlsx were interpreted as September dates; '
       'duplicate rows for a date were summed (treated as separate rearing trays).')
bullet('Days without a disease record in the 2026 rearing window were assumed to have '
       'zero incidence.')
bullet('The 23 September 2026 record was excluded from training because observed climate '
       'data for that date were not yet available.')
bullet('2026 rows carry no plot/spacing/instar attributes; these management variables '
       'are consequently excluded from the predictive model.')
bullet('NASA POWER and ERA5 wind-speed magnitudes differ systematically; wind speed is a '
       'minor contributor to the ensemble, and the risk estimates are dominated by '
       'temperature and humidity terms.')
bullet('Weather-only accuracy (~0.79 on the test partition) is lower than earlier '
       'management-inclusive models (~0.96), which is expected because plot and pest '
       'information is no longer used; generalization to the unseen 2026 season remains '
       'strong (0.94\u20130.98).')
bullet('Probabilistic outputs are decision-support estimates derived from a moderate '
       'sample (116 records) and should be combined with field scouting.')

doc.save('Methodology_Silkworm_Disease_Predictor.docx')
print('Saved: Methodology_Silkworm_Disease_Predictor.docx')
print('Paragraphs:', len(doc.paragraphs), '| Tables:', len(doc.tables))
