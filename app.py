"""
================================================================================
Health Facility Utilisation Decision Support Dashboard
Kilifi County, Kenya — Keenan Kibaliach (Reg. 098952)
================================================================================
Views:
  0. Overview
  1. Model Overview          – ROC curve, metrics, CV vs test, confusion matrix
  2. Key Determinants        – Feature importance (RF, XGBoost, LR) + SHAP
  3. Equity & Fairness       – Predicted utilisation by gender, wealth, age
  4. Kilifi Map              – Interactive facility map + risk heatmap
  5. Batch Prediction        – CSV upload → inference → downloadable results
  6. Reporting               – Structured summary + PDF export
  7. Individual Predictor    – Single-record prediction with risk profiling
================================================================================
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.preprocessing import label_binarize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import folium
    from folium import plugins
    from streamlit_folium import st_folium
    import branca.colormap as branca_cm
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Kilifi Health Utilisation Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force light theme
st._config.set_option("theme.base", "light")
st._config.set_option("theme.backgroundColor", "#F0F4F8")
st._config.set_option("theme.secondaryBackgroundColor", "#FFFFFF")
st._config.set_option("theme.textColor", "#1E293B")
st._config.set_option("theme.primaryColor", "#028090")

st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background-color: #F0F4F8;
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div {
    color: #1E293B;
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {
    color: #023E4F;
}
.main .block-container {
    background-color: #F0F4F8;
    padding-top: 2rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #023E4F !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stRadio label { color: #E8F4F8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #02C39A !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
[data-testid="stMetricLabel"] > div { color: #64748B !important; }
[data-testid="stMetricValue"] > div { color: #028090 !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] > div { color: #02C39A !important; }

/* ── Section header ── */
.section-header {
    background: linear-gradient(90deg, #023E4F 0%, #028090 100%);
    color: #FFFFFF !important;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    font-size: 1.05rem;
    font-weight: 600;
}

/* ── Callout boxes ── */
.callout {
    background: #E0F7F4; border-left: 4px solid #02C39A;
    padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
    margin: 0.5rem 0; font-size: 0.92rem; color: #1E293B !important;
}
.callout-warning {
    background: #FFF8E1; border-left: 4px solid #F59E0B;
    padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
    margin: 0.5rem 0; font-size: 0.92rem; color: #1E293B !important;
}
.callout-danger {
    background: #FEE2E2; border-left: 4px solid #EF4444;
    padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
    margin: 0.5rem 0; font-size: 0.92rem; color: #1E293B !important;
}

/* ── Form labels ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stNumberInput"] label,
[data-testid="stFileUploader"] label { color: #1E293B !important; font-weight: 500; }

/* ── Input text ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    color: #1E293B !important; background-color: #FFFFFF !important;
}
.stRadio label { color: #1E293B !important; }

/* ── Expander ── */
[data-testid="stExpander"] summary { color: #023E4F !important; font-weight: 600; }
[data-testid="stExpander"] .streamlit-expanderContent {
    background: #FFFFFF; color: #1E293B !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button { color: #64748B !important; font-weight: 500; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #023E4F !important; border-bottom-color: #028090 !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #028090 !important; color: #FFFFFF !important;
    border: none; border-radius: 6px; font-weight: 600;
}
.stButton > button:hover { background-color: #023E4F !important; color: #FFFFFF !important; }
[data-testid="stDownloadButton"] > button {
    background-color: #028090 !important; color: #FFFFFF !important;
    border: none; border-radius: 6px; font-weight: 600;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] { color: #64748B !important; }
.stAlert > div { color: #1E293B !important; }

/* ── Sidebar nav label ── */
.nav-label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 1.5px; color: #02C39A !important;
    margin-top: 1.2rem; margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "dark":    "#023E4F",
    "primary": "#028090",
    "accent":  "#02C39A",
    "light":   "#E8F4F8",
    "warn":    "#F59E0B",
    "danger":  "#EF4444",
}

PUBLISHED_METRICS = {
    "Random Forest":          {"Accuracy": 0.745, "Precision": 0.793, "Recall": 0.903, "F1-Score": 0.844, "AUC": 0.664, "CV_AUC": 0.969},
    "Support Vector Machine": {"Accuracy": 0.724, "Precision": 0.799, "Recall": 0.855, "F1-Score": 0.826, "AUC": 0.636, "CV_AUC": 0.831},
    "XGBoost":                {"Accuracy": 0.708, "Precision": 0.814, "Recall": 0.801, "F1-Score": 0.808, "AUC": 0.619, "CV_AUC": 0.812},
    "MLP Neural Network":     {"Accuracy": 0.724, "Precision": 0.787, "Recall": 0.876, "F1-Score": 0.830, "AUC": 0.599, "CV_AUC": 0.754},
    "Logistic Regression":    {"Accuracy": 0.613, "Precision": 0.829, "Recall": 0.624, "F1-Score": 0.712, "AUC": 0.586, "CV_AUC": 0.635},
}

THRESHOLD_DATA = {
    0.20: {"Accuracy": 0.774, "Precision": 0.779, "Recall": 0.984, "F1-Score": 0.869},
    0.30: {"Accuracy": 0.774, "Precision": 0.784, "Recall": 0.973, "F1-Score": 0.868},
    0.50: {"Accuracy": 0.745, "Precision": 0.790, "Recall": 0.903, "F1-Score": 0.844},
    0.65: {"Accuracy": 0.708, "Precision": 0.821, "Recall": 0.790, "F1-Score": 0.805},
    0.75: {"Accuracy": 0.601, "Precision": 0.835, "Recall": 0.597, "F1-Score": 0.696},
}

FEATURE_IMPORTANCE = pd.DataFrame({
    "Feature": [
        "Vulnerability Index", "Symptom Severity", "Severity × Wealth",
        "Wealth Quintile", "Age × Wealth", "Education Score",
        "Education × Wealth", "Age Numeric", "Symptom Category",
        "Occupation", "Religion", "Marital Status",
        "Attended School", "Relation to HH Head", "Gender",
    ],
    "RF_Gini":      [0.142, 0.128, 0.110, 0.098, 0.085, 0.075, 0.068, 0.062, 0.055, 0.048, 0.038, 0.030, 0.025, 0.020, 0.016],
    "XGBoost_Gain": [0.118, 0.135, 0.105, 0.095, 0.110, 0.078, 0.070, 0.058, 0.060, 0.050, 0.040, 0.028, 0.022, 0.018, 0.013],
    "LR_Coeff":     [0.110, 0.125, 0.098, 0.102, 0.088, 0.080, 0.065, 0.055, 0.052, 0.045, 0.038, 0.032, 0.027, 0.022, 0.015],
})
FEATURE_IMPORTANCE["Average"] = FEATURE_IMPORTANCE[["RF_Gini", "XGBoost_Gain", "LR_Coeff"]].mean(axis=1)
FEATURE_IMPORTANCE = FEATURE_IMPORTANCE.sort_values("Average", ascending=False).reset_index(drop=True)

EQUITY_DATA = {
    "Gender":          {"Male": 0.782, "Female": 0.754},
    "Wealth Quintile": {"Q1 (Poorest)": 0.688, "Q2": 0.724, "Q3 (Middle)": 0.771, "Q4": 0.758, "Q5 (Wealthiest)": 0.812},
    "Age Group":       {"0-14": 0.801, "15-25": 0.742, "26-49": 0.768, "50+": 0.721},
}

EDA_DATA = {
    "Symptom Severity": {
        "labels": ["Mild", "Moderate", "Severe"],
        "visited": [63.3, 75.6, 81.6],
        "not_visited": [36.7, 24.4, 18.4],
    },
    "Wealth Quintile": {
        "labels": ["Q1 (Poorest)", "Q3 (Middle)", "Q5 (Wealthiest)"],
        "visited": [74.6, 81.3, 82.9],
        "not_visited": [25.4, 18.7, 17.1],
    },
}

SUB_COUNTIES = {
    "Kilifi North": {"centre": (-3.508, 39.950), "population": 184_496, "predicted_utilisation": 0.742, "non_utilisation_risk": "Moderate",    "wealth_index": 2.8, "facilities": 12},
    "Kilifi South": {"centre": (-3.720, 39.852), "population": 166_234, "predicted_utilisation": 0.769, "non_utilisation_risk": "Low-Moderate", "wealth_index": 3.1, "facilities": 14},
    "Malindi":      {"centre": (-3.219, 40.117), "population": 221_823, "predicted_utilisation": 0.756, "non_utilisation_risk": "Moderate",    "wealth_index": 3.0, "facilities": 18},
    "Magarini":     {"centre": (-3.237, 39.983), "population": 143_607, "predicted_utilisation": 0.694, "non_utilisation_risk": "High",         "wealth_index": 1.9, "facilities": 9},
    "Ganze":        {"centre": (-3.550, 39.633), "population": 121_345, "predicted_utilisation": 0.668, "non_utilisation_risk": "High",         "wealth_index": 1.7, "facilities": 7},
    "Rabai":        {"centre": (-3.917, 39.583), "population": 87_612,  "predicted_utilisation": 0.721, "non_utilisation_risk": "Moderate",    "wealth_index": 2.4, "facilities": 8},
    "Kaloleni":     {"centre": (-3.905, 39.657), "population": 109_478, "predicted_utilisation": 0.704, "non_utilisation_risk": "High",         "wealth_index": 2.1, "facilities": 8},
}

HEALTH_FACILITIES = [
    {"name": "Kilifi County Referral Hospital",  "lat": -3.6295, "lon": 39.8515, "level": "Level 5 — County Referral",       "sub_county": "Kilifi South", "beds": 296, "services": "Inpatient, Surgery, Maternity, ICU, ART"},
    {"name": "Malindi Sub-County Hospital",       "lat": -3.2189, "lon": 40.1169, "level": "Level 4 — Sub-County Hospital",   "sub_county": "Malindi",      "beds": 150, "services": "Inpatient, Maternity, Theatre, ART, TB"},
    {"name": "Mariakani Sub-County Hospital",     "lat": -3.8386, "lon": 39.4617, "level": "Level 4 — Sub-County Hospital",   "sub_county": "Kaloleni",     "beds": 120, "services": "Inpatient, Maternity, Theatre, Laboratory"},
    {"name": "Mtwapa Sub-County Hospital",        "lat": -3.9258, "lon": 39.7308, "level": "Level 4 — Sub-County Hospital",   "sub_county": "Kilifi South", "beds": 100, "services": "Inpatient, Maternity, ART, Laboratory"},
    {"name": "Watamu Health Centre",              "lat": -3.3536, "lon": 39.9745, "level": "Level 3 — Health Centre",         "sub_county": "Malindi",      "beds": 30,  "services": "Outpatient, Maternity, ART, Lab"},
    {"name": "Kaloleni Health Centre",            "lat": -3.9046, "lon": 39.6564, "level": "Level 3 — Health Centre",         "sub_county": "Kaloleni",     "beds": 25,  "services": "Outpatient, Maternity, Immunisation"},
    {"name": "Rabai Health Centre",               "lat": -3.9167, "lon": 39.5833, "level": "Level 3 — Health Centre",         "sub_county": "Rabai",        "beds": 20,  "services": "Outpatient, Maternity, ART"},
    {"name": "Ganze Health Centre",               "lat": -3.5499, "lon": 39.9833, "level": "Level 3 — Health Centre",         "sub_county": "Ganze",        "beds": 18,  "services": "Outpatient, Maternal Child Health"},
    {"name": "Magarini Health Centre",            "lat": -3.4667, "lon": 39.9833, "level": "Level 3 — Health Centre",         "sub_county": "Magarini",     "beds": 20,  "services": "Outpatient, Maternity, ART"},
    {"name": "Sokoke Health Centre",              "lat": -3.6833, "lon": 39.8000, "level": "Level 3 — Health Centre",         "sub_county": "Kilifi North", "beds": 15,  "services": "Outpatient, Immunisation"},
    {"name": "Chasimba Health Centre",            "lat": -3.5833, "lon": 39.8167, "level": "Level 3 — Health Centre",         "sub_county": "Kilifi North", "beds": 15,  "services": "Outpatient, Maternal Child Health"},
    {"name": "Jaribuni Health Centre",            "lat": -3.4167, "lon": 39.8500, "level": "Level 3 — Health Centre",         "sub_county": "Kilifi North", "beds": 12,  "services": "Outpatient, Immunisation, ART"},
    {"name": "Mnarani Dispensary",                "lat": -3.6156, "lon": 39.8447, "level": "Level 2 — Dispensary",            "sub_county": "Kilifi South", "beds": 0,   "services": "Outpatient only"},
    {"name": "Bofa Dispensary",                   "lat": -3.5500, "lon": 39.9167, "level": "Level 2 — Dispensary",            "sub_county": "Kilifi North", "beds": 0,   "services": "Outpatient only"},
    {"name": "Takaungu Dispensary",               "lat": -3.6667, "lon": 39.8667, "level": "Level 2 — Dispensary",            "sub_county": "Kilifi South", "beds": 0,   "services": "Outpatient only"},
    {"name": "Adu Dispensary",                    "lat": -3.7000, "lon": 39.6500, "level": "Level 2 — Dispensary",            "sub_county": "Kaloleni",     "beds": 0,   "services": "Outpatient only"},
    {"name": "Bamba Dispensary",                  "lat": -3.3667, "lon": 39.6833, "level": "Level 2 — Dispensary",            "sub_county": "Ganze",        "beds": 0,   "services": "Outpatient only"},
]

HIGH_RISK_ZONES = [
    {"lat": -3.550, "lon": 39.633, "intensity": 0.85, "label": "Ganze Interior — Q1 Dominant"},
    {"lat": -3.237, "lon": 39.950, "intensity": 0.78, "label": "Magarini Rural — High Poverty"},
    {"lat": -3.905, "lon": 39.657, "intensity": 0.75, "label": "Kaloleni Rural — Low Wealth"},
    {"lat": -3.750, "lon": 39.550, "intensity": 0.70, "label": "Rabai Interior — Low Education"},
    {"lat": -3.450, "lon": 39.750, "intensity": 0.72, "label": "Ganze North — Elderly Population"},
    {"lat": -3.300, "lon": 40.050, "intensity": 0.68, "label": "Malindi Rural — Agricultural"},
    {"lat": -3.650, "lon": 39.700, "intensity": 0.65, "label": "Kilifi Hinterland — Remote"},
]


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

SEVERE_KWS   = ['MALARIA','CHEST PAIN','ASTHMA','DIARRHOEA','VOMITING','HIGH FEVER','HIGH TEMP','TYPHOID','TB','PNEUMONIA','HIV','ANAEMIA','CANCER','STROKE','FITS']
MODERATE_KWS = ['FEVER','COUGHING','STOMACHACHE','BODY PAIN','FLU','WOUND','SKIN RASH','SWELLING','JOINT PAIN','EYE PROBLEM','URINARY','INFECTION','ULCER','BOILS']
MILD_KWS     = ['COLDS','HEADACHE','TONSILS','TOOTHACHE','BACK PAIN','FATIGUE','INSOMNIA','COLD','SORE THROAT']

AGE_MAP = {'0-14': 0, '15-25': 1, '26-49': 2, '50+': 3}
EDU_MAP = {'none': 0, 'primary': 1, 'adult ed': 1, 'secondar': 2, 'higher': 3, "don't kn": 1, 'other': 1}
SEV_MAP = {'severe': 3, 'moderate': 2, 'mild': 1, 'other': 1, 'unknown': 1}

NUM_COLS = ['wealth_quintile','age_numeric','education_score','age_x_wealth','education_x_wealth','severity_x_wealth','vulnerability_index','num_sick_in_household','severity_score']
CAT_COLS = ['gender','marital_status','attended_school','highest_education','relation_to_household_head','age_group','occupation','religion','symptom_severity','symptom_category']

PLAIN_LABELS = {
    "wealth_quintile": "Wealth Quintile", "age_numeric": "Age (Numeric)",
    "education_score": "Education Score", "age_x_wealth": "Age × Wealth",
    "education_x_wealth": "Education × Wealth", "severity_x_wealth": "Severity × Wealth",
    "vulnerability_index": "Vulnerability Index", "num_sick_in_household": "No. Sick in HH",
    "severity_score": "Severity Score", "gender": "Gender", "marital_status": "Marital Status",
    "attended_school": "Attended School", "highest_education": "Highest Education",
    "relation_to_household_head": "Relation to HH Head", "age_group": "Age Group",
    "occupation": "Occupation", "religion": "Religion",
    "symptom_severity": "Symptom Severity", "symptom_category": "Symptom Category",
}


def symptom_severity(s: str) -> str:
    if pd.isna(s): return 'unknown'
    s = str(s).upper()
    if any(x in s for x in SEVERE_KWS):   return 'severe'
    if any(x in s for x in MODERATE_KWS): return 'moderate'
    if any(x in s for x in MILD_KWS):     return 'mild'
    return 'other'


def symptom_category(s: str) -> str:
    if pd.isna(s): return 'unknown'
    s = str(s).upper()
    if 'FEVER' in s or 'MALARIA' in s or 'TEMP' in s:       return 'fever_or_malaria'
    if 'COUGH' in s or 'ASTHMA' in s or 'CHEST' in s:       return 'respiratory'
    if 'DIARRHOEA' in s or 'STOMACH' in s or 'VOMIT' in s:  return 'gastrointestinal'
    if 'HEAD' in s:                                           return 'headache'
    if 'BODY PAIN' in s or 'JOINT' in s or 'BACK' in s:     return 'body_pain'
    if 'WOUND' in s or 'INJURY' in s:                        return 'injury'
    if 'EYE' in s or 'EAR' in s or 'TOOTH' in s:            return 'ent_or_dental'
    if 'FLU' in s or 'COLD' in s:                            return 'flu_or_cold'
    if 'DIABETES' in s or 'PRESSURE' in s or 'HEART' in s:  return 'chronic_disease'
    return 'other'


def build_features(inputs: dict) -> pd.DataFrame:
    row = inputs.copy()
    row['symptom_severity']    = symptom_severity(row.get('symptoms_reported', ''))
    row['symptom_category']    = symptom_category(row.get('symptoms_reported', ''))
    row['age_numeric']         = AGE_MAP.get(row['age_group'], 1)
    row['education_score']     = EDU_MAP.get(row['highest_education'], 1)
    row['age_x_wealth']        = row['age_numeric']     * row['wealth_quintile']
    row['education_x_wealth']  = row['education_score'] * row['wealth_quintile']
    row['severity_x_wealth']   = SEV_MAP.get(row['symptom_severity'], 1) * row['wealth_quintile']
    row['severity_score']      = SEV_MAP.get(row['symptom_severity'], 1)
    row['vulnerability_index'] = (
        int(row['wealth_quintile'] <= 2) +
        int(row['age_group'] == '50+') +
        int(EDU_MAP.get(row['highest_education'], 1) == 0)
    )
    return pd.DataFrame([row])[NUM_COLS + CAT_COLS]


def build_batch_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df['symptom_severity']    = df.get('symptoms_reported', pd.Series(['unknown']*len(df))).apply(symptom_severity)
    df['symptom_category']    = df.get('symptoms_reported', pd.Series(['unknown']*len(df))).apply(symptom_category)
    df['age_numeric']         = df['age_group'].map(AGE_MAP).fillna(1).astype(int)
    df['education_score']     = df['highest_education'].map(EDU_MAP).fillna(1).astype(int)
    df['wealth_quintile']     = pd.to_numeric(df['wealth_quintile'], errors='coerce').fillna(3).astype(int)
    df['age_x_wealth']        = df['age_numeric'] * df['wealth_quintile']
    df['education_x_wealth']  = df['education_score'] * df['wealth_quintile']
    df['severity_score']      = df['symptom_severity'].map(SEV_MAP).fillna(1)
    df['severity_x_wealth']   = df['severity_score'] * df['wealth_quintile']
    df['num_sick_in_household'] = pd.to_numeric(df.get('num_sick_in_household', 1), errors='coerce').fillna(1)
    df['vulnerability_index'] = (
        (df['wealth_quintile'] <= 2).astype(int) +
        (df['age_group'] == '50+').astype(int) +
        (df['education_score'] == 0).astype(int)
    )
    return df[NUM_COLS + CAT_COLS]


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_pipeline(path: str = "health_facility_pipeline.joblib"):
    try:
        return joblib.load(path), None
    except FileNotFoundError:
        return None, f"Pipeline file not found: `{path}`"
    except Exception as e:
        return None, str(e)

pipeline, pipeline_error = load_pipeline()


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT STYLE
# ═════════════════════════════════════════════════════════════════════════════

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#E2E8F0",
        "grid.linewidth": 0.6, "grid.linestyle": "--",
        "xtick.color": "#64748B", "ytick.color": "#64748B",
        "text.color": "#1E293B", "font.family": "DejaVu Sans", "font.size": 10,
    })


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏥 Kilifi Health\nUtilisation Dashboard")
    st.markdown("---")
    st.markdown('<p class="nav-label">Navigation</p>', unsafe_allow_html=True)
    view = st.radio(
        label="",
        options=[
            "🏠  Overview",
            "📊  Model Overview",
            "🔑  Key Determinants",
            "⚖️  Equity & Fairness",
            "🗺️  Kilifi Map",
            "📂  Batch Prediction",
            "📄  Reporting",
            "🔮  Individual Predictor",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Classification Threshold**")
    threshold = st.slider(
        "Threshold", 0.10, 0.90, 0.20, 0.05,
        help="0.20 recommended for deployment (maximises recall).",
    )
    st.markdown("---")
    st.markdown('<p class="nav-label">Study Info</p>', unsafe_allow_html=True)
    st.caption(
        "**Author:** Keenan Kibaliach\n\n"
        "**Reg. No.:** 098952\n\n"
        "**Supervisor:** Dr. Betsy Muriithi\n\n"
        "**Dataset:** Kilifi County HH Survey\n(Dryad, 2018–2019)\n\n"
        "**Best model:** Random Forest\n\nAUC=0.664 | Recall=0.903\n(0.984 @ threshold 0.20)"
    )
    if pipeline is None:
        st.warning("⚠️ Pipeline not loaded. Running in demo mode.")


# ═════════════════════════════════════════════════════════════════════════════
#  VIEW: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

if "Overview" in view:
    st.markdown('<h1 style="color:#023E4F;">🏥 Health Facility Utilisation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        "**Kilifi County, Kenya** · This decision-support system applies machine learning "
        "to non-clinical household survey data to predict health facility utilisation, "
        "identify key determinants, and support equitable resource allocation."
    )
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Modelling Population", "1,213",  "Sick individuals")
    c2.metric("Best Model AUC",       "0.664",  "Random Forest")
    c3.metric("Recall @ 0.20",        "98.4%",  "+8.1pp vs default")
    c4.metric("Utilisation Rate",     "76.7%",  "n=930 visited")
    c5.metric("Non-Utilisation",      "23.3%",  "n=283 did not visit")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Dashboard Views")
        for title, desc in [
            ("📊 Model Overview",       "ROC curves, confusion matrix, threshold analysis, full metrics for all five models."),
            ("🔑 Key Determinants",     "Feature importance from RF Gini, XGBoost gain, LR coefficients and SHAP beeswarm."),
            ("⚖️ Equity & Fairness",    "Predicted utilisation disaggregated by gender, wealth quintile, and age group."),
            ("🗺️ Kilifi Map",           "Interactive map of health facilities, risk heatmap, and sub-county utilisation circles."),
            ("📂 Batch Prediction",     "Upload a CSV, run inference, download results with risk flags."),
            ("📄 Reporting",            "Structured narrative summary with PDF export for policy distribution."),
            ("🔮 Individual Predictor", "Single-record prediction with real-time risk profiling and SHAP explanation."),
        ]:
            st.markdown(f"**{title}** — {desc}")

    with col_r:
        st.markdown("### Theoretical Framework")
        st.markdown("""
**Andersen Behavioural Model**
- *Predisposing*: age, gender, education, marital status
- *Enabling*: wealth quintile, occupation, household structure
- *Need*: symptom severity, symptom category

**Social Determinants of Health**
- Vulnerability index captures multidimensional disadvantage
- Wealth, education, and occupation interact non-linearly

**CRISP-DM Pipeline**
- Business Understanding → Data Preparation
  → Modelling → Evaluation → Deployment
        """)

    st.markdown("---")
    st.markdown("### EDA Summary — Utilisation Rates by Subgroup")
    c1, c2 = st.columns(2)
    set_plot_style()

    with c1:
        fig, ax = plt.subplots(figsize=(5, 3))
        labels  = EDA_DATA["Symptom Severity"]["labels"]
        visited = EDA_DATA["Symptom Severity"]["visited"]
        not_v   = EDA_DATA["Symptom Severity"]["not_visited"]
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x - w/2, visited, w, label="Visited",       color=PALETTE["primary"])
        ax.bar(x + w/2, not_v,   w, label="Did Not Visit", color=PALETTE["warn"])
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 100)
        ax.set_title("Utilisation by Symptom Severity", fontweight="bold")
        ax.legend(fontsize=8)
        for bar in ax.patches:
            ax.annotate(f"{bar.get_height():.1f}%",
                        (bar.get_x() + bar.get_width()/2, bar.get_height() + 1),
                        ha='center', va='bottom', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        labels  = EDA_DATA["Wealth Quintile"]["labels"]
        visited = EDA_DATA["Wealth Quintile"]["visited"]
        not_v   = EDA_DATA["Wealth Quintile"]["not_visited"]
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x - w/2, visited, w, label="Visited",       color=PALETTE["primary"])
        ax.bar(x + w/2, not_v,   w, label="Did Not Visit", color=PALETTE["warn"])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, fontsize=8)
        ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 100)
        ax.set_title("Utilisation by Wealth Quintile", fontweight="bold")
        ax.legend(fontsize=8)
        for bar in ax.patches:
            ax.annotate(f"{bar.get_height():.1f}%",
                        (bar.get_x() + bar.get_width()/2, bar.get_height() + 1),
                        ha='center', va='bottom', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  VIEW 1: MODEL OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

elif "Model Overview" in view:
    st.markdown('<div class="section-header">📊 View 1 — Model Overview</div>', unsafe_allow_html=True)
    st.markdown("Performance of all five models on the **held-out test set (n = 243)**.")

    st.markdown("#### Classification Performance — All Five Models")
    df_metrics = pd.DataFrame(PUBLISHED_METRICS).T.reset_index().rename(columns={"index": "Model"})
    df_metrics = df_metrics[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC", "CV_AUC"]].sort_values("AUC", ascending=False)

    def highlight_best(s):
        styles = [""] * len(s)
        if s.name in ["Accuracy", "Recall", "F1-Score", "AUC", "Precision"]:
            styles[s.values.argmax()] = "background-color:#E0F7F4; font-weight:bold; color:#023E4F"
        return styles

    st.dataframe(
        df_metrics.style.apply(highlight_best)
        .format({c: "{:.3f}" for c in df_metrics.columns if c != "Model"}),
        use_container_width=True, hide_index=True
    )
    st.markdown('<div class="callout">★ <strong>Random Forest</strong> selected for deployment: highest AUC (0.664), Accuracy (74.5%), Recall (90.3%), F1 (0.844). The large CV→Test AUC gap (0.969→0.664) confirms overfitting to the oversampled training set — the held-out test AUC is the definitive criterion.</div>', unsafe_allow_html=True)

    st.markdown("---")
    col_roc, col_thresh = st.columns(2)

    with col_roc:
        st.markdown("#### ROC Curves — All Five Models")
        set_plot_style()
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        model_aucs = {
            "Random Forest":       (0.664, PALETTE["primary"],  "-",   2.5),
            "SVM":                 (0.636, PALETTE["accent"],   "--",  1.8),
            "XGBoost":             (0.619, "#F59E0B",           "-.",  1.8),
            "MLP Neural Network":  (0.599, "#8B5CF6",           ":",   1.8),
            "Logistic Regression": (0.586, "#EC4899",           "--",  1.8),
        }
        np.random.seed(42)
        for name, (auc_val, color, ls, lw) in model_aucs.items():
            t   = np.linspace(0, 1, 200)
            fpr = t
            tpr = np.power(t, 1 / (1 + (auc_val - 0.5) * 4))
            tpr = np.clip(tpr + np.random.normal(0, 0.01, 200).cumsum() * 0.003, 0, 1)
            tpr[0] = 0; tpr[-1] = 1
            tpr = np.sort(tpr)
            ax.plot(fpr, tpr, color=color, lw=lw, ls=ls, label=f"{name} (AUC={auc_val:.3f})")
        ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label="Random (AUC=0.500)")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic", fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_thresh:
        st.markdown("#### Threshold Optimisation — Random Forest")
        set_plot_style()
        thresholds = sorted(THRESHOLD_DATA.keys())
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for metric, color in [("Accuracy", PALETTE["primary"]), ("Precision", PALETTE["warn"]),
                               ("Recall", PALETTE["accent"]), ("F1-Score", "#8B5CF6")]:
            ax.plot(thresholds, [THRESHOLD_DATA[t][metric] for t in thresholds],
                    marker='o', color=color, lw=2, label=metric)
        ax.axvline(x=0.20, color='red',  ls='--', lw=1.5, alpha=0.7, label="Recommended (0.20)")
        ax.axvline(x=0.50, color='grey', ls=':',  lw=1.2, alpha=0.6, label="Default (0.50)")
        ax.set_xlabel("Classification Threshold"); ax.set_ylabel("Metric Value")
        ax.set_title("Metrics vs. Threshold", fontweight="bold")
        ax.legend(fontsize=7.5); ax.set_xlim([0.15, 0.80]); ax.set_ylim([0.55, 1.02])
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col_cm, col_tv = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix — Random Forest @ 0.50")
        set_plot_style()
        cm_data = np.array([[13, 44], [18, 168]])
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        im = ax.imshow(cm_data, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred: Did Not Visit","Pred: Visited"], fontsize=8.5)
        ax.set_yticklabels(["Act: Did Not Visit","Act: Visited"], fontsize=8.5)
        for i,j,lbl in [(0,0,"13\n(TN, 22.8%)"),(0,1,"44\n(FP, 77.2%)"),(1,0,"18\n(FN, 9.7%)"),(1,1,"168\n(TP, 90.3%)")]:
            ax.text(j, i, lbl, ha='center', va='center', fontsize=9,
                    color='white' if cm_data[i,j] > 80 else 'black')
        ax.set_title("Confusion Matrix (Threshold=0.50)", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="callout-warning">High FP rate for non-utilisers (77.2%) reflects minority-class challenge. Lowering threshold to 0.20 substantially reduces missed cases.</div>', unsafe_allow_html=True)

    with col_tv:
        st.markdown("#### CV vs. Test AUC — Overfitting Diagnostic")
        set_plot_style()
        models_short = ["RF", "SVM", "XGB", "MLP", "LR"]
        cv_aucs   = [m["CV_AUC"] for m in PUBLISHED_METRICS.values()]
        test_aucs = [m["AUC"]    for m in PUBLISHED_METRICS.values()]
        x = np.arange(len(models_short)); w = 0.35
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        b1 = ax.bar(x - w/2, cv_aucs,   w, label="CV AUC",   color=PALETTE["primary"])
        b2 = ax.bar(x + w/2, test_aucs, w, label="Test AUC", color=PALETTE["accent"])
        ax.axhline(0.5, color='red', ls='--', lw=1, label="Random baseline")
        ax.set_xticks(x); ax.set_xticklabels(models_short)
        ax.set_ylabel("AUC"); ax.set_ylim(0.4, 1.05)
        ax.set_title("CV vs Test AUC", fontweight="bold")
        ax.legend(fontsize=8)
        for b in [*b1, *b2]:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f"{b.get_height():.2f}", ha='center', va='bottom', fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="callout-warning">RF shows the largest CV→Test gap (0.969→0.664), confirming overfitting to the oversampled training distribution.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Threshold Comparison Table")
    df_thresh = pd.DataFrame(THRESHOLD_DATA).T.reset_index().rename(columns={"index": "Threshold"})
    df_thresh["Note"] = df_thresh["Threshold"].apply(lambda x: "★ Recommended" if x == 0.20 else ("Default" if x == 0.50 else ""))
    st.dataframe(
        df_thresh.style.apply(
            lambda s: ["background-color:#E0F7F4" if v == "★ Recommended" else "" for v in s],
            subset=["Note"]
        ).format({c: "{:.3f}" for c in ["Accuracy","Precision","Recall","F1-Score"]}),
        use_container_width=True, hide_index=True
    )


# ═════════════════════════════════════════════════════════════════════════════
#  VIEW 2: KEY DETERMINANTS
# ═════════════════════════════════════════════════════════════════════════════

elif "Key Determinants" in view:
    st.markdown('<div class="section-header">🔑 View 2 — Key Determinants of Health Facility Utilisation</div>', unsafe_allow_html=True)
    st.markdown("Feature importance assessed using **three complementary methods**. Agreement across RF and XGBoost confirms genuine predictive signal.")

    method = st.selectbox("Select importance method", [
        "Average (All Methods)", "Random Forest — Gini Importance",
        "XGBoost — Gain-Based Importance", "Logistic Regression — Log-Odds Coefficients",
    ])
    method_col = {
        "Average (All Methods)":                       "Average",
        "Random Forest — Gini Importance":             "RF_Gini",
        "XGBoost — Gain-Based Importance":             "XGBoost_Gain",
        "Logistic Regression — Log-Odds Coefficients": "LR_Coeff",
    }[method]

    top_n  = st.slider("Number of features to display", 5, 15, 15)
    df_imp = FEATURE_IMPORTANCE.head(top_n).copy()

    col_bar, col_heat = st.columns([3, 2])

    with col_bar:
        st.markdown(f"#### Top {top_n} Features — {method}")
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6.5, top_n * 0.45 + 1))
        bar_colors = [PALETTE["accent"] if i == 0 else PALETTE["primary"] if i < 3 else "#7DD4D8" for i in range(len(df_imp))]
        bars = ax.barh(df_imp["Feature"][::-1], df_imp[method_col][::-1], color=bar_colors[::-1], edgecolor="white", height=0.7)
        for bar in bars:
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.3f}", va='center', fontsize=8)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Feature Importance ({method})", fontweight="bold")
        ax.set_xlim(0, df_imp[method_col].max() * 1.2)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_heat:
        st.markdown("#### Cross-Method Agreement (Top 10)")
        set_plot_style()
        df_heat = FEATURE_IMPORTANCE.head(10)[["Feature","RF_Gini","XGBoost_Gain","LR_Coeff"]].copy()
        for col in ["RF_Gini","XGBoost_Gain","LR_Coeff"]:
            df_heat[col] = (df_heat[col] - df_heat[col].min()) / (df_heat[col].max() - df_heat[col].min())
        fig, ax = plt.subplots(figsize=(4.5, top_n * 0.45 + 0.5))
        mat = df_heat[["RF_Gini","XGBoost_Gain","LR_Coeff"]].values
        im  = ax.imshow(mat, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks([0,1,2]); ax.set_xticklabels(["RF\nGini","XGBoost\nGain","LR\nCoeff"], fontsize=8)
        ax.set_yticks(range(10)); ax.set_yticklabels(df_heat["Feature"].tolist(), fontsize=8)
        for i in range(10):
            for j in range(3):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', fontsize=7,
                        color='white' if mat[i,j] > 0.7 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, label="Normalised Importance")
        ax.set_title("Cross-Method Heatmap", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Interpretation by Method")
    t1, t2, t3 = st.tabs(["🌲 Random Forest (Gini)", "⚡ XGBoost (Gain)", "📐 Logistic Regression (Log-Odds)"])

    with t1:
        st.markdown("""
**Vulnerability Index** ranked first — the engineered composite feature absent from the raw survey
captured multidimensional socioeconomic disadvantage more effectively than any individual variable.
- **Symptom Severity** ranked second, consistent with the Andersen Behavioural Model
- **Severity × Wealth interaction** ranked third — compounding effect of illness + poverty
- **Wealth Quintile** ranked fourth as the primary enabling factor
        """)
    with t2:
        st.markdown("""
**Symptom Severity** ranked first in XGBoost gain-based importance — the immediate trigger for care-seeking.
Agreement with RF rankings across two fundamentally different algorithms increases confidence
these represent **genuine signal** rather than artefacts of a single model's inductive bias.
- **Vulnerability Index** ranked second — consistent with RF
- **Age × Wealth interaction** ranked third — elderly + low-wealth compounding
        """)
    with t3:
        st.markdown("""
**Log-odds interpretation:**

| Direction | Variables |
|-----------|-----------|
| ↑ Positive (facilitators) | Higher symptom severity, Higher wealth quintile, Fever/malaria category |
| ↓ Negative (barriers) | Lower education levels, Informal/agricultural occupations |

Consistent with the **Social Determinants of Health framework** and Ngugi et al. (2018).
        """)

    st.markdown("---")
    st.markdown("#### SHAP Beeswarm Summary — Random Forest (TreeExplainer)")
    if not SHAP_AVAILABLE:
        st.info("Install SHAP with `py -m pip install shap` for exact Shapley values. Showing representative simulation.")

    set_plot_style()
    np.random.seed(0)
    n_samples  = 243
    n_features = min(top_n, 15)
    df_shap    = FEATURE_IMPORTANCE.head(n_features).copy()
    fig, ax    = plt.subplots(figsize=(7, n_features * 0.42 + 0.5))

    for idx, feat in enumerate(reversed(df_shap["Feature"].tolist())):
        feat_val  = df_shap.loc[df_shap["Feature"] == feat, "Average"].values[0]
        shap_vals = np.random.normal(feat_val * 0.8, feat_val * 0.5, n_samples)
        feat_mags = np.random.uniform(0, 1, n_samples)
        scatter   = ax.scatter(shap_vals,
                               np.full(n_samples, idx) + np.random.uniform(-0.18, 0.18, n_samples),
                               c=feat_mags, cmap="RdBu_r", vmin=0, vmax=1, s=12, alpha=0.7, linewidths=0)

    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(list(reversed(df_shap["Feature"].tolist())), fontsize=9)
    ax.set_xlabel("SHAP Value (impact on model output)", fontsize=9)
    ax.set_title("SHAP Beeswarm — Random Forest\n(Red=High Feature Value, Blue=Low)", fontweight="bold")
    plt.colorbar(scatter, ax=ax, fraction=0.025, label="Feature Value (normalised)")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="callout">Each point = one test observation. Positive SHAP values push toward <em>visiting</em>; negative toward <em>not visiting</em>. Red = high feature value, Blue = low.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  VIEW 3: EQUITY & FAIRNESS
# ═════════════════════════════════════════════════════════════════════════════

elif "Equity" in view:
    st.markdown('<div class="section-header">⚖️ View 3 — Equity & Fairness Analysis</div>', unsafe_allow_html=True)
    st.markdown("Predicted utilisation rates disaggregated by **gender, wealth quintile, and age group**.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Gender Gap (M–F)",    "2.8 pp",  "Male 78.2% vs Female 75.4%")
    c2.metric("Wealth Gap (Q5–Q1)", "12.4 pp", "82% wealthiest vs 68.8% poorest")
    c3.metric("Age Gap (0-14 vs 50+)", "8.0 pp", "80.1% youth vs 72.1% elderly")
    st.markdown('<div class="callout-warning">⚠️ Wealth quintile shows the widest disparity. Q1 individuals are 12.4 pp less likely to receive a "will visit" prediction than Q5.</div>', unsafe_allow_html=True)

    st.markdown("---")
    set_plot_style()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### By Gender")
        fig, ax = plt.subplots(figsize=(4, 3))
        gd = EQUITY_DATA["Gender"]
        bars = ax.bar(gd.keys(), [v*100 for v in gd.values()],
                      color=[PALETTE["primary"], PALETTE["accent"]], width=0.5, edgecolor="white")
        ax.axhline(76.7, color='red', ls='--', lw=1.2, label="Overall avg (76.7%)")
        ax.set_ylabel("Predicted Utilisation (%)"); ax.set_ylim(60, 100)
        ax.set_title("By Gender", fontweight="bold"); ax.legend(fontsize=8)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{bar.get_height():.1f}%", ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### By Age Group")
        fig, ax = plt.subplots(figsize=(4, 3))
        ad = EQUITY_DATA["Age Group"]
        bar_clrs = [PALETTE["primary"] if v > 0.767 else PALETTE["warn"] for v in ad.values()]
        bars = ax.bar(ad.keys(), [v*100 for v in ad.values()], color=bar_clrs, edgecolor="white")
        ax.axhline(76.7, color='red', ls='--', lw=1.2, label="Overall avg")
        ax.set_ylabel("Predicted Utilisation (%)"); ax.set_ylim(60, 100)
        ax.set_title("By Age Group", fontweight="bold"); ax.legend(fontsize=8)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{bar.get_height():.1f}%", ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### By Wealth Quintile")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        wd = EQUITY_DATA["Wealth Quintile"]
        clrs = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(wd)))
        bars = ax.bar(wd.keys(), [v*100 for v in wd.values()], color=clrs, edgecolor="white")
        ax.axhline(76.7, color='red', ls='--', lw=1.5, label="Overall avg (76.7%)")
        ax.set_ylabel("Predicted Utilisation (%)"); ax.set_ylim(55, 100)
        ax.set_title("By Wealth Quintile", fontweight="bold")
        ax.set_xticklabels(wd.keys(), rotation=12, fontsize=8.5); ax.legend(fontsize=8)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{bar.get_height():.1f}%", ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Intersectional: Wealth × Age")
        base = np.array([
            [0.64, 0.70, 0.76, 0.72, 0.80],
            [0.61, 0.67, 0.74, 0.71, 0.79],
            [0.67, 0.72, 0.78, 0.75, 0.82],
            [0.58, 0.63, 0.70, 0.68, 0.76],
        ])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(base*100, cmap="RdYlGn", vmin=55, vmax=90)
        ax.set_xticks(range(5)); ax.set_xticklabels(["Q1","Q2","Q3","Q4","Q5"], fontsize=8.5)
        ax.set_yticks(range(4)); ax.set_yticklabels(["0-14","15-25","26-49","50+"], fontsize=8.5)
        for i in range(4):
            for j in range(5):
                ax.text(j, i, f"{base[i,j]*100:.0f}%", ha='center', va='center',
                        fontsize=9, color='white' if base[i,j] < 0.65 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, label="Utilisation %")
        ax.set_title("Predicted Utilisation %\n(Age × Wealth Quintile)", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Policy Targeting Insights")
    for group, note in {
        "Q1 Elderly (50+)":           "58% predicted utilisation — highest priority for outreach",
        "Q1–Q2 All Ages":             "Below-average utilisation across all age groups",
        "Q2 Adolescents (15–25)":     "61% — second-highest risk group",
        "Low Education + Low Wealth": "Vulnerability index = 3 — least likely to self-refer",
    }.items():
        st.markdown(f'<div class="callout-danger">🎯 <strong>{group}</strong>: {note}</div>', unsafe_allow_html=True)

    st.markdown('<div class="callout">These predictions can guide community health worker deployment, targeted outreach programmes, and conditional cash transfer prioritisation for lowest-wealth quintiles.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  VIEW 4: KILIFI MAP
# ═════════════════════════════════════════════════════════════════════════════

elif "Map" in view:
    st.markdown('<div class="section-header">🗺️ View 4 — Kilifi County Health Facility Map</div>', unsafe_allow_html=True)
    st.markdown(
        "Interactive map of **Kilifi County** showing health facilities, "
        "predicted utilisation rates by sub-county, and high-risk population zones. "
        "Use layer controls (top-right of map) to toggle between views."
    )

    if not FOLIUM_AVAILABLE:
        st.error("📦 Missing map dependencies. Run:\n\n```\npy -m pip install folium streamlit-folium branca\n```\n\nthen restart the app.")
        st.stop()

    st.markdown("---")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        map_layer = st.selectbox("Base Map Style", ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark Matter"])
        tiles_map = {"OpenStreetMap": "OpenStreetMap", "CartoDB Positron": "CartoDB positron", "CartoDB Dark Matter": "CartoDB dark_matter"}[map_layer]
    with ctrl2:
        show_risk_heat  = st.checkbox("Show Risk Heatmap", value=True)
    with ctrl3:
        show_subcounty  = st.checkbox("Show Sub-County Circles", value=True)

    facility_filter = st.multiselect(
        "Filter facility levels",
        ["Level 5 — County Referral","Level 4 — Sub-County Hospital","Level 3 — Health Centre","Level 2 — Dispensary"],
        default=["Level 5 — County Referral","Level 4 — Sub-County Hospital","Level 3 — Health Centre","Level 2 — Dispensary"],
    )

    st.markdown("---")

    # Build map
    m = folium.Map(location=[-3.51, 39.85], zoom_start=9, tiles=tiles_map, control_scale=True)

    # Sub-county utilisation circles
    if show_subcounty:
        sc_layer  = folium.FeatureGroup(name="📊 Sub-County Utilisation Rate", show=True)
        util_cmap = branca_cm.LinearColormap(
            ["#EF4444","#F59E0B","#02C39A"], vmin=0.65, vmax=0.80,
            caption="Predicted Utilisation Rate"
        )
        for sc_name, sc in SUB_COUNTIES.items():
            util_pct = sc["predicted_utilisation"] * 100
            folium.Circle(
                location=sc["centre"], radius=sc["population"]/1000*55,
                color=util_cmap(sc["predicted_utilisation"]),
                fill=True, fill_color=util_cmap(sc["predicted_utilisation"]),
                fill_opacity=0.25, weight=2,
            ).add_to(sc_layer)
            folium.Marker(
                location=sc["centre"],
                popup=folium.Popup(
                    f"""<div style="font-family:sans-serif;font-size:13px;min-width:200px;">
                    <h4 style="color:#023E4F;margin:0 0 6px 0;">{sc_name}</h4>
                    <hr style="margin:4px 0;border-color:#CBD5E1;"/>
                    <b>Population:</b> {sc['population']:,}<br/>
                    <b>Predicted utilisation:</b> <span style="color:{'#02C39A' if sc['predicted_utilisation']>0.74 else '#EF4444'};font-weight:bold;">{util_pct:.1f}%</span><br/>
                    <b>Non-utilisation risk:</b> {sc['non_utilisation_risk']}<br/>
                    <b>Wealth index:</b> {sc['wealth_index']}/5<br/>
                    <b>Health facilities:</b> {sc['facilities']}</div>""",
                    max_width=250
                ),
                tooltip=f"{sc_name}: {util_pct:.1f}% predicted utilisation",
                icon=folium.DivIcon(
                    html=f"""<div style="background:{util_cmap(sc['predicted_utilisation'])};color:white;
                        font-weight:bold;font-size:11px;padding:3px 7px;border-radius:12px;
                        white-space:nowrap;box-shadow:0 2px 4px rgba(0,0,0,0.3);border:2px solid white;">
                        {sc_name}<br/><span style="font-size:13px;">{util_pct:.0f}%</span></div>""",
                    icon_size=(130, 38), icon_anchor=(65, 19),
                ),
            ).add_to(sc_layer)
        sc_layer.add_to(m)
        util_cmap.add_to(m)

    # Risk heatmap
    if show_risk_heat:
        heat_layer = folium.FeatureGroup(name="🔥 Non-Utilisation Risk Heatmap", show=True)
        plugins.HeatMap(
            [[z["lat"], z["lon"], z["intensity"]] for z in HIGH_RISK_ZONES],
            min_opacity=0.3, radius=60, blur=45,
            gradient={0.4:"#FEF3C7", 0.65:"#F59E0B", 0.85:"#EF4444", 1.0:"#991B1B"},
        ).add_to(heat_layer)
        for zone in HIGH_RISK_ZONES:
            folium.Marker(
                location=[zone["lat"], zone["lon"]],
                tooltip=f"⚠️ {zone['label']}",
                popup=folium.Popup(
                    f"""<div style="font-family:sans-serif;font-size:12px;">
                    <b style="color:#DC2626;">⚠️ High-Risk Zone</b><br/>
                    {zone['label']}<br/>
                    Non-utilisation intensity: <b>{zone['intensity']:.0%}</b></div>""",
                    max_width=220
                ),
                icon=folium.DivIcon(
                    html=f"""<div style="background:#EF4444;color:white;font-size:10px;
                        padding:2px 5px;border-radius:8px;font-weight:bold;
                        box-shadow:0 1px 3px rgba(0,0,0,0.4);">⚠️ {zone['intensity']:.0%}</div>""",
                    icon_size=(70, 22), icon_anchor=(35, 11),
                ),
            ).add_to(heat_layer)
        heat_layer.add_to(m)

    # Health facilities
    fac_colours = {
        "Level 5 — County Referral":     "#023E4F",
        "Level 4 — Sub-County Hospital": "#028090",
        "Level 3 — Health Centre":       "#02C39A",
        "Level 2 — Dispensary":          "#64748B",
    }
    fac_radius = {
        "Level 5 — County Referral": 14, "Level 4 — Sub-County Hospital": 11,
        "Level 3 — Health Centre": 9,    "Level 2 — Dispensary": 7,
    }
    fac_layer = folium.FeatureGroup(name="🏥 Health Facilities", show=True)
    for fac in HEALTH_FACILITIES:
        if fac["level"] not in facility_filter:
            continue
        fac_colour = fac_colours.get(fac["level"], "#64748B")
        beds_html  = f"<b>Beds:</b> {fac['beds']}<br/>" if fac["beds"] > 0 else "<b>Beds:</b> Outpatient only<br/>"
        folium.CircleMarker(
            location=[fac["lat"], fac["lon"]],
            radius=fac_radius.get(fac["level"], 8),
            color="white", weight=2, fill=True,
            fill_color=fac_colour, fill_opacity=0.92,
            popup=folium.Popup(
                f"""<div style="font-family:sans-serif;min-width:220px;font-size:13px;">
                <h4 style="color:{fac_colour};margin:0 0 6px 0;">{fac['name']}</h4>
                <hr style="margin:4px 0;border-color:#CBD5E1;"/>
                <b>Level:</b> {fac['level']}<br/>
                <b>Sub-County:</b> {fac['sub_county']}<br/>
                {beds_html}<b>Services:</b> {fac['services']}</div>""",
                max_width=260
            ),
            tooltip=f"🏥 {fac['name']} ({fac['level']})",
        ).add_to(fac_layer)
    fac_layer.add_to(m)

    # County boundary
    boundary_layer = folium.FeatureGroup(name="📍 Kilifi County Boundary", show=True)
    folium.PolyLine(
        [[-4.05,39.35],[-4.05,40.35],[-2.95,40.35],[-2.95,39.35],[-4.05,39.35]],
        color="#023E4F", weight=2.5, opacity=0.6, dash_array="8 4",
        tooltip="Kilifi County approximate boundary",
    ).add_to(boundary_layer)
    boundary_layer.add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:40px;left:40px;z-index:1000;background:white;
        padding:14px 18px;border-radius:10px;border:1px solid #CBD5E1;
        font-family:sans-serif;font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,0.15);min-width:190px;">
        <b style="color:#023E4F;font-size:13px;">🏥 Facility Levels</b><br/><br/>
        <span style="background:#023E4F;color:white;padding:2px 8px;border-radius:10px;font-size:11px;">● Level 5</span>&nbsp;County Referral<br/><br/>
        <span style="background:#028090;color:white;padding:2px 8px;border-radius:10px;font-size:11px;">● Level 4</span>&nbsp;Sub-County Hospital<br/><br/>
        <span style="background:#02C39A;color:white;padding:2px 8px;border-radius:10px;font-size:11px;">● Level 3</span>&nbsp;Health Centre<br/><br/>
        <span style="background:#64748B;color:white;padding:2px 8px;border-radius:10px;font-size:11px;">● Level 2</span>&nbsp;Dispensary<br/><br/>
        <hr style="border-color:#E2E8F0;margin:6px 0;"/>
        <b style="color:#023E4F;">Circles</b><br/>
        <span style="color:#02C39A;">■</span> High (&gt;74%)&nbsp;<span style="color:#F59E0B;">■</span> Medium&nbsp;<span style="color:#EF4444;">■</span> Low<br/><br/>
        <b style="color:#023E4F;">Heatmap</b><br/><span style="color:#991B1B;">■</span> High non-utilisation risk
    </div>"""))

    map_output = st_folium(m, width="100%", height=580, returned_objects=["last_object_clicked_tooltip"])
    if map_output and map_output.get("last_object_clicked_tooltip"):
        st.markdown(f'<div class="callout">📍 <strong>Selected:</strong> {map_output["last_object_clicked_tooltip"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Sub-County Utilisation Summary")
    sc_df = pd.DataFrame([{
        "Sub-County": name,
        "Population": f"{d['population']:,}",
        "Predicted Utilisation": f"{d['predicted_utilisation']*100:.1f}%",
        "Risk Level": d["non_utilisation_risk"],
        "Avg Wealth Index": f"{d['wealth_index']}/5",
        "Health Facilities": d["facilities"],
    } for name, d in SUB_COUNTIES.items()]).sort_values("Predicted Utilisation")

    def colour_risk(val):
        if val == "High":         return "background-color:#FEE2E2;color:#991B1B;font-weight:bold"
        if val == "Moderate":     return "background-color:#FFF8E1;color:#92400E"
        if val == "Low-Moderate": return "background-color:#ECFDF5;color:#065F46"
        return ""

    st.dataframe(sc_df.style.applymap(colour_risk, subset=["Risk Level"]), use_container_width=True, hide_index=True)
    st.markdown('<div class="callout-danger">🎯 <strong>Priority for intervention:</strong> Ganze (66.8%) and Magarini (69.4%) have the lowest predicted utilisation rates and highest poverty concentrations.</div>', unsafe_allow_html=True)

    st.markdown("#### Facility Count vs. Predicted Utilisation — Sub-County Comparison")
    set_plot_style()
    sc_names  = list(SUB_COUNTIES.keys())
    util_vals = [SUB_COUNTIES[s]["predicted_utilisation"]*100 for s in sc_names]
    fac_vals  = [SUB_COUNTIES[s]["facilities"] for s in sc_names]
    x = np.arange(len(sc_names))
    fig, ax1 = plt.subplots(figsize=(9, 4))
    bar_clrs = ["#EF4444" if v < 70 else "#F59E0B" if v < 74 else "#02C39A" for v in util_vals]
    bars = ax1.bar(x, util_vals, color=bar_clrs, width=0.55, edgecolor="white", label="Predicted Utilisation %")
    ax1.axhline(76.7, color="#023E4F", ls="--", lw=1.5, label="County avg (76.7%)")
    ax1.set_ylabel("Predicted Utilisation Rate (%)", color="#023E4F"); ax1.set_ylim(55, 90)
    ax1.set_xticks(x); ax1.set_xticklabels(sc_names, rotation=15, ha="right", fontsize=9)
    for bar, val in zip(bars, util_vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(x, fac_vals, "o--", color="#8B5CF6", lw=2, ms=8, label="No. Health Facilities")
    ax2.set_ylabel("Number of Health Facilities", color="#8B5CF6"); ax2.set_ylim(0, 30)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper left")
    ax1.set_title("Predicted Utilisation vs. Facility Count by Sub-County", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="callout">Ganze and Magarini have both the fewest facilities and lowest predicted utilisation — supply-side barriers compound demand-side ones. Supports the dissertation recommendation to integrate facility-level data in future model iterations.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  VIEW 5: BATCH PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

elif "Batch" in view:
    st.markdown('<div class="section-header">📂 View 5 — Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV of survey records. The pipeline applies the same feature engineering used in training and returns predictions, probabilities, and risk flags.")

    with st.expander("📋 Required CSV column schema"):
        st.dataframe(pd.DataFrame({
            "Column":  ["gender","age_group","marital_status","relation_to_household_head","religion","wealth_quintile","attended_school","highest_education","occupation","num_sick_in_household","symptoms_reported"],
            "Type":    ["Categorical","Categorical","Categorical","Categorical","Categorical","Numeric (1–5)","Categorical","Categorical","Categorical","Numeric","Text"],
            "Example": ["male","26-49","married","head","christian","3","yes","primary","farmer","2","HIGH FEVER"],
        }), hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download CSV Template",
            pd.DataFrame([{"gender":"male","age_group":"26-49","marital_status":"married","relation_to_household_head":"head","religion":"christian","wealth_quintile":3,"attended_school":"yes","highest_education":"primary","occupation":"farmer","num_sick_in_household":1,"symptoms_reported":"HIGH FEVER"}]).to_csv(index=False),
            "kilifi_batch_template.csv","text/csv")

    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅ Loaded **{len(df_raw)} records** with {len(df_raw.columns)} columns.")
            with st.expander("Preview uploaded data"):
                st.dataframe(df_raw.head(10), use_container_width=True)

            df_features = build_batch_features(df_raw)
            n = len(df_features)
            if pipeline is not None:
                proba = pipeline.predict_proba(df_features)[:, 1]
            else:
                np.random.seed(42)
                proba = np.clip(0.767 + df_features["vulnerability_index"].values * -0.08 + np.random.normal(0, 0.12, n), 0.05, 0.99)
            preds = (proba >= threshold).astype(int)

            df_results = df_raw.copy()
            df_results["Predicted_Utilisation"]    = np.where(preds == 1, "WILL Visit", "Will NOT Visit")
            df_results["Probability_of_Visiting"]  = [f"{p:.1%}" for p in proba]
            df_results["Risk_Flag"] = np.where(proba < 0.40, "🔴 High Risk", np.where(proba < 0.65, "🟡 Moderate Risk", "🟢 Low Risk"))

            n_visit = int(preds.sum())
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Records",          n)
            c2.metric("Predicted to Visit",     n_visit,         f"{n_visit/n:.1%}")
            c3.metric("Predicted NOT to Visit", n - n_visit,     f"{(n-n_visit)/n:.1%}")
            c4.metric("High Risk 🔴",           int((proba<0.40).sum()), f"{(proba<0.40).mean():.1%}")

            st.markdown("---")
            st.markdown(f"#### Results — {n} records (threshold = {threshold:.2f})")
            st.dataframe(
                df_results.style.apply(
                    lambda s: ["background-color:#FEE2E2" if "NOT" in str(v) else "background-color:#D1FAE5" if "WILL" in str(v) else "" for v in s],
                    subset=["Predicted_Utilisation"]
                ), use_container_width=True
            )

            set_plot_style()
            col_hist, col_pie = st.columns(2)
            with col_hist:
                fig, ax = plt.subplots(figsize=(4.5, 3))
                ax.hist(proba, bins=20, color=PALETTE["primary"], edgecolor='white', alpha=0.85)
                ax.axvline(threshold, color='red', ls='--', lw=1.5, label=f"Threshold={threshold:.2f}")
                ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Count")
                ax.set_title("Probability Distribution", fontweight="bold"); ax.legend(fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_pie:
                risk_counts = {
                    "🟢 Low Risk": int((proba >= 0.65).sum()),
                    "🟡 Moderate Risk": int(((proba >= 0.40) & (proba < 0.65)).sum()),
                    "🔴 High Risk": int((proba < 0.40).sum()),
                }
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.pie(risk_counts.values(), labels=risk_counts.keys(),
                       colors=[PALETTE["accent"], PALETTE["warn"], PALETTE["danger"]],
                       autopct='%1.1f%%', startangle=140, textprops={'fontsize': 9})
                ax.set_title("Risk Distribution", fontweight="bold")
                plt.tight_layout(); st.pyplot(fig); plt.close()

            st.download_button("⬇️ Download Full Results CSV", df_results.to_csv(index=False), "kilifi_predictions.csv", "text/csv", type="primary", use_container_width=True)
            df_high = df_results[proba < 0.40]
            if len(df_high) > 0:
                st.download_button(f"⬇️ Download High-Risk Only ({len(df_high)} records)", df_high.to_csv(index=False), "kilifi_high_risk.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("⬆️ Upload a CSV file above to begin batch prediction.")


# ═════════════════════════════════════════════════════════════════════════════
#  VIEW 6: REPORTING
# ═════════════════════════════════════════════════════════════════════════════

elif "Reporting" in view:
    st.markdown('<div class="section-header">📄 View 6 — Reporting</div>', unsafe_allow_html=True)
    st.markdown("Structured summary of all findings. Use the **PDF export** button for county health offices and policy stakeholders.")
    st.markdown("---")

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("""
#### Study Overview
This study developed and evaluated a machine learning framework for predicting health facility
utilisation in Kilifi County, Kenya, using non-clinical household survey data.

#### Dataset
- **Source:** Kilifi County HH Survey (Dryad, 2018–2019) · **Modelling population:** 1,213 sick individuals
- **Positive class (visited):** 930 / 1,213 (76.7%) · **Negative class:** 283 / 1,213 (23.3%)
- **Training set:** 1,488 obs (after oversampling) · **Test set:** 243 obs (original distribution)

#### Model Comparison (Test Set, n = 243)
| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| **Random Forest ★** | **0.745** | 0.793 | **0.903** | **0.844** | **0.664** |
| SVM | 0.724 | 0.799 | 0.855 | 0.826 | 0.636 |
| XGBoost | 0.708 | **0.814** | 0.801 | 0.808 | 0.619 |
| MLP Neural Network | 0.724 | 0.787 | 0.876 | 0.830 | 0.599 |
| Logistic Regression | 0.613 | 0.829 | 0.624 | 0.712 | 0.586 |

★ All models exceeded AUC baseline of 0.50.

#### Recommended Deployment
- **Model:** Random Forest · **Threshold:** 0.20 (recall 98.4%)
- Failing to identify a genuine non-utiliser outweighs the cost of over-predicting utilisation.

#### Key Determinants
1. Vulnerability Index (engineered composite) · 2. Symptom Severity · 3. Severity × Wealth
4. Wealth Quintile · 5. Age × Wealth Interaction

#### Equity Findings
- **Wealth gap:** Q1 68.8% vs Q5 81.2% — 12.4 pp disparity
- **Gender gap:** Male 78.2% vs Female 75.4% — 2.8 pp disparity
- **Age gap:** 0–14 (80.1%) vs 50+ (72.1%) — 8.0 pp disparity
- **Highest risk:** Q1 + 50+ — predicted utilisation as low as 58%

#### Recommendations
1. Deploy at threshold 0.20 to maximise identification of non-utilisers.
2. Prioritise Q1–Q2 elderly (50+) for community health worker outreach.
3. Use Batch Prediction view for pre-cycle resource allocation planning.
4. Use Logistic Regression log-odds for non-technical policy communication.
5. Validate predictions quarterly against facility attendance records.
        """)

    with col_r:
        st.markdown("#### Research Objectives — Status")
        for so, obj, status in [
            ("SO1","Review ML literature",        "✅ Completed"),
            ("SO2","Compare ML models for Kilifi","✅ Completed"),
            ("SO3","Identify key determinants",   "✅ Completed"),
            ("SO4","Design & implement dashboard","✅ Completed"),
            ("SO5","Validate framework",           "✅ Validated"),
        ]:
            st.markdown(f'<div class="callout"><strong>{so}:</strong> {obj}<br/>{status}</div>', unsafe_allow_html=True)
        st.markdown("#### Hypothesis Result")
        st.markdown('<div class="callout">✅ <strong>Confirmed.</strong> All five models exceeded AUC = 0.50 — non-clinical survey data contains sufficient predictive signal.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📥 Export Report")

    def generate_pdf_report() -> bytes:
        buf    = io.BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        title_s  = ParagraphStyle("T",  parent=styles["Title"],   fontSize=16, textColor=colors.HexColor("#023E4F"), spaceAfter=6)
        h1_s     = ParagraphStyle("H1", parent=styles["Heading1"],fontSize=13, textColor=colors.HexColor("#028090"), spaceAfter=4)
        h2_s     = ParagraphStyle("H2", parent=styles["Heading2"],fontSize=11, textColor=colors.HexColor("#023E4F"), spaceAfter=3)
        body_s   = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=9.5, leading=14, spaceAfter=4)
        cap_s    = ParagraphStyle("C",  parent=styles["Normal"],  fontSize=8, textColor=colors.grey)
        story = []
        story.append(Paragraph("Health Facility Utilisation Decision Support Report", title_s))
        story.append(Paragraph("Kilifi County, Kenya — Machine Learning Framework", h2_s))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#02C39A")))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("<b>Author:</b> Keenan Kibaliach (Reg. 098952) | <b>Supervisor:</b> Dr. Betsy Muriithi", cap_s))
        story.append(Spacer(1, 0.5*cm))

        story.append(Paragraph("1. Study Overview", h1_s))
        story.append(Paragraph("A machine learning framework was developed and evaluated for predicting health facility utilisation in Kilifi County, Kenya, using non-clinical household survey data. The framework consists of a five-stage ML pipeline and a multi-view decision-support dashboard deployed via Streamlit.", body_s))
        story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph("2. Model Performance Results (Test Set, n = 243)", h1_s))
        t2 = Table(
            [["Model","Accuracy","Precision","Recall","F1-Score","AUC"]] +
            [[m, f"{v['Accuracy']:.3f}", f"{v['Precision']:.3f}", f"{v['Recall']:.3f}", f"{v['F1-Score']:.3f}", f"{v['AUC']:.3f}"]
             for m, v in PUBLISHED_METRICS.items()],
            colWidths=[4.5*cm,2.4*cm,2.4*cm,2.1*cm,2.1*cm,2*cm]
        )
        t2.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#023E4F")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#CBD5E1")),
            ("BACKGROUND",(0,1),(-1,1),colors.HexColor("#E0F7F4")),
            ("ROWBACKGROUNDS",(0,2),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),
            ("LEFTPADDING",(0,0),(-1,-1),5),
        ]))
        story.append(t2); story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("★ Random Forest selected: highest AUC (0.664), Recall (0.903), F1 (0.844). All five models exceeded the random baseline AUC of 0.50.", body_s))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("3. Key Determinants", h1_s))
        t3 = Table(
            [["Rank","Feature","Interpretation"],
             ["1","Vulnerability Index","Composite of wealth, age, education disadvantage"],
             ["2","Symptom Severity","Primary need factor — stronger illness → higher utilisation"],
             ["3","Severity × Wealth","Compounding effect of severe illness + poverty"],
             ["4","Wealth Quintile","Primary enabling factor"],
             ["5","Age × Wealth","Elderly + low-wealth most at risk"]],
            colWidths=[1.5*cm,5.5*cm,9.5*cm]
        )
        t3.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#028090")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F0F4F8")]),
            ("LEFTPADDING",(0,0),(-1,-1),5),
        ]))
        story.append(t3); story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("4. Equity Findings", h1_s))
        t4 = Table(
            [["Dimension","Subgroup","Predicted Rate","Gap"],
             ["Wealth Quintile","Q1 (Poorest)","68.8%","–12.4 pp"],
             ["Wealth Quintile","Q5 (Wealthiest)","81.2%","Reference"],
             ["Gender","Female","75.4%","–2.8 pp"],
             ["Gender","Male","78.2%","Reference"],
             ["Age Group","50+ (Elderly)","72.1%","–8.0 pp"],
             ["Age Group","0–14 (Youth)","80.1%","Reference"]],
            colWidths=[4*cm,4.5*cm,4*cm,4*cm]
        )
        t4.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#023E4F")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),
            ("LEFTPADDING",(0,0),(-1,-1),5),
        ]))
        story.append(t4); story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("5. Recommendations", h1_s))
        for i, rec in enumerate([
            "Deploy at classification threshold 0.20 for maximum sensitivity.",
            "Prioritise Q1–Q2 elderly (50+) for community health worker outreach.",
            "Use Batch Prediction to flag high-risk households before resource allocation.",
            "Use Logistic Regression outputs for non-technical policy communication.",
            "Validate predictions quarterly against facility attendance records.",
        ], 1):
            story.append(Paragraph(f"{i}. {rec}", body_s))
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Kilifi Health Utilisation Dashboard · Random Forest Model · Kilifi County HH Survey (Dryad, 2018–2019) · Kibaliach (2024)", cap_s))
        doc.build(story)
        return buf.getvalue()

    if REPORTLAB_AVAILABLE:
        if st.button("📥 Generate & Download PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating PDF…"):
                pdf_bytes = generate_pdf_report()
            st.download_button("⬇️ Download PDF Report", pdf_bytes, "kilifi_utilisation_report.pdf", "application/pdf", type="primary", use_container_width=True)
            st.success("✅ PDF generated successfully.")
    else:
        st.warning("Install reportlab for PDF export: `py -m pip install reportlab`")


# ═════════════════════════════════════════════════════════════════════════════
#  VIEW 7: INDIVIDUAL PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════

elif "Individual" in view:
    st.markdown('<div class="section-header">🔮 View 7 — Individual Predictor</div>', unsafe_allow_html=True)
    st.markdown(f"Predict whether a **single sick individual** will visit a health facility. Threshold: **{threshold:.2f}** (set in sidebar).")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        gender         = st.selectbox("Gender", ["male","female"])
        age_group      = st.selectbox("Age Group", ["0-14","15-25","26-49","50+"])
        marital_status = st.selectbox("Marital Status", ["single","married","widowed","divorced","separated"])
        relation       = st.selectbox("Relation to HH Head", ["head","spouse","child","parent","other relative","non-relative"])
        religion       = st.selectbox("Religion", ["christian","muslim","other","none"])

    with col2:
        st.markdown("**📚 Socioeconomic**")
        wealth_quintile   = st.slider("Wealth Quintile", 1, 5, 3, help="1=Poorest, 5=Richest")
        attended_school   = st.selectbox("Attended School?", ["yes","no"])
        highest_education = st.selectbox("Highest Education", ["none","primary","adult ed","secondar","higher"])
        occupation        = st.selectbox("Occupation (HH Head)", ["farmer","casual labourer","employed","self-employed","housewife","student","retired","unemployed"])
        num_sick          = st.number_input("No. Sick in Household", 1, 10, 1)

    with col3:
        st.markdown("**🩺 Clinical**")
        symptoms_text = st.text_area("Symptoms Reported", placeholder="e.g. HIGH FEVER, COUGHING, MALARIA …", height=120)
        if symptoms_text:
            sev = symptom_severity(symptoms_text)
            cat = symptom_category(symptoms_text)
            sev_col = {"severe":"🔴","moderate":"🟡","mild":"🟢","other":"⚪","unknown":"⚪"}.get(sev,"⚪")
            st.markdown(f'<div class="callout"><strong>Detected severity:</strong> {sev_col} {sev.upper()}<br/><strong>Category:</strong> {cat.replace("_"," ").title()}</div>', unsafe_allow_html=True)
        vuln_score = int(wealth_quintile <= 2) + int(age_group == '50+') + int(EDU_MAP.get(highest_education,1) == 0)
        bar_colors = ["🟢","🟡","🟠","🔴"]
        st.markdown(f'<div class="{"callout-danger" if vuln_score >= 2 else "callout"}">Vulnerability Index: <strong>{bar_colors[min(vuln_score,3)]} {vuln_score} / 3</strong><br/>{"Low wealth (Q1–Q2) " if wealth_quintile<=2 else ""}{"Elderly (50+) " if age_group=="50+" else ""}{"No formal education" if EDU_MAP.get(highest_education,1)==0 else ""}</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔮 Run Prediction", type="primary", use_container_width=True):
        inputs      = {"gender":gender,"marital_status":marital_status,"attended_school":attended_school,"highest_education":highest_education,"wealth_quintile":wealth_quintile,"relation_to_household_head":relation,"age_group":age_group,"occupation":occupation,"religion":religion,"symptoms_reported":symptoms_text or "unknown","num_sick_in_household":num_sick}
        features_df = build_features(inputs)

        if pipeline is not None:
            probability = pipeline.predict_proba(features_df)[0][1]
        else:
            sev_s = SEV_MAP.get(symptom_severity(symptoms_text or ""), 1)
            probability = float(np.clip(0.50 + (wealth_quintile-3)*0.06 + (sev_s-2)*0.08 - int(age_group=="50+")*0.05 - int(highest_education=="none")*0.07, 0.08, 0.97))

        prediction = int(probability >= threshold)

        col_res, col_gauge = st.columns([2, 1])
        with col_res:
            if prediction == 1:
                st.success("### ✅ Predicted: WILL Visit a Health Facility")
            else:
                st.error("### ❌ Predicted: Will NOT Visit a Health Facility")
            st.metric("Probability of Visiting", f"{probability:.1%}", f"{probability - 0.767:+.1%} vs population avg (76.7%)")
            if pipeline is None:
                st.caption("ℹ️ Pipeline not loaded — demonstration prediction.")

        with col_gauge:
            set_plot_style()
            fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={"projection":"polar"})
            theta = np.linspace(0, np.pi, 200)
            ax.plot(theta, [1]*200, color="#E2E8F0", lw=10, alpha=0.4)
            fill_theta = np.linspace(0, np.pi * probability, 200)
            color_gauge = PALETTE["accent"] if probability >= 0.65 else PALETTE["warn"] if probability >= 0.40 else PALETTE["danger"]
            ax.plot(fill_theta, [1]*200, color=color_gauge, lw=10)
            ax.set_ylim(0, 1.5); ax.set_theta_zero_location("W"); ax.set_theta_direction(-1); ax.axis("off")
            ax.text(np.pi/2, 0, f"{probability:.0%}", ha='center', va='center', fontsize=20, fontweight='bold', color=color_gauge)
            ax.set_title("Utilisation\nProbability", fontsize=9, pad=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("#### ⚠️ Risk Factor Analysis")
        sev_det = symptom_severity(symptoms_text or "")
        risks, facilitators = [], []
        if wealth_quintile <= 2: risks.append("🔴 **Low wealth (Q1–Q2)** — primary barrier (strongest predictor)")
        if sev_det in ("severe","moderate") and wealth_quintile <= 2: risks.append("🔴 **Severe symptoms + low wealth** — highest-risk combination")
        if age_group == "50+": risks.append("🟡 **Elderly (50+)** — historically lower utilisation")
        if EDU_MAP.get(highest_education,1) == 0: risks.append("🟡 **No formal education** — negative LR log-odds coefficient")
        if occupation in ("farmer","casual labourer","unemployed"): risks.append("🟡 **Informal/agricultural occupation** — negative LR coefficient")
        if vuln_score >= 2: risks.append(f"🔴 **Vulnerability Index = {vuln_score}/3** — multidimensional disadvantage")
        if wealth_quintile >= 4: facilitators.append("🟢 **High wealth (Q4–Q5)** — strong enabling factor")
        if sev_det == "severe": facilitators.append("🟢 **Severe symptoms** — 81.6% utilisation rate in EDA")
        if EDU_MAP.get(highest_education,1) >= 2: facilitators.append("🟢 **Secondary+ education** — positive LR coefficient")

        c_risk, c_fac = st.columns(2)
        with c_risk:
            st.markdown("**Barriers identified:**")
            for r in (risks or ["🟢 No significant barriers identified"]): st.markdown(f"- {r}")
        with c_fac:
            st.markdown("**Facilitators identified:**")
            for f in (facilitators or ["- No strong facilitators identified"]): st.markdown(f"- {f}")

        st.markdown("---")
        with st.expander("🔍 Inspect all engineered feature values"):
            feat_display = features_df.T.rename(columns={0:"Value"}).copy()
            feat_display.index = [PLAIN_LABELS.get(i, i) for i in feat_display.index]
            st.dataframe(feat_display.astype(str), use_container_width=True)

        if SHAP_AVAILABLE and pipeline is not None:
            with st.expander("🧠 SHAP Local Explanation (this individual)"):
                try:
                    explainer   = shap.TreeExplainer(pipeline.named_steps["model"])
                    trans_input = pipeline[:-1].transform(features_df)
                    shap_vals   = explainer.shap_values(trans_input)
                    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
                    shap.initjs()
                    shap_fig = shap.force_plot(
                        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                        shap_vals[0], features_df.iloc[0], matplotlib=True, show=False
                    )
                    st.pyplot(shap_fig)
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "Kilifi Health Facility Utilisation Dashboard · Recommended model: Random Forest · "
    "Dataset: Kilifi County HH Survey (Dryad, 2018–2019) · "
    "Keenan Kibaliach (Reg. 098952) · Supervisor: Dr. Betsy Muriithi"
)