import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Health Facility Utilization Predictor",
    page_icon="🏥",
    layout="centered"
)

# ── Load pipeline ──────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("health_facility_pipeline.joblib")

pipeline = load_model()

# ── Feature engineering helpers (same as notebook) ─────────
severe   = ['MALARIA','CHEST PAIN','ASTHMA','DIARRHOEA','VOMITING','HIGH FEVER','HIGH TEMP']
moderate = ['FEVER','COUGHING','STOMACHACHE','BODY PAIN','FLU','WOUND']
mild_syms = ['COLDS','HEADACHE','TONSILS','TOOTHACHE','BACK PAIN']

def symptom_severity(s):
    if pd.isna(s): return 'unknown'
    s = str(s).upper()
    if any(x in s for x in severe):    return 'severe'
    if any(x in s for x in moderate):  return 'moderate'
    if any(x in s for x in mild_syms): return 'mild'
    return 'other'

def symptom_category(s):
    if pd.isna(s): return 'unknown'
    s = str(s).upper()
    if 'FEVER' in s or 'MALARIA' in s or 'TEMP' in s:      return 'fever_or_malaria'
    if 'COUGH' in s or 'ASTHMA' in s or 'CHEST' in s:      return 'respiratory'
    if 'DIARRHOEA' in s or 'STOMACH' in s or 'VOMIT' in s: return 'gastrointestinal'
    if 'HEAD' in s:                                          return 'headache'
    if 'BODY PAIN' in s or 'JOINT' in s or 'BACK' in s:    return 'body_pain'
    if 'WOUND' in s or 'INJURY' in s:                       return 'injury'
    if 'EYE' in s or 'EAR' in s or 'TOOTH' in s:           return 'ent_or_dental'
    if 'FLU' in s or 'COLD' in s:                           return 'flu_or_cold'
    if 'DIABETES' in s or 'PRESSURE' in s or 'HEART' in s: return 'chronic_disease'
    return 'other'

age_map = {'0-14': 0, '15-25': 1, '26-49': 2, '50+': 3}
edu_map = {'none': 0, 'primary': 1, 'adult ed': 1,
           'secondar': 2, 'higher': 3, "don't kn": 1, 'other': 1}
sev_map = {'severe': 3, 'moderate': 2, 'mild': 1, 'other': 1, 'unknown': 1}

NUM_COLS = [
    'wealth_quintile', 'age_numeric', 'education_score',
    'age_x_wealth', 'education_x_wealth', 'severity_x_wealth',
    'vulnerability_index', 'num_sick_in_household'
]
CAT_COLS = [
    'gender', 'marital_status', 'attended_school', 'highest_education',
    'relation_to_household_head', 'age_group', 'occupation', 'religion',
    'symptom_severity', 'symptom_category'
]

def build_features(inputs: dict) -> pd.DataFrame:
    """Convert raw form inputs into the engineered feature DataFrame."""
    row = inputs.copy()

    # Symptom-derived
    symptoms_text = row.get('symptoms_reported', '')
    row['symptom_severity'] = symptom_severity(symptoms_text)
    row['symptom_category'] = symptom_category(symptoms_text)

    # Ordinal encodings
    row['age_numeric']     = age_map.get(row['age_group'], 1)
    row['education_score'] = edu_map.get(row['highest_education'], 1)

    # Interaction features
    row['age_x_wealth']       = row['age_numeric']     * row['wealth_quintile']
    row['education_x_wealth'] = row['education_score'] * row['wealth_quintile']
    row['severity_x_wealth']  = sev_map.get(row['symptom_severity'], 1) * row['wealth_quintile']

    # Composite flags
    row['vulnerability_index'] = (
        int(row['wealth_quintile'] <= 2) +
        int(row['age_group'] == '50+') +
        int(row['education_score'] == 0)
    )

    df = pd.DataFrame([row])
    return df[NUM_COLS + CAT_COLS]

# ── UI ──────────────────────────────────────────────────────
st.title("🏥 Health Facility Utilization Predictor")
st.markdown(
    "**Kilifi County, Kenya** — Predicts whether a sick individual will visit "
    "a health facility, based on sociodemographic and clinical features."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographics")
    gender = st.selectbox("Gender", ["male", "female"])
    age_group = st.selectbox("Age Group", ["0-14", "15-25", "26-49", "50+"])
    marital_status = st.selectbox(
        "Marital Status", ["single", "married", "widowed", "divorced", "separated"]
    )
    relation = st.selectbox(
        "Relation to Household Head",
        ["head", "spouse", "child", "parent", "other relative", "non-relative"]
    )
    religion = st.selectbox("Religion", ["christian", "muslim", "other", "none"])

with col2:
    st.subheader("📚 Socioeconomic")
    wealth_quintile = st.slider("Wealth Quintile", 1, 5, 3,
                                 help="1 = Poorest, 5 = Richest")
    attended_school = st.selectbox("Attended School?", ["yes", "no"])
    highest_education = st.selectbox(
        "Highest Education", ["none", "primary", "adult ed", "secondar", "higher"]
    )
    occupation = st.selectbox(
        "Occupation (Household Head)",
        ["farmer", "casual labourer", "employed", "self-employed",
         "housewife", "student", "retired", "unemployed"]
    )
    num_sick = st.number_input("Number of Sick in Household", 1, 10, 1)

st.subheader("🩺 Clinical")
symptoms_text = st.text_input(
    "Symptoms Reported (free text)",
    placeholder="e.g. HIGH FEVER, COUGHING, MALARIA …",
    help="Enter symptoms as described. The model will classify severity automatically."
)

if symptoms_text:
    sev  = symptom_severity(symptoms_text)
    cat  = symptom_category(symptoms_text)
    sev_colors = {'severe': '🔴', 'moderate': '🟡', 'mild': '🟢',
                  'other': '⚪', 'unknown': '⚪'}
    st.info(f"Detected severity: **{sev_colors[sev]} {sev.upper()}** | Category: **{cat}**")

st.divider()

if st.button("🔮 Predict", type="primary", use_container_width=True):
    inputs = {
        'gender':                    gender,
        'marital_status':            marital_status,
        'attended_school':           attended_school,
        'highest_education':         highest_education,
        'wealth_quintile':           wealth_quintile,
        'relation_to_household_head': relation,
        'age_group':                 age_group,
        'occupation':                occupation,
        'religion':                  religion,
        'symptoms_reported':         symptoms_text or 'unknown',
        'num_sick_in_household':     num_sick,
    }

    features_df = build_features(inputs)
    prediction  = pipeline.predict(features_df)[0]
    probability = pipeline.predict_proba(features_df)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"### ✅ Predicted: WILL Visit a Health Facility")
    else:
        st.error(f"### ❌ Predicted: Will NOT Visit a Health Facility")

    st.metric(
        label="Probability of Visiting",
        value=f"{probability:.1%}",
        delta=f"{probability - 0.5:+.1%} vs 50% baseline"
    )

    # Risk factors
    age_num = age_map.get(age_group, 1)
    edu     = edu_map.get(highest_education, 1)
    vuln    = int(wealth_quintile <= 2) + int(age_group == '50+') + int(edu == 0)

    st.subheader("⚠️ Risk Factor Summary")
    risks = []
    if wealth_quintile <= 2:
        risks.append("🔴 Low wealth quintile — strongest barrier to care-seeking")
    if age_group == '50+':
        risks.append("🟡 Elderly age group — lower utilization historically")
    if edu == 0:
        risks.append("🟡 No formal education — associated with lower health-seeking")
    if symptom_severity(symptoms_text or '') in ('severe', 'moderate') and wealth_quintile <= 2:
        risks.append("🔴 Severe symptoms + low wealth — highest risk combination")
    if not risks:
        risks.append("🟢 No major risk factors identified")

    for r in risks:
        st.markdown(f"- {r}")

    with st.expander("🔍 View engineered feature values"):
        st.dataframe(features_df.T.rename(columns={0: 'Value'}))

st.divider()
st.caption(
    "Model: XGBoost · Dataset: Kilifi County Household Survey (Dryad, 2018–2019) · "
    "AUC ≈ 0.65"
)