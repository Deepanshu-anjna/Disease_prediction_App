import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>
.big-title {
    font-size: 50px;
    font-weight: 800;
    text-align: center;
    color: white;
}
.sub-title {
    font-size: 18px;
    text-align: center;
    color: #b0b0b0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------
# SAFE CSV LOADER
# ---------------------------
def safe_read_csv(file):
    """Tries multiple encodings to avoid UnicodeDecodeError"""
    try:
        return pd.read_csv(file)
    except:
        try:
            return pd.read_csv(file, encoding="latin1")
        except:
            return pd.read_csv(file, encoding="cp1252")


# ---------------------------
# LOAD DATASETS
# ---------------------------
@st.cache_data
def load_data():
    df_main = safe_read_csv("Original_Dataset.csv")
    df_desc = safe_read_csv("Disease_Description.csv")
    df_weights = safe_read_csv("Symptom_Weights.csv")
    df_doctor = safe_read_csv("Doctor_Versus_Disease.csv")
    return df_main, df_desc, df_weights, df_doctor


df_main, df_desc, df_weights, df_doctor = load_data()


# ---------------------------
# CLEAN COLUMN NAMES
# ---------------------------
df_main.columns = [c.strip().lower() for c in df_main.columns]
df_desc.columns = [c.strip().lower() for c in df_desc.columns]
df_weights.columns = [c.strip().lower() for c in df_weights.columns]
df_doctor.columns = [c.strip().lower() for c in df_doctor.columns]


# ---------------------------
# AUTO DETECT DISEASE COLUMN
# ---------------------------
possible_disease_cols = ["disease", "prognosis"]
disease_col = None

for col in possible_disease_cols:
    if col in df_main.columns:
        disease_col = col
        break

if disease_col is None:
    disease_col = df_main.columns[0]  # fallback


# ---------------------------
# AUTO DETECT SYMPTOM COLUMNS
# ---------------------------
symptom_cols = [col for col in df_main.columns if "symptom" in col]

if len(symptom_cols) == 0:
    st.error("❌ Symptom columns not detected.")
    st.write("Columns found:", df_main.columns)
    st.stop()


# ---------------------------
# CLEAN MAIN DATASET
# ---------------------------
df_main = df_main.fillna("")
df_main[disease_col] = df_main[disease_col].astype(str).str.strip().str.title()

df_main["symptom_list"] = df_main[symptom_cols].values.tolist()
df_main["symptom_list"] = df_main["symptom_list"].apply(
    lambda x: [str(i).strip().lower() for i in x if str(i).strip() != "" and str(i).strip().lower() != "nan"]
)

df_main = df_main[df_main["symptom_list"].apply(len) > 0]


# ---------------------------
# UNIQUE SYMPTOMS
# ---------------------------
all_symptoms = sorted(set([sym for row in df_main["symptom_list"] for sym in row]))


# ---------------------------
# ENCODE SYMPTOMS
# ---------------------------
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_main["symptom_list"])
y = df_main[disease_col]


# ---------------------------
# TRAIN MODEL
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# ---------------------------
# DESCRIPTION DICTIONARY
# ---------------------------
desc_dict = {}

if "disease" in df_desc.columns and "description" in df_desc.columns:
    df_desc["disease"] = df_desc["disease"].astype(str).str.strip().str.title()
    df_desc["description"] = df_desc["description"].astype(str).str.strip()
    desc_dict = dict(zip(df_desc["disease"], df_desc["description"]))


# ---------------------------
# DOCTOR SPECIALIST DICTIONARY
# ---------------------------
# Doctor_Versus_Disease.csv usually contains:
# Disease, Specialist
doctor_dict = {}

# auto detect columns
doctor_disease_col = None
doctor_specialist_col = None

for col in df_doctor.columns:
    if "disease" in col:
        doctor_disease_col = col
    if "specialist" in col or "doctor" in col:
        doctor_specialist_col = col

# fallback if not found
if doctor_disease_col is None:
    doctor_disease_col = df_doctor.columns[0]
if doctor_specialist_col is None:
    doctor_specialist_col = df_doctor.columns[1]

df_doctor[doctor_disease_col] = df_doctor[doctor_disease_col].astype(str).str.strip().str.title()
df_doctor[doctor_specialist_col] = df_doctor[doctor_specialist_col].astype(str).str.strip().str.title()

doctor_dict = dict(zip(df_doctor[doctor_disease_col], df_doctor[doctor_specialist_col]))


# ---------------------------
# SYMPTOM WEIGHT DICTIONARY
# ---------------------------
weight_dict = {}

if df_weights.shape[1] >= 2:
    w_sym_col = df_weights.columns[0]
    w_weight_col = df_weights.columns[1]

    df_weights[w_sym_col] = df_weights[w_sym_col].astype(str).str.strip().str.lower()
    df_weights[w_weight_col] = pd.to_numeric(df_weights[w_weight_col], errors="coerce").fillna(0)

    weight_dict = dict(zip(df_weights[w_sym_col], df_weights[w_weight_col]))


# ---------------------------
# SEVERITY FUNCTION
# ---------------------------
def calculate_severity(symptoms):
    raw_score = 0
    for sym in symptoms:
        raw_score += weight_dict.get(sym.lower(), 5)

    max_possible = sum(sorted(weight_dict.values(), reverse=True)[:10]) if len(weight_dict) > 0 else 1
    normalized_score = (raw_score / max_possible) * 100
    normalized_score = min(normalized_score, 100)

    if normalized_score < 30:
        risk = "🟢 Low Risk (Monitor at Home)"
    elif normalized_score < 70:
        risk = "🟠 Moderate Risk (Consult Doctor)"
    else:
        risk = "🔴 High Risk (Seek Immediate Medical Attention)"

    return raw_score, round(normalized_score, 2), risk


# ---------------------------
# TOP 3 PREDICTIONS
# ---------------------------
def predict_top3(symptoms):
    input_vector = mlb.transform([symptoms])
    probs = model.predict_proba(input_vector)[0]

    top3_idx = np.argsort(probs)[-3:][::-1]

    results = []
    for idx in top3_idx:
        disease_name = model.classes_[idx]
        probability = probs[idx] * 100
        results.append((disease_name, round(probability, 2)))

    return results


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.markdown("<div class='big-title'>🩺 AI Disease Prediction & Specialist Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Select symptoms to predict disease, description, specialist, and risk level</div>", unsafe_allow_html=True)
st.write("")


# ---------------------------
# SIDEBAR INFO
# ---------------------------
st.sidebar.title("⚙️ Model Info")
st.sidebar.success(f"Model Accuracy: {accuracy*100:.2f}%")
st.sidebar.write("Model: Multinomial Naive Bayes")
st.sidebar.write(f"Detected Disease Column: {disease_col}")
st.sidebar.write(f"Detected Symptom Columns: {len(symptom_cols)}")
st.sidebar.write(f"Unique Symptoms: {len(all_symptoms)}")


# ---------------------------
# USER INPUT
# ---------------------------
selected_symptoms = st.multiselect("✅ Select Symptoms:", all_symptoms)

if st.button("🔍 Predict Disease"):
    if len(selected_symptoms) < 2:
        st.warning("⚠ Please select at least 2 symptoms for better prediction.")
    else:
        top3 = predict_top3(selected_symptoms)
        best_disease = top3[0][0]

        description = desc_dict.get(best_disease, "Description not available.")
        specialist = doctor_dict.get(best_disease, "Specialist not available.")

        raw_score, normalized_score, risk = calculate_severity(selected_symptoms)

        # ---------------------------
        # DISPLAY RESULTS
        # ---------------------------
        st.markdown("## 📌 Top 3 Predicted Diseases")

        for i, (disease, prob) in enumerate(top3, start=1):
            st.markdown(f"### {i}. **{disease}** — {prob}%")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("## 📖 Disease Description")
            st.info(description)

        with col2:
            st.markdown("## 👨‍⚕ Recommended Specialist")
            st.success(specialist)

        st.markdown("---")

        st.markdown("## 🚨 Risk Level Analysis")
        st.metric("Raw Severity Score", raw_score)
        st.metric("Normalized Severity (0-100)", normalized_score)

        st.progress(int(normalized_score))

        if "Low" in risk:
            st.success(risk)
        elif "Moderate" in risk:
            st.warning(risk)
        else:
            st.error(risk)

        st.markdown("---")
        st.caption("⚠ Disclaimer: This tool is for educational/demo purposes only and is not a replacement for professional medical advice.")
