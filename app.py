import streamlit as st
import joblib
import numpy as np

# Load models
model_early = joblib.load("early_model.pkl")
model_mid = joblib.load("mid_model.pkl")
model_late = joblib.load("late_model.pkl")

st.title("🎓 Student Performance Predictor")

st.write("Predict your likelihood of passing based on your academic progress.")

# ---------------------------
# Stage Selection
# ---------------------------
stage = st.selectbox(
    "Select Academic Stage",
    ["Early (No Assessments)", "Mid (Some Assessments)", "Late (Most Assessments)"]
)

# ---------------------------
# Common Inputs
# ---------------------------
studytime = st.slider("Study Time (1-4)", 1, 4)
failures = st.slider("Past Failures", 0, 5)
absences = st.number_input("Absences", 0, 100)

schoolsup = st.selectbox("School Support", ["No", "Yes"])
famsup = st.selectbox("Family Support", ["No", "Yes"])
internet = st.selectbox("Internet Access", ["No", "Yes"])

# Convert categorical
schoolsup = 1 if schoolsup == "Yes" else 0
famsup = 1 if famsup == "Yes" else 0
internet = 1 if internet == "Yes" else 0

# ---------------------------
# Weighted Grade Function
# ---------------------------
def calculate_weighted_grade(scores, weights):
    total_weight = sum(weights)
    if total_weight == 0:
        return 0
    weighted_sum = sum([s * w for s, w in zip(scores, weights)])
    return weighted_sum / total_weight

# ---------------------------
# Assessment Input (Dynamic)
# ---------------------------
G1 = 0
G2 = 0

if stage != "Early (No Assessments)":
    st.subheader("📊 Enter Assessment Details")

    num_assessments = st.number_input("Number of assessments completed", 1, 10)

    scores = []
    weights = []

    for i in range(int(num_assessments)):
        score = st.number_input(f"Score for Assessment {i+1}", 0.0, 100.0, key=f"s{i}")
        weight = st.number_input(f"Weight (%) for Assessment {i+1}", 0.0, 100.0, key=f"w{i}")

        scores.append(score)
        weights.append(weight)

    current_grade = calculate_weighted_grade(scores, weights)

    st.success(f"📈 Current Weighted Grade: {round(current_grade, 2)}%")

    if stage == "Mid (Some Assessments)":
        G1 = current_grade

    elif stage == "Late (Most Assessments)":
        G1 = current_grade
        G2 = current_grade  # simplified (can improve later)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Outcome"):

    if stage == "Early (No Assessments)":
        features = np.array([[studytime, failures, absences, schoolsup, famsup, internet]])
        prediction = model_early.predict(features)[0]

    elif stage == "Mid (Some Assessments)":
        features = np.array([[studytime, failures, absences, schoolsup, famsup, internet, G1]])
        prediction = model_mid.predict(features)[0]

    else:
        features = np.array([[studytime, failures, absences, schoolsup, famsup, internet, G1, G2]])
        prediction = model_late.predict(features)[0]

    # ---------------------------
    # Output Result
    # ---------------------------
    if prediction == 1:
        st.success("✅ You are likely to PASS!")
    else:
        st.error("⚠️ You are at risk of FAILING.")

        st.subheader("💡 Suggestions:")
        st.write("- Increase study time")
        st.write("- Reduce absences")
        st.write("- Seek academic support")
