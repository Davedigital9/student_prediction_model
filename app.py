import streamlit as st
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

# ---------------------------
# Load models
# ---------------------------
if not os.path.exists("early_model.pkl"):
    st.error("Model files not found. Please upload .pkl files.")
    st.stop()

model_early = joblib.load("early_model.pkl")
model_mid = joblib.load("mid_model.pkl")
model_late = joblib.load("late_model.pkl")

# ---------------------------
# App Title & Description
# ---------------------------
st.title("🎓 Student Performance Predictor")

st.write("Predict your likelihood of passing based on your academic progress.")

st.markdown("""
### 📘 About This Tool

This system predicts whether a student is likely to pass or fail a module based on academic behaviour, engagement, and performance over time.

The model uses machine learning trained on real student data to identify patterns linked to academic success. It considers factors such as study time, attendance, past academic performance, and support systems.

A key feature of this system is its ability to adapt to different stages of the semester. Students can enter their assessment scores and corresponding weightings, allowing the system to calculate a running weighted grade.

### 🎯 How This Helps You

- Identify risk of failing early  
- Track your academic progress over time  
- Understand how your performance impacts outcomes  
- Take action before it’s too late  
""")

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
studytime = st.slider(
    "Study Time per Week (1 = <2hrs, 2 = 2–5hrs, 3 = 5–10hrs, 4 = >10hrs)", 
    1, 4
)

failures = st.slider(
    "Number of Past Academic Failures (historical, not current module)", 
    0, 5
)

absences = st.number_input(
    "Number of Classes Missed This Semester (Max 13)", 
    0, 13
)

schoolsup = st.selectbox("Do you receive academic support from your institution?", ["No", "Yes"])
famsup = st.selectbox("Do you receive family support for your studies?", ["No", "Yes"])
internet = st.selectbox(
    "Do you have reliable internet access outside the University?", 
    ["No", "Yes"]
)

# Convert categorical inputs
schoolsup = 1 if schoolsup == "Yes" else 0
famsup = 1 if famsup == "Yes" else 0
internet = 1 if internet == "Yes" else 0

# ---------------------------
# Weighted Grade Function
# ---------------------------
def calculate_weighted_grade(scores, weights):
    total_weight = sum(weights)
    weighted_sum = sum([s * w for s, w in zip(scores, weights)])
    
    if total_weight == 0:
        return 0, 0
    
    current_grade = weighted_sum / total_weight
    return current_grade, total_weight

# ---------------------------
# Assessment Input
# ---------------------------
G1 = 0
G2 = 0
current_grade = 0
total_weight = 0

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

    current_grade, total_weight = calculate_weighted_grade(scores, weights)

    st.success(f"📈 Current Weighted Grade: {round(current_grade, 2)}%")

    # ---------------------------
    # Pass Requirement Calculator
    # ---------------------------
    remaining_weight = 100 - total_weight

    if remaining_weight > 0:
        required_score = (50 - (current_grade * (total_weight / 100))) / (remaining_weight / 100)

        st.info(f"📌 To PASS the module, you need an average of {round(required_score, 2)}% in the remaining assessments.")

        if required_score > 100:
            st.error("⚠️ It is mathematically impossible to pass based on current performance.")
        elif required_score < 0:
            st.success("🎉 You are already guaranteed to pass!")

    # Assign grades to model inputs
    if stage == "Mid (Some Assessments)":
        G1 = current_grade

    elif stage == "Late (Most Assessments)":
        G1 = current_grade
        G2 = current_grade  # simplified

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Outcome"):

    st.session_state.results = {}
    st.session_state.probabilities = {}

    # Early prediction
    early_features = np.array([[studytime, failures, absences, schoolsup, famsup, internet]])
    early_pred = model_early.predict(early_features)[0]
    early_prob = model_early.predict_proba(early_features)[0][1]
    
    st.session_state.results["Early Stage"] = early_pred
    st.session_state.probabilities["Early Stage"] = early_prob

    # Mid prediction
    if G1 > 0:
        mid_features = np.array([[studytime, failures, absences, schoolsup, famsup, internet, G1]])
        mid_pred = model_mid.predict(mid_features)[0]
        mid_prob = model_mid.predict_proba(mid_features)[0][1]
        
        st.session_state.results["Mid Stage"] = mid_pred
        st.session_state.probabilities["Mid Stage"] = mid_prob

    # Late prediction
    if G2 > 0:
        late_features = np.array([[studytime, failures, absences, schoolsup, famsup, internet, G1, G2]])
        late_pred = model_late.predict(late_features)[0]
        late_prob = model_late.predict_proba(late_features)[0][1]
        
        st.session_state.results["Late Stage"] = late_pred
        st.session_state.probabilities["Late Stage"] = late_prob

    # ---------------------------
    # Display Results
    # ---------------------------
    st.subheader("📊 Prediction Progression")

    for stage_name in st.session_state.results:
        pred = st.session_state.results[stage_name]
        prob = st.session_state.probabilities[stage_name]

        if pred ==1:
            st.success(f"{stage_name}: PASS ✅ ({round(prob*100, 2)}% confidence)")
        else:
            st.error(f"{stage_name}: FAIL ⚠️ ({round((1-prob)*100, 2)}% risk)")

    # ---------------------------
    # Progression Insight
    # ---------------------------
    if "Early Stage" in st.session_state.results and "Late Stage" in st.session_state.results:
        if results["Early Stage"] == 0 and results["Late Stage"] == 1:
            st.success("🎉 Improvement detected! You moved from risk to passing.")
        elif results["Early Stage"] == 1 and results["Late Stage"] == 0:
            st.warning("⚠️ Performance dropped over time. Immediate action is recommended.")

    # ---------------------------
    # Final Result (Latest Stage)
    # ---------------------------
    latest_stage = list(st.session_state.results.keys())[-1]
    latest_prediction = st.session_state.results[latest_stage]
    latest_prob = st.session_state.probabilities[latest_stage]  

    st.subheader("🎯 Final Prediction")

    if latest_prediction == 1:
        st.success(f"✅ You are likely to PASS ({round(latest_prob*100,2)}% confidence)")
    else:
        st.error("⚠️ You are at risk of FAILING ({round((1-latest_prob)*100,2)}% risk)")

        st.subheader("💡 Suggestions:")
        st.write("- Increase study time")
        st.write("- Reduce absences")
        st.write("- Seek academic support")

# ---------------------------
# Visual Dashboard
# ---------------------------
if "results" in st.session_state:

    st.subheader("📈 Performance Analytics")
    st.write("### Grade & Prediction Trend")

    stage_labels = list(st.session_state.results.keys())
    pass_probs = [st.session_state.probabilities[s] for s in stage_labels]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(stage_labels, pass_probs, marker='o')
    ax.set_title("Prediction Confidence Over Time")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Probability of Passing")

    st.pyplot(fig)
