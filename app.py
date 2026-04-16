import streamlit as st
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

#Setting the page and title
st.set_page_config(
    page_title="Student Performance",
    page_icon="🎓"
)
#-------------------------
# Defined Functions
#-------------------------
# Load Models Safely
def load_models():
    try:
        model_early = joblib.load("early_model.pkl")
        model_mid = joblib.load("mid_model.pkl")
        model_late = joblib.load("late_model.pkl")
        return model_early, model_mid, model_late
    except Exception:
        st.error("Error loading model files. Make sure all .pkl files exist.")
        st.stop()

# SAFE PASS CLASS DETECTION
def get_pass_label(model):
    for i, c in enumerate(model.classes_):
        if str(c).lower() == "pass" or str(c) == "1":
            return i
    raise ValueError("PASS class not found in model classes")

# Weighted Grade Function
#this will be for G1 and G2
def calculate_weighted_grade(scores, weights):
    if not scores or not weights:
        return 0.0

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    # Convert weights to decimals
    normalized = [w / total_weight for w in weights]
    return sum(s * w for s, w in zip(scores, normalized))

#this is for the pass requirement of the module
def module_contribution(scores, weights):
    return sum(s * (w/100) for s, w in zip(scores, weights))

#
def calculate_required_score(scores, weights):
    total_weight = sum(weights)

    #Prevent calculations when weights exceed 100%
    if total_weight > 100:
        st.error("Total weight exceeds 100%. Please fix your inputs.")
        return None

    remaining_weight = 100 - total_weight

    if remaining_weight <= 0:
        return None

    contribution_so_far = module_contribution(scores, weights)
    required_score = (50 - contribution_so_far) / (remaining_weight / 100)

    return required_score, remaining_weight

def get_stage_grade(stage_data):
    return calculate_weighted_grade(stage_data["scores"], stage_data["weights"])
# ---------------------------
# Session State Initialization
# ---------------------------
if "data_store" not in st.session_state:
    st.session_state.data_store = {
        "early": {"scores": [], "weights": []},
        "mid": {"scores": [], "weights": []},
        "late": {"scores": [], "weights": []},
    }

if "results" not in st.session_state:
    st.session_state.results = {}

if "probabilities" not in st.session_state:
    st.session_state.probabilities = {}

model_early, model_mid, model_late = load_models()

early_pass_label = get_pass_label(model_early)
mid_pass_label = get_pass_label(model_mid)
late_pass_label = get_pass_label(model_late)

pass_labels = {
    "Early": early_pass_label,
    "Mid": mid_pass_label,
    "Late": late_pass_label
}

# ---------------------------
# Model Header
# ---------------------------
st.title("🎓 Student Performance Predictor")

st.markdown("""
### 📘 About This Tool

This system predicts whether a student is likely to pass or fail a module based on academic behaviour, engagement, and performance over time.

The model uses machine learning trained on real student data to identify patterns linked to academic success. It considers factors such as study time, attendance, past academic performance, and support systems.

A key feature of this system is its ability to adapt to different stages of the semester. Students can enter their assessment scores and corresponding weightings, allowing the system to calculate a running weighted grade.

Enter only the assessments that belong to each stage:

- **Early** → early coursework
- **Mid** → mid-term assessments
- **Late** → final assessments

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

stage_key_map = {
    "Early (No Assessments)": "early",
    "Mid (Some Assessments)": "mid",
    "Late (Most Assessments)": "late"
}

current_key = stage_key_map[stage]
current_data = st.session_state.data_store[current_key]

# ---------------------------
# Common Inputs
# ---------------------------
studytime = st.slider("Study Time per Week (1 = <2hrs, 2 = 2–5hrs, 3 = 5–10hrs, 4 = >10hrs)", 1, 4)
failures = st.slider("Number of Past Academic Failures (historical, not current module)", 0, 5)
absences = st.number_input("Number of Classes Missed This Semester (Max 13)", 0, 13)

def encode_binary(val):
    return 1 if val == "Yes" else 0

schoolsup = encode_binary(st.selectbox("Do you receive academic support from your institution?", ["No", "Yes"]))
famsup = encode_binary(st.selectbox("Do you receive family support for your studies?", ["No", "Yes"]))
internet = encode_binary(st.selectbox("Do you have reliable internet access outside the University?", ["No", "Yes"]))

understanding_map = {"Poor": 0, "Average": 1, "Good": 2}
understanding_str = st.selectbox(
    "How well do you currently understand this module?",
    ["Poor", "Average", "Good"]
)
understanding = understanding_map[understanding_str]

# ---------------------------
# Assessment Inputs
# ---------------------------
G1 = 0.0
G2 = 0.0

if stage != "Early (No Assessments)":
    st.subheader("📊 Assessments")

    num_assessments = st.number_input(
        "Number of assessments",
        1, 11,
        value=len(current_data["scores"]) or 1
    )

    scores = []
    weights = []

    for i in range(int(num_assessments)):

        score = st.number_input(
            f"Score {i+1}",
            0.0, 100.0,
            value=current_data["scores"][i] if i < len(current_data["scores"]) else 0.0,
            key=f"{stage}_score_{i}"
        )

        weight = st.number_input(
            f"Weight {i+1} (%)",
            0.0, 100.0,
            value=current_data["weights"][i] if i < len(current_data["weights"]) else 0.0,
            key=f"{stage}_weight_{i}"
        )

        scores.append(score)
        weights.append(weight)

    # Warn if weights ≠ 100
    if abs(sum(weights) - 100) > 0.01:
        st.warning(f"⚠️ Total weight is {sum(weights)}%. It should sum to 100%.")
        st.caption(f"Current contribution to final grade: {module_contribution(scores, weights):.2f}%")
    # Save state
    st.session_state.data_store[current_key] = {
        "scores": scores,
        "weights": weights
    }

    # Compute weighted grade
    current_grade = calculate_weighted_grade(scores, weights)
    st.success(f"Current Weighted Grade: {current_grade:.2f}%")

    #Implementing a hybrid decision logic to stop conflicting prediction results
    def rule_based_override(weighted_grade, ml_pred, pass_label):
        if weighted_grade >= 50:
            return pass_label, "Rule-based PASS (grade ≥ 50%)"
        elif weighted_grade < 40:
            return 1 - pass_label, "Rule-based FAIL (grade < 40%)"
        else:
            return ml_pred, "ML-based decision (40–50% zone)"


    # ---------------------------
    # PASS REQUIREMENT
    # ---------------------------
    total_weight = sum(weights)
    remaining_weight = 100 - total_weight

    if remaining_weight > 0:
        weighted_score_so_far = current_grade * (total_weight / 100)

        required_score = (50 - weighted_score_so_far) / (remaining_weight / 100)
        required_score = max(0, required_score)

        st.info(f"Required avg in remaining work: {required_score:.2f}%")

        if required_score > 100:
            st.error("Mathematically impossible to pass.")
        elif module_contribution(scores, weights) >= 50:
            st.success("You are already passing.")

    # ---------------------------
    # Assign G1 / G2 CONSISTENTLY
    # ---------------------------
    def get_stable_grade(data):
        return calculate_weighted_grade(data["scores"], data["weights"])

    G1 = get_stable_grade(st.session_state.data_store["mid"])
    G2 = get_stable_grade(st.session_state.data_store["late"])

    st.caption("ℹ️ G1 = Mid-stage performance (continuous assessment average)")
    st.caption("ℹ️ G2 = Late-stage performance (near-final grade estimate)")

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Outcome"):

    st.session_state.results = {}
    st.session_state.probabilities = {}

    # ---------------- Early Stage ----------------
    early_features = np.array([[studytime, failures, absences, schoolsup, famsup, internet, understanding]])
    early_prob = model_early.predict_proba(early_features)[0]

    early_pred = np.argmax(early_prob)
    #applied the new hybrid decision system
    #early_final, early_reason = current_grade, early_pred, early_pass_label
    #st.session_state.results["Early"] = early_final
    st.session_state.results["Early"] = early_pred #reversed to previous logic without the hybrid decision making
    st.session_state.probabilities["Early"] = early_prob[early_pass_label]
    #st.session_state[f"reason_Early"] = early_reason

    # ---------------- Mid Stage ----------------
    #Check for actual data instead of G1>0
    if len(st.session_state.data_store["mid"]["scores"]) > 0:
        mid_features = np.array([[studytime, failures, absences, schoolsup, famsup, internet, understanding, G1]])

        mid_prob = model_mid.predict_proba(mid_features)[0]
        mid_pred = np.argmax(mid_prob)
        #applied the new hybrid decision system
        mid_final, mid_reason = rule_based_override(G1, mid_pred, mid_pass_label)
        st.session_state.results["Mid"] = mid_final
        st.session_state.probabilities["Mid"] = mid_prob[mid_pass_label]
        st.session_state[f"reason_Mid"] = mid_reason

    # ---------------- Late Stage ----------------
    #Check for actual data instead of G2>0
    if len(st.session_state.data_store["late"]["scores"]) > 0:
        late_features = np.array([[studytime, failures, absences, schoolsup, famsup, internet, understanding, G1, G2]])

        late_prob = model_late.predict_proba(late_features)[0]
        late_pred = np.argmax(late_prob)
        #applied the new hybrid decision system
        late_final, late_reason = rule_based_override(G2, late_pred, late_pass_label)
        st.session_state.results["Late"] = late_final
        st.session_state.probabilities["Late"] = late_prob[late_pass_label]
        st.session_state[f"reason_Late"] = late_reason

    # ---------------------------
    # Display Results
    # ---------------------------
    st.subheader("📊 Results")

    for stage_name in st.session_state.results:
        pred = st.session_state.results[stage_name]
        prob = st.session_state.probabilities[stage_name]
        reason = st.session_state.get(f"reason_{stage_name}", "ML-based decision")

        if pred == pass_labels[stage_name]:
            st.success(f"{stage_name}: PASS ({prob*100:.2f}%)")
        else:
            st.error(f"{stage_name}: FAIL ({(1-prob)*100:.2f}% risk)")

        reason = st.session_state.get(f"reason_{stage_name}", "ML-based decision")
        st.caption(f"Decision source: {reason}")

    # ---------------------------
    # Progress Insight
    # ---------------------------
    if "Early" in st.session_state.results and "Late" in st.session_state.results:

        early_pred = st.session_state.results["Early"]
        late_pred = st.session_state.results["Late"]

        #removed hardcoded 0/1 assumptions ---
        if early_pred != early_pass_label and late_pred == late_pass_label:
            st.success("Improvement detected 📈")
        elif early_pred == early_pass_label and late_pred != late_pass_label:
            st.warning("Performance drop detected ⚠️")

    # ---------------------------
    # Final Stage Result
    # ---------------------------
    latest = list(st.session_state.results.keys())[-1]
    st.subheader("🎯 Final Prediction")

    final_pred = st.session_state.results[latest]
    final_prob = st.session_state.probabilities[latest]
    final_reason = st.session_state.get(f"reason_{latest}", "ML-based decision")

    if final_pred == pass_labels[latest]:
        st.success(f"Likely PASS ({final_prob*100:.2f}%)")
    else:
        st.error(f"Risk of FAIL ({(1-final_prob)*100:.2f}%)")

        st.write("💡 Suggestions:")
        st.write("- Increase study time")
        st.write("- Reduce absences")
        st.write("- Seek support")

    st.caption(f"Decision source: {final_reason}")

# ---------------------------
# Visualization
# ---------------------------
if "results" in st.session_state and len(st.session_state.results) > 0:

    st.subheader("📈 Trend Analysis")

    labels = list(st.session_state.results.keys())
    probs = [st.session_state.probabilities[l] for l in labels]

    if len(probs) > 1:
        fig, ax = plt.subplots()
        ax.plot(labels, probs, marker="o")
        ax.set_ylabel("Pass Probability")
        ax.set_xlabel("Stage")
        ax.set_title("Performance Trend")
        st.pyplot(fig)

# ---------------------------
# Support Recommendation System
# ---------------------------
#Added a support system ONLY if the student is predicted to FAIL
if "results" in st.session_state and len(st.session_state.results) > 0:
    latest = list(st.session_state.results.keys())[-1]
    final_pred = st.session_state.results[latest]

    # Check if final prediction is FAIL
    if final_pred != pass_labels[latest]:
        st.subheader("🆘 Get Support")
        st.write("We’ve identified that you may be at risk of failing.")
        st.write("Tell us what challenges you're facing so we can guide you to the right support services.")
       
        # User input
        user_input = st.text_area("Describe your challenges:")
        if user_input:

            # Keyword mapping from the user's input to services
            support_services = {
                "time": ("Time Management Support", "timemanagement@university.edu"),
                "stress": ("Wellbeing / Counselling Service", "wellbeing@university.edu"),
                "anxiety": ("Wellbeing / Counselling Service", "wellbeing@university.edu"),
                "mental": ("Mental Health Support", "mentalhealth@university.edu"),
                "financial": ("Financial Support Team", "financehelp@university.edu"),
                "money": ("Financial Support Team", "financehelp@university.edu"),
                "family": ("Student Support Services", "studentsupport@university.edu"),
                "motivation": ("Academic Skills Support", "academicskills@university.edu"),
                "study": ("Academic Skills Support", "academicskills@university.edu"),
                "understanding": ("Module Tutor Support", "moduletutor@university.edu"),
                "lecture": ("Module Tutor Support", "moduletutor@university.edu"),
                "attendance": ("Student Engagement Team", "engagement@university.edu"),
                "absence": ("Student Engagement Team", "engagement@university.edu")
            }

            user_input_lower = user_input.lower()

            matched_services = set()

            #Keyword matching 
            for keyword in support_services:
                if keyword in user_input_lower:
                    matched_services.add(support_services[keyword])

            #Display support suggestions if there is a keyword match
            if matched_services:
                st.success("We recommend the following support services:")

                for service_name, email in matched_services:
                    st.write(f"**{service_name}**")
                    st.write(f"📧 {email}")
                    st.write("---")

            else:
                #Direct contact to programme leader if no keyword matched
                st.warning("We couldn't identify a specific issue, but support is available.")

                st.write("Please contact your Programme Leader for guidance:")
                st.write("📧 programmeleader@university.edu")
