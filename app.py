import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("admission_model.pkl")

st.title("🎓 Graduate Admission Prediction App")

# --- User Inputs ---
gre = st.number_input("GRE (Graduate Record Examination)  Score (260 - 340)", min_value=260, max_value=340, step=1)
toefl = st.number_input("TOEFL (Test of English as a Foreign Language) Score (0 - 120)", min_value=0, max_value=120, step=1)
uni_rating = st.slider("University Rating (1-5)", 1, 5)
sop = st.slider("SOP (Statement of Purpose) Strength (1-5)", 1.0, 5.0, step=0.5)
lor = st.slider("LOR (Letter of Recommendation) Strength (1-5)", 1.0, 5.0, step=0.5)
cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, step=0.1)
research = st.radio("Research Experience", ["No", "Yes"])
research_val = 1 if research == "Yes" else 0

# --- Prediction Button ---
if st.button("Predict Admission Chance"):
    # --- Input validation ---
    missing_fields = []
    if gre == 0:
        missing_fields.append("GRE Score")
    if toefl == 0:
        missing_fields.append("TOEFL Score")
    if uni_rating == 0:
        missing_fields.append("University Rating")
    if sop == 0:
        missing_fields.append("SOP Strength")
    if lor == 0:
        missing_fields.append("LOR Strength")
    if cgpa == 0:
        missing_fields.append("CGPA")
    if research not in ["Yes", "No"]:
        missing_fields.append("Research Experience")

    if missing_fields:
        st.error(f"❌ Please fill all required fields: {', '.join(missing_fields)}")
    else:
        # All fields filled, proceed with prediction
        input_data = np.array([[gre, toefl, uni_rating, sop, lor, cgpa, research_val]])
        prediction = model.predict(input_data)[0]

        # Prediction in percentage
        percentage = round(prediction * 100, 2)
        percentage = max(0, min(percentage, 100))  # Clamp between 0-100

        st.subheader(f"🎯 Predicted Chance of Admission: {percentage}%")

        # --- Gradient Color Meter ---
        if percentage >= 70:
            color = "green"
        elif percentage >= 50:
            color = "orange"
        else:
            color = "red"

        st.markdown(
            f"""
            <div style="width: 100%; height: 30px; 
                 background: linear-gradient(90deg, {color} {percentage}%, lightgray {percentage}%); 
                 border-radius: 10px;">
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Result Message ---
        if percentage >= 70:
            st.success(f"✅ Possible! You have a strong chance with {percentage}%.")
        elif percentage >= 50:
            st.warning(f"⚠️ Moderate chance ({percentage}%). Improve SOP, LOR, or Research for better results.")
        else:
            st.error(f"❌ Not Possible. Only {percentage}%. Work on improving your profile.")
