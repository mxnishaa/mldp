import streamlit as st
import pandas as pd
import joblib

# 1. Load the single-file deployment bundle
@st.cache_resource
def load_model():
    # Using your specific filename provided
    bundle = joblib.load('income_prediction_model_final.pkl')
    return bundle

bundle = load_model()
model = bundle['model']
feature_names = bundle['features']
threshold = bundle['threshold']

# 2. App Header (Business Branding)
st.title("🚀 SmartLead Fintech")
st.subheader("Income Prediction Engine")
st.write("Determine customer earning potential instantly to provide personalized financial recommendations.")

# 3. User Inputs (User-Friendly Interface)
st.sidebar.header("Customer Profile")

# Feature: age
age = st.sidebar.slider("Age", 17, 90, 30)

# Feature: education.num (User-friendly Dropdown)
# Mapping text labels to the education.num values from the dataset
edu_options = {
    "Preschool/Primary": 3,
    "Secondary School (9th-12th)": 7,
    "High School Grad": 9,
    "No Degree / Currently Enrolled": 10,
    "Associates Degree (Vocational/Academic)": 11,
    "Bachelors Degree": 13,
    "Masters Degree": 14,
    "Professional School (JD, MD, etc.)": 15,
    "Doctorate (PhD)": 16
}
selected_edu = st.sidebar.selectbox("Education Level", list(edu_options.keys()))
edu_num = edu_options[selected_edu]

# Feature: hours.per.week
hours = st.sidebar.slider("Typical Working Hours Per Week", 1, 99, 40)

# Feature: is_married (Detailed Options)
# Mapping marital status to binary 1 (Married) or 0 (Not Married)
marital_status = st.sidebar.selectbox(
    "Marital Status", 
    ["Married (Living with Spouse)", "Single / Divorced / Separated / Widowed"]
)
is_married = 1 if "Married" in marital_status else 0

# Feature: net_capital (Simplified into Gain and Loss)
st.sidebar.divider()
st.sidebar.write("**Financial History (Annual)**")
cap_gain = st.sidebar.number_input("Total Investment Gains ($)", min_value=0, value=0, help="Profits from selling assets like stocks or property etc.")
cap_loss = st.sidebar.number_input("Total Investment Losses ($)", min_value=0, value=0, help="Losses from selling assets etc.")
net_capital = cap_gain - cap_loss

# 4. Prediction Logic
if st.button("Predict Income"):
    # Ensure the order matches exactly: ['age', 'education.num', 'hours.per.week', 'is_married', 'net_capital']
    input_data = pd.DataFrame([[age, edu_num, hours, is_married, net_capital]], 
                              columns=feature_names)
    
    # Get probability using the optimized model
    prob = model.predict_proba(input_data)[0, 1]
    
    # Apply your optimized threshold of 0.56
    is_high_income = prob >= threshold

    # 5. Display Results
    st.divider()
    if is_high_income:
        st.success("✅ **High-Value Prospect**")
        st.write(f"This individual is likely to earn >$50k annually (Model Confidence: {prob:.2%}).")
        st.info("**Marketing Action:** Direct to Premium Wealth Management & Investment Portfolio services.")
    else:
        st.warning("⚠️ **Standard-Value Prospect**")
        st.write(f"This individual is likely to earn <=$50k annually (Model Confidence: {1-prob:.2%}).")
        st.write("**Marketing Action:** Assign to basic automated savings and budget-tracking tools.")