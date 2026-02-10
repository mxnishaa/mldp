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

# --- UI DESIGN: CENTERED LAYOUT & BACKGROUND IMAGE ---
st.set_page_config(page_title="SmartLead Fintech", layout="centered")

# Background image implementation via CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("https://img.freepik.com/free-vector/black-arrow-background-abstract-border-gold-design-vector_53876-140557.jpg?semt=ais_hybrid&w=740&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Force ALL text to White with a shadow for readability */
    h1, h2, h3, h4, p, label, .stMarkdown p {{
        color: #FFFFFF !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8) !important;
    }}

    /* Create a Glassmorphism Card for input separation */
    [data-testid="stVerticalBlock"] > div:has(div[data-testid="stHorizontalBlock"]) {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 20px;
    }}

    /* Input Widget Styling: Darker background to make white text pop */
    .stSelectbox div[data-baseweb="select"], .stNumberInput div[data-baseweb="input"] {{
        background-color: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
    }}
    
    /* FORCE TEXT INSIDE INPUTS TO BE WHITE */
    input, div[data-baseweb="select"] div {{
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}
    
    /* Ensure the dropdown arrow and icons are visible */
    svg {{
        fill: white !important;
    }}

    /* UPDATED FIX: Multiple selectors to force Tooltip Icon visibility */
    [data-testid="stTooltipIcon"], 
    .stTooltipIcon,
    div[data-testid="stTooltipIcon"] svg {{
        color: #d4af37 !important;
        fill: #d4af37 !important;
        filter: drop-shadow(0px 0px 3px rgba(255, 255, 255, 0.8)) !important;
    }}

    /* Slider specific styling to make the red bar and white text pop */
    .stSlider [data-testid="stMetricValue"] {{
        color: white !important;
    }}
    
    /* Success/Warning box text color fix */
    .stAlert p {{
        color: white !important;
        text-shadow: none !important;
    }}

    /* STYLE THE BUTTON TO BE GREEN */
    div.stButton > button:first-child {{
        background-color: #208f4f !important;
        color: white !important;
        border: none;
    }}
    
    div.stButton > button:hover {{
        background-color: #27ae60 !important;
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# 2. App Header (Business Branding)
st.title("🚀 SmartLead Fintech")
st.subheader("Income Prediction Engine")
st.write("Determine customer earning potential instantly to provide personalized financial recommendations.")
st.divider()

# 3. User Inputs (Centralized Interface)
# Section 1: Demographic Info
with st.container():
    st.markdown("### 📋 Step 1: Customer Profile")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 17, 90, 30)
        
        edu_options = {
            "Preschool/Primary": 3, "Secondary School (9th-12th)": 7, 
            "High School Grad": 9, "No Degree / Currently Enrolled": 10,
            "Associates Degree": 11, "Bachelors Degree": 13,
            "Masters Degree": 14, "Professional School (Law/Med School etc)": 15, "Doctorate": 16
        }
        selected_edu = st.selectbox("Education Level", list(edu_options.keys()))
        edu_num = edu_options[selected_edu]

    with col2:
        hours = st.slider("Typical Working Hours Per Week", 1, 99, 40)
        marital_status = st.selectbox(
            "Marital Status", 
            ["Married (Spouse Civillian)", "Married (Spouse Armed Forces)", "Never Married", "Divorced", "Separated", "Widowed"]
        )
        is_married = 1 if "Married" in marital_status else 0

st.write("") # Aesthetic Spacer

# Section 2: Financial Info
with st.container():
    st.markdown("### 💰 Step 2: Financial History (Annual)")
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        # Help hint for investment gains
        cap_gain = st.number_input("Total Investment Gains ($)", min_value=0, value=0, help="Total profits from selling assets like stocks, bonds, or real estate.")

    with f_col2:
        # Help hint for investment losses
        cap_loss = st.number_input("Total Investment Losses ($)", min_value=0, value=0, help="Total losses from selling assets like stocks, bonds, or real estate.")

    net_capital = cap_gain - cap_loss

# 4. Prediction Logic
st.write("") 
if st.button("Predict Income", use_container_width=True):
    input_data = pd.DataFrame([[age, edu_num, hours, is_married, net_capital]], 
                              columns=feature_names)
    prob = model.predict_proba(input_data)[0, 1]
    is_high_income = prob >= threshold

    # 5. Display Results
    st.divider()
    if is_high_income:
        st.success("✅ **High-Value Prospect**")
        st.write(f"This individual is likely to earn >$50k annually.")
        st.info("**Marketing Action:** Direct to Premium Wealth Management & Investment Portfolio services.")
    else:
        st.warning("⚠️ **Standard-Value Prospect**")
        st.write(f"This individual is likely to earn <=$50k annually.")
        st.write("**Marketing Action:** Assign to basic automated savings and budget-tracking tools.")