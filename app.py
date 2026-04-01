import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('heart_model.pkl')

model = load_model()

st.title("AI Heart Disease Risk Assessment")
st.markdown("""
This tool uses Machine Learning to predict the likelihood of heart disease. 
**Disclaimer:** For educational purposes only. Not for medical diagnosis.
""")

# Sidebar for inputs
st.sidebar.header("Patient Metrics")

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 50)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol', 126, 564, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120', ('False', 'True'))
    restecg = st.sidebar.slider('Resting ECG Results', 0, 2, 1)
    thalach = st.sidebar.slider('Max Heart Rate', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('No', 'Yes'))
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.2, 1.0)
    slope = st.sidebar.slider('Slope of ST Peak', 0, 2, 1)
    ca = st.sidebar.slider('Major Vessels Colored', 0, 3, 1)
    thal = st.sidebar.slider('Thalassemia', 1, 3, 2)

    # Map string inputs to integers expected by model
    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'True' else 0
    exang = 1 if exang == 'Yes' else 0

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button('Predict Risk'):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)
    
    if prediction[0] == 1:
        st.error(f"**High Risk Detected** ({proba[0][1]*100:.2f}% probability)")
        st.warning("Recommendation: Consult a cardiologist immediately.")
    else:
        st.success(f"**Low Risk Detected** ({proba[0][0]*100:.2f}% probability)")
        st.info("Recommendation: Maintain healthy lifestyle.")
    
    with st.expander("Show Input Data"):
        st.write(input_df)