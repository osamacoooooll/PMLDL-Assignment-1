import streamlit as st
import requests

st.title("Diabetes Prediction App")

# Collect input from the user for all the features
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=80)
skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=30.0, format="%.1f")
diabetes_pedigree_function = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Create the input data dictionary
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }
    
    # Make a request to the FastAPI endpoint
    response = requests.post("http://backend:8001/predict", json=input_data)
    
    # Get the prediction from the response
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        
        # Interpret the prediction
        if prediction == 1:
            st.write("Prediction: The person is likely to have diabetes.")
        else:
            st.write("Prediction: The person is not likely to have diabetes.")
    else:
        st.write("Error: Unable to get prediction.")
