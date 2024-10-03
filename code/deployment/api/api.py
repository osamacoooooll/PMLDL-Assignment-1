from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load the saved model
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define the input data structure (schema)
class InputData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Define a prediction route
@app.post("/predict")
def predict(data: InputData):
    # Convert input data into a numpy array
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
                            data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age]])

    # Standardize the input data
    input_data_standardized = scaler.fit_transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data_standardized)
    
    # Return the prediction (if it's binary, make sure the result is clear)
    return {"prediction": int(prediction[0])}  # You can check if this needs adjustment.
