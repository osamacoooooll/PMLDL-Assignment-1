import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from collections import Counter
from sklearn.preprocessing import StandardScaler
import joblib  # To save and load the model

current_path = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_path, '..', '..'))
data_dir = os.path.join(base_dir, 'data', 'raw')

# Load the dataset with the full path
df = pd.read_csv(os.path.join(data_dir, 'diabetes.csv'))

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardize the data
scaler = StandardScaler()
d = scaler.fit_transform(x)

# Resample the data to handle imbalance
r = RandomOverSampler()
x_data, y_data = r.fit_resample(d, y)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=23)

# Train RandomForestClassifier
f1 = RandomForestClassifier()
f1.fit(x_train, y_train)

# Make predictions and evaluate accuracy
f1_pred = f1.predict(x_test)
f1_score = accuracy_score(y_test, f1_pred) * 100
print("Accuracy:", f1_score)

# Calculate precision and recall
precision = precision_score(y_test, f1_pred)
recall = recall_score(y_test, f1_pred)
print("Precision:", precision)
print("Recall:", recall)

# Create the relative path to save the model
model_dir = os.path.join(current_path, 'code', 'models')  # Create the path to 'code/models'

# Save the trained model
model_path = os.path.join(current_path, 'model.pkl')  # Full path to the model
joblib.dump(f1, model_path)
print(f"Model saved at: {model_path}")

# Save the scaler model
scaler_path = os.path.join(current_path, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved at: {scaler_path}")
