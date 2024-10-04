import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load processed data
current_path = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_path, '..', '..'))
processed_data_dir = os.path.join(base_dir, 'data', 'processed')

x_train = pd.read_csv(os.path.join(processed_data_dir, 'x_train.csv'))
x_test = pd.read_csv(os.path.join(processed_data_dir, 'x_test.csv'))
y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv'))

y_train = y_train.values.ravel()
y_test = y_test.values.ravel() 

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model
model_path = os.path.join(current_path, 'model.pkl')
joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")
