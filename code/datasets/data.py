import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
current_path = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_path, '..', '..'))
data_dir = os.path.join(base_dir, 'data', 'raw')
processed_dir = os.path.join(base_dir, 'data', 'processed')

# Load the raw data
df = pd.read_csv(os.path.join(data_dir, 'diabetes.csv'))

# Handle missing values
df = df.fillna(df.mean())  # Simple imputation by mean

# Split data
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pd.DataFrame(x_train).to_csv(os.path.join(processed_dir, 'x_train.csv'), index=False)
pd.DataFrame(x_test).to_csv(os.path.join(processed_dir, 'x_test.csv'), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)

# Save the scaler for later use
joblib.dump(scaler, os.path.join(processed_dir, 'scaler.pkl'))
