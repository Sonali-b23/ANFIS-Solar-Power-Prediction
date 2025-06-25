import pandas as pd
import pickle
from mppt_model import mppt_algorithm
from src.anfis_model import ANFIS

# Load preprocessed data
data = pd.read_csv('/data/solar_data_preprocessed.csv')

# Load the trained ANFIS model
with open('models/anfis_model.pkl', 'rb') as file:
    anfis_model = pickle.load(file)

# Load the scaler
with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Make predictions on the test data
X = data.drop(columns=['SystemProduction'])
predictions = anfis_model.predict(X.values)

# Apply MPPT optimization on each prediction (simulated for this example)
data['OptimizedVoltage'] = data.apply(lambda row: mppt_algorithm(row['Radiation'], row['SystemProduction']), axis=1)

# Add predictions and optimized voltages to the results
data['PredictedPower'] = predictions
data['ActualPower'] = data['SystemProduction']

# Save the results
data.to_csv('../data/predicted_power_with_mppt.csv', index=False)
print("Final results with MPPT optimization and ANFIS predictions saved.")