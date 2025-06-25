import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load the dataset
data = pd.read_csv('data/solar_power_generation.csv')

# Drop the 'Date-Hour(NMT)' column as it is not useful for the model
data = data.drop(columns=['Date-Hour(NMT)'])

# Handle missing values by filling them with the mean of each column
data = data.fillna(data.mean())

# Normalize features
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data.drop(columns=['SystemProduction']))

# Convert back to DataFrame
normalized_data_df = pd.DataFrame(normalized_data, columns=data.columns[:-1])

# Add the target variable (SystemProduction) back to the data
normalized_data_df['SystemProduction'] = data['SystemProduction']

# Save the scaler for future use
with open('models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save the preprocessed data to a new CSV
normalized_data_df.to_csv('data/solar_data_preprocessed.csv', index=False)

print("Data preprocessing is complete and scaler has been saved.")
