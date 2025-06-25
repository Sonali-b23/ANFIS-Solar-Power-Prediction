# import pandas as pd
# from anfis import ANFIS
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Load the preprocessed data
# data = pd.read_csv('data/solar_data_preprocessed.csv')

# # Split data into features and target
# X = data.drop(columns=['SystemProduction'])
# y = data['SystemProduction']

# # Split into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize ANFIS model
# model = ANFIS()

# # Train the model
# model.fit(X_train.values, y_train.values)

# # Save the trained model
# with open('models/anfis_model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# # Evaluate the model
# y_pred = model.predict(X_test.values)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')


import pandas as pd
from src.anfis import ANFIS  # or from src.anfis import ANFIS if you have local module
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the preprocessed data
data = pd.read_csv('data/solar_data_preprocessed.csv')

# Split data into features and target
X = data.drop(columns=['SystemProduction'])
y = data['SystemProduction']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize ANFIS model
# model = ANFIS()
model = ANFIS(n_inputs=X_train.shape[1], n_mfs=2, epochs=10, learning_rate=0.01)


# Train the model
model.fit(X_train.values, y_train.values)

# Save the trained model
with open('models/anfis_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Evaluate the model
y_pred = model.predict(X_test.values)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
