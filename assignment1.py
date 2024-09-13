
import pandas as pd
import numpy as np
pip install pygam
# Import necessary libraries
import pandas as pd
import numpy as np
from pygam import LinearGAM, s, f
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle

# Load the train and test data
train_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')
test_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')

# Convert the 'Timestamp' column to datetime and set it as the index
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])
train_data.set_index('Timestamp', inplace=True)

# Check if 'trips' column is correct, update if needed
print(train_data.columns)

# Prepare the input features (X) and target variable (y)
# For simplicity, we'll use the hour of the day and the day of the week as features
train_data['hour'] = train_data.index.hour
train_data['day_of_week'] = train_data.index.dayofweek

X = train_data[['hour', 'day_of_week']].values  # Input features
y = train_data['trips'].values  # Target variable

# Initialize and fit a GAM model with smoothing splines for hour and day_of_week
gam = LinearGAM(s(0) + s(1))  # s(0) for hour, s(1) for day of week
gam.fit(X, y)

# Print model summary
print(gam.summary())

# Predict for the test period (744 hours)
test_data['hour'] = pd.to_datetime(test_data['Timestamp']).dt.hour
test_data['day_of_week'] = pd.to_datetime(test_data['Timestamp']).dt.dayofweek

X_test = test_data[['hour', 'day_of_week']].values
pred = gam.predict(X_test)

# Evaluate the model using RMSE or any other metrics
rmse = np.sqrt(mean_squared_error(test_data['trips'], pred))
print(f'RMSE: {rmse}')

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(test_data['Timestamp'], test_data['trips'], label='True Trips')
plt.plot(test_data['Timestamp'], pred, label='Predicted Trips', color='red')
plt.legend()
plt.title('GAM Forecast of Taxi Trips')
plt.show()

# Save the model to a file
with open('gam_model.pkl', 'wb') as f:
    pickle.dump(gam, f)

# Save predictions to a CSV file
pred_df = pd.DataFrame({'Timestamp': test_data['Timestamp'], 'Predicted_Trips': pred})
pred_df.to_csv('gam_predictions.csv', index=False)





