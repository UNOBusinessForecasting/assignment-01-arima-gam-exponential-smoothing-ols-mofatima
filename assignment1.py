import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
import pickle
# Load train data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train_data = pd.read_csv(train_url)

# Load test data
test_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
test_data = pd.read_csv(test_url)

# Check the head of the training dataset
print(train_data.head())
# Features: year, month, day, hour
X_train = train_data[['year', 'month', 'day', 'hour']]

# Target: trips
y_train = train_data['trips']

# Prepare test data in the same way
X_test = test_data[['year', 'month', 'day', 'hour']]
# Define the GAM model
model = LinearGAM(s(0) + s(1) + s(2) + s(3))
# Fit the model
modelFit = model.fit(X_train, y_train)
# Generate predictions using the fitted model
pred = modelFit.predict(X_test)

# Store predictions in the test dataframe
test_data['predicted_trips'] = pred

# check the test data with predictions
print(test_data[['Timestamp', 'predicted_trips']].head())


# Save the modelFit (the fitted model) to a file
with open('gam_model.pkl', 'wb') as f:
    pickle.dump(modelFit, f)
