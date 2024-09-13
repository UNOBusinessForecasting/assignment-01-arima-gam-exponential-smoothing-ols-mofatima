import pandas as pd

# Load the training data
url_train = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train_data = pd.read_csv(url_train)

# Inspect the data
print(train_data.head())TEST

import statsmodels.api as sm

# Set up the ARIMA model (p, d, q) based on the data
model = sm.tsa.ARIMA(train_data['trips'], order=(1,1,1))
modelFit = model.fit()

# Print summary to inspect the results
print(modelFit.summary())
