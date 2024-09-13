import pandas as pd
import statsmodels.api as sm

url_train = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train_data = pd.read_csv(url_train)


print(train_data.head())TEST


from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit an Exponential Smoothing model
model = ExponentialSmoothing(train_data['num_trips'], seasonal='add', seasonal_periods=24)
modelFit = model.fit()

# Forecast for January (744 hours)
pred = modelFit.forecast(steps=744)

# Print or inspect the predictions
print(pred)


