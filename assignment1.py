import pandas as pd
import statsmodels.api as sm

url_train = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train_data = pd.read_csv(url_train)


print(train_data.head())TEST


model = sm.tsa.ARIMA(train_data['trips'], order=(1,1,1))
modelFit = model.fit()

print(modelFit.summary())


