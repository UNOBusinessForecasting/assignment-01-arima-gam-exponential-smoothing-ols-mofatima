import pandas as pd

# Load the training data
url_train = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train_data = pd.read_csv(url_train)

# Inspect the data
print(train_data.head())TEST
