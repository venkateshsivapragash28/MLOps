import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

x_train = pd.read_csv("data/x_train.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()

model = LinearRegression()
model.fit(x_train, y_train)

joblib.dump(model, "models/model.pkl")