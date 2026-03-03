import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

df = pd.read_csv("data/raw.csv")
df = df.drop(columns=['id'])

X = df.drop(columns=['exam_score', 'age'])
y = df['exam_score']

categorical_cols = X.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pd.DataFrame(x_train).to_csv("data/x_train.csv", index=False)
pd.DataFrame(x_test).to_csv("data/x_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

joblib.dump(scaler, "models/scaler.pkl")