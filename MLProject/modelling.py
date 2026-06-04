import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    current_dir,
    "diabetes_preprocessing",
    "train_preprocessing.csv"
)

data = pd.read_csv(csv_path)

X = data.drop("Credit_Score", axis=1)
y = data["Credit_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model,
        "model"
    )

print("Training selesai")
