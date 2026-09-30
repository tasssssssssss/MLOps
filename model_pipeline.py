"""Modular functions for the customer churn ML pipeline."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_FILE = "Churn_Modelling.csv"
MODEL_FILE = "classifier.joblib"
DROP_COLS = ["RowNumber", "CustomerId", "Surname", "Geography"]


def prepare_data(path=DATA_FILE, test_size=0.2, random_state=1):
    """Load the CSV, clean it and return x_train, x_test, y_train, y_test."""
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS)
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    x = df.drop(columns=["Exited"])
    y = df["Exited"]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_model(x_train, y_train, n_estimators=100):
    """Train and return a RandomForest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Print metrics and return the accuracy."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy * 100, 2))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return accuracy


def save_model(model, path=MODEL_FILE):
    """Save the trained model with joblib."""
    joblib.dump(model, path)


def load_model(path=MODEL_FILE):
    """Load a saved model."""
    return joblib.load(path)


def predict(model, features):
    """Predict churn for one customer.

    features: [CreditScore, Gender, Age, Tenure, Balance,
               NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
    """
    return model.predict([features])
