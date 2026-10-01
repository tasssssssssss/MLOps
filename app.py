from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_pipeline import MODEL_FILE, load_model, predict

MODEL_PATH = MODEL_FILE

app = FastAPI(title="Churn Prediction API")

# Load the model once, at startup
model = load_model(MODEL_PATH)
print("Model loaded from", MODEL_PATH)

# Same encoding as LabelEncoder in prepare_data (alphabetical order)
GENDER_MAP = {"Female": 0, "Male": 1}


class Features(BaseModel):
    CreditScore: int
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.post("/predict")
def predict_churn(features: Features):
    """
    Make a prediction from the given features.
    Args: features: the customer's characteristics.
    Returns: dict: the model's prediction (1 = churn, 0 = stays).
    Raises: HTTPException: if an error occurs during prediction.
    """
    try:
        if features.Gender not in GENDER_MAP:
            raise ValueError("Gender must be 'Female' or 'Male'")
        row = [
            features.CreditScore,
            GENDER_MAP[features.Gender],
            features.Age,
            features.Tenure,
            features.Balance,
            features.NumOfProducts,
            features.HasCrCard,
            features.IsActiveMember,
            features.EstimatedSalary,
        ]
        prediction = predict(model, row)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
