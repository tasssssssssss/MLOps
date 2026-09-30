import pandas as pd

from model_pipeline import prepare_data, train_model


def make_csv(path, n=40):
    pd.DataFrame(
        {
            "RowNumber": range(n),
            "CustomerId": range(n),
            "Surname": ["x"] * n,
            "CreditScore": [600 + i for i in range(n)],
            "Geography": ["France"] * n,
            "Gender": ["Male", "Female"] * (n // 2),
            "Age": [20 + i for i in range(n)],
            "Tenure": [i % 10 for i in range(n)],
            "Balance": [1000.0 * i for i in range(n)],
            "NumOfProducts": [1] * n,
            "HasCrCard": [1] * n,
            "IsActiveMember": [i % 2 for i in range(n)],
            "EstimatedSalary": [50000.0] * n,
            "Exited": [i % 2 for i in range(n)],
        }
    ).to_csv(path, index=False)


def test_prepare_data(tmp_path):
    csv = tmp_path / "data.csv"
    make_csv(csv)
    x_train, x_test, y_train, y_test = prepare_data(str(csv))
    assert "Surname" not in x_train.columns
    assert "Exited" not in x_train.columns
    assert len(x_train) + len(x_test) == 40


def test_train_model(tmp_path):
    csv = tmp_path / "data.csv"
    make_csv(csv)
    x_train, x_test, y_train, y_test = prepare_data(str(csv))
    model = train_model(x_train, y_train, n_estimators=10)
    assert len(model.predict(x_test)) == len(x_test)
