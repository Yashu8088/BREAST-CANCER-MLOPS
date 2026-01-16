import mlflow.pyfunc
import pandas as pd

#Replace this with an actual run_id from MLflow UI
MODEL_URI = "runs:/<RUN_ID>/model"

def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)

def predict():
    model = load_model()

    df = pd.read_csv("data/breast_cancer_clean.csv")
    sample = df.drop("diagnosis", axis=1).iloc[:1]

    prediction = model.predict(sample)
    return prediction


if __name__ == "__main__":
    pred = predict()
    print("Prediction:", pred)
