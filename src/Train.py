import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Breast_Cancer_Experiment")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier

data_path = r"C:\Users\yashu\OneDrive\MLOPs\MLOPS_BREAST_CANCER_PROJECT\original_data\BREAST_CANCER1.csv"


def load_data(path):
    return pd.read_csv(path)


def train_model(model_type="logistic"):

    df = load_data(data_path)

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000,random_state=42)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        pipeline = Pipeline([
            ("model", model)
        ])

    else:
        raise ValueError("Invalid model type")
    mlflow.set_experiment("Breast_Cancer_Experiment")
    with mlflow.start_run(run_name=model_type) as run:

        pipeline.fit(X_train, y_train)
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        if model_type == "logistic":
            mlflow.sklearn.log_model(pipeline, "model")
        else:
            mlflow.xgboost.log_model(model, "model")

        run_id = run.info.run_id

    return accuracy, roc_auc, run_id

if __name__ == "__main__":

    acc_lr, roc_lr, run_lr = train_model("logistic")
    print("Logistic Regression Accuracy:", acc_lr)
    print("Logistic Regression ROC-AUC:", roc_lr)
    print("Run ID:", run_lr)

    acc_xgb, roc_xgb, run_xgb = train_model("xgboost")
    print("XGBoost Accuracy:", acc_xgb)
    print("XGBoost ROC-AUC:", roc_xgb)
    print("Run ID:", run_xgb)