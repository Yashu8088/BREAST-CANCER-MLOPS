import mlflow
import mlflow.pyfunc
import pandas as pd
import os

def test_tc5_deployment():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(
        project_root,
        "original_data",
        "BREAST_CANCER1.csv"
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Breast_Cancer_Experiment")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    model = mlflow.pyfunc.load_model(model_uri)

    df = pd.read_csv(data_path)
    sample = df.drop(["id", "diagnosis"], axis=1).iloc[:1]

    prediction = model.predict(sample)

    assert prediction is not None
