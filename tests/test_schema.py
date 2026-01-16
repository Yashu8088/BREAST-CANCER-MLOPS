import pandas as pd

data_path = r"C:\Users\yashu\OneDrive\MLOPs\MLOPS_BREAST_CANCER_PROJECT\original_data\BREAST_CANCER1.csv"

def load_data(path):
    return pd.read_csv(path)


def test_schema():
    df = load_data(data_path)

    df = df.drop(columns=["id"])

    expected_columns = 31  # 30 features + 1 target

    assert df.shape[1] == expected_columns


