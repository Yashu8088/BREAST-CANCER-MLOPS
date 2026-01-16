import pytest
import pandas as pd

data_path = r"C:\Users\yashu\OneDrive\MLOPs\MLOPS_BREAST_CANCER_PROJECT\original_data\BREAST_CANCER1.csv"

def load_data(path):
    return pd.read_csv(path)

def test_failure_handling():
    with pytest.raises(Exception):
        load_data(
            r"C:\Users\yashu\OneDrive\MLOPs\MLOPS_BREAST_CANCER_PROJECT\original_data\wdbc.names"
        )
