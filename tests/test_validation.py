from src.Train import train_model

baseline_roc_auc = 0.90


def test_validation(model_type="logistic"):
    accuracy, roc_auc, run_id = train_model(model_type)

    print(f"Model Type: {model_type}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Run ID: {run_id}")

    if roc_auc < baseline_roc_auc:
        raise ValueError(
            f"{model_type} model performance below baseline ROC-AUC ({baseline_roc_auc})"
        )

    print("Performance meets baseline threshold")
    return roc_auc, run_id


if __name__ == "__main__":
    test_validation("logistic")
    test_validation("xgboost")
