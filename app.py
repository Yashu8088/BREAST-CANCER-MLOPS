import gradio as gr
import joblib
import pandas as pd


lr_model = joblib.load("model/logistic.pkl")
xgb_model = joblib.load("model/xgboost.pkl")


FEATURE_COLUMNS = [
    "radius_mean","texture_mean","perimeter_mean","area_mean",
    "smoothness_mean","compactness_mean","concavity_mean","concave_points_mean",
    "symmetry_mean","fractal_dimension_mean",

    "radius_se","texture_se","perimeter_se","area_se",
    "smoothness_se","compactness_se","concavity_se","concave_points_se",
    "symmetry_se","fractal_dimension_se",

    "radius_worst","texture_worst","perimeter_worst","area_worst",
    "smoothness_worst","compactness_worst","concavity_worst","concave_points_worst",
    "symmetry_worst","fractal_dimension_worst"
]


def predict(
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    concave_points_mean,
    symmetry_mean,
    fractal_dimension_mean,

    radius_se,
    texture_se,
    perimeter_se,
    area_se,
    smoothness_se,
    compactness_se,
    concavity_se,
    concave_points_se,
    symmetry_se,
    fractal_dimension_se,

    radius_worst,
    texture_worst,
    perimeter_worst,
    area_worst,
    smoothness_worst,
    compactness_worst,
    concavity_worst,
    concave_points_worst,
    symmetry_worst,
    fractal_dimension_worst
):
    # Create input DataFrame
    df = pd.DataFrame([[

        radius_mean, texture_mean, perimeter_mean, area_mean,
        smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
        symmetry_mean, fractal_dimension_mean,

        radius_se, texture_se, perimeter_se, area_se,
        smoothness_se, compactness_se, concavity_se, concave_points_se,
        symmetry_se, fractal_dimension_se,

        radius_worst, texture_worst, perimeter_worst, area_worst,
        smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
        symmetry_worst, fractal_dimension_worst

    ]], columns=FEATURE_COLUMNS)

    # Logistic Regression
    lr_prob = lr_model.predict_proba(df)[0][1]
    lr_pred = 1 if lr_prob > 0.5 else 0

    # XGBoost
    xgb_prob = xgb_model.predict_proba(df)[0][1]
    xgb_pred = 1 if xgb_prob > 0.5 else 0

    # Average risk
    avg_risk = (lr_prob + xgb_prob) / 2

    return f"""
Risk Probability: {avg_risk*100:.2f} %
Logistic Regression: {"Malignant" if lr_pred == 1 else "Benign"}
XGBoost: {"Malignant" if xgb_pred == 1 else "Benign"}
"""


interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="radius_mean"),
        gr.Number(label="texture_mean"),
        gr.Number(label="perimeter_mean"),
        gr.Number(label="area_mean"),
        gr.Number(label="smoothness_mean"),
        gr.Number(label="compactness_mean"),
        gr.Number(label="concavity_mean"),
        gr.Number(label="concave_points_mean"),
        gr.Number(label="symmetry_mean"),
        gr.Number(label="fractal_dimension_mean"),

        gr.Number(label="radius_se"),
        gr.Number(label="texture_se"),
        gr.Number(label="perimeter_se"),
        gr.Number(label="area_se"),
        gr.Number(label="smoothness_se"),
        gr.Number(label="compactness_se"),
        gr.Number(label="concavity_se"),
        gr.Number(label="concave_points_se"),
        gr.Number(label="symmetry_se"),
        gr.Number(label="fractal_dimension_se"),

        gr.Number(label="radius_worst"),
        gr.Number(label="texture_worst"),
        gr.Number(label="perimeter_worst"),
        gr.Number(label="area_worst"),
        gr.Number(label="smoothness_worst"),
        gr.Number(label="compactness_worst"),
        gr.Number(label="concavity_worst"),
        gr.Number(label="concave_points_worst"),
        gr.Number(label="symmetry_worst"),
        gr.Number(label="fractal_dimension_worst"),
    ],
    outputs="text",
    title="Breast Cancer Prediction App",
    description="Logistic Regression + XGBoost | UCI Breast Cancer Dataset | MLOps Deployment",
)

interface.launch()
