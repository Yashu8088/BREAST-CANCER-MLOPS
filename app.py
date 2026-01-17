import gradio as gr
import mlflow.pyfunc
import pandas as pd

# Load MLflow model
model = mlflow.pyfunc.load_model(".")

feature_names = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean"
]

def predict(*inputs):
    df = pd.DataFrame([inputs], columns=feature_names)
    pred = model.predict(df)[0]
    return "Malignant" if pred == 1 else "Benign"

inputs = [gr.Number(label=f) for f in feature_names]

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Breast Cancer Prediction (MLOps Project)",
    description="Predicts whether a tumor is Benign or Malignant using an MLflow-trained model."
)

if __name__ == "__main__":
    app.launch()
