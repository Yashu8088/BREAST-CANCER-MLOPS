---
title: Breast Cancer Prediction (MLOps)
emoji: 🎗️
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

Hugging Face Live Demo :https://huggingface.co/spaces/yashuchouhan/breast-cancer-deployement


This app predicts whether a tumor is benign or malignant based on 30 medical features.

Enter the feature values to predict whether the tumor is Benign or Malignant.

To test **Malignant** prediction Use These Values:

**Mean Features**

radius_mean: 20.5

texture_mean: 25.0

perimeter_mean: 135.0

area_mean: 1250.0

smoothness_mean: 0.145

compactness_mean: 0.260

concavity_mean: 0.300

concave_points_mean: 0.155

symmetry_mean: 0.230

fractal_dimension_mean: 0.075

**Standard Error Features**

radius_se: 1.20

texture_se: 2.50

perimeter_se: 8.0

area_se: 150.0

smoothness_se: 0.020

compactness_se: 0.060

concavity_se: 0.080

concave_points_se: 0.030

symmetry_se: 0.040

fractal_dimension_se: 0.012

**Worst Features**

radius_worst: 25.0

texture_worst: 32.0

perimeter_worst: 170.0

area_worst: 1800.0

smoothness_worst: 0.180

compactness_worst: 0.420

concavity_worst: 0.520

concave_points_worst: 0.220

symmetry_worst: 0.350

fractal_dimension_worst: 0.110

To test **Benign** Prediction Use These Values:

**Mean Features**

radius_mean: 11.8

texture_mean: 14.0

perimeter_mean: 75.0

area_mean: 450.0

smoothness_mean: 0.090

compactness_mean: 0.070

concavity_mean: 0.040

concave_points_mean: 0.020

symmetry_mean: 0.160

fractal_dimension_mean: 0.060

**Standard Error Features**

radius_se: 0.30

texture_se: 1.10

perimeter_se: 2.5

area_se: 25.0

smoothness_se: 0.006

compactness_se: 0.020

concavity_se: 0.030

concave_points_se: 0.010

symmetry_se: 0.020

fractal_dimension_se: 0.005

**Worst Features**

radius_worst: 13.0

texture_worst: 18.0

perimeter_worst: 85.0

area_worst: 520.0

smoothness_worst: 0.110

compactness_worst: 0.120

concavity_worst: 0.090

concave_points_worst: 0.040

symmetry_worst: 0.220

fractal_dimension_worst: 0.080

Enter the values above and click **Predict** to see the result instantly.



