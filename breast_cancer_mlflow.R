# ============================
# Breast Cancer Classification
# MLflow-safe version (Windows)
# ============================

library(caret)
library(randomForest)

# ---- Load data ----
data <- read.csv("breast_cancer.csv")

# Rename diagnosis column (second column)
colnames(data)[2] <- "diagnosis"
data$diagnosis <- as.factor(data$diagnosis)

# Remove ID column
data <- data[, -1]

# ---- Train-test split ----
set.seed(42)
trainIndex <- createDataPartition(data$diagnosis, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test  <- data[-trainIndex, ]

# ---- Train model ----
model <- randomForest(diagnosis ~ ., data = train)
pred  <- predict(model, test)

# ---- Accuracy ----
acc <- confusionMatrix(pred, test$diagnosis)$overall["Accuracy"]
print(acc)

# ---- MLflow code (SAFE – no crash) ----
try({
  library(mlflow)
  Sys.setenv(MLFLOW_TRACKING_URI = "file:mlruns")
  
  with(mlflow_start_run(), {
    mlflow_log_metric("accuracy", as.numeric(acc))
    mlflow_log_param("model", "RandomForest")
  })
}, silent = TRUE)
