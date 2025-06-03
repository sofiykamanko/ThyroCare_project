# ThyroCare
ThyroCare is an AI-based platform for predicting thyroid cancer risk using machine learning algorithms.

# Features:
- Predicts the likelihood of thyroid cancer (risk levels 1, 2, or 3).
- Differentiates between malignant and benign tumors.
- User-friendly interface for quick predictions.
- Models trained on imbalanced medical datasets for high accuracy.

## Models:
- `model_diag1.pkl`: Model for diagnosing thyroid cancer.
- `model_diag2.pkl`: Alternative model for diagnosing without considering countries and ethnicity.
- `model_risk.pkl`: Model for risk assessment.
- `scaler.pkl`: Scaler object for normalizing input data.

## Uploaded Files:
- `threshold_diag.npy`: Thresholds for diagnosing.
- `thresholds_risk.npy`: Thresholds for determining risk levels.

## Important Files:
- `thyrocare_app.py`: Main file for running the web application.

