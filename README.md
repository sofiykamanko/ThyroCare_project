# ThyroCare
ThyroCare is an AI-based platform for predicting thyroid cancer risk using machine learning algorithms.

# Features:
- Predicts the likelihood of thyroid cancer (risk levels 1, 2, or 3).
- Differentiates between malignant and benign tumors.
- User-friendly interface for quick predictions.
- Models trained on imbalanced medical datasets for high accuracy.

## Models:
- `model_diag1.pkl`: Модель для діагностики раку щитоподібної залози.
- `model_diag2.pkl`: Альтернативна модель для оцінки діагнозу без врахування країн та етнічності.
- `model_risk.pkl`: Модель для оцінки ризику.
- `scaler.pkl`: Масштабуючий об'єкт для нормалізації вхідних даних.

## Uploaded Files:
- `threshold_diag.npy`: Пороги для визначення діагнозу.
- `thresholds_risk.npy`: Пороги для визначення рівня ризику.

## Important Files:
- `thyrocare_app.py`: Основний файл для запуску веб-додатку.

