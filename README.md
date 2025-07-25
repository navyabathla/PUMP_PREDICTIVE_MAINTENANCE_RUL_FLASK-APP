# Pump Predictive Maintenance – Remaining Useful Life (RUL) Prediction

This project predicts the **Remaining Useful Life (RUL)** of industrial pumps in hours using machine learning models.  
A **Random Forest regressor**, trained on sensor data, is deployed through a **Flask web interface** to provide real-time predictions.

---

## Features
- Predicts **Remaining Useful Life (RUL)** of pumps.
- Compares multiple ML models (Random Forest, Decision Trees, etc.) to find the best performer.
- Flask-based web app for user-friendly predictions.
- Visualizations and performance metrics:
  - **MAE:** 10.31
  - **RMSE:** 27.72
  - **R²:** 0.985
  - **SMAPE:** 9.23%

---

## Dataset
The dataset includes **sensor readings** such as pressure, temperature, vibration, and flow rate, collected over the operational lifetime of pumps.  
Datadet link - https://www.kaggle.com/datasets/anseldsouza/water-pump-rul-predictive-maintenance

---
## Model Training
The complete data preprocessing and model training code can be found in 
[pump_maintenance_prediction.ipynb] & [pump_maintenance_prediction_rul.ipynb]
---


## Model & Results
After testing multiple ML models, **Random Forest** achieved the best results:
- **MAE:** 10.31
- **RMSE:** 27.72
- **R²:** 0.985
- **MAPE:** 78.98%
- **SMAPE:** 9.23%

---

