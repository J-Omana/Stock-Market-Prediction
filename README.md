Stock Market Price Prediction using LSTM and XGBoost

Project Overview
This project proposes a **hybrid ensemble model** that integrates **Long Short-Term Memory (LSTM)** neural networks and **XGBoost** to predict stock market prices. 

- LSTM captures long-term temporal dependencies in financial time series.
- XGBoost enhances performance by modeling residual errors and reducing overfitting.
- The hybrid model improves prediction accuracy and generalization compared to traditional models.

This approach is evaluated against standalone models including **LSTM**, **GRU**, and **GRU+XGBoost**.

---

Models Used

LSTM (Long Short-Term Memory) 
  A type of RNN ideal for sequential and time-series data.

GRU (Gated Recurrent Unit)
  A lightweight alternative to LSTM with fewer parameters.

XGBoost (Extreme Gradient Boosting)
  A powerful gradient boosting algorithm used to refine residuals from LSTM/GRU predictions.

---

Dataset

The dataset includes:

1. **Historical stock price data** with:
   - `Open`, `High`, `Low`, `Close`, `Volume`
   - Technical indicators such as Moving Averages and RSI (Relative Strength Index)
   
2. **Time-stamped entries** suitable for training deep learning-based time series models.

---

Problem Statement

- Predict future stock prices using historical data.
- Reduce forecast error and overfitting.
- Provide robust real-time predictions for decision-making in financial trading.

---

Key Features

LSTM
- Learns sequential patterns.
- Effective for long-term time dependencies.

XGBoost
- Models the residual errors from LSTM predictions.
- Enhances accuracy and robustness.

Hybrid Model (LSTM + XGBoost)
- Outperforms LSTM, GRU, and GRU+XGBoost models.
- Achieves lower **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.


Business Value

- Assists investors and analysts with **data-driven decision-making**.
- Improves **risk management** through better forecasts.
- Provides **insightful trends** for developing trading strategies.



Conclusion

The **LSTM + XGBoost hybrid model** delivers enhanced forecasting performance by combining the sequential learning of LSTM with the boosting capabilities of XGBoost. 

Compared to standalone models, this hybrid:
- Achieves higher **generalization**
- Provides **more accurate predictions**
- Performs better than **GRU + XGBoost**



Future Work

- Incorporate additional **technical indicators** and **macroeconomic factors**.
- Explore **Transformer models** for further improvements.
- Apply **Explainable AI (XAI)** for better model transparency.
- Extend the model for **multi-stock** or **sector-level predictions**.

---

Technologies Used

| **Technology**        | **Purpose**                                  |
|------------------------|-----------------------------------------------|
| Python                | Programming and data analysis                 |
| NumPy / Pandas        | Data cleaning and manipulation                |
| Matplotlib / Seaborn  | Data visualization                            |
| TensorFlow / Keras    | Deep learning (LSTM, GRU models)              |
| XGBoost               | Residual modeling and ensemble learning       |
| Scikit-learn          | Preprocessing and model evaluation            |


