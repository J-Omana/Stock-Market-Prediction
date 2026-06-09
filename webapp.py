import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pandas.tseries.offsets import BDay  # To skip weekends

st.title("Stock Market Price Predictor")

# Default stock ticker
stock = st.text_input("Enter the stock ticker:", "AAPL")

try:
    # Fetch stock data
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    stock_data = yf.download(stock, start, end)

    if stock_data.empty:
        st.error("⚠️ No data found for the given ticker. Please check the symbol and try again.")
        st.stop()

    st.subheader("Stock Data")
    st.write(stock_data)

    # Compute Moving Averages
    stock_data["MA_100"] = stock_data["Close"].rolling(window=100).mean()
    stock_data["MA_200"] = stock_data["Close"].rolling(window=200).mean()

    # Plot Moving Averages
    st.subheader("Moving Averages (100-day & 200-day)")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(stock_data["Close"], label="Closing Price", color="blue", alpha=0.5)
    ax.plot(stock_data["MA_100"], label="100-Day MA", color="orange", linestyle="dashed")
    ax.plot(stock_data["MA_200"], label="200-Day MA", color="red", linestyle="dashed")
    ax.legend()
    st.pyplot(fig)

    # Load pre-trained model (ensure the file exists in the working directory)
    model = load_model("lstm_xgb.keras")

    # Split data: use last 10% for testing
    splitting_len = int(len(stock_data) * 0.9)
    test_data = stock_data.Close[splitting_len:]

    st.subheader("Test Close Price")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(test_data, label="Test Data", color="blue")
    ax.legend()
    st.pyplot(fig)

    # Preprocess test data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test_data = scaler.fit_transform(test_data.values.reshape(-1, 1))

    # Create sequences using a sliding window of 100 days
    x_data, y_data = [], []
    for i in range(100, len(scaled_test_data)):
        x_data.append(scaled_test_data[i-100:i])
        y_data.append(scaled_test_data[i])
    x_data, y_data = np.array(x_data), np.array(y_data)

    # Generate predictions using the loaded model
    predictions = model.predict(x_data)

    # Inverse transform predictions and actual values to original scale
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_data = scaler.inverse_transform(y_data)

    # Create a DataFrame to compare predictions with actual values
    results = pd.DataFrame({
        "Original": inv_y_data.flatten(),
        "Predicted": inv_predictions.flatten()
    }, index=stock_data.index[splitting_len + 100:])
    st.subheader("Original vs Predicted Prices")
    st.write(results)

    # Plot the original and predicted prices
    st.subheader("Comparison Plot")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(results["Original"], label="Original Test Data", color="blue")
    ax.plot(results["Predicted"], label="Predicted Data", color="red")
    ax.legend()
    st.pyplot(fig)

    # Future Predictions
    st.subheader("Future Price Predictions")
    last_100 = stock_data[['Close']].tail(100)
    last_100_scaled = scaler.transform(last_100.values.reshape(-1, 1)).reshape(1, -1, 1)

    def predict_future(no_of_days, prev_100):
        future_predictions = []
        for _ in range(no_of_days):
            next_day = model.predict(prev_100)
            future_predictions.append(scaler.inverse_transform(next_day)[0][0])
            prev_100 = np.append(prev_100[:, 1:, :], [[next_day[0]]], axis=1)
        return future_predictions

    try:
        no_of_days = int(st.text_input("Enter the number of future days to predict:", "10"))
    except ValueError:
        st.error("⚠️ Please enter a valid integer for future days.")
        st.stop()

    future_results = predict_future(no_of_days, last_100_scaled)

    # Generate future dates skipping weekends using Business Days (BDay)
    future_dates = [datetime.now() + BDay(i) for i in range(1, no_of_days + 1)]
    future_dates = [date.strftime('%Y-%m-%d') for date in future_dates]

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Close Price": future_results
    })
    st.subheader("Predicted Future Prices Table")
    st.write("The predicted future closing prices for the next days are:")
    st.dataframe(future_df)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(future_results, marker="o", color="green", label="Predicted Close Prices")
    for i, value in enumerate(future_results):
        ax.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    ax.set_title("Future Close Prices")
    ax.set_xlabel("Days")
    ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)

    # Calculate Evaluation Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

    mse = mean_squared_error(inv_y_data, inv_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(inv_y_data, inv_predictions)
    mape = mean_absolute_percentage_error(inv_y_data, inv_predictions) * 100
    r2 = r2_score(inv_y_data, inv_predictions)

    mean_actual = np.mean(inv_y_data)
    mae_percent = (mae / mean_actual) * 100
    rmse_percent = (rmse / mean_actual) * 100
    r2_percent = r2 * 100
    accuracy = r2 * 100


    st.subheader("Model Accuracy Metrics")
    st.write(f"📊 **Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
    st.write(f"📏 **Mean Absolute Error (MAE):** {mae:.4f} ({mae_percent:.2f}%)")
    st.write(f"📈 **Root Mean Squared Error (RMSE):** {rmse:.4f} ({rmse_percent:.2f}%)")
    st.write(f"🔍 **R² Score:** {r2:.4f} ({r2_percent:.2f}%)")
    print("Overall prediction accuracy as % based on R²")
    st.write(f"✅ **Overall Accuracy:** {accuracy:.2f}%")

except Exception as e:
    st.error(f"An error occurred: {e}")
